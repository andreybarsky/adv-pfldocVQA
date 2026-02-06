"""Custom attack for Pix2Struct."""

import configparser
import math

import numpy as np
import torch
from PIL import Image
from secmlt.adv.evasion.modular_attack import ModularEvasionAttackFixedEps
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.constraints import (
    LInfConstraint,
    MaskConstraint,
)
from secmlt.optimization.gradient_processing import (
    LinearProjectionGradientProcessing,
)
from secmlt.optimization.initializer import Initializer
from torch.utils.data import DataLoader, TensorDataset
from transformers.models.pix2struct.image_processing_pix2struct import render_header

from attacks.constraints import QuantizationConstraintWithMask

from models.processing.pix2struct_processor import Pix2StructImageProcessor
from models.pix2struct import Pix2StructModel, Pix2StructModelProcessor
from transformers.image_utils import to_numpy_array
from typing import Union, List
from .masks import mask_include_all
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import Manipulation
from secmlt.models.base_model import BaseModel
from secmlt.optimization.constraints import Constraint
from secmlt.optimization.gradient_processing import GradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.trackers.trackers import Tracker
from secmlt.utils.tensor_utils import atleast_kd
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from typing import Literal, Union
from functools import partial

CE_LOSS = "ce_loss"
LOGIT_LOSS = "logit_loss"

LOSS_FUNCTIONS = {
    CE_LOSS: CrossEntropyLoss,
}

class LoRaPGDAttack(ModularEvasionAttackFixedEps):
    def __init__(
        self,
        rank: int,
        y_target: int | None,
        num_steps: int,
        step_size: float,
        loss_function: Union[str, torch.nn.Module],
        optimizer_cls: str | partial[Optimizer],
        manipulation_function: Manipulation,
        initializer: Initializer,
        gradient_processing: GradientProcessing,
        trackers: list[Tracker] | Tracker | None = None,
    ) -> None:
        self.rank = rank
        self.y_target = y_target
        self.num_steps = num_steps
        self.step_size = step_size
        self.trackers = trackers
        if isinstance(loss_function, str):
            if loss_function in LOSS_FUNCTIONS:
                self.loss_function = LOSS_FUNCTIONS[loss_function](reduction="none")
            else:
                msg = (
                    f"Loss function not found. Use one among {LOSS_FUNCTIONS.values()}"
                )
                raise ValueError(msg)
        else:
            self.loss_function = loss_function

        if isinstance(optimizer_cls, str):
            optimizer_cls = OptimizerFactory.create_from_name(
                optimizer_cls,
                lr=step_size,
            )

        self.optimizer_cls = optimizer_cls

        self._manipulation_function = manipulation_function
        self.initializer = initializer
        self.gradient_processing = gradient_processing

        super().__init__(
            y_target=y_target,
            num_steps=num_steps,
            step_size=step_size,
            loss_function=loss_function,
            optimizer_cls=optimizer_cls,
            manipulation_function=manipulation_function,
            initializer=initializer,
            gradient_processing=gradient_processing,
            trackers=trackers
        )

    @property
    def manipulation_function(self) -> Manipulation:
        """
        Get the manipulation function for the attack.

        Returns
        -------
        Manipulation
            The manipulation function used in the attack.
        """
        return self._manipulation_function

    @manipulation_function.setter
    def manipulation_function(self, manipulation_function: Manipulation) -> None:
        """
        Set the manipulation function for the attack.

        Parameters
        ----------
        manipulation_function : Manipulation
            The manipulation function to be used in the attack.
        """
        self._manipulation_function = manipulation_function

    @classmethod
    def get_perturbation_models(cls) -> set[str]:
        """
        Check if a given perturbation model is implemented.

        Returns
        -------
        set[str]
            Set of perturbation models available for this attack.
        """
        return {
            LpPerturbationModels.L1,
            LpPerturbationModels.L2,
            LpPerturbationModels.LINF,
        }

    @classmethod
    def _trackers_allowed(cls) -> Literal[True]:
        return True

    def _init_perturbation_constraints(self) -> list[Constraint]:
        msg = "Must be implemented accordingly"
        raise NotImplementedError(msg)

    def _create_optimizer(self, delta: torch.Tensor, **kwargs) -> Optimizer:
        return self.optimizer_cls([delta], lr=self.step_size, **kwargs)

    def forward_loss(
        self, model: BaseModel, x: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the forward for the loss function.

        Parameters
        ----------
        model : BaseModel
            Model used by the attack run.
        x : torch.Tensor
            Input sample.
        target : torch.Tensor
            Target for computing the loss.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output scores and loss.
        """
        scores = model.decision_function(x)
        target = target.to(scores.device)
        losses = self.loss_function(scores, target)
        return scores, losses

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        init_deltas: torch.Tensor = None,
        optim_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if optim_kwargs is None:
            optim_kwargs = {}
        multiplier = 1 if self.y_target is not None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)

        import torch.nn as nn
        from torch.utils.checkpoint import checkpoint
        model._model.eval()
        for param in model._model.parameters():
            param.requires_grad = False
        # LoRA matrixes
        samples = samples.detach()
        b,h,w,c = samples.shape
        delta_B = torch.zeros(b, c, w,self.rank, device=samples.device, requires_grad=True)
        delta_A = torch.randn(b, c, self.rank ,h, device=samples.device, requires_grad=True)
        nn.init.kaiming_uniform_(delta_A, a=math.sqrt(5))

        optimizer = self.optimizer_cls([delta_B, delta_A], lr=self.step_size, **optim_kwargs)
        best_loss = torch.zeros(1).fill_(torch.inf)
        best_delta_B = torch.zeros_like(delta_B)
        best_delta_A = torch.zeros_like(delta_A)

        processor = Pix2StructImageProcessor()
        for i in range(self.num_steps):
            # delta = delta_B  @ delta_A
            # delta = torch.permute(delta, (0,3,2,1))
            #x_adv = samples + (delta_B  @ delta_A).permute((0,3,2,1))
            x_header, _, _ = processor.my_preprocess_image(samples + (delta_B  @ delta_A).permute((0,3,2,1)),
                                                           "How much is the total?")
            patches = self.processor.extract_flattened_patches(x_header)
            _, losses = model.loss_fn(patches.unsqueeze(0), torch.tensor([[500, 4,  4, 5, 8, 7,7,7,1]]))
            #scores, losses = self.forward_loss(model=model, x=x_adv, target=target)
            current_loss = (losses.squeeze()*multiplier)
            print('loss = ', losses.item())
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            print('GRAD =',delta_A.grad)
            # track the loss
            loss_val = losses.squeeze().item()
            if loss_val < best_loss:
                best_delta_B.data.copy_(delta_B.data)
                best_delta_A.data.copy_(delta_A.data)
        
        with torch.no_grad():
            final_best_delta = best_delta_B @ best_delta_A
            x_adv, final_best_delta = self.manipulation_function(samples.data,final_best_delta)
        return x_adv, final_best_delta

class E2EPix2StructAttack(LoRaPGDAttack):
    """Attack implementing a custom forward for Pix2Stuct."""
    def __init__(self, 
                 auto_processor:Pix2StructModelProcessor,
                 processor:Pix2StructImageProcessor, 
                 questions: Union[str, List[str]], 
                 *args, 
                 **kwargs):
        super().__init__(rank=1,*args, **kwargs)
        self.auto_processor = auto_processor
        self.processor = processor
        self.questions = questions
    
    def forward_loss(self, model:Pix2StructModel, x, target):
        # Reconstruct is necessary because multiple questions are
        # concatenated with a separator in an unique tensor
        x = x.detach()
        targets = self.auto_processor.reconstruct_targets(target)
        
        losses = []
        for question, answer in zip(self.questions, targets):
            # Preprocess and extract patches
            x_header, _, _ = self.processor.my_preprocess_image(x, question)
            patches = self.processor.extract_flattened_patches(x_header)
            _, loss = model.loss_fn(patches.unsqueeze(0), answer.unsqueeze(0))
            losses.append(loss)
        
        total_loss = torch.vstack(losses).mean(dim=0)
        
        return None, total_loss.mean()

def test_processing(image, question):
    # get an instance of our processor
    processor = Pix2StructImageProcessor()
    output = processor.preprocess(image=image, header_text=question, requires_grad=True)
    
    # get an instance of auto processor
    auto_processor = Pix2StructModelProcessor()
    output_auto = auto_processor(image, question)["flattened_patches"].squeeze(0)

    return torch.equal(output, output_auto)

def test_grad1(
        image_pil:Image, 
        question
        ):
    processor = Pix2StructImageProcessor()

    image = image_pil.convert("RGB")
    image_np = np.asarray(image).astype(np.float32)
    image_np = render_header(image_np, question)
    image_np = processor.normalize(image_np)
    input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
    input_tensor.requires_grad = True

    output = processor.extract_flattened_patches(input_tensor, requires_grad=True)
    loss = output.sum()
    loss.backward()

    print("Gradients on input:", input_tensor.grad)

def test_grad2(
        image_pil:Image, 
        question
        ):
    processor = Pix2StructImageProcessor()
    image, patches = processor.preprocess(image=image_pil, header_text=question, requires_grad=True)
    loss = patches.sum()
    loss.backward()
    print("Gradients on inputs: ", image.grad)

def questions_target_validation(questions, targets):
    assert (isinstance(questions, str) and isinstance(targets, str)) or \
           (isinstance(questions, list) and isinstance(targets, list) and len(questions) == len(targets)) or \
           (isinstance(questions, list) or isinstance(questions, str) and targets is None), \
           "Both questions and targets must either be both strings or both lists of the same length."
    
    if isinstance(questions, str) and isinstance(targets, str):
        questions, targets = [questions], [targets]
    
    return questions, targets

def lora_attack_pix2struct(
        model: Pix2StructModel, 
        processor: Pix2StructImageProcessor,
        auto_processor: Pix2StructModelProcessor, # TODO remove and write only one processor
        image, 
        questions, 
        targets, 
        args,
        is_targeted=True,
        mask_function=mask_include_all
        ):
    """Compute adversarial perturbation for Pix2Struct model."""
    questions, targets = questions_target_validation(questions, targets)

    input_tensor = torch.tensor(to_numpy_array(image).astype(np.float32))
    labels = auto_processor.get_input_ids(targets)

    perturbation_mask = mask_function(input_tensor)

    perturbation_constraints = [
        MaskConstraint(mask=perturbation_mask),
        LInfConstraint(radius=float(args.eps)),
    ]
    domain_constraints = [
        QuantizationConstraintWithMask(
            mask=perturbation_mask.unsqueeze(0),
            levels=torch.arange(0, 256),
        ),
    ]
    gradient_processing = LinearProjectionGradientProcessing(LpPerturbationModels.LINF)
    
    attack = E2EPix2StructAttack(
        y_target=labels if is_targeted else None,
        num_steps=int(args.steps),
        step_size=float(args.step_size),
        loss_function=model.loss_fn,
        optimizer_cls=torch.optim.Adam,
        manipulation_function=AdditiveManipulation(
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
        ),
        initializer=Initializer(),
        gradient_processing=gradient_processing,
        auto_processor=auto_processor, # TODO: refactor and make only one processor
        processor = processor,
        questions=questions
    )
    test_loader = DataLoader(TensorDataset(input_tensor.unsqueeze(0), labels))

    native_adv_ds = attack(model, test_loader)
    advx, _ = next(iter(native_adv_ds))

    advx = advx.squeeze(0)
    np_img = advx.clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    
    return Image.fromarray(np_img)
