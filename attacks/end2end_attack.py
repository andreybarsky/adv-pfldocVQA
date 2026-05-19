"""Custom attack for Pix2Struct."""

import configparser
import math

import numpy as np
import torch
from PIL import Image
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

# custom modular evasion attack definition
from .modular_attack_grad_accumulation import ModularEvasionAttackFixedEps

class E2EPix2StructAttack(ModularEvasionAttackFixedEps):
    """Attack implementing a custom forward for Pix2Stuct."""
    def __init__(self, 
                 auto_processor:Pix2StructModelProcessor,
                 processor:Pix2StructImageProcessor, 
                 questions: Union[str, List[str]], 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_processor = auto_processor
        self.processor = processor
        self.questions = questions
    
    def forward_loss(self, model:Pix2StructModel, x, target):
        x.requires_grad_(True)

        # Reconstruct is necessary because multiple questions are
        # concatenated with a separator in an unique tensor
        targets = self.auto_processor.reconstruct_targets(target)
        
        total_loss = 0
        for question, answer in zip(self.questions, targets):
            # Preprocess and extract patches
            x_header, _, _ = self.processor.my_preprocess_image(x, question)
            patches = self.processor.extract_flattened_patches(x_header)
            answer_logits, loss = model.loss_fn(patches.unsqueeze(0), answer.unsqueeze(0))
            loss /= len(self.questions)
            loss.backward()
            total_loss += loss.item()

            output_ids = answer_logits["logits"].argmax(dim=2)
            output_scores = answer_logits["logits"].max(dim=2).values
            
            
        #total_loss = torch.vstack(losses).mean(dim=0)
        
        return output_scores, total_loss

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

def e2e_attack_pix2struct(
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

    input_tensor = torch.tensor(to_numpy_array(image).astype(np.float32), requires_grad=True)
    labels = auto_processor.get_input_ids(targets)


######
    if args.tracking:
        trackers = [LossTracker(),
                       PredictionTracker(),
                       # PerturbationNormTracker("linf"),
                       PerturbationNormTracker("l1"),
                       GradientNormTracker(),]
        tensorboard_tracker = TensorboardTracker("sec_logs/", trackers)

    else:
        trackers = []
        tensorboard_tracker = None

    gradient_processing = LinearProjectionGradientProcessing(LpPerturbationModels.LINF)


    
    if args.soft_mask:
        # soft mask additionally clamps the domain-space perturbation to be a fraction of the fixed epsilon

        assert mask_function == mask_robust_sobel # only implemented for this function for now
        
        # the hard mask itself stays boolean so we don't break the quantization by accident:
        # the soft mask is the same but without thresholding applied:
        soft_mask_kwargs = {k:v for k,v in args.mask_kwargs.items() if k!='threshold'}

        print(f' ++ creating HARD MASK')
        perturbation_mask = mask_function(input_tensor, threshold=0, **soft_mask_kwargs)

        print(f' ++ creating SOFT MASK')
        soft_perturbation_mask = mask_function(input_tensor, threshold=None, **soft_mask_kwargs)
    
        perturbation_constraints = [
            MaskConstraint(mask=perturbation_mask),
            # LInfConstraint(radius=float(args.eps)),
            PerPixelSymmetricConstraint(soft_perturbation_mask, eps=float(args.eps)),
            ]
        qc_mask = None
        
    else:
        # hard mask, acts as the boolean surface for the fixed epsilon threshold
        perturbation_mask = mask_function(input_tensor, **args.mask_kwargs)
        # qc_mask = perturbation_mask.unsqueeze(0)
        qc_mask = None # for perceptual attack we must re-quantize after the lab-rgb conversion
        perturbation_constraints = [
            MaskConstraint(mask=perturbation_mask),
            LInfConstraint(radius=float(args.eps)),
            ]

    print(f'Setting up QuantizationConstraintWithMask using perturbation_mask of type: {perturbation_mask.dtype}, shape: {perturbation_mask.shape} and L0 mean: {perturbation_mask.bool().float().mean()}')
    domain_constraints = [
            # ClipConstraint(0, 255), # is this needed??
            QuantizationConstraintWithMask(
                # mask=perturbation_mask.unsqueeze(0),
                mask=qc_mask,
                levels=torch.arange(0, 256),
            ),
        ]    
    
    if args.perceptual:
        manip_func = PerceptualAdditiveManipulation(
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
            x_clean_rgb = input_tensor,
        )
    else:
        manip_func = AdditiveManipulation(
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
        )
######
    
    # perturbation_mask = mask_function(input_tensor)

    # perturbation_constraints = [
    #     MaskConstraint(mask=perturbation_mask),
    #     LInfConstraint(radius=float(args.eps)),
    # ]
    # domain_constraints = [
    #     QuantizationConstraintWithMask(
    #         mask=perturbation_mask.unsqueeze(0),
    #         levels=torch.arange(0, 256),
    #     ),
    # ]
    print(f'Initialising attack with scheduler: {args.scheduler}')
    
    attack = E2EPix2StructAttack(
        y_target=labels if is_targeted else None,
        num_steps=int(args.steps),
        step_size=float(args.step_size),
        loss_function=model.loss_fn,
        optimizer_cls=torch.optim.Adam,
        # manipulation_function=AdditiveManipulation(
        #     domain_constraints=domain_constraints,
        #     perturbation_constraints=perturbation_constraints,
        # ),
        scheduler_cls=args.scheduler,
        manipulation_function=manip_func,        
        initializer=Initializer(),
        gradient_processing=gradient_processing,
        auto_processor=auto_processor, # TODO: refactor and make only one processor
        processor = processor,
        questions=questions
    )

###

    print(f"{input_tensor.shape=}")
    test_loader = DataLoader(TensorDataset(input_tensor.unsqueeze(0), labels))

    native_adv_ds = attack(model, test_loader)
    # advx, _ = next(iter(native_adv_ds))
    advx, best_delta, target_answer_tokens = next(iter(native_adv_ds))


    advx = advx.squeeze(0)
    np_img = advx.clamp(0, 255).detach().cpu().numpy().astype(np.uint8)

    adv_image = Image.fromarray(np_img)


    
    # adv_image = Image.fromarray(np_img / 255)
    
    return adv_image, best_delta #, target_answer_str    
    
    # return Image.fromarray(np_img)
