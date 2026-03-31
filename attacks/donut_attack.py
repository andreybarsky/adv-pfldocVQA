"""Custom attack for Pix2Struct."""

import numpy as np
import torch
from PIL import Image
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
# from secmlt.manipulations.manipulation import AdditiveManipulation

from secmlt.optimization.constraints import (
    LInfConstraint,
    MaskConstraint,
    ClipConstraint
)
from secmlt.optimization.gradient_processing import (
    LinearProjectionGradientProcessing,
)
from secmlt.optimization.initializer import Initializer
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoProcessor
from attacks.constraints import  QuantizationConstraintWithMask
from models.processing.donut_processor import DonutImageProcessor
from models.donut import DonutModel, DonutModelProcessor
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import to_numpy_array
from typing import Union, List
import torch.nn.utils.rnn as rnn_utils
from .masks import mask_include_all, mask_robust_sobel

from .modular_attack_grad_accumulation import ModularEvasionAttackFixedEps

from secmlt.trackers import LossTracker, PredictionTracker, PerturbationNormTracker, GradientNormTracker, TensorboardTracker

from jnd.perceptual import AdditiveManipulation, PerceptualAdditiveManipulation, PerPixelSymmetricConstraint


class DonutAttack(ModularEvasionAttackFixedEps):
    """Attack implementing a custom forward for Pix2Stuct."""
    def __init__(self,
                 auto_processor: DonutModelProcessor,
                 processor: DonutImageProcessor,
                 questions: Union[str, List[str]],
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_processor = auto_processor
        self.processor = processor
        self.questions = questions

    def forward_loss(self, model: DonutModel, 
                     x, # input tensor version of pil image
                     target, # used for loss calculation and teacher forcing
                    ):
        """Compute the loss for the given input."""
        x.requires_grad_(True)
        # Reconstruct is necessary because multiple task-prompts are
        # concatenated with a separator in an unique tensor
        targets = self.auto_processor.reconstruct_targets(target)

        total_loss = 0
        
        for y, question in zip(targets, self.questions):
            x_preproc = self.processor(x.squeeze(0))
            x_input = x_preproc["pixel_values"][0]
            answer_logits, loss = model.loss_fn(x_input.unsqueeze(0), y.unsqueeze(0), question) # teacher-forced!
            loss /= len(self.questions)
            loss.backward()
            total_loss += loss.item()

            output_ids = answer_logits.argmax(dim=2)
            output_scores = answer_logits.max(dim=2).values
            


        
        return output_scores, total_loss

def test_torch_transformation(image):
    # instatiate torchified processor and auto processor
    torch_processor = DonutImageProcessor()
    auto_processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = AutoModelForVision2Seq.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

    # prepare inputs
    question = "How much is the total?"
    task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    inputs = torch_processor.preprocess(image)
    auto_inputs = auto_processor(image, task_prompt, return_tensors="pt")
    labels = auto_inputs["labels"]

    # try inference
    outputs = model.generate(
        input_ids=labels,
        pixel_values=inputs["pixel_values"].unsqueeze(0),
        max_length=50
    )

    # finally compare torch_processor and auto_processor outputs
    answer = auto_processor.decode(outputs[0], skip_special_tokens=True)
    if question in answer:
        answer = answer.split(question)[-1].strip()
    print(answer)

# TODO: put in an utils file    
def questions_target_validation(questions, targets):
    assert (isinstance(questions, str) and isinstance(targets, str)) or \
           (isinstance(questions, list) and isinstance(targets, list) and len(questions) == len(targets)) or \
           (isinstance(questions, list) or isinstance(questions, str) and targets is None), \
           "Both questions and targets must either be both strings or both lists of the same length."
    
    if isinstance(questions, str) and isinstance(targets, str):
        questions, targets = [questions], [targets]
    
    return questions, targets

def attack_donut(
        model, 
        processor: DonutImageProcessor,
        auto_processor: DonutModelProcessor, 
        image, 
        questions, 
        targets,
        args,
        is_targeted=True,
        mask_function=mask_include_all,
        ):
    """Compute adversarial perturbation for Donut model."""
    questions, targets = questions_target_validation(questions, targets)

    input_tensor = torch.tensor(to_numpy_array(image).astype(np.float32), requires_grad=True)
    labels = auto_processor.get_input_ids(targets)


    gradient_processing = LinearProjectionGradientProcessing(LpPerturbationModels.LINF)

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



    
    if args.soft_mask:
        # soft mask additionally clamps the domain-space perturbation to be a fraction of the fixed epsilon

        assert mask_function == mask_robust_sobel # only implemented for this function for now
        
        # the hard mask itself stays boolean so we don't break the quantization by accident:
        # the soft mask is the same but without thresholding applied:
        soft_mask_kwargs = {k:v for k,v in args.mask_kwargs.items() if k!='threshold'}
        perturbation_mask = mask_function(input_tensor, threshold=0, **soft_mask_kwargs)

        
        soft_perturbation_mask = mask_function(input_tensor, threshold=None, **soft_mask_kwargs)
    
        perturbation_constraints = [
            MaskConstraint(mask=perturbation_mask),
            # LInfConstraint(radius=float(args.eps)),
            PerPixelSymmetricConstraint(soft_perturbation_mask, eps=float(args.eps)),
            ]
        
    else:
        # hard mask, acts as the boolean surface for the fixed epsilon threshold
        perturbation_mask = mask_function(input_tensor, **args.mask_kwargs)
    
        perturbation_constraints = [
            MaskConstraint(mask=perturbation_mask),
            LInfConstraint(radius=float(args.eps)),
            ]

    print(f'Setting up QuantizationConstraintWithMask using perturbation_mask of type: {perturbation_mask.dtype}, shape: {perturbation_mask.shape} and L0 mean: {perturbation_mask.bool().float().mean()}')
    domain_constraints = [
            # ClipConstraint(0, 255), # is this needed??
            QuantizationConstraintWithMask(
                mask=perturbation_mask.unsqueeze(0),
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

    print(f'Initialising attack with scheduler: {args.scheduler}')
    attack = DonutAttack(
        y_target=labels if is_targeted else None,
        num_steps=int(args.steps),
        step_size=float(args.step_size),
        loss_function=model.loss_fn,
        optimizer_cls=torch.optim.Adam,
        scheduler_cls=args.scheduler,
        manipulation_function=manip_func,
        initializer=Initializer(),
        gradient_processing=gradient_processing,
        auto_processor=auto_processor,
        processor = processor,
        questions = questions,
        trackers = tensorboard_tracker,
    )
    test_loader = DataLoader(TensorDataset(input_tensor.unsqueeze(0), labels))

    native_adv_ds = attack(model, test_loader)
    advx, best_delta, target_answer_tokens = next(iter(native_adv_ds))
    # target_answer_str = model.model_processor.decode(*target_answer_tokens)
    # print(f'target_answer: {target_answer_str}')

    # advx = advx.squeeze(0)
    # np_img = advx.clamp(0, 255).detach().cpu().numpy()

    np_img = advx.squeeze(0).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    adv_image = Image.fromarray(np_img)

    # adv_image = Image.fromarray(np_img / 255)
    
    return adv_image, best_delta #, target_answer_str