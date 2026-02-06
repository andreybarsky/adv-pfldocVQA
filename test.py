import sys
import os
import logging
import random
import numpy as np
import torch
from abc import ABC, abstractmethod
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration, AutoModelForVision2Seq
from config import MODEL_NAMES
from dataset import Imdb_Dataloader
from tqdm import tqdm
import editdistance

def get_exp(name:str, args):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}. Available models: {MODEL_NAMES}")

    if name == 'pix2struct':
        model_name = "google/pix2struct-docvqa-base"
        auto_processor = AutoProcessor.from_pretrained(model_name)
        model = (Pix2StructForConditionalGeneration.from_pretrained(model_name)).to(args.device)
        exp = Exp_Pix2Struct(model=model, processor=auto_processor, args=args)
    elif name == 'donut':
        model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
        auto_processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name).to(args.device)
        exp = Exp_Donut(model=model, processor=auto_processor, args=args)

    return exp

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import editdistance


class Evaluator:
    def __init__(self, case_sensitive=False):

        self.case_sensitive = case_sensitive
        self.get_edit_distance = editdistance.eval
        self.anls_threshold = 0.5

        self.total_accuracies = []
        self.total_anls = []

        self.best_accuracy = 0
        # self.best_anls = 0
        self.best_epoch = 0

    def get_metrics(self, gt_answers, preds, answer_types=None, update_global_metrics=True):
        answer_types = answer_types if answer_types is not None else ['string' for batch_idx in range(len(gt_answers))]
        batch_accuracy = []
        batch_anls = []
        for batch_idx in range(len(preds)):
            gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[batch_idx]]
            pred = self._preprocess_str(preds[batch_idx])

            batch_accuracy.append(self._calculate_accuracy(gt, pred, answer_types[batch_idx]))
            batch_anls.append(self._calculate_anls(gt, pred, answer_types[batch_idx]))

        # if accumulate_metrics:
        #     self.total_accuracies.extend(batch_accuracy)
        #     self.total_anls.extend(batch_anls)

        return {'accuracy': batch_accuracy, 'anls': batch_anls}

    def get_retrieval_metric(self, gt_answer_page, pred_answer_page):
        retrieval_precision = [1 if gt == pred else 0 for gt, pred in zip(gt_answer_page, pred_answer_page)]
        return retrieval_precision

    def update_global_metrics(self, accuracy, anls, current_epoch):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = current_epoch
            return True

        else:
            return False

    def _preprocess_str(self, string):
        if not self.case_sensitive:
            string = string.lower()

        return string.strip()

    def _calculate_accuracy(self, gt, pred, answer_type):

        if answer_type == 'not-answerable':
            return 1 if pred in ['', 'none', 'NA', None, []] else 0

        if pred == 'none' and answer_type != 'not-answerable':
            return 0

        for gt_elm in gt:
            if gt_elm == pred:
                return 1

        return 0

    def _calculate_anls(self, gt, pred, answer_type):
        if len(pred) == 0:
            return 0

        if answer_type == 'not-answerable':
            return 1 if pred in ['', 'none', 'NA', None, []] else 0

        if pred == 'none' and answer_type != 'not-answerable':
            return 0

        answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
        max_similarity = max(answers_similarity)

        anls = max_similarity if max_similarity >= self.anls_threshold else 0
        return anls


class Exp_Test(ABC):
    RESULTS_FOLDER = 'results_test_models'

    def __init__(self, model, processor, args):
        self.model = model
        self.processor = processor
        self.args = args
        self.max_out_tokens = 50  # Max tokens for generation
        self.evaluator = Evaluator(case_sensitive=False) 
        
        self.global_accuracies = []
        self.global_anls = []

    def set_seed(self, seed:int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup(self):
        self.set_seed(self.args.seed)

        # Prepare the results' directory
        self.res_directory = (f'{Exp_Test.RESULTS_FOLDER}/{self.__class__.__name__}/{self.model.__class__.__name__}/')
        if not os.path.exists(self.res_directory):
            os.makedirs(self.res_directory)

    def test(self, dataset):
        results = {}

        logger.info(f"Testing {self.model.__class__.__name__} with {len(dataset)} samples.")
        dataset = dict(list(dataset.items())[:5])
        for key, sample in tqdm(dataset.items(), total=len(dataset), desc="Tested"):
            image = sample["image"]

            for question in sample["questions"]:
                # Process the image and question
                inputs = self.prepare_inputs(image=image, question=question)

                raw_answer = self.generate_answer(inputs)
                answer = self.clean_answer(raw_answer, question)

                # Compute accuracy and ANLS
                ground_truth = sample["questions"][question]

                batch_gt_answers = [ground_truth]   
                batch_preds = [answer]

                metrics_batch = self.evaluator.get_metrics(
                    gt_answers=batch_gt_answers, 
                    preds=batch_preds
                )
                
                acc_val = metrics_batch['accuracy'][0]
                anls_val = metrics_batch['anls'][0]
                
                self.global_accuracies.append(acc_val)
                self.global_anls.append(anls_val)

        self.report()
        #self.save_results(results)

    def report(self):
        final_acc = sum(self.global_accuracies) / len(self.global_accuracies)
        final_anls = sum(self.global_anls) / len(self.global_anls)

        logger.info("Test completed.")
        logger.info(f"Accuracy: {final_acc * 100:.2f}%")
        logger.info(f"ANLS: {final_anls * 100:.2f}%")
    
    def save_results(self, results):
        import pickle
        filename = f"results_{self.model.__class__.__name__}.pkl"
        results_path = os.path.join(self.res_directory, filename)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

    @abstractmethod
    def prepare_inputs(self, image, question):
        pass

    @abstractmethod
    def generate_answer(self, inputs):
        pass
    
    @abstractmethod
    def clean_answer(self, raw_answer, question):
        pass

class Exp_Pix2Struct(Exp_Test):
    def __init__(self, model, processor, args):
        super().__init__(model, processor, args)

    def prepare_inputs(self, image, question):
        return self.processor(images=image, text=question, return_tensors="pt")

    def generate_answer(self, inputs):
        inputs = inputs.to(self.args.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_out_tokens)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def clean_answer(self, raw_answer, question=None):
        return raw_answer.strip()

class Exp_Donut(Exp_Test):
    def __init__(self, model, processor, args):
        super().__init__(model, processor, args)
    
    def get_task_prompt(self, question):
        return f"<s_docvqa><s_question>{question}</s_question><s_answer>"

    def prepare_inputs(self, image, question):
        prompt = self.get_task_prompt(question)
        return self.processor(image, prompt, return_tensors="pt")

    def generate_answer(self, inputs):
        input_ids = inputs.input_ids.to(self.args.device)
        pixel_values = inputs.pixel_values.to(self.args.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_length=self.max_out_tokens
        )
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def clean_answer(self, raw_answer, question):
        if question in raw_answer:
            return raw_answer.split(question)[-1].strip()
        return raw_answer.strip()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test models on document understanding tasks.")
    parser.add_argument('--model', type=str, default="donut", help='Model to test (pix2struct or donut)')
    parser.add_argument('--dataset_path', type=str, default="/home-local/mpintore/adv_docVQA/utils/advdoc_data_nsampl1000_nqst5.pkl", help='Path to the dataset')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use (cuda:index or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility') 
    args = parser.parse_args()
    print(args.__dict__, flush=True)
    return args

if __name__ == '__main__':
    args = parse_args()
    data_loader = Imdb_Dataloader(args.dataset_path).load_data()

    logger.info(f'Testing model {args.model} on dataset {args.dataset_path}')

    exp = get_exp(args.model, args)
    exp.setup()
    exp.test(dataset=data_loader)