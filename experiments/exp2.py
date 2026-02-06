# Exp2 = Perturbing a document with k multiple pair of questions and targeted answers
# Evaluation of the perturbed documents on n-k questions-answer pairs

from tqdm import tqdm
import random
from collections import defaultdict
import logging

from .experiment_base import Experiment, Metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Exp2(Experiment):
    def __init__(self, n_questions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_questions = n_questions
        assert len(self.args.target_answer) == self.n_questions , "The number of questions (self.n_questions) must match the number of target answers"

        self.results = defaultdict(dict)

    def create_adv_examples(self):
        dataset = self.data_loader
        # from itertools import islice
        # dataset = dict(islice(dataset.items(), 5))
        for key, sample in tqdm(dataset.items(), total=len(dataset), desc="Processed"):
            image = sample["image"]
            targets = self.args.target_answer
            questions = random.sample(list(sample["questions"].keys()),k=self.n_questions)

            advx = self.attack(model=self.model,
                              processor=self.processor,
                              auto_processor=self.autoprocessor, # TODO: remove autoprocessor, it is not needed
                              image=image,
                              questions=questions,
                              targets=targets,
                              is_targeted=True,
                              mask_function=self.mask_function,
                              args=self.args)
            
            assert image.size == advx.size
            
            # save adv example
            advx.save(f"{self.res_directory}/{key}.jpg")

            # update the predictions, ground truths and targets
            y_pred = self.model.torch_predict(image, questions)
            y_pred_adv = self.model.torch_predict(advx, questions)
            print(y_pred, y_pred_adv, flush=True)

            assert key not in self.results.keys(), "Found sample in results"
            self.results[key]["questions"] = tuple(questions)
            self.results[key]["gt"] = tuple([sample["questions"][q] for q in questions]) # one question can have multiple answers
            self.results[key]["y_pred"] = tuple(y_pred)
            self.results[key]["y_pred_adv"] = tuple(y_pred_adv)
            self.results[key]["target"] = tuple(targets)
        
        # save the results
        self.save_results(self.results)

    def report(self):
        asr, cdmg = 0, 0
        metrics = Metrics() # TODO: now useful just for anls. include also asr and cdmg
        
        samples_count = len(self.results)

        logger.info('*********** Starting report ***********')
        for _, value in self.results.items():
            y_pred, y_pred_adv ,target, gt = value['y_pred'], value['y_pred_adv'], value['target'], value['gt']
            asr += 1 if y_pred_adv == target else 0
            cdmg += 1 if y_pred_adv != y_pred else 0
            # print(f'y_pred_adv = {y_pred_adv}')
            # print(f'y_pred = {y_pred}')
            # print(f'target = {target}')
            # print(f'gt = {gt}')
            # print('******************')
            metrics.update_batch_anls(gt_batch=gt, pred_batch=y_pred_adv)
        
        asr /= samples_count if samples_count != 0 else 0
        cdmg /= samples_count if samples_count != 0 else 0
        logger.info('*********** Report ***********')
        logger.info(f'# Number of samples = {samples_count}')
        logger.info(f'# ASR = {asr}')
        logger.info(f'# CDMG = {cdmg}')
        logger.info(f"# ANLS: {metrics.get_anls() * 100:.2f}%")
        logger.info('******************************')