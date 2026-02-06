# Exp3 = Perturbing a document with k good question in a untargeted manner (Denial of Service)
# Evaluation of the perturbed documents on n question-answer pairs

from tqdm import tqdm
from collections import defaultdict
import random
import logging

from .experiment_base import Experiment, Metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Exp3(Experiment):
    def __init__(self, n_questions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_questions = n_questions

        self.results = defaultdict(dict)

    def _extract_questions_answers(self, questions_answer_pairs:list):
        questions = []
        answers = []
        for question, answer in questions_answer_pairs:
            questions.append(question)
            # In case of multiple answers, for this attack just take the first one. 
            # At test time, we rule out the all the correct answers TODO
            answers.append(list(answer)[0]) 
        
        return questions, answers

    def create_adv_examples(self):
        dataset = self.data_loader

        for key, sample in tqdm(dataset.items(), total=len(dataset), desc="Processed"):
            image = sample["image"]
            # extract the GT
            if self.args.n_questions > 0:
                items = random.sample(list(sample["questions"].items()), self.args.n_questions)
            questions, targets = self._extract_questions_answers(items)

            advx = self.attack(model=self.model,
                              processor=self.processor,
                              auto_processor=self.autoprocessor, # TODO: remove autoprocessor, it is not needed
                              image=image,
                              questions=questions,
                              targets=targets,
                              is_targeted=False,
                              mask_function=self.mask_function,
                              args=self.args)
            
            assert image.size == advx.size
            
            advx.save(f"{self.res_directory}/{key}.jpg")

            y_pred = self.model.torch_predict(image, questions)
            y_pred_adv = self.model.torch_predict(advx, questions)
            
            assert key not in self.results.keys(), "Found sample in results"
            self.results[key]["questions"] = tuple(questions)
            self.results[key]["gt"] = tuple([sample["questions"][q] for q in questions]) # one question can have multiple answers
            self.results[key]["y_pred"] = tuple(y_pred)
            self.results[key]["y_pred_adv"] = tuple(y_pred_adv)

            print(f'y_pred={y_pred}, y_pred_adv={y_pred_adv}, GT={self.results[key]["gt"]}', flush=True)
            #self.results[key]["target"] = tuple(targets)
        
        self.save_results(self.results)


    def report(self):
        #asr, cdmg = 0, 0
        cdmg = 0
        metrics = Metrics() # TODO: now useful just for anls. include also asr and cdmg
        
        samples_count = len(self.results)

        logger.info('*********** Starting report ***********')
        for _, value in self.results.items():
            y_pred, y_pred_adv, gt = value['y_pred'], value['y_pred_adv'], value['gt']
            #asr += 1 if y_pred_adv == target else 0
            cdmg += 1 if y_pred_adv != y_pred else 0
            # print(f'y_pred_adv = {y_pred_adv}')
            # print(f'y_pred = {y_pred}')
            # print(f'target = {target}')
            # print(f'gt = {gt}')
            # print('******************')
            metrics.update_batch_anls(gt_batch=gt, pred_batch=y_pred_adv)
        
        #asr /= samples_count
        cdmg /= samples_count
        logger.info('*********** Report ***********')
        logger.info(f'# Number of samples = {samples_count}')
        #logger.info(f'# ASR = {asr}')
        logger.info(f'# CDMG = {cdmg}')
        logger.info(f"# ANLS: {metrics.get_anls() * 100:.2f}%")
        logger.info('******************************')