from tqdm import tqdm
from collections import defaultdict
import logging

from .experiment_base import Experiment
import pickle
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Exp4(Experiment):
    def __init__(self, n_questions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_questions = n_questions
        assert len(self.args.target_answer) == self.n_questions , "The number of questions (self.n_questions) must match the number of target answers"

        self.results = defaultdict(dict)

        import os
        self.advx_path = f'/home-local/mpintore/recovery_2109/adv_docVQA/resultsACCUMULATION/Exp2/{self.n_questions}-{self.args.mask}-{self.model.__class__.__name__}'
        print('ADVX path = ',self.advx_path)
        self.files = {f for f in os.listdir(self.advx_path) if os.path.isfile(os.path.join(self.advx_path, f))}
        print('File counter = ',len(self.files))

        self.PATH_PRE_COMPUTED_RESULTS = f'{self.advx_path}/results_{self.model.__class__.__name__}.pkl'
        with open(self.PATH_PRE_COMPUTED_RESULTS, 'rb') as file:
            self.PRE_COMPUTED_RESULTS = pickle.load(file)

    def create_adv_examples(self):
        dataset = self.data_loader
        # from itertools import islice
        # dataset = dict(islice(dataset.items(), 1))

        for key, sample in tqdm(dataset.items(), total=len(dataset), desc="Processed"):
            # simple set difference between the question used during the optimization and the dataset
            questions = list(dataset[key]['questions'].keys() - set(self.PRE_COMPUTED_RESULTS[key]['questions']))

            if f'{key}.jpg' in self.files:
                image = sample['image']

                advx = Image.open(f'{self.advx_path}/{key}.jpg')

                y_pred = self.model.torch_predict(image, questions)
                y_pred_adv = self.model.torch_predict(advx, questions)
                print(y_pred, y_pred_adv, flush=True)

                assert key not in self.results.keys(), "Found sample in results"
                self.results[key]["questions"] = tuple(questions)
                self.results[key]["gt"] = tuple([sample["questions"][q] for q in questions]) # one question can have multiple answers
                self.results[key]["y_pred"] = tuple(y_pred)
                self.results[key]["y_pred_adv"] = tuple(y_pred_adv)
            else:
                print('ERROR KEY NOT IN ADVX FOLDER')

        # save the results
        self.save_results(self.results)

    def report(self):
        pass