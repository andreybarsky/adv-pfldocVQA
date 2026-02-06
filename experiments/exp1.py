# Exp1 = Perturbing an image with a question 
# to cause a target answer.
# Evaluation of the perturbed image on n-1 questions
from tqdm import tqdm
import random

from .experiment_base import Experiment

class Exp1(Experiment):
    def create_adv_examples(self):
        """
        TODO
        """
        dataset = self.data_loader
        for key, sample in tqdm(dataset.items(), total=len(dataset), desc="Processed"):
            image = sample["image"]
            question = [random.choice(list(sample["questions"].keys()))]

            advx = self.attack(model=self.model,
                              processor=self.processor,
                              auto_processor=self.autoprocessor, # TODO: remove autoprocessor, it is not needed
                              image=image,
                              questions=question,
                              targets=self.args.target_answer,
                              mask_function=self.mask_function,
                              args=self.args)
            
            assert image.size == advx.size
            
            # Save results
            advx.save(f"{self.res_directory}/{key}.jpg")

    def report(self):
        pass