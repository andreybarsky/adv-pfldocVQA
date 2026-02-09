import sys
import os
import logging

from experiments import Exp2
from dataset import Imdb_Dataloader
from config import MODEL_NAMES
from models.model_registry import get_model
from utils import parse_args

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = parse_args()
    data_loader = Imdb_Dataloader(args.dataset_path).load_data()

    for kwarg, val in args._get_kwargs():
        print(f"{kwarg}: {val}")

    logger.info(f'Starting Experiment n.2')

    processor, autoprocessor, model, attack, mask_function = get_model(args.model, args)
    exp = Exp2(attack=attack, 
               model=model, 
               autoprocessor=autoprocessor, 
               processor=processor, 
               data_loader=data_loader,
               n_questions=args.n_questions,
               mask_function=mask_function,
               args=args)
    exp.setup()
    exp.create_adv_examples()
    exp.report()