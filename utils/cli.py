import argparse
from config import AVAILABLE_MASKS

def parse_args():
    parser = argparse.ArgumentParser()
    # TODO: remove my path here
    parser.add_argument("--dataset_path", type=str, default="/home-local/mpintore/adv_docVQA/utils/advdoc_data_nsampl1000_nqst5.pkl", help="The dataset .pkl you want to perturb")
    parser.add_argument("--target_answer", type=str, nargs='+', default=["No answer"], help="The answer the model should output")
    parser.add_argument("--n_questions", type=int, default=1, help="The number of questions on which the sample should be optimized")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use (cuda:index/cpu)")
    parser.add_argument("--model", type=str, default="donut", help="Model to be used (donut or pix2struct)")
    parser.add_argument("--eps", type=float, default=32, help="Attack Perturbation size")
    parser.add_argument("--step_size", type=float, default=2, help="Attack Step Size")
    parser.add_argument("--mask", type=str, default="include_all", help=f"Currently implemented masks {AVAILABLE_MASKS.keys()}")
    parser.add_argument("--steps", type=int, default=100, help="PGD number of iterations")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for experiments")
    args = parser.parse_args()
    print(args.__dict__, flush=True)
    return args