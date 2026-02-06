import subprocess
import datetime

all_target_answers = ["No answer", "Unclear", "Retry", "Try later", "I won't tell you"]
n_questions_list = [4,5]
masks = ["include_all", "bottom_right_corner"]
SEED = 42
STEPS_DONUT = 100

include_all_pix2struct = {
    "script": "main_exp2.py",
    "dataset_path": "/home-local/mpintore/github/adv_docVQA/utils/advdoc_data_nsampl1000_nqst5.pkl",
    "device": "cuda:1",
    "model": "pix2struct",
    "eps": 8,
    "step_size": 2,
    "steps": 20,
    "seed": SEED,
    "mask":"include_all"
}

patched_pix2struct= {
    "script": "main_exp2.py",
    "dataset_path": "/home-local/mpintore/adv_docVQA/utils/advdoc_data_nsampl1000_nqst5.pkl",
    "device": "cuda:0",
    "model": "donut",
    "eps": 96,
    "step_size": 24,
    "steps": 25,
    "seed": SEED,
    "mask": "bottom_right_corner"
}

include_all_donut = {
    "script": "main_exp2.py",
    "dataset_path": "/home-local/mpintore/adv_docVQA/utils/advdoc_data_nsampl1000_nqst5.pkl",
    "device": "cuda:4",
    "model": "donut",
    "eps": 32,
    "step_size": 2,
    "steps": STEPS_DONUT,
    "seed": SEED,
    "mask":"include_all"
}

patched_donut = {
    "script": "main_exp2.py",
    "dataset_path": "/home-local/mpintore/adv_docVQA/utils/advdoc_data_nsampl1000_nqst5.pkl",
    "device": "cuda:0",
    "model": "donut",
    "eps": 96,
    "step_size": 24,
    "steps": STEPS_DONUT,
    "seed": SEED,
    "mask": "bottom_right_corner"
}

model_cmd = {
    "include_all_pix2struct": include_all_pix2struct,
    "patched_pix2struct": patched_pix2struct,
    "include_all_donut": include_all_donut,
    "patched_donut": patched_donut
}

# extract attack setting
model = "include_all_donut"
base_cmd = model_cmd[model]

for n_q in n_questions_list:
    targets = all_target_answers[:n_q]
    target_str = " ".join(f'"{t}"' for t in targets)
    logname = f"logs_gradacc/Exp2/{base_cmd['model']}_{n_q}questions_{base_cmd["mask"]}.log"

    cmd = (
        f'python3 {base_cmd["script"]} '
        f'--dataset_path="{base_cmd["dataset_path"]}" '
        f'--target_answer {target_str} '
        f'--n_questions={n_q} '
        f'--device="{base_cmd["device"]}" '
        f'--model="{base_cmd["model"]}" '
        f'--eps={base_cmd["eps"]} '
        f'--step_size={base_cmd["step_size"]} '
        f'--steps={base_cmd["steps"]} '
        f'--mask={base_cmd["mask"]} '
        f'--seed={base_cmd["seed"]} '
        f'> {logname} 2>&1'
    )

    print(f"[RUN] {cmd} - {datetime.datetime.now()}",flush=True)
    subprocess.call(cmd, shell=True) 

print("Exp finished")
