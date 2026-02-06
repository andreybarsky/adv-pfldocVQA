import subprocess
import datetime

n_questions_list = [1,2,3,4,5]
masks = ["include_all", "bottom_right_corner"]
script = "main_exp3.py"
SEED = 42
STEPS_DOA = 1

include_all_pix2struct = {
    "script": script,
    "dataset_path": "/home-local/mpintore/adv_docVQA/utils/advdoc_data_nsampl1000_nqst5.pkl",
    "device": "cuda:0",
    "model": "pix2struct",
    "eps": 8,
    "step_size": 2,
    "steps": STEPS_DOA,
    "seed": 42,
    "mask": "include_all"
}

patched_pix2struct = {
    "script": script,
    "dataset_path": "/home-local/mpintore/adv_docVQA/utils/advdoc_data_nsampl1000_nqst5.pkl",
    "device": "cuda:1",
    "model": "pix2struct",
    "eps": 8,
    "step_size": 2,
    "steps": STEPS_DOA,
    "seed": 42,
    "mask": "bottom_right_corner"
}

include_all_donut = {
    "script": script,
    "dataset_path": "/home-local/mpintore/adv_docVQA/utils/advdoc_data_nsampl1000_nqst5.pkl",
    "device": "cuda:1",
    "model": "donut",
    "eps": 32,
    "step_size": 2,
    "steps": STEPS_DOA,
    "seed": SEED,
    "mask":"include_all"
}

patched_donut = {
    "script": script,
    "dataset_path": "/home-local/mpintore/adv_docVQA/utils/advdoc_data_nsampl1000_nqst5.pkl",
    "device": "cuda:0",
    "model": "donut",
    "eps": 32,
    "step_size": 2,
    "steps": STEPS_DOA,
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
    targets = ["Doesn't matter"]
    target_str = " ".join(f'"{t}"' for t in targets)
    logname = f"logs_gradacc/Exp3/{base_cmd['model']}_{n_q}questions_{base_cmd['mask']}.log"

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
