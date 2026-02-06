# Creating adversarial documents
The exp_launcher.py file contains all the commands that were launched to obtain the adversarial examples reported in the paper (https://arxiv.org/abs/2512.04554).

If you notice, inside ```exp_launcher.py```, the ```main_exp2.py``` script is always called, which uses ```experiments/exp2.py```.

The useful files are ```experiments/exp2.py``` (which turns out to be the targeted attack with M questions) and ```experiments/exp3.py``` (which turns out to be an untargeted attack aka Denial of Answer (DoA)). 
From these experiments, you can verify the entire internal functioning of the attack functions.

Dataset available here:
```
https://drive.google.com/file/d/17PDqk8wzqiF1_bggxQhiLSqnVXJz5xJC/view?usp=drive_link
```