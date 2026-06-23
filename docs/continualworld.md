# ContinualWorld Path

Use this when you want continual reinforcement learning on a standard robot
benchmark.

ContinualWorld is built on Meta-World robot manipulation tasks and includes
scripts for single-task, continual-learning, and multitask experiments.

Reference:

- <https://github.com/awarelab/continual_world>

## Setup

Linux or WSL2 is recommended.

```bash
conda create -n cw python=3.8 -y
conda activate cw

git clone https://github.com/awarelab/continual_world.git
cd continual_world
pip install -e .
```

You may need MuJoCo/mujoco-py system dependencies depending on your platform.

## Verify one task

```bash
python run_single.py \
  --seed 0 \
  --steps 2e3 \
  --log_every 250 \
  --task hammer-v1 \
  --logger_output tsv tensorboard
```

## Run continual learning

```bash
python run_cl.py \
  --seed 0 \
  --steps_per_task 2e3 \
  --log_every 250 \
  --tasks CW20 \
  --cl_method ewc \
  --cl_reg_coef 1e4 \
  --logger_output tsv tensorboard
```

Increase `--steps_per_task` when the pipeline is stable. The paper-scale setup
uses far more training steps per task than the quick smoke-test command above.

## What to export

For each method, save:

- task sequence
- success or return after each task
- performance matrix `R[i, j]`
- average success on seen tasks
- forgetting per task
- rollout videos or screenshots if rendering is enabled

Then compare:

- finetuning / sequential baseline
- EWC
- replay
- multitask upper bound
