# LIBERO Path

Use this when you want a richer robot simulator with visuomotor policies,
language-conditioned tasks, robosuite scenes, and official lifelong imitation
learning baselines.

Reference:

- <https://github.com/Lifelong-Robot-Learning/LIBERO>
- <https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html>

## Setup

Linux or WSL2 is recommended.

```bash
conda create -n libero python=3.8.13 -y
conda activate libero

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

pip install -r requirements.txt
pip install robosuite
pip install -e .
```

If you hit robosuite compatibility issues, check the current LIBERO issues and
pin the robosuite version used by the repo.

## Download demonstrations

```bash
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
```

Other suites include:

```text
libero_spatial
libero_object
libero_goal
libero_100
```

## Run lifelong imitation learning

```bash
python libero/lifelong/main.py \
  seed=0 \
  benchmark_name=LIBERO_SPATIAL \
  policy=bc_rnn_policy \
  lifelong=base
```

Compare with:

```bash
python libero/lifelong/main.py seed=0 benchmark_name=LIBERO_SPATIAL policy=bc_rnn_policy lifelong=er
python libero/lifelong/main.py seed=0 benchmark_name=LIBERO_SPATIAL policy=bc_rnn_policy lifelong=ewc
python libero/lifelong/main.py seed=0 benchmark_name=LIBERO_SPATIAL policy=bc_rnn_policy lifelong=packnet
```

## Evaluation view

By default, LIBERO evaluates during training. You can also evaluate checkpoints:

```bash
python libero/lifelong/evaluate.py \
  --benchmark LIBERO_SPATIAL \
  --task_id 0 \
  --algo er \
  --policy bc_rnn_policy \
  --seed 0 \
  --ep 10 \
  --load_task 0 \
  --device_id 0
```

Use the same matrix logic as the lightweight simulator:

```text
R[i, j] = success on task j after training through task i
```
