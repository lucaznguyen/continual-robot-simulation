# continual-robot-simulation

A lightweight robot continual-learning simulation repo.

The first goal is practical: run one command and see a robot policy learn tasks
sequentially, then inspect whether it forgets earlier tasks. The built-in
simulator is intentionally small so the pipeline runs on CPU without MuJoCo.
Once the idea works, use `docs/continualworld.md` or `docs/libero.md` to move to
research-grade robot benchmarks.

## What this simulates

The robot is a 2D tabletop end-effector. It learns a sequence of manipulation
tasks:

1. `reach_red`
2. `push_blue`
3. `slide_green`
4. `open_drawer`
5. `close_drawer`

The policy is trained with behavior cloning from an expert. After each new task,
the repo evaluates all tasks seen so far and writes:

- a success matrix heatmap
- method comparison plot
- per-task rollout trajectory PNGs
- animated rollout GIFs for the final checkpoint
- a browser-viewable demo gallery
- forgetting metrics in JSON

Implemented continual-learning baselines:

- `finetune`: sequential training, usually forgets earlier tasks
- `replay`: keeps a small memory of previous demonstrations
- `ewc`: Elastic Weight Consolidation regularizes important parameters

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m continual_robot_sim.run_demo --clean
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m continual_robot_sim.run_demo --clean
```

Open the generated report:

```text
runs/quickstart/REPORT.md
```

Open the animated robot viewer:

```text
runs/quickstart/DEMO_GALLERY.html
```

The most useful files are:

```text
runs/quickstart/method_comparison.png
runs/quickstart/finetune/success_matrix.png
runs/quickstart/replay/success_matrix.png
runs/quickstart/ewc/success_matrix.png
runs/quickstart/*/rollouts/after_05_close_drawer/*.gif
```

## Faster smoke test

Use fewer demos and epochs:

```bash
python -m continual_robot_sim.run_demo \
  --clean \
  --episodes-per-task 20 \
  --epochs-per-task 3 \
  --eval-episodes 5 \
  --animations final \
  --methods finetune replay ewc
```

Use `--animations all` if you want GIFs after every training checkpoint, not
only after the final task.

## How to read the result

The success matrix is:

```text
R[i, j] = success on evaluation task j after training through task i
```

If values in earlier columns drop after later rows, the robot forgot old tasks.
Replay and EWC should usually retain old-task performance better than pure
finetuning.

The generated `metrics.json` contains:

```text
forgetting(task) = best_success_seen_for_task - final_success_for_task
```

## Research-grade upgrade paths

- Use `docs/continualworld.md` for continual reinforcement learning on
  Meta-World robot tasks.
- Use `docs/libero.md` for lifelong robot imitation learning with robosuite
  scenes, language tasks, and official lifelong baselines.

## Development

```bash
pytest
python -m continual_robot_sim.run_demo --clean --episodes-per-task 10 --epochs-per-task 1 --eval-episodes 2
```
