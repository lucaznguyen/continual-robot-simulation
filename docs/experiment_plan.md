# Experiment Plan

This repo starts with a quick CPU-friendly simulator, then gives you a path to
standard robot benchmarks.

## Phase 1: Fast feasibility

Run:

```bash
python -m continual_robot_sim.run_demo --clean
```

Questions this answers:

- Does sequential robot learning show forgetting?
- Do EWC or replay reduce forgetting?
- Can we produce the core plots and rollout artifacts?

Expected outputs:

```text
runs/quickstart/REPORT.md
runs/quickstart/method_comparison.png
runs/quickstart/{method}/success_matrix.png
runs/quickstart/{method}/rollouts/
```

## Phase 2: Research-grade RL

Move the same evaluation logic to ContinualWorld:

- task sequence: CW10 or CW20
- method set: finetuning, EWC, replay, multitask
- metrics: average success, forgetting, forward transfer

## Phase 3: Research-grade imitation learning

Move the same evaluation logic to LIBERO:

- benchmark: `LIBERO_SPATIAL`, `LIBERO_OBJECT`, `LIBERO_GOAL`, or `LIBERO_10`
- policy: `bc_rnn_policy`, then transformer/VILT if needed
- methods: `base`, `er`, `ewc`, `packnet`, `multitask`

## Phase 4: Continual domain generalization

Replace task sequence with domain sequence:

```text
normal objects
new colors/textures
camera shift
mass/friction shift
lighting shift
held-out mixed shift
```

Then evaluate both:

- forgetting on old domains
- generalization to held-out domains
