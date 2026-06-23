from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from continual_robot_sim.data import collect_expert_dataset
from continual_robot_sim.evaluate import collect_rollout, evaluate_policy
from continual_robot_sim.models import PolicyNet
from continual_robot_sim.reporting import forgetting_summary, save_json, save_matrix_csv
from continual_robot_sim.tasks import task_catalog
from continual_robot_sim.trainers import ContinualTrainer, TrainerConfig
from continual_robot_sim.visualize import (
    save_forgetting_plot,
    save_rollout_gif,
    save_rollout_plot,
    save_success_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight robot continual-learning simulation.",
    )
    parser.add_argument("--out", default="runs/quickstart", help="Output directory.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["finetune", "replay", "ewc"],
        choices=["finetune", "replay", "ewc"],
        help="Continual-learning methods to compare.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names. Defaults to the full built-in sequence.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=80)
    parser.add_argument("--epochs-per-task", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=18)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--animations",
        choices=["none", "final", "all"],
        default="final",
        help="Write animated GIF rollouts. 'final' animates only the last checkpoint.",
    )
    parser.add_argument("--clean", action="store_true", help="Delete the output directory before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


def run_experiment(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if not hasattr(args, "animations"):
        args.animations = "final"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.out)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = task_catalog(args.tasks)
    task_names = [task.name for task in tasks]
    print(f"Collecting expert demonstrations for {len(tasks)} tasks...")

    datasets = [
        collect_expert_dataset(
            task=task,
            task_index=index,
            num_tasks=len(tasks),
            episodes=args.episodes_per_task,
            seed=args.seed + 31 * index,
            max_steps=args.max_steps,
        )
        for index, task in enumerate(tasks)
    ]

    all_matrices: dict[str, np.ndarray] = {}
    for method in args.methods:
        print(f"\nRunning method: {method}")
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        matrix = _run_method(method=method, args=args, tasks=tasks, datasets=datasets, output_dir=method_dir)
        all_matrices[method] = matrix

    save_forgetting_plot(all_matrices, task_names=task_names, output_path=output_dir / "method_comparison.png")
    save_json(
        {
            "tasks": task_names,
            "methods": args.methods,
            "episodes_per_task": args.episodes_per_task,
            "epochs_per_task": args.epochs_per_task,
            "eval_episodes": args.eval_episodes,
            "max_steps": args.max_steps,
            "animations": args.animations,
        },
        output_dir / "run_config.json",
    )
    _write_gallery(output_dir, methods=args.methods, task_names=task_names)
    _write_report(output_dir, all_matrices, task_names)
    print(f"\nDone. Open {output_dir / 'REPORT.md'}")
    return all_matrices


def _run_method(
    method: str,
    args: argparse.Namespace,
    tasks,
    datasets,
    output_dir: Path,
) -> np.ndarray:
    probe_env = collect_expert_dataset(
        task=tasks[0],
        task_index=0,
        num_tasks=len(tasks),
        episodes=1,
        seed=args.seed,
        max_steps=1,
    )
    obs_dim = probe_env.observations.shape[1]
    action_dim = probe_env.actions.shape[1]

    model = PolicyNet(obs_dim=obs_dim, action_dim=action_dim)
    trainer = ContinualTrainer(
        model=model,
        config=TrainerConfig(
            method=method,
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size,
            device=args.device,
        ),
    )

    matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=np.float32)
    losses = []

    for train_index, dataset in enumerate(datasets):
        print(f"  train task {train_index + 1}/{len(tasks)}: {dataset.name}")
        losses.append(trainer.train_task(dataset, task_number=train_index, seed=args.seed))
        scores = evaluate_policy(
            model=model,
            tasks=tasks,
            eval_task_indices=list(range(train_index + 1)),
            episodes=args.eval_episodes,
            seed=args.seed + 10_000 + train_index,
            max_steps=args.max_steps,
            device=args.device,
        )
        for eval_index in range(train_index + 1):
            matrix[train_index, eval_index] = scores[tasks[eval_index].name]

        rollout_dir = output_dir / "rollouts" / f"after_{train_index + 1:02d}_{tasks[train_index].name}"
        for eval_index in range(train_index + 1):
            rollout = collect_rollout(
                model=model,
                task=tasks[eval_index],
                task_index=eval_index,
                num_tasks=len(tasks),
                seed=args.seed + 20_000 + train_index * 100 + eval_index,
                max_steps=args.max_steps,
                device=args.device,
            )
            save_rollout_plot(
                rollout,
                task=tasks[eval_index],
                output_path=rollout_dir / f"{tasks[eval_index].name}.png",
            )
            should_animate = args.animations == "all" or (
                args.animations == "final" and train_index == len(tasks) - 1
            )
            if should_animate:
                save_rollout_gif(
                    rollout,
                    task=tasks[eval_index],
                    output_path=rollout_dir / f"{tasks[eval_index].name}.gif",
                )

    task_names = [task.name for task in tasks]
    save_matrix_csv(matrix, task_names, output_dir / "success_matrix.csv")
    save_success_matrix(matrix, task_names, output_dir / "success_matrix.png", title=f"{method} success matrix")
    save_json(
        {
            "losses": losses,
            "forgetting": forgetting_summary(matrix, task_names),
            "final_seen_task_mean_success": float(np.nanmean(matrix[-1])),
        },
        output_dir / "metrics.json",
    )
    torch.save(model.state_dict(), output_dir / "final_policy.pt")
    return matrix


def _write_report(output_dir: Path, matrices: dict[str, np.ndarray], task_names: list[str]) -> None:
    lines = [
        "# Continual Robot Simulation Report",
        "",
        "This report was generated by `python -m continual_robot_sim.run_demo`.",
        "",
        "![Method comparison](method_comparison.png)",
        "",
        "Open `DEMO_GALLERY.html` in a browser to watch animated robot rollouts.",
        "",
        "## Final mean success on seen tasks",
        "",
    ]
    for method, matrix in matrices.items():
        lines.append(f"- `{method}`: {float(np.nanmean(matrix[-1])):.3f}")

    lines.extend(["", "## Artifacts", ""])
    for method in matrices:
        lines.append(f"- `{method}/success_matrix.png`: task-by-task success heatmap")
        lines.append(f"- `{method}/rollouts/`: rollout trajectory plots after each task")
        lines.append(f"- `{method}/metrics.json`: forgetting summary and final score")

    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gallery(output_dir: Path, methods: list[str], task_names: list[str]) -> None:
    html = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "  <title>Continual Robot Simulation Demo</title>",
        "  <style>",
        "    body { font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f4; color: #202020; }",
        "    h1 { margin-bottom: 4px; }",
        "    h2 { margin-top: 32px; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }",
        "    figure { margin: 0; background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }",
        "    img { width: 100%; height: auto; display: block; }",
        "    figcaption { margin-top: 8px; font-size: 14px; font-weight: 600; }",
        "    .note { color: #555; margin-top: 0; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Continual Robot Simulation Demo</h1>",
        "  <p class=\"note\">Animated final-checkpoint rollouts after the policy has learned the full task sequence.</p>",
        "  <p><a href=\"REPORT.md\">Open report</a> | <a href=\"method_comparison.png\">Method comparison</a></p>",
    ]

    final_task = task_names[-1]
    for method in methods:
        html.append(f"  <h2>{method}</h2>")
        html.append("  <div class=\"grid\">")
        for task_name in task_names:
            gif_path = f"{method}/rollouts/after_{len(task_names):02d}_{final_task}/{task_name}.gif"
            png_path = f"{method}/rollouts/after_{len(task_names):02d}_{final_task}/{task_name}.png"
            html.extend(
                [
                    "    <figure>",
                    f"      <a href=\"{gif_path}\"><img src=\"{gif_path}\" alt=\"{method} {task_name} rollout\" onerror=\"this.src='{png_path}'\"></a>",
                    f"      <figcaption>{task_name}</figcaption>",
                    "    </figure>",
                ]
            )
        html.append("  </div>")

    html.extend(["</body>", "</html>"])
    (output_dir / "DEMO_GALLERY.html").write_text("\n".join(html) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
