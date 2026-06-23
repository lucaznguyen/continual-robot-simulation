from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--replay-samples-per-task",
        type=int,
        default=4096,
        help="Maximum demonstrations retained per previous task for replay.",
    )
    parser.add_argument(
        "--replay-epoch-multiplier",
        type=float,
        default=1.6,
        help="Replay-only multiplier for epochs per task.",
    )
    parser.add_argument(
        "--no-replay-balance",
        action="store_true",
        help="Disable equal weighting across current and replayed tasks.",
    )
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
    live_rollouts: dict[str, dict[str, dict]] = {}
    for method in args.methods:
        print(f"\nRunning method: {method}")
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        matrix, method_rollouts = _run_method(
            method=method,
            args=args,
            tasks=tasks,
            datasets=datasets,
            output_dir=method_dir,
        )
        all_matrices[method] = matrix
        live_rollouts[method] = method_rollouts

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
            "replay_samples_per_task": args.replay_samples_per_task,
            "replay_epoch_multiplier": args.replay_epoch_multiplier,
            "replay_balance_tasks": not args.no_replay_balance,
        },
        output_dir / "run_config.json",
    )
    _write_gallery(output_dir, methods=args.methods, task_names=task_names)
    _write_live_sim(output_dir, tasks=tasks, live_rollouts=live_rollouts)
    _write_report(output_dir, all_matrices, task_names)
    print(f"\nDone. Open {output_dir / 'REPORT.md'}")
    return all_matrices


def _run_method(
    method: str,
    args: argparse.Namespace,
    tasks,
    datasets,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, dict]]:
    # Keep method comparisons fair: each method starts from the same initial policy.
    torch.manual_seed(args.seed)
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
            replay_samples_per_task=args.replay_samples_per_task,
            replay_epoch_multiplier=args.replay_epoch_multiplier,
            replay_balance_tasks=not args.no_replay_balance,
            device=args.device,
        ),
    )

    matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=np.float32)
    losses = []
    final_rollouts: dict[str, dict] = {}

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
            if train_index == len(tasks) - 1:
                final_rollouts[tasks[eval_index].name] = _rollout_to_live_dict(rollout)

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
    return matrix, final_rollouts


def _write_report(output_dir: Path, matrices: dict[str, np.ndarray], task_names: list[str]) -> None:
    lines = [
        "# Continual Robot Simulation Report",
        "",
        "This report was generated by `python -m continual_robot_sim.run_demo`.",
        "",
        "![Method comparison](method_comparison.png)",
        "",
        "Open `LIVE_SIM.html` for an interactive canvas simulation of the robot arm.",
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


def _rollout_to_live_dict(rollout) -> dict:
    frames = []
    for snap in rollout.snapshots:
        frames.append(
            {
                "ee": _array_to_xy(snap["ee"]),
                "obj": _array_to_xy(snap["obj"]),
                "goal": _array_to_xy(snap["goal"]),
                "success": bool(snap["success"]),
            }
        )
    return {
        "taskName": rollout.task_name,
        "success": bool(rollout.success),
        "frames": frames,
    }


def _array_to_xy(value) -> list[float]:
    return [round(float(value[0]), 4), round(float(value[1]), 4)]


def _write_live_sim(output_dir: Path, tasks, live_rollouts: dict[str, dict[str, dict]]) -> None:
    task_meta = {
        task.name: {
            "name": task.name,
            "kind": task.kind,
            "color": task.color,
            "goal": [float(task.goal[0]), float(task.goal[1])],
            "objectStart": [float(task.object_start[0]), float(task.object_start[1])],
        }
        for task in tasks
    }
    payload = json.dumps({"tasks": task_meta, "rollouts": live_rollouts})
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Robot Simulation</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f4ef;
      color: #1f2420;
    }}
    body {{ margin: 0; }}
    main {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(280px, 360px) 1fr;
      gap: 0;
    }}
    aside {{
      padding: 22px;
      border-right: 1px solid #d7dacd;
      background: #fbfbf7;
    }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    p {{ color: #536052; line-height: 1.45; margin: 0 0 18px; }}
    label {{ display: block; font-size: 12px; font-weight: 700; text-transform: uppercase; margin: 16px 0 6px; }}
    select, input[type="range"], button {{
      width: 100%;
      box-sizing: border-box;
      font: inherit;
    }}
    select {{
      height: 38px;
      border: 1px solid #bfc6b9;
      border-radius: 6px;
      background: white;
      padding: 0 10px;
    }}
    button {{
      height: 40px;
      border: 0;
      border-radius: 6px;
      background: #243326;
      color: white;
      font-weight: 700;
      cursor: pointer;
      margin-top: 10px;
    }}
    button.secondary {{
      background: #dde3d8;
      color: #243326;
    }}
    .stats {{
      margin-top: 18px;
      padding: 12px;
      border: 1px solid #d7dacd;
      border-radius: 8px;
      background: white;
      font-size: 14px;
      line-height: 1.7;
    }}
    .stage {{
      display: grid;
      place-items: center;
      padding: 24px;
    }}
    canvas {{
      width: min(88vw, 860px);
      height: min(88vh, 860px);
      max-height: calc(100vh - 48px);
      aspect-ratio: 1;
      background: #ffffff;
      border: 1px solid #d7dacd;
      border-radius: 8px;
      box-shadow: 0 12px 36px rgba(38, 50, 34, 0.12);
    }}
    @media (max-width: 820px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid #d7dacd; }}
      canvas {{ width: calc(100vw - 24px); height: calc(100vw - 24px); }}
    }}
  </style>
</head>
<body>
<main>
  <aside>
    <h1>Live Robot Simulation</h1>
    <p>Watch the learned policy as a two-link robot arm. The square is the object or drawer handle, and the star is the goal.</p>

    <label for="method">Method</label>
    <select id="method"></select>

    <label for="task">Task</label>
    <select id="task"></select>

    <label for="speed">Speed</label>
    <input id="speed" type="range" min="0.25" max="3" value="1" step="0.25">

    <label for="scrub">Timeline</label>
    <input id="scrub" type="range" min="0" max="1" value="0" step="1">

    <button id="play">Pause</button>
    <button id="reset" class="secondary">Reset</button>

    <div class="stats">
      <div><strong>Status:</strong> <span id="status">-</span></div>
      <div><strong>Step:</strong> <span id="step">0</span></div>
      <div><strong>End effector:</strong> <span id="ee">-</span></div>
      <div><strong>Object:</strong> <span id="obj">-</span></div>
    </div>
  </aside>
  <section class="stage">
    <canvas id="canvas" width="900" height="900"></canvas>
  </section>
</main>
<script>
const DATA = {payload};
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const methodSelect = document.getElementById("method");
const taskSelect = document.getElementById("task");
const speedInput = document.getElementById("speed");
const scrubInput = document.getElementById("scrub");
const playButton = document.getElementById("play");
const resetButton = document.getElementById("reset");
const statusEl = document.getElementById("status");
const stepEl = document.getElementById("step");
const eeEl = document.getElementById("ee");
const objEl = document.getElementById("obj");

let playing = true;
let frameIndex = 0;
let lastTick = 0;

const methods = Object.keys(DATA.rollouts);
for (const method of methods) methodSelect.add(new Option(method, method));

function tasksForMethod(method) {{
  return Object.keys(DATA.rollouts[method] || {{}});
}}

function fillTasks() {{
  const method = methodSelect.value;
  taskSelect.innerHTML = "";
  for (const task of tasksForMethod(method)) taskSelect.add(new Option(task, task));
  frameIndex = 0;
  updateScrubMax();
  draw();
}}

function currentRollout() {{
  return DATA.rollouts[methodSelect.value][taskSelect.value];
}}

function currentTask() {{
  return DATA.tasks[taskSelect.value];
}}

function updateScrubMax() {{
  const rollout = currentRollout();
  scrubInput.max = String(Math.max(0, rollout.frames.length - 1));
  scrubInput.value = String(frameIndex);
}}

function worldToCanvas(p) {{
  const margin = 70;
  const size = canvas.width - margin * 2;
  return [
    margin + ((p[0] + 0.9) / 1.8) * size,
    margin + ((0.9 - p[1]) / 1.8) * size,
  ];
}}

function armJoints(ee) {{
  const base = [0, 0];
  const l1 = 0.52;
  const l2 = 0.58;
  let x = ee[0], y = ee[1];
  const dist = Math.max(0.001, Math.min(Math.hypot(x, y), l1 + l2 - 0.001));
  const angle = Math.atan2(y, x);
  const cosA = (l1*l1 + dist*dist - l2*l2) / (2*l1*dist);
  const shoulder = angle - Math.acos(Math.max(-1, Math.min(1, cosA)));
  const elbow = [l1 * Math.cos(shoulder), l1 * Math.sin(shoulder)];
  return [base, elbow, ee];
}}

function drawStar(cx, cy, spikes, outer, inner) {{
  let rot = Math.PI / 2 * 3;
  let x = cx;
  let y = cy;
  const step = Math.PI / spikes;
  ctx.beginPath();
  ctx.moveTo(cx, cy - outer);
  for (let i = 0; i < spikes; i++) {{
    x = cx + Math.cos(rot) * outer;
    y = cy + Math.sin(rot) * outer;
    ctx.lineTo(x, y);
    rot += step;
    x = cx + Math.cos(rot) * inner;
    y = cy + Math.sin(rot) * inner;
    ctx.lineTo(x, y);
    rot += step;
  }}
  ctx.lineTo(cx, cy - outer);
  ctx.closePath();
  ctx.fill();
}}

function drawGrid() {{
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#edf0e9";
  ctx.lineWidth = 1;
  for (let i = -9; i <= 9; i++) {{
    const x = worldToCanvas([i / 10, 0])[0];
    const y = worldToCanvas([0, i / 10])[1];
    ctx.beginPath(); ctx.moveTo(x, 50); ctx.lineTo(x, 850); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(850, y); ctx.stroke();
  }}
  ctx.strokeStyle = "#cfd7ca";
  ctx.lineWidth = 2;
  const xAxisY = worldToCanvas([0, 0])[1];
  const yAxisX = worldToCanvas([0, 0])[0];
  ctx.beginPath(); ctx.moveTo(50, xAxisY); ctx.lineTo(850, xAxisY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(yAxisX, 50); ctx.lineTo(yAxisX, 850); ctx.stroke();
}}

function draw() {{
  const rollout = currentRollout();
  const task = currentTask();
  const frame = rollout.frames[frameIndex];
  drawGrid();

  const goal = worldToCanvas(frame.goal);
  ctx.fillStyle = "#111111";
  drawStar(goal[0], goal[1], 5, 18, 8);

  if (task.kind.startsWith("drawer")) {{
    const start = worldToCanvas([-0.75, task.objectStart[1]]);
    const end = worldToCanvas([0.75, task.objectStart[1]]);
    ctx.strokeStyle = "#d4d8d0";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(start[0], start[1]); ctx.lineTo(end[0], end[1]); ctx.stroke();
  }}

  const joints = armJoints(frame.ee).map(worldToCanvas);
  ctx.strokeStyle = "#334238";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(joints[0][0], joints[0][1]);
  ctx.lineTo(joints[1][0], joints[1][1]);
  ctx.lineTo(joints[2][0], joints[2][1]);
  ctx.stroke();

  ctx.fillStyle = "#1d261f";
  for (const joint of joints) {{
    ctx.beginPath(); ctx.arc(joint[0], joint[1], 16, 0, Math.PI * 2); ctx.fill();
  }}

  const obj = worldToCanvas(frame.obj);
  if (task.kind !== "reach") {{
    ctx.fillStyle = task.color;
    ctx.strokeStyle = "#202020";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(obj[0] - 18, obj[1] - 18, 36, 36, 6);
    ctx.fill();
    ctx.stroke();
  }}

  ctx.fillStyle = frame.success ? "#2e8b57" : "#8a4b18";
  ctx.font = "700 22px system-ui";
  ctx.fillText(frame.success ? "success" : "running", 60, 60);

  statusEl.textContent = rollout.success ? "success rollout" : "failed rollout";
  stepEl.textContent = `${{frameIndex}} / ${{rollout.frames.length - 1}}`;
  eeEl.textContent = `[${{frame.ee[0].toFixed(2)}}, ${{frame.ee[1].toFixed(2)}}]`;
  objEl.textContent = `[${{frame.obj[0].toFixed(2)}}, ${{frame.obj[1].toFixed(2)}}]`;
  scrubInput.value = String(frameIndex);
}}

function tick(timestamp) {{
  const rollout = currentRollout();
  const speed = Number(speedInput.value);
  if (playing && timestamp - lastTick > 90 / speed) {{
    frameIndex = (frameIndex + 1) % rollout.frames.length;
    lastTick = timestamp;
    draw();
  }}
  requestAnimationFrame(tick);
}}

methodSelect.addEventListener("change", fillTasks);
taskSelect.addEventListener("change", () => {{ frameIndex = 0; updateScrubMax(); draw(); }});
scrubInput.addEventListener("input", () => {{ frameIndex = Number(scrubInput.value); draw(); }});
playButton.addEventListener("click", () => {{
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
}});
resetButton.addEventListener("click", () => {{ frameIndex = 0; draw(); }});

fillTasks();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
    (output_dir / "LIVE_SIM.html").write_text(html, encoding="utf-8")


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
