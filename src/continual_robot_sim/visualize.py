from __future__ import annotations

from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

from continual_robot_sim.evaluate import Rollout
from continual_robot_sim.tasks import RobotTask


def save_success_matrix(
    matrix: np.ndarray,
    task_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots(figsize=(1.2 * len(task_names) + 3, 1.0 * len(task_names) + 2))
    im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("After training task")
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names, rotation=35, ha="right")
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax, label="Success rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_forgetting_plot(
    matrices: dict[str, np.ndarray],
    task_names: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for method, matrix in matrices.items():
        averages = []
        for row in range(matrix.shape[0]):
            seen = matrix[row, : row + 1]
            averages.append(float(np.nanmean(seen)))
        ax.plot(range(1, len(averages) + 1), averages, marker="o", label=method)

    ax.set_title("Continual-learning performance over the task sequence")
    ax.set_xlabel("Number of tasks learned")
    ax.set_ylabel("Mean success on seen tasks")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(range(1, len(task_names) + 1))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_rollout_plot(
    rollout: Rollout,
    task: RobotTask,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ee = np.array([snap["ee"] for snap in rollout.snapshots], dtype=np.float32)
    obj = np.array([snap["obj"] for snap in rollout.snapshots], dtype=np.float32)
    goal = np.array(task.goal, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f"{rollout.task_name}: {'success' if rollout.success else 'failed'}")
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.plot(ee[:, 0], ee[:, 1], color="#222222", linewidth=2, label="end-effector")
    ax.scatter(ee[0, 0], ee[0, 1], color="#222222", marker="o", label="ee start")
    ax.scatter(ee[-1, 0], ee[-1, 1], color="#222222", marker="x", label="ee end")

    if task.kind != "reach":
        ax.plot(obj[:, 0], obj[:, 1], color=task.color, linewidth=2, label="object/handle")
        ax.scatter(obj[0, 0], obj[0, 1], color=task.color, marker="o", label="object start")
        ax.scatter(obj[-1, 0], obj[-1, 1], color=task.color, marker="x", label="object end")

    ax.scatter(goal[0], goal[1], color="#111111", marker="*", s=180, label="goal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_rollout_gif(
    rollout: Rollout,
    task: RobotTask,
    output_path: Path,
    fps: int = 12,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ee = np.array([snap["ee"] for snap in rollout.snapshots], dtype=np.float32)
    obj = np.array([snap["obj"] for snap in rollout.snapshots], dtype=np.float32)
    goal = np.array(task.goal, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{rollout.task_name}: {'success' if rollout.success else 'failed'}")
    ax.scatter(goal[0], goal[1], color="#111111", marker="*", s=180, label="goal")

    if task.kind.startswith("drawer"):
        drawer_y = np.array(task.object_start, dtype=np.float32)[1]
        ax.plot([-0.75, 0.75], [drawer_y, drawer_y], color="#cccccc", linewidth=7, solid_capstyle="round")

    ee_trace, = ax.plot([], [], color="#222222", linewidth=2, label="end-effector path")
    ee_point = ax.scatter([], [], color="#222222", s=80, zorder=4, label="end-effector")
    arm_line, = ax.plot([], [], color="#666666", linewidth=3, alpha=0.8)

    obj_trace = None
    obj_point = None
    if task.kind != "reach":
        obj_trace, = ax.plot([], [], color=task.color, linewidth=2, label="object/handle path")
        obj_point = ax.scatter([], [], color=task.color, s=110, zorder=5, label="object/handle")

    step_text = ax.text(-0.86, 0.83, "", fontsize=10, color="#222222")
    ax.legend(loc="upper right", fontsize=8)

    def update(frame: int):
        ee_trace.set_data(ee[: frame + 1, 0], ee[: frame + 1, 1])
        ee_point.set_offsets(ee[frame])
        arm_line.set_data([0.0, ee[frame, 0]], [0.0, ee[frame, 1]])
        artists = [ee_trace, ee_point, arm_line, step_text]

        if obj_trace is not None and obj_point is not None:
            obj_trace.set_data(obj[: frame + 1, 0], obj[: frame + 1, 1])
            obj_point.set_offsets(obj[frame])
            artists.extend([obj_trace, obj_point])

        step_text.set_text(f"step {frame}/{len(ee) - 1}")
        return artists

    animation = FuncAnimation(fig, update, frames=len(ee), interval=1000 / fps, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
