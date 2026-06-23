from argparse import Namespace

from continual_robot_sim.run_demo import run_experiment


def test_smoke_experiment(tmp_path):
    args = Namespace(
        out=str(tmp_path / "run"),
        methods=["finetune"],
        tasks=["reach_red", "push_blue"],
        episodes_per_task=4,
        epochs_per_task=1,
        eval_episodes=1,
        max_steps=20,
        batch_size=16,
        seed=3,
        device="cpu",
        animations="none",
        clean=True,
    )
    matrices = run_experiment(args)

    assert "finetune" in matrices
    assert matrices["finetune"].shape == (2, 2)
    assert (tmp_path / "run" / "finetune" / "success_matrix.csv").exists()
    assert (tmp_path / "run" / "REPORT.md").exists()
