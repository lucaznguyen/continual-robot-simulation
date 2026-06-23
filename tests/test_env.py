import numpy as np

from continual_robot_sim.envs import TabletopRobotEnv
from continual_robot_sim.tasks import task_catalog


def test_env_observation_and_step_shapes():
    tasks = task_catalog()
    env = TabletopRobotEnv(tasks[0], task_index=0, num_tasks=len(tasks), seed=0)
    obs = env.reset()
    action = env.expert_action()
    next_obs, reward, done, info = env.step(action)

    assert obs.shape == next_obs.shape
    assert action.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "success" in info
    assert np.isfinite(next_obs).all()
