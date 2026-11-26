import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pirl import ppo_train, video
from pirl.ppo import TrainingConfig


class StubEnvironment:
    def __init__(self):
        self.closed = False
        self.completed_successes = [None, None]

    def reset(self):
        return {"state": torch.zeros(2, 1)}

    def step(self, actions):
        self.completed_successes = [True, False]
        return self.reset(), torch.ones(2), torch.ones(2)

    def close(self):
        self.closed = True


class StubPPO:
    def __init__(self, action_dim, device, obs_shapes, config):
        self.mse_losses = []
        self.entropy_losses = []
        self.surrogate_losses = []
        self.obj_pred_losses = []

    def select_action(self, obs):
        return torch.zeros(2, 1), torch.zeros(2), torch.zeros(2, 1), obs

    def save(self, path):
        Path(path).touch()

    def update(self, buffer, next_obs):
        pass


def training_config(tmp_path):
    return TrainingConfig(
        checkpoint_dir=str(tmp_path),
        runid="test-run",
        num_envs=2,
        max_ep_len=1,
        max_training_timesteps=0,
        K_epochs=1,
        use_object_obs=False,
    )


def prepare_training(monkeypatch, environment):
    monkeypatch.setattr(
        ppo_train, "get_env_infos", lambda *args, **kwargs: (1, {"state": (1,)})
    )
    monkeypatch.setattr(ppo_train, "SubprocVecEnv", lambda *args, **kwargs: environment)
    monkeypatch.setattr(ppo_train, "PPO", StubPPO)
    monkeypatch.setattr(ppo_train, "video_render", lambda *args, **kwargs: None)


def test_startup_error_is_not_hidden_by_cleanup(tmp_path, monkeypatch):
    config = training_config(tmp_path)
    config.use_wandb = True
    monkeypatch.setattr(
        ppo_train.wandb, "init", lambda **kwargs: SimpleNamespace(id="test-run")
    )
    monkeypatch.setattr(
        ppo_train.wandb, "config", SimpleNamespace(update=lambda *args, **kwargs: None)
    )
    exits = []
    monkeypatch.setattr(ppo_train.wandb, "finish", lambda **kwargs: exits.append(kwargs))

    def fail(*args, **kwargs):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(ppo_train, "get_env_infos", fail)
    with pytest.raises(RuntimeError, match="initialization failed"):
        ppo_train.main(config)
    assert exits == [{"exit_code": 1}]


def test_offline_training_logs_each_finished_episode(tmp_path, monkeypatch):
    environment = StubEnvironment()
    prepare_training(monkeypatch, environment)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def unexpected_tracking(*args, **kwargs):
        raise AssertionError("tracking was called")

    for name in ("init", "log", "finish"):
        monkeypatch.setattr(ppo_train.wandb, name, unexpected_tracking)

    ppo_train.main(training_config(tmp_path))
    rows = [
        json.loads(line)
        for line in (tmp_path / "test-run" / "episodes.jsonl").read_text().splitlines()
    ]
    assert rows == [
        {"step": 2, "env_id": 0, "success": True},
        {"step": 2, "env_id": 1, "success": False},
    ]
    assert environment.closed


def test_environment_is_closed_after_training_error(tmp_path, monkeypatch):
    environment = StubEnvironment()
    prepare_training(monkeypatch, environment)

    def fail():
        raise RuntimeError("reset failed")

    monkeypatch.setattr(environment, "reset", fail)
    with pytest.raises(RuntimeError, match="reset failed"):
        ppo_train.main(training_config(tmp_path))
    assert environment.closed


def test_tracking_is_used_when_enabled(tmp_path, monkeypatch):
    environment = StubEnvironment()
    prepare_training(monkeypatch, environment)
    config = training_config(tmp_path)
    config.use_wandb = True
    calls = []
    monkeypatch.setattr(
        ppo_train.wandb,
        "init",
        lambda **kwargs: calls.append(("init", kwargs)) or SimpleNamespace(id="test-run"),
    )
    monkeypatch.setattr(
        ppo_train.wandb,
        "config",
        SimpleNamespace(update=lambda *args, **kwargs: calls.append(("config", args))),
    )
    monkeypatch.setattr(
        ppo_train.wandb, "log", lambda *args, **kwargs: calls.append(("log", kwargs))
    )
    monkeypatch.setattr(
        ppo_train.wandb, "finish", lambda **kwargs: calls.append(("finish", kwargs))
    )

    ppo_train.main(config)
    assert [name for name, _ in calls] == ["init", "config", "log", "finish"]
    assert calls[0][1]["resume"] == "allow"
    assert calls[-1][1] == {"exit_code": 0}


def test_video_does_not_append_run_id_twice(tmp_path, monkeypatch):
    config = training_config(tmp_path)
    config.update_path()

    def fail(*args, **kwargs):
        raise RuntimeError("stop before rendering")

    monkeypatch.setattr(video, "get_env_infos", fail)
    with pytest.raises(RuntimeError, match="stop before rendering"):
        video.video_render(config, agent=object())
    assert config.checkpoint_dir == str(tmp_path / "test-run")
