"""Unit tests for ``TrainingConfig`` validation and the ``PPO`` agent."""

import pytest
import torch

from pirl.ppo import PPO, TrainingConfig
from pirl.rollout import RolloutBuffer

ACTION_DIM = 4


def _make_config(**overrides):
    defaults = dict(max_ep_len=2, num_envs=2, K_epochs=2)
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# --------------------------------------------------------------------------- #
# TrainingConfig
# --------------------------------------------------------------------------- #


def test_update_timestep_is_derived_from_ep_len_and_num_envs():
    config = TrainingConfig(max_ep_len=10, num_envs=4)

    assert config.update_timestep == 40


@pytest.mark.parametrize(
    "overrides",
    [
        {"lr_actor": 0.0},
        {"lr_critic": -1.0},
        {"shared_lr": 0.0},
        {"gamma": 0.0},
        {"gamma": 1.5},
        {"lam": -0.1},
        {"lam": 1.1},
        {"eps_clip": 0.0},
        {"K_epochs": 0},
        {"max_ep_len": 0},
        {"num_envs": 0},
        {"embed_dim": 0},
    ],
)
def test_rejects_invalid_hyperparameters(overrides):
    with pytest.raises(AssertionError):
        TrainingConfig(**overrides)


def test_action_std_bounds_only_enforced_for_fixed_variance():
    with pytest.raises(AssertionError):
        TrainingConfig(fixed_policy_variance=True, action_std=0.0)
    with pytest.raises(AssertionError):
        TrainingConfig(fixed_policy_variance=True, action_std=1.5)

    # No constraint on action_std when the policy learns its own variance.
    TrainingConfig(fixed_policy_variance=False, action_std=1.5)


def test_load_checkpoint_path_must_exist(tmp_path):
    missing = tmp_path / "missing.pth"

    with pytest.raises(AssertionError):
        TrainingConfig(load_checkpoint_path=str(missing))


def test_unused_loss_coefficients_are_zeroed_out():
    config = TrainingConfig(use_object_obs=False, obj_pred_coef=2.0)
    assert config.obj_pred_coef == 0.0

    config = TrainingConfig(fixed_policy_variance=True, entropy_coef=0.5)
    assert config.entropy_coef == 0.0


def test_update_path_creates_the_checkpoint_directory(tmp_path):
    config = TrainingConfig(checkpoint_dir=str(tmp_path / "ckpts"), runid="abc123")

    config.update_path()

    expected_dir = str(tmp_path / "ckpts" / "abc123")
    assert config.checkpoint_dir == expected_dir
    assert (tmp_path / "ckpts" / "abc123").is_dir()


def test_set_id_updates_runid():
    config = TrainingConfig()

    config.set_id("my-run")

    assert config.runid == "my-run"


# --------------------------------------------------------------------------- #
# PPO agent
# --------------------------------------------------------------------------- #


def test_select_action_shapes(obs_shapes, make_obs, img_size, embed_dim):
    config = _make_config(img_size=img_size, embed_dim=embed_dim, use_object_obs=False)
    shapes = obs_shapes(use_object_obs=False)
    agent = PPO(ACTION_DIM, torch.device("cpu"), shapes, config)
    obs = make_obs(shapes, config.num_envs)

    action, log_prob, values, obs_on_device = agent.select_action(obs)

    assert action.shape == (config.num_envs, ACTION_DIM)
    assert log_prob.shape == (config.num_envs,)
    assert values.shape == (config.num_envs, 1)
    assert set(obs_on_device.keys()) == set(shapes.keys())


def test_update_runs_and_resyncs_old_policy(obs_shapes, make_obs, img_size, embed_dim):
    config = _make_config(img_size=img_size, embed_dim=embed_dim, use_object_obs=True)
    shapes = obs_shapes(use_object_obs=True)
    agent = PPO(ACTION_DIM, torch.device("cpu"), shapes, config)
    buffer = RolloutBuffer(
        config.max_ep_len, config.num_envs, shapes, ACTION_DIM, torch.device("cpu")
    )

    obs = make_obs(shapes, config.num_envs)
    for _ in range(config.max_ep_len):
        action, log_prob, values, obs = agent.select_action(obs)
        rewards = torch.rand(config.num_envs)
        dones = torch.zeros(config.num_envs)
        buffer.add(obs, action, rewards, dones, values, log_prob)
        obs = make_obs(shapes, config.num_envs)

    assert len(buffer) == config.update_timestep

    agent.update(buffer, obs)

    assert len(agent.mse_losses) == 1
    assert len(agent.surrogate_losses) == 1
    assert len(agent.obj_pred_losses) == 1
    # policy_old must be resynced with the freshly-updated policy after each update.
    for p_new, p_old in zip(agent.policy.parameters(), agent.policy_old.parameters()):
        assert torch.equal(p_new, p_old)


def test_save_and_load_round_trip(tmp_path, obs_shapes, img_size, embed_dim):
    config = _make_config(img_size=img_size, embed_dim=embed_dim, use_object_obs=False)
    shapes = obs_shapes(use_object_obs=False)
    agent = PPO(ACTION_DIM, torch.device("cpu"), shapes, config)
    checkpoint_path = tmp_path / "ppo.pth"

    agent.save(str(checkpoint_path))
    assert checkpoint_path.is_file()

    reloaded = PPO(ACTION_DIM, torch.device("cpu"), shapes, config)
    reloaded.load(str(checkpoint_path))

    for p1, p2 in zip(agent.policy.parameters(), reloaded.policy.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(agent.policy.parameters(), reloaded.policy_old.parameters()):
        assert torch.equal(p1, p2)
