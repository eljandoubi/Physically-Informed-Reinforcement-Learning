"""Unit tests for ``RolloutBuffer`` (storage + GAE computation)."""

import pytest
import torch

from pirl.rollout import RolloutBuffer


def _make_buffer(T, N, action_dim, obs_shapes=None):
    obs_shapes = obs_shapes or {"feat": (4,)}
    return RolloutBuffer(T, N, obs_shapes, action_dim, device=torch.device("cpu"))


def _add_dummy_step(buffer, N, action_dim, reward=0.0, done=0.0):
    obs = {k: torch.zeros(N, *shape) for k, shape in buffer.obs_shapes.items()}
    actions = torch.zeros(N, action_dim)
    rewards = torch.full((N,), reward)
    dones = torch.full((N,), done)
    values = torch.zeros(N, 1)
    logprobs = torch.zeros(N)
    buffer.add(obs, actions, rewards, dones, values, logprobs)


def test_len_and_ptr_track_number_of_steps_added():
    buffer = _make_buffer(T=3, N=2, action_dim=3)

    assert len(buffer) == 0

    _add_dummy_step(buffer, N=2, action_dim=3)

    assert buffer.ptr == 1
    assert len(buffer) == 2  # ptr * N


def test_add_raises_once_buffer_is_full():
    buffer = _make_buffer(T=1, N=2, action_dim=3)
    _add_dummy_step(buffer, N=2, action_dim=3)

    with pytest.raises(AssertionError):
        _add_dummy_step(buffer, N=2, action_dim=3)


def test_reset_clears_pointer_but_keeps_storage():
    buffer = _make_buffer(T=2, N=2, action_dim=3)
    _add_dummy_step(buffer, N=2, action_dim=3)

    buffer.reset()

    assert buffer.ptr == 0
    assert len(buffer) == 0


def test_get_flattens_time_and_env_dimensions():
    T, N, action_dim = 2, 3, 4
    buffer = _make_buffer(T=T, N=N, action_dim=action_dim, obs_shapes={"feat": (5,)})
    for _ in range(T):
        _add_dummy_step(buffer, N=N, action_dim=action_dim)

    batch = buffer.get()

    assert batch["obs"]["feat"].shape == (T * N, 5)
    assert batch["actions"].shape == (T * N, action_dim)
    assert batch["logprobs"].shape == (T * N,)


def test_compute_gae_matches_hand_computation_without_dones():
    buffer = _make_buffer(T=3, N=1, action_dim=1)
    buffer.rewards = torch.tensor([[1.0], [1.0], [1.0]])
    buffer.values = torch.zeros(3, 1)
    buffer.dones = torch.zeros(3, 1)

    returns, advantages = buffer.compute_gae(
        last_value=torch.zeros(1), gamma=1.0, lam=1.0
    )

    # With gamma = lambda = 1 and zero value estimates, the return at each step
    # is simply the sum of the rewards still to come.
    expected = torch.tensor([[3.0], [2.0], [1.0]])
    assert torch.allclose(returns, expected)
    assert torch.allclose(advantages, expected)  # values are all zero


def test_compute_gae_resets_bootstrap_at_done_flags():
    buffer = _make_buffer(T=2, N=1, action_dim=1)
    buffer.rewards = torch.tensor([[5.0], [1.0]])
    buffer.values = torch.zeros(2, 1)
    buffer.dones = torch.tensor([[1.0], [0.0]])

    returns, _ = buffer.compute_gae(last_value=torch.zeros(1), gamma=1.0, lam=1.0)

    # The done flag at t=0 must stop the advantage at t=1 from leaking backwards.
    assert torch.allclose(returns, torch.tensor([[5.0], [1.0]]))
