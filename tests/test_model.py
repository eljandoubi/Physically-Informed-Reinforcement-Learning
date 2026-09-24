"""Unit tests for the multimodal ``Backbone``/``ActorCritic`` networks."""

import math

import torch

from pirl.model import ActorCritic, Backbone

BATCH_SIZE = 3
ACTION_DIM = 4


def test_backbone_forward_returns_fused_embedding(
    obs_shapes, make_obs, img_size, proprio_dim, embed_dim
):
    backbone = Backbone(img_size=img_size, proprio_dim=proprio_dim, embed_dim=embed_dim)
    obs = make_obs(obs_shapes(), BATCH_SIZE)

    fused = backbone(obs)

    assert fused.shape == (BATCH_SIZE, embed_dim)
    assert torch.isfinite(fused).all()


def test_actor_critic_fixed_variance_shapes_and_value(
    obs_shapes, make_obs, img_size, proprio_dim, embed_dim
):
    action_std = 0.5
    model = ActorCritic(
        ACTION_DIM,
        img_size=img_size,
        proprio_dim=proprio_dim,
        fixed_policy_variance=True,
        action_std=action_std,
        embed_dim=embed_dim,
    )
    obs = make_obs(obs_shapes(), BATCH_SIZE)

    action_mean, action_var, state_value, object_pred = model(obs)

    assert action_mean.shape == (BATCH_SIZE, ACTION_DIM)
    assert action_var.shape == (BATCH_SIZE, ACTION_DIM)
    assert state_value.shape == (BATCH_SIZE, 1)
    assert object_pred is None
    # The fixed-variance buffer is broadcast identically to every sample/action.
    assert torch.allclose(action_var, torch.full_like(action_var, action_std**2))


def test_actor_critic_learned_variance_is_clamped(
    obs_shapes, make_obs, img_size, proprio_dim, embed_dim
):
    model = ActorCritic(
        ACTION_DIM,
        img_size=img_size,
        proprio_dim=proprio_dim,
        fixed_policy_variance=False,
        embed_dim=embed_dim,
    )
    obs = make_obs(obs_shapes(), BATCH_SIZE)

    action_mean, action_var, _, _ = model(obs)

    assert action_mean.shape == (BATCH_SIZE, ACTION_DIM)
    assert action_var.shape == (BATCH_SIZE, ACTION_DIM)
    # log-variance is clamped to [-5, -0.5], so variance must live in [e^-5, e^-0.5].
    assert torch.all(action_var >= math.exp(-5.0) - 1e-6)
    assert torch.all(action_var <= math.exp(-0.5) + 1e-6)


def test_actor_critic_predicts_object_state_when_enabled(
    obs_shapes, make_obs, img_size, proprio_dim, embed_dim, object_dim
):
    model = ActorCritic(
        ACTION_DIM,
        img_size=img_size,
        proprio_dim=proprio_dim,
        embed_dim=embed_dim,
        predict_object_state=True,
        object_dim=object_dim,
    )
    obs = make_obs(obs_shapes(use_object_obs=True), BATCH_SIZE)

    *_, object_pred = model(obs)

    assert object_pred is not None
    assert object_pred.shape == (BATCH_SIZE, object_dim)


def test_actor_critic_object_state_is_none_when_disabled(
    obs_shapes, make_obs, img_size, proprio_dim, embed_dim
):
    model = ActorCritic(
        ACTION_DIM,
        img_size=img_size,
        proprio_dim=proprio_dim,
        embed_dim=embed_dim,
        predict_object_state=False,
    )
    obs = make_obs(obs_shapes(), BATCH_SIZE)

    *_, object_pred = model(obs)

    assert object_pred is None
