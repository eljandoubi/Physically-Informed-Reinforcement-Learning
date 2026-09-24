"""Shared fixtures for the test suite.

Tests in this repository avoid instantiating real Robosuite/MuJoCo
environments (they require a rendering backend and heavy assets that are not
appropriate for a fast, deterministic CI run). Instead, they exercise the
pure PyTorch model/algorithm code with small, hand-built observation
dictionaries that match the shapes Robosuite would normally produce.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Make randomly-initialized tensors reproducible across test runs."""
    torch.manual_seed(0)


@pytest.fixture
def img_size() -> int:
    return 64


@pytest.fixture
def proprio_dim() -> int:
    return 12


@pytest.fixture
def object_dim() -> int:
    return 7


@pytest.fixture
def embed_dim() -> int:
    return 32


@pytest.fixture
def obs_shapes(img_size, proprio_dim, object_dim):
    """Factory returning the observation-shape dict PPO/ActorCritic expect."""

    def _shapes(use_object_obs: bool = False) -> dict[str, tuple[int, ...]]:
        shapes = {
            "robot0_eye_in_hand_image": (img_size, img_size, 3),
            "robot0_eye_in_hand_depth": (img_size, img_size, 1),
            "robot0_proprio-state": (proprio_dim,),
        }
        if use_object_obs:
            shapes["object-state"] = (object_dim,)
        return shapes

    return _shapes


@pytest.fixture
def make_obs():
    """Factory building a batch of random observations for the given shapes."""

    def _make(
        shapes: dict[str, tuple[int, ...]], batch_size: int
    ) -> dict[str, torch.Tensor]:
        obs = {}
        for key, shape in shapes.items():
            if "image" in key:
                obs[key] = torch.randint(0, 256, (batch_size, *shape)).float()
            elif "depth" in key:
                obs[key] = torch.rand(batch_size, *shape)
            else:
                obs[key] = torch.randn(batch_size, *shape)
        return obs

    return _make
