"""Unit tests for the pure data-conversion helpers in ``robotenv``.

Environment-creation helpers (``make_env``, ``get_env_infos``,
``SubprocVecEnv``) require a real Robosuite/MuJoCo rendering backend and are
exercised manually / in integration testing rather than in this fast unit
suite. Importing ``robotenv`` itself requires Robosuite to be installed, so
the whole module is skipped gracefully if it isn't available.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("robosuite")

from pirl.robotenv import filter_state, stack_obs, worker  # noqa: E402


def test_filter_state_only_keeps_requested_keys_as_float32_tensors():
    state = {
        "a": np.zeros((2, 2), dtype=np.float64),
        "b": np.ones(3, dtype=np.uint8),
        "c": np.array([1.0, 2.0], dtype=np.float32),
    }

    result = filter_state(state, ["a", "c"])

    assert set(result.keys()) == {"a", "c"}
    assert all(t.dtype == torch.float32 for t in result.values())
    assert torch.equal(result["a"], torch.zeros(2, 2))
    assert torch.equal(result["c"], torch.tensor([1.0, 2.0]))


def test_stack_obs_stacks_each_key_along_a_new_batch_dimension():
    obs_list = [
        {"feat": torch.tensor([1.0, 2.0]), "scalar": torch.tensor(1.0)},
        {"feat": torch.tensor([3.0, 4.0]), "scalar": torch.tensor(2.0)},
    ]

    stacked = stack_obs(obs_list)

    assert stacked["feat"].shape == (2, 2)
    assert torch.equal(stacked["feat"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(stacked["scalar"], torch.tensor([1.0, 2.0]))


def test_worker_reports_success_for_completed_episodes_and_resets_it():
    class Remote:
        def __init__(self):
            self.commands = iter(
                [
                    ("step", None),
                    ("step", None),
                    ("step", None),
                    ("reset", None),
                    ("step", None),
                    ("close", None),
                ]
            )
            self.sent = []

        def recv(self):
            return next(self.commands)

        def send(self, value):
            self.sent.append(value)

    class Environment:
        def __init__(self):
            self.steps = iter(
                [(False, True), (True, False), (False, True), (True, False)]
            )
            self.success = False

        def step(self, action):
            done, self.success = next(self.steps)
            return {"state": np.array([1.0])}, 0.0, done, {}

        def _check_success(self):
            return self.success

        def reset(self):
            return {"state": np.array([0.0])}

    remote = Remote()
    environment = Environment()
    worker(remote, lambda: environment, {}, ["state"])

    assert remote.sent[0][3] is None
    assert remote.sent[1][3] is True
    assert remote.sent[2][3] is None
    assert remote.sent[4][3] is False
