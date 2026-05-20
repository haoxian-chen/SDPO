# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    _compute_neg_g_u,
    compute_self_distillation_loss,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)
from verl.workers.config.actor import DIVERGENCE_TYPES


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


@pytest.mark.parametrize("divergence_type", DIVERGENCE_TYPES)
def test_compute_neg_g_u_matches_tinker_formulas(divergence_type: str):
    """Keep SDPO's sampled-token divergence helper aligned with tinker-cookbook."""
    log_u = torch.tensor([[-20.0, -1.0, 0.0, 1.0, 20.0]], dtype=torch.float64)
    actual = _compute_neg_g_u(log_u, divergence_type)

    # reverse_kl and improved_reverse_kl don't use exp() — they read log_u directly.
    if divergence_type == "reverse_kl":
        expected = log_u
    elif divergence_type == "improved_reverse_kl":
        expected = log_u - 1.0
    else:
        log_u_clamped = torch.clamp(log_u, min=-10.0, max=10.0)
        u = torch.exp(log_u_clamped)
        log2 = torch.tensor(np.log(2.0), dtype=log_u.dtype)
        if divergence_type == "forward_kl":
            expected = -u * log_u_clamped
        elif divergence_type == "jsd":
            expected = -0.5 * (u * log_u_clamped - (u + 1.0) * (torch.log1p(u) - log2))
        elif divergence_type == "improved_forward_kl":
            expected = u
        elif divergence_type == "improved_jsd":
            expected = 0.5 * (torch.log1p(u) - log2)
        else:
            raise AssertionError(f"Unhandled divergence_type: {divergence_type}")

    assert torch.allclose(actual, expected)


def test_compute_self_distillation_loss_reverse_kl_matches_prior_sampled_loss():
    student_log_probs = torch.tensor(
        [[-0.2, -1.3, -2.0], [-0.7, -0.4, -1.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_log_probs = torch.tensor(
        [[-0.5, -1.0, -1.7], [-0.9, -0.3, -1.4]],
        dtype=torch.float32,
    )
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32)
    cfg = SimpleNamespace(
        full_logit_distillation=False,
        alpha=1.0,
        divergence_type="reverse_kl",
        is_clip=None,
    )

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        loss_agg_mode="token-mean",
    )

    prior_per_token_loss = (student_log_probs - teacher_log_probs).detach() * student_log_probs
    expected_loss = (prior_per_token_loss * response_mask).sum() / response_mask.sum()
    assert torch.equal(loss, expected_loss)

    actual_grad = torch.autograd.grad(loss, student_log_probs, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_loss, student_log_probs)[0]
    assert torch.equal(actual_grad, expected_grad)

    expected_teacher_kl = ((student_log_probs - teacher_log_probs) * response_mask).sum() / response_mask.sum()
    assert metrics.keys() == {"actor/distill_div/reverse_kl", "actor/distill_teacher_kl"}
    assert metrics["actor/distill_div/reverse_kl"] == pytest.approx(expected_teacher_kl.item())
    assert metrics["actor/distill_teacher_kl"] == pytest.approx(expected_teacher_kl.item())


# At student==teacher (log_u=0, u=1):
#   reverse_kl, forward_kl, jsd, improved_jsd all have g(1) = 0
#       -> per-token signal vanishes, loss is 0, metric is 0.
#   improved_forward_kl uses g(u)=-u, so g(1) = -1, -g(1) = 1
#       -> per_token_loss = -student_log_probs, metric = -1.
#   improved_reverse_kl uses g(u)=-ln u + 1, so g(1) = 1, -g(1) = -1
#       -> per_token_loss =  student_log_probs, metric = 1.
# These constant offsets do not change the expected gradient (REINFORCE
# absorbs constant baselines) but they affect the loss-value scale.
_NEG_G_AT_MATCH = {
    "reverse_kl": 0.0,
    "forward_kl": 0.0,
    "jsd": 0.0,
    "improved_forward_kl": 1.0,
    "improved_reverse_kl": -1.0,
    "improved_jsd": 0.0,
}


@pytest.mark.parametrize("divergence_type", DIVERGENCE_TYPES)
def test_compute_self_distillation_loss_at_match_and_metrics(divergence_type: str):
    student_log_probs = torch.tensor(
        [[-0.2, -1.3, -2.0], [-0.7, -0.4, -1.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_log_probs = student_log_probs.detach().clone()
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32)
    cfg = SimpleNamespace(
        full_logit_distillation=False,
        alpha=0.5,
        divergence_type=divergence_type,
        is_clip=None,
    )

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        loss_agg_mode="token-mean",
    )

    neg_g_const = _NEG_G_AT_MATCH[divergence_type]
    expected_per_token_loss = -neg_g_const * student_log_probs
    expected_loss = (expected_per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
    # actor/distill_teacher_kl = mean(-log_u) = 0 at student==teacher for all divergences.
    # The selected divergence metric is mean(g(u)) = -neg_g_const.
    expected_div_metric = -neg_g_const

    assert loss.shape == ()
    assert torch.allclose(loss, expected_loss, atol=1e-7)
    assert metrics.keys() == {f"actor/distill_div/{divergence_type}", "actor/distill_teacher_kl"}
    assert metrics[f"actor/distill_div/{divergence_type}"] == pytest.approx(expected_div_metric, abs=1e-7)
    assert metrics["actor/distill_teacher_kl"] == pytest.approx(0.0, abs=1e-7)


def test_compute_self_distillation_loss_invalid_divergence_type_raises():
    cfg = SimpleNamespace(
        full_logit_distillation=False,
        alpha=1.0,
        divergence_type="not_a_divergence",
        is_clip=None,
    )
    log_probs = torch.zeros(1, 2)
    response_mask = torch.ones(1, 2)

    with pytest.raises(ValueError, match="self_distillation.divergence_type"):
        compute_self_distillation_loss(
            student_log_probs=log_probs,
            teacher_log_probs=log_probs,
            response_mask=response_mask,
            self_distillation_config=cfg,
        )


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
