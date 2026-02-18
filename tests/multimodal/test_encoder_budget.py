# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MultiModalBudget.get_modality_with_max_tokens().

Verifies deterministic modality selection across TP ranks.
Run with: pytest tests/multimodal/test_encoder_budget.py -v
"""

import pytest

from vllm.multimodal.encoder_budget import MultiModalBudget


def _make_budget_with_toks(toks_per_item: dict[str, int]) -> MultiModalBudget:
    """Create a minimal MultiModalBudget with only mm_max_toks_per_item set."""
    obj = object.__new__(MultiModalBudget)
    obj.mm_max_toks_per_item = toks_per_item
    return obj


class TestGetModalityWithMaxTokens:
    """Ensure get_modality_with_max_tokens() is deterministic across ranks."""

    def test_single_modality(self):
        budget = _make_budget_with_toks({"image": 1024})
        assert budget.get_modality_with_max_tokens() == "image"

    def test_different_token_counts(self):
        budget = _make_budget_with_toks({"image": 1024, "video": 4096})
        assert budget.get_modality_with_max_tokens() == "video"

    def test_equal_counts_deterministic_regardless_of_insertion_order(self):
        """When token counts tie, result must not depend on dict order.

        This is the core regression test: tower_modalities is a set,
        so dict iteration order varies between processes. Without the
        secondary sort key, different TP ranks can pick different
        modalities, causing mismatched NCCL collectives and a hang.
        """
        budget_a = _make_budget_with_toks({"image": 1024, "video": 1024})
        budget_b = _make_budget_with_toks({"video": 1024, "image": 1024})
        assert (
            budget_a.get_modality_with_max_tokens()
            == budget_b.get_modality_with_max_tokens()
        )
        # Secondary key is modality name; "video" > "image" alphabetically
        assert budget_a.get_modality_with_max_tokens() == "video"

    def test_three_modalities_clear_winner(self):
        budget = _make_budget_with_toks(
            {
                "image": 1024,
                "video": 4096,
                "audio": 2048,
            }
        )
        assert budget.get_modality_with_max_tokens() == "video"

    def test_three_modalities_all_tied(self):
        """All three tie — alphabetically last name wins."""
        budget = _make_budget_with_toks(
            {
                "audio": 1024,
                "image": 1024,
                "video": 1024,
            }
        )
        assert budget.get_modality_with_max_tokens() == "video"

    @pytest.mark.parametrize(
        "order",
        [
            ["audio", "image", "video"],
            ["video", "image", "audio"],
            ["image", "video", "audio"],
            ["video", "audio", "image"],
            ["audio", "video", "image"],
            ["image", "audio", "video"],
        ],
    )
    def test_all_insertion_orders_give_same_result(self, order: list[str]):
        """Exhaustively verify all 6 permutations of 3 tied modalities."""
        toks = {mod: 1024 for mod in order}
        budget = _make_budget_with_toks(toks)
        assert budget.get_modality_with_max_tokens() == "video"
