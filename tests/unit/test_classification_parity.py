"""Tests verifying SHACL and ShEx classification produce identical results.

This test suite ensures that for semantically equivalent shapes expressed in
both SHACL and ShEx, the classification metrics (precision, recall, F1)
produce exactly the same results.

The test uses simplified YAGO-based shapes as ground truth and "noisy" predicted
shapes that are conceptually equivalent in both languages.
"""

import pytest
from pathlib import Path

from shapespresso.metrics.classification import (
    predicate_match,
    cardinality_match,
    node_constraint_match,
    exact_constraint_match,
    datatype_match,
    loosened_cardinality_match,
    count_true_positives,
    constraint_match,
)
from shapespresso.metrics.utils import (
    extract_shex_constraints,
    extract_shacl_constraints,
)


# Paths to fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SHEX_FIXTURES = FIXTURES_DIR / "shex"
SHACL_FIXTURES = FIXTURES_DIR / "shacl"


class TestClassificationParity:
    """Test suite verifying SHACL and ShEx classification produce identical results."""

    # =========================================================================
    # Fixture Loading
    # =========================================================================

    @pytest.fixture
    def simple_person_shex_gt(self) -> list:
        """Load simple person ShEx ground truth constraints."""
        shex_text = (SHEX_FIXTURES / "simple_person_gt.shex").read_text()
        return extract_shex_constraints(shex_text)

    @pytest.fixture
    def simple_person_shex_pred(self) -> list:
        """Load simple person ShEx predicted constraints."""
        shex_text = (SHEX_FIXTURES / "simple_person_pred.shex").read_text()
        return extract_shex_constraints(shex_text)

    @pytest.fixture
    def simple_person_shacl_gt(self) -> list:
        """Load simple person SHACL ground truth constraints."""
        shacl_text = (SHACL_FIXTURES / "simple_person_gt.ttl").read_text()
        return extract_shacl_constraints(shacl_text)

    @pytest.fixture
    def simple_person_shacl_pred(self) -> list:
        """Load simple person SHACL predicted constraints."""
        shacl_text = (SHACL_FIXTURES / "simple_person_pred.ttl").read_text()
        return extract_shacl_constraints(shacl_text)

    @pytest.fixture
    def scientist_shex_gt(self) -> list:
        """Load scientist ShEx ground truth constraints."""
        shex_text = (SHEX_FIXTURES / "scientist_gt.shex").read_text()
        return extract_shex_constraints(shex_text)

    @pytest.fixture
    def scientist_shex_pred(self) -> list:
        """Load scientist ShEx predicted constraints."""
        shex_text = (SHEX_FIXTURES / "scientist_pred.shex").read_text()
        return extract_shex_constraints(shex_text)

    @pytest.fixture
    def scientist_shacl_gt(self) -> list:
        """Load scientist SHACL ground truth constraints."""
        shacl_text = (SHACL_FIXTURES / "scientist_gt.ttl").read_text()
        return extract_shacl_constraints(shacl_text)

    @pytest.fixture
    def scientist_shacl_pred(self) -> list:
        """Load scientist SHACL predicted constraints."""
        shacl_text = (SHACL_FIXTURES / "scientist_pred.ttl").read_text()
        return extract_shacl_constraints(shacl_text)

    # =========================================================================
    # Simple Person Tests - Constraint Counting
    # =========================================================================

    def test_simple_person_constraint_count_parity(
        self,
        simple_person_shex_gt,
        simple_person_shex_pred,
        simple_person_shacl_gt,
        simple_person_shacl_pred
    ):
        """Verify same number of constraints extracted from both formats."""
        assert len(simple_person_shex_gt) == len(simple_person_shacl_gt), \
            f"GT constraint count mismatch: ShEx={len(simple_person_shex_gt)}, SHACL={len(simple_person_shacl_gt)}"
        assert len(simple_person_shex_pred) == len(simple_person_shacl_pred), \
            f"Pred constraint count mismatch: ShEx={len(simple_person_shex_pred)}, SHACL={len(simple_person_shacl_pred)}"

    def test_simple_person_true_positives_parity(
        self,
        simple_person_shex_gt,
        simple_person_shex_pred,
        simple_person_shacl_gt,
        simple_person_shacl_pred
    ):
        """Verify identical true positive counts for both formats."""
        shex_tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=simple_person_shex_gt,
            y_pred=simple_person_shex_pred
        )

        shacl_tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=simple_person_shacl_gt,
            y_pred=simple_person_shacl_pred
        )

        assert shex_tp_count == shacl_tp_count, \
            f"True positive count mismatch: ShEx={shex_tp_count}, SHACL={shacl_tp_count}"

    def test_simple_person_precision_recall_parity(
        self,
        simple_person_shex_gt,
        simple_person_shex_pred,
        simple_person_shacl_gt,
        simple_person_shacl_pred
    ):
        """Verify identical precision and recall for both formats."""
        # ShEx metrics
        shex_tp, _ = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=simple_person_shex_gt,
            y_pred=simple_person_shex_pred
        )
        shex_precision = shex_tp / len(simple_person_shex_pred) if simple_person_shex_pred else 0
        shex_recall = shex_tp / len(simple_person_shex_gt) if simple_person_shex_gt else 0

        # SHACL metrics
        shacl_tp, _ = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=simple_person_shacl_gt,
            y_pred=simple_person_shacl_pred
        )
        shacl_precision = shacl_tp / len(simple_person_shacl_pred) if simple_person_shacl_pred else 0
        shacl_recall = shacl_tp / len(simple_person_shacl_gt) if simple_person_shacl_gt else 0

        assert abs(shex_precision - shacl_precision) < 0.001, \
            f"Precision mismatch: ShEx={shex_precision:.3f}, SHACL={shacl_precision:.3f}"
        assert abs(shex_recall - shacl_recall) < 0.001, \
            f"Recall mismatch: ShEx={shex_recall:.3f}, SHACL={shacl_recall:.3f}"

    def test_simple_person_f1_parity(
        self,
        simple_person_shex_gt,
        simple_person_shex_pred,
        simple_person_shacl_gt,
        simple_person_shacl_pred
    ):
        """Verify identical F1 score for both formats."""
        # ShEx F1
        shex_tp, _ = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=simple_person_shex_gt,
            y_pred=simple_person_shex_pred
        )
        shex_p = shex_tp / len(simple_person_shex_pred) if simple_person_shex_pred else 0
        shex_r = shex_tp / len(simple_person_shex_gt) if simple_person_shex_gt else 0
        shex_f1 = 2 * shex_p * shex_r / (shex_p + shex_r) if (shex_p + shex_r) > 0 else 0

        # SHACL F1
        shacl_tp, _ = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=simple_person_shacl_gt,
            y_pred=simple_person_shacl_pred
        )
        shacl_p = shacl_tp / len(simple_person_shacl_pred) if simple_person_shacl_pred else 0
        shacl_r = shacl_tp / len(simple_person_shacl_gt) if simple_person_shacl_gt else 0
        shacl_f1 = 2 * shacl_p * shacl_r / (shacl_p + shacl_r) if (shacl_p + shacl_r) > 0 else 0

        assert abs(shex_f1 - shacl_f1) < 0.001, \
            f"F1 mismatch: ShEx={shex_f1:.3f}, SHACL={shacl_f1:.3f}"

    # =========================================================================
    # Scientist Tests - Constraint Counting
    # =========================================================================

    def test_scientist_constraint_count_parity(
        self,
        scientist_shex_gt,
        scientist_shex_pred,
        scientist_shacl_gt,
        scientist_shacl_pred
    ):
        """Verify same number of constraints extracted from both formats."""
        assert len(scientist_shex_gt) == len(scientist_shacl_gt), \
            f"GT constraint count mismatch: ShEx={len(scientist_shex_gt)}, SHACL={len(scientist_shacl_gt)}"
        assert len(scientist_shex_pred) == len(scientist_shacl_pred), \
            f"Pred constraint count mismatch: ShEx={len(scientist_shex_pred)}, SHACL={len(scientist_shacl_pred)}"

    def test_scientist_true_positives_parity(
        self,
        scientist_shex_gt,
        scientist_shex_pred,
        scientist_shacl_gt,
        scientist_shacl_pred
    ):
        """Verify identical true positive counts for scientist shapes."""
        shex_tp_count, shex_tps = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=scientist_shex_gt,
            y_pred=scientist_shex_pred
        )

        shacl_tp_count, shacl_tps = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=scientist_shacl_gt,
            y_pred=scientist_shacl_pred
        )

        assert shex_tp_count == shacl_tp_count, \
            f"True positive count mismatch: ShEx={shex_tp_count}, SHACL={shacl_tp_count}"

    def test_scientist_precision_recall_f1_parity(
        self,
        scientist_shex_gt,
        scientist_shex_pred,
        scientist_shacl_gt,
        scientist_shacl_pred
    ):
        """Verify identical precision, recall, and F1 for scientist shapes."""
        # ShEx metrics
        shex_tp, _ = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=scientist_shex_gt,
            y_pred=scientist_shex_pred
        )
        shex_p = shex_tp / len(scientist_shex_pred) if scientist_shex_pred else 0
        shex_r = shex_tp / len(scientist_shex_gt) if scientist_shex_gt else 0
        shex_f1 = 2 * shex_p * shex_r / (shex_p + shex_r) if (shex_p + shex_r) > 0 else 0

        # SHACL metrics
        shacl_tp, _ = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=scientist_shacl_gt,
            y_pred=scientist_shacl_pred
        )
        shacl_p = shacl_tp / len(scientist_shacl_pred) if scientist_shacl_pred else 0
        shacl_r = shacl_tp / len(scientist_shacl_gt) if scientist_shacl_gt else 0
        shacl_f1 = 2 * shacl_p * shacl_r / (shacl_p + shacl_r) if (shacl_p + shacl_r) > 0 else 0

        assert abs(shex_p - shacl_p) < 0.001, \
            f"Precision mismatch: ShEx={shex_p:.3f}, SHACL={shacl_p:.3f}"
        assert abs(shex_r - shacl_r) < 0.001, \
            f"Recall mismatch: ShEx={shex_r:.3f}, SHACL={shacl_r:.3f}"
        assert abs(shex_f1 - shacl_f1) < 0.001, \
            f"F1 mismatch: ShEx={shex_f1:.3f}, SHACL={shacl_f1:.3f}"

    # =========================================================================
    # Matching Level Tests
    # =========================================================================

    def test_scientist_datatype_matching_parity(
        self,
        scientist_shex_gt,
        scientist_shex_pred,
        scientist_shacl_gt,
        scientist_shacl_pred
    ):
        """Verify identical results with datatype matching level."""
        shex_tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=scientist_shex_gt,
            y_pred=scientist_shex_pred,
            node_constraint_matching_level="datatype"
        )

        shacl_tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=scientist_shacl_gt,
            y_pred=scientist_shacl_pred,
            node_constraint_matching_level="datatype"
        )

        assert shex_tp_count == shacl_tp_count, \
            f"Datatype matching TP count mismatch: ShEx={shex_tp_count}, SHACL={shacl_tp_count}"

    def test_scientist_loosened_cardinality_parity(
        self,
        scientist_shex_gt,
        scientist_shex_pred,
        scientist_shacl_gt,
        scientist_shacl_pred
    ):
        """Verify identical results with loosened cardinality matching."""
        shex_tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=scientist_shex_gt,
            y_pred=scientist_shex_pred,
            cardinality_matching_level="loosened"
        )

        shacl_tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=scientist_shacl_gt,
            y_pred=scientist_shacl_pred,
            cardinality_matching_level="loosened"
        )

        assert shex_tp_count == shacl_tp_count, \
            f"Loosened cardinality TP count mismatch: ShEx={shex_tp_count}, SHACL={shacl_tp_count}"

    def test_scientist_combined_matching_levels_parity(
        self,
        scientist_shex_gt,
        scientist_shex_pred,
        scientist_shacl_gt,
        scientist_shacl_pred
    ):
        """Verify identical results with combined matching levels."""
        shex_tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=scientist_shex_gt,
            y_pred=scientist_shex_pred,
            node_constraint_matching_level="datatype",
            cardinality_matching_level="loosened"
        )

        shacl_tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=scientist_shacl_gt,
            y_pred=scientist_shacl_pred,
            node_constraint_matching_level="datatype",
            cardinality_matching_level="loosened"
        )

        assert shex_tp_count == shacl_tp_count, \
            f"Combined matching TP count mismatch: ShEx={shex_tp_count}, SHACL={shacl_tp_count}"

    # =========================================================================
    # Edge Cases: Perfect Match
    # =========================================================================

    def test_perfect_match_shex(self, simple_person_shex_gt):
        """Verify perfect match when comparing shape to itself (ShEx)."""
        tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=simple_person_shex_gt,
            y_pred=simple_person_shex_gt
        )
        precision = tp_count / len(simple_person_shex_gt)
        recall = tp_count / len(simple_person_shex_gt)

        assert precision == 1.0, f"ShEx perfect match should have precision 1.0, got {precision}"
        assert recall == 1.0, f"ShEx perfect match should have recall 1.0, got {recall}"

    def test_perfect_match_shacl(self, simple_person_shacl_gt):
        """Verify perfect match when comparing shape to itself (SHACL)."""
        tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=simple_person_shacl_gt,
            y_pred=simple_person_shacl_gt
        )
        precision = tp_count / len(simple_person_shacl_gt)
        recall = tp_count / len(simple_person_shacl_gt)

        assert precision == 1.0, f"SHACL perfect match should have precision 1.0, got {precision}"
        assert recall == 1.0, f"SHACL perfect match should have recall 1.0, got {recall}"

    def test_perfect_match_parity(
        self,
        scientist_shex_gt,
        scientist_shacl_gt
    ):
        """Verify perfect match produces identical results for both formats."""
        shex_tp, _ = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=scientist_shex_gt,
            y_pred=scientist_shex_gt
        )
        shex_p = shex_tp / len(scientist_shex_gt)
        shex_r = shex_tp / len(scientist_shex_gt)

        shacl_tp, _ = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=scientist_shacl_gt,
            y_pred=scientist_shacl_gt
        )
        shacl_p = shacl_tp / len(scientist_shacl_gt)
        shacl_r = shacl_tp / len(scientist_shacl_gt)

        assert shex_p == shacl_p == 1.0, "Perfect match precision should be 1.0"
        assert shex_r == shacl_r == 1.0, "Perfect match recall should be 1.0"

    # =========================================================================
    # Edge Cases: No Match
    # =========================================================================

    def test_no_match_shex(self, simple_person_shex_gt):
        """Verify zero match with completely different shapes (ShEx)."""
        # Create completely different constraint
        different_constraints = [{
            "type": "TripleConstraint",
            "predicate": "http://example.org/nonexistent",
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/2001/XMLSchema#boolean"
            },
            "min": 5,
            "max": 10
        }]

        tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=simple_person_shex_gt,
            y_pred=different_constraints
        )

        assert tp_count == 0, f"ShEx should have zero matches with different predicates, got {tp_count}"

    def test_no_match_shacl(self, simple_person_shacl_gt):
        """Verify zero match with completely different shapes (SHACL)."""
        # Create completely different constraint
        different_constraints = [{
            "http://www.w3.org/ns/shacl#path": [
                {"@id": "http://example.org/nonexistent"}
            ],
            "http://www.w3.org/ns/shacl#datatype": [
                {"@id": "http://www.w3.org/2001/XMLSchema#boolean"}
            ],
            "http://www.w3.org/ns/shacl#minCount": [{"@value": 5}],
            "http://www.w3.org/ns/shacl#maxCount": [{"@value": 10}]
        }]

        tp_count, _ = count_true_positives(
            dataset="yagos",
            syntax="SHACL",
            y_true=simple_person_shacl_gt,
            y_pred=different_constraints
        )

        assert tp_count == 0, f"SHACL should have zero matches with different predicates, got {tp_count}"


class TestConstraintExtraction:
    """Tests for constraint extraction from both formats."""

    def test_shex_extraction_returns_list(self):
        """Verify ShEx constraint extraction returns a list."""
        shex_text = (SHEX_FIXTURES / "simple_person_gt.shex").read_text()
        constraints = extract_shex_constraints(shex_text)
        assert isinstance(constraints, list)
        assert len(constraints) > 0

    def test_shacl_extraction_returns_list(self):
        """Verify SHACL constraint extraction returns a list."""
        shacl_text = (SHACL_FIXTURES / "simple_person_gt.ttl").read_text()
        constraints = extract_shacl_constraints(shacl_text)
        assert isinstance(constraints, list)
        assert len(constraints) > 0

    def test_shex_constraints_have_predicate(self):
        """Verify each ShEx constraint has a predicate."""
        shex_text = (SHEX_FIXTURES / "simple_person_gt.shex").read_text()
        constraints = extract_shex_constraints(shex_text)
        for c in constraints:
            assert "predicate" in c, f"ShEx constraint missing predicate: {c}"

    def test_shacl_constraints_have_path(self):
        """Verify each SHACL constraint has sh:path."""
        shacl_text = (SHACL_FIXTURES / "simple_person_gt.ttl").read_text()
        constraints = extract_shacl_constraints(shacl_text)
        for c in constraints:
            assert "http://www.w3.org/ns/shacl#path" in c, \
                f"SHACL constraint missing sh:path: {list(c.keys())}"


class TestMatchingFunctionParity:
    """Test individual matching functions for consistency."""

    def test_cardinality_defaults_shex(self):
        """Verify ShEx cardinality defaults (1,1)."""
        from shapespresso.metrics.classification import cardinality_match

        c1 = {}  # defaults to min=1, max=1
        c2 = {"min": 1, "max": 1}

        assert cardinality_match(c1, c2, "ShEx") is True

    def test_cardinality_defaults_shacl(self):
        """Verify SHACL cardinality defaults (0, inf)."""
        from shapespresso.metrics.classification import cardinality_match

        c1 = {}  # defaults to min=0, max=inf
        c2 = {}

        assert cardinality_match(c1, c2, "SHACL") is True

    def test_datatype_normalization_langstring(self):
        """Verify langString is normalized to string in both formats."""
        from shapespresso.metrics.classification import datatype_match

        # ShEx
        shex_langstring = {
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
            }
        }
        shex_string = {
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/2001/XMLSchema#string"
            }
        }

        assert datatype_match(shex_langstring, shex_string, "ShEx") is True
