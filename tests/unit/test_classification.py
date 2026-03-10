"""Unit tests for classification metrics."""

import math
import pytest


class TestPredicateMatch:
    """Tests for predicate_match function."""

    def test_predicate_match_shex_exact(self, shex_constraint_simple):
        """Test exact predicate match for ShEx."""
        from shapespresso.metrics.classification import predicate_match

        y_true = shex_constraint_simple.copy()
        y_pred = shex_constraint_simple.copy()

        assert predicate_match(y_true, y_pred, "ShEx") is True

    def test_predicate_match_shex_different(self, shex_constraint_simple):
        """Test predicate mismatch for ShEx."""
        from shapespresso.metrics.classification import predicate_match

        y_true = shex_constraint_simple.copy()
        y_pred = shex_constraint_simple.copy()
        y_pred["predicate"] = "http://schema.org/email"

        assert predicate_match(y_true, y_pred, "ShEx") is False

    def test_predicate_match_shacl_exact(self, shacl_constraint_simple):
        """Test exact predicate match for SHACL."""
        from shapespresso.metrics.classification import predicate_match

        y_true = shacl_constraint_simple.copy()
        y_pred = shacl_constraint_simple.copy()

        assert predicate_match(y_true, y_pred, "SHACL") is True

    def test_predicate_match_shacl_different(self, shacl_constraint_simple):
        """Test predicate mismatch for SHACL."""
        from shapespresso.metrics.classification import predicate_match

        y_true = shacl_constraint_simple.copy()
        y_pred = shacl_constraint_simple.copy()
        y_pred["http://www.w3.org/ns/shacl#path"] = [{"@id": "http://schema.org/email"}]

        assert predicate_match(y_true, y_pred, "SHACL") is False

    def test_predicate_match_shex_missing_predicate(self):
        """Test predicate match when predicate is missing."""
        from shapespresso.metrics.classification import predicate_match

        y_true = {"type": "TripleConstraint"}
        y_pred = {"type": "TripleConstraint", "predicate": "http://schema.org/name"}

        assert predicate_match(y_true, y_pred, "ShEx") is False


class TestCardinalityMatch:
    """Tests for cardinality_match function."""

    def test_cardinality_match_shex_exact(self):
        """Test exact cardinality match for ShEx."""
        from shapespresso.metrics.classification import cardinality_match

        y_true = {"min": 1, "max": 1}
        y_pred = {"min": 1, "max": 1}

        assert cardinality_match(y_true, y_pred, "ShEx") is True

    def test_cardinality_match_shex_different(self):
        """Test cardinality mismatch for ShEx."""
        from shapespresso.metrics.classification import cardinality_match

        y_true = {"min": 1, "max": 1}
        y_pred = {"min": 0, "max": -1}

        assert cardinality_match(y_true, y_pred, "ShEx") is False

    def test_cardinality_match_shex_defaults(self):
        """Test cardinality match with defaults for ShEx."""
        from shapespresso.metrics.classification import cardinality_match

        y_true = {}  # defaults to min=1, max=1
        y_pred = {}  # defaults to min=1, max=1

        assert cardinality_match(y_true, y_pred, "ShEx") is True

    def test_cardinality_match_shacl_exact(self):
        """Test exact cardinality match for SHACL."""
        from shapespresso.metrics.classification import cardinality_match

        y_true = {
            "http://www.w3.org/ns/shacl#minCount": [{"@value": 1}],
            "http://www.w3.org/ns/shacl#maxCount": [{"@value": 1}]
        }
        y_pred = {
            "http://www.w3.org/ns/shacl#minCount": [{"@value": 1}],
            "http://www.w3.org/ns/shacl#maxCount": [{"@value": 1}]
        }

        assert cardinality_match(y_true, y_pred, "SHACL") is True

    def test_cardinality_match_shacl_different(self):
        """Test cardinality mismatch for SHACL."""
        from shapespresso.metrics.classification import cardinality_match

        y_true = {
            "http://www.w3.org/ns/shacl#minCount": [{"@value": 1}],
            "http://www.w3.org/ns/shacl#maxCount": [{"@value": 1}]
        }
        y_pred = {
            "http://www.w3.org/ns/shacl#minCount": [{"@value": 0}]
            # no maxCount = infinity
        }

        assert cardinality_match(y_true, y_pred, "SHACL") is False

    def test_cardinality_match_shacl_no_max(self):
        """Test SHACL cardinality with no maxCount (unbounded)."""
        from shapespresso.metrics.classification import cardinality_match

        y_true = {
            "http://www.w3.org/ns/shacl#minCount": [{"@value": 0}]
        }
        y_pred = {
            "http://www.w3.org/ns/shacl#minCount": [{"@value": 0}]
        }

        # Both have no maxCount, so both are infinity
        assert cardinality_match(y_true, y_pred, "SHACL") is True


class TestLoosenedCardinalityMatch:
    """Tests for loosened_cardinality_match function."""

    def test_loosened_cardinality_match_shex_broader(self):
        """Test loosened match when prediction is broader."""
        from shapespresso.metrics.classification import loosened_cardinality_match

        y_true = {"min": 1, "max": 5}
        y_pred = {"min": 0, "max": -1}  # -1 = unbounded

        # pred_min <= true_min and true_max <= pred_max
        assert loosened_cardinality_match(y_true, y_pred, "ShEx") is True

    def test_loosened_cardinality_match_shex_narrower(self):
        """Test loosened match fails when prediction is narrower."""
        from shapespresso.metrics.classification import loosened_cardinality_match

        y_true = {"min": 0, "max": -1}  # unbounded
        y_pred = {"min": 1, "max": 5}

        # Prediction is narrower than ground truth
        assert loosened_cardinality_match(y_true, y_pred, "ShEx") is False

    def test_loosened_cardinality_match_rejected_property(self):
        """Test that rejected properties (0,0) must be matched exactly."""
        from shapespresso.metrics.classification import loosened_cardinality_match

        y_true = {"min": 0, "max": 0}  # rejected
        y_pred = {"min": 0, "max": -1}  # not rejected

        assert loosened_cardinality_match(y_true, y_pred, "ShEx") is False

    def test_loosened_cardinality_match_both_rejected(self):
        """Test that both rejected properties match."""
        from shapespresso.metrics.classification import loosened_cardinality_match

        y_true = {"min": 0, "max": 0}
        y_pred = {"min": 0, "max": 0}

        assert loosened_cardinality_match(y_true, y_pred, "ShEx") is True


class TestNodeConstraintMatch:
    """Tests for node_constraint_match function."""

    def test_node_constraint_match_shex_exact(self, shex_constraint_simple):
        """Test exact node constraint match for ShEx."""
        from shapespresso.metrics.classification import node_constraint_match

        y_true = shex_constraint_simple.copy()
        y_pred = shex_constraint_simple.copy()

        assert node_constraint_match(y_true, y_pred, "ShEx") is True

    def test_node_constraint_match_shex_different(self):
        """Test node constraint mismatch for ShEx."""
        from shapespresso.metrics.classification import node_constraint_match

        y_true = {
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/2001/XMLSchema#string"
            }
        }
        y_pred = {
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/2001/XMLSchema#integer"
            }
        }

        assert node_constraint_match(y_true, y_pred, "ShEx") is False

    def test_node_constraint_match_shacl_exact(self, shacl_constraint_simple):
        """Test exact node constraint match for SHACL."""
        from shapespresso.metrics.classification import node_constraint_match

        y_true = shacl_constraint_simple.copy()
        y_pred = shacl_constraint_simple.copy()

        assert node_constraint_match(y_true, y_pred, "SHACL") is True


class TestExactConstraintMatch:
    """Tests for exact_constraint_match function."""

    def test_exact_constraint_match_shex_all_match(self, shex_constraint_simple):
        """Test exact constraint match when all components match."""
        from shapespresso.metrics.classification import exact_constraint_match

        y_true = shex_constraint_simple.copy()
        y_pred = shex_constraint_simple.copy()

        assert exact_constraint_match(y_true, y_pred, "ShEx") is True

    def test_exact_constraint_match_shex_predicate_mismatch(self, shex_constraint_simple):
        """Test exact constraint match fails when predicate differs."""
        from shapespresso.metrics.classification import exact_constraint_match

        y_true = shex_constraint_simple.copy()
        y_pred = shex_constraint_simple.copy()
        y_pred["predicate"] = "http://schema.org/email"

        assert exact_constraint_match(y_true, y_pred, "ShEx") is False

    def test_exact_constraint_match_shex_cardinality_mismatch(self, shex_constraint_simple):
        """Test exact constraint match fails when cardinality differs."""
        from shapespresso.metrics.classification import exact_constraint_match

        y_true = shex_constraint_simple.copy()
        y_pred = shex_constraint_simple.copy()
        y_pred["min"] = 0
        y_pred["max"] = -1

        assert exact_constraint_match(y_true, y_pred, "ShEx") is False


class TestDatatypeMatch:
    """Tests for datatype_match function."""

    def test_datatype_match_shex_same(self):
        """Test datatype match for ShEx with same datatype."""
        from shapespresso.metrics.classification import datatype_match

        y_true = {
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/2001/XMLSchema#string"
            }
        }
        y_pred = {
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/2001/XMLSchema#string"
            }
        }

        assert datatype_match(y_true, y_pred, "ShEx") is True

    def test_datatype_match_shex_langstring_to_string(self):
        """Test that langString matches string (normalization)."""
        from shapespresso.metrics.classification import datatype_match

        y_true = {
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
            }
        }
        y_pred = {
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/2001/XMLSchema#string"
            }
        }

        # Both should normalize to string
        assert datatype_match(y_true, y_pred, "ShEx") is True

    def test_datatype_match_shex_float_to_decimal(self):
        """Test that float matches decimal (normalization)."""
        from shapespresso.metrics.classification import datatype_match

        y_true = {
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/2001/XMLSchema#float"
            }
        }
        y_pred = {
            "valueExpr": {
                "type": "NodeConstraint",
                "datatype": "http://www.w3.org/2001/XMLSchema#decimal"
            }
        }

        # Both should normalize to decimal
        assert datatype_match(y_true, y_pred, "ShEx") is True


class TestCountTruePositives:
    """Tests for count_true_positives function."""

    def test_count_true_positives_all_match(self, shex_constraint_simple):
        """Test counting true positives when all match."""
        from shapespresso.metrics.classification import count_true_positives

        y_true = [shex_constraint_simple]
        y_pred = [shex_constraint_simple]

        count, tps = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=y_true,
            y_pred=y_pred
        )

        assert count == 1
        assert len(tps) == 1

    def test_count_true_positives_none_match(self, shex_constraint_simple):
        """Test counting true positives when none match."""
        from shapespresso.metrics.classification import count_true_positives

        y_true = [shex_constraint_simple]
        y_pred_different = shex_constraint_simple.copy()
        y_pred_different["predicate"] = "http://schema.org/email"
        y_pred = [y_pred_different]

        count, tps = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=y_true,
            y_pred=y_pred
        )

        assert count == 0
        assert len(tps) == 0

    def test_count_true_positives_partial_match(self, shex_constraint_simple):
        """Test counting true positives with partial matches."""
        from shapespresso.metrics.classification import count_true_positives

        constraint2 = shex_constraint_simple.copy()
        constraint2["predicate"] = "http://schema.org/email"

        y_true = [shex_constraint_simple, constraint2]
        y_pred = [shex_constraint_simple]  # Only one matches

        count, tps = count_true_positives(
            dataset="yagos",
            syntax="ShEx",
            y_true=y_true,
            y_pred=y_pred
        )

        assert count == 1
