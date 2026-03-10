"""Unit tests for ShEx Pydantic models."""

import pytest
from pydantic import ValidationError


class TestNodeConstraint:
    """Tests for NodeConstraint Pydantic model."""

    def test_node_constraint_with_datatype(self):
        """Test NodeConstraint with datatype."""
        from shapespresso.syntax.shexj import NodeConstraint

        nc = NodeConstraint(
            type="NodeConstraint",
            datatype="http://www.w3.org/2001/XMLSchema#string"
        )
        assert nc.type == "NodeConstraint"
        assert nc.datatype == "http://www.w3.org/2001/XMLSchema#string"
        assert nc.nodeKind is None
        assert nc.values is None

    def test_node_constraint_with_nodekind(self):
        """Test NodeConstraint with nodeKind."""
        from shapespresso.syntax.shexj import NodeConstraint

        nc = NodeConstraint(
            type="NodeConstraint",
            nodeKind="iri"
        )
        assert nc.type == "NodeConstraint"
        assert nc.nodeKind == "iri"
        assert nc.datatype is None

    def test_node_constraint_with_values(self):
        """Test NodeConstraint with value set."""
        from shapespresso.syntax.shexj import NodeConstraint

        nc = NodeConstraint(
            type="NodeConstraint",
            values=["http://example.org/Male", "http://example.org/Female"]
        )
        assert nc.type == "NodeConstraint"
        assert nc.values is not None
        assert len(nc.values) == 2

    def test_node_constraint_mutual_exclusion(self):
        """Test that exactly one of nodeKind, datatype, or values must be set."""
        from shapespresso.syntax.shexj import NodeConstraint

        # Multiple constraints set should raise error
        with pytest.raises(ValidationError):
            NodeConstraint(
                type="NodeConstraint",
                nodeKind="iri",
                datatype="http://www.w3.org/2001/XMLSchema#string"
            )

    def test_node_constraint_none_set(self):
        """Test that at least one constraint must be set."""
        from shapespresso.syntax.shexj import NodeConstraint

        # No constraints set should raise error
        with pytest.raises(ValidationError):
            NodeConstraint(type="NodeConstraint")


class TestTripleConstraint:
    """Tests for TripleConstraint Pydantic model."""

    def test_triple_constraint_basic(self):
        """Test basic TripleConstraint."""
        from shapespresso.syntax.shexj import TripleConstraint, NodeConstraint

        tc = TripleConstraint(
            type="TripleConstraint",
            predicate="http://schema.org/name",
            valueExpr=NodeConstraint(
                type="NodeConstraint",
                datatype="http://www.w3.org/2001/XMLSchema#string"
            )
        )
        assert tc.type == "TripleConstraint"
        assert tc.predicate == "http://schema.org/name"
        assert tc.min is None  # Default
        assert tc.max is None  # Default

    def test_triple_constraint_with_cardinality(self):
        """Test TripleConstraint with cardinality."""
        from shapespresso.syntax.shexj import TripleConstraint, NodeConstraint

        tc = TripleConstraint(
            type="TripleConstraint",
            predicate="http://schema.org/email",
            valueExpr=NodeConstraint(
                type="NodeConstraint",
                datatype="http://www.w3.org/2001/XMLSchema#string"
            ),
            min=0,
            max=-1  # unbounded
        )
        assert tc.min == 0
        assert tc.max == -1

    def test_triple_constraint_with_shape_reference(self):
        """Test TripleConstraint with shape reference as valueExpr."""
        from shapespresso.syntax.shexj import TripleConstraint

        tc = TripleConstraint(
            type="TripleConstraint",
            predicate="http://schema.org/knows",
            valueExpr="http://example.org/Person"  # Shape reference as string
        )
        assert isinstance(tc.valueExpr, str)
        assert tc.valueExpr == "http://example.org/Person"

    def test_triple_constraint_inverse(self):
        """Test TripleConstraint with inverse predicate."""
        from shapespresso.syntax.shexj import TripleConstraint, NodeConstraint

        tc = TripleConstraint(
            type="TripleConstraint",
            predicate="http://schema.org/memberOf",
            valueExpr=NodeConstraint(
                type="NodeConstraint",
                nodeKind="iri"
            ),
            inverse=True
        )
        assert tc.inverse is True


class TestIriStem:
    """Tests for IriStem Pydantic model."""

    def test_iri_stem_basic(self):
        """Test basic IriStem."""
        from shapespresso.syntax.shexj import IriStem

        stem = IriStem(
            type="IriStem",
            stem="http://www.wikidata.org/entity/"
        )
        assert stem.type == "IriStem"
        assert str(stem.stem) == "http://www.wikidata.org/entity/"


class TestShapeExpression:
    """Tests for ShapeExpression Pydantic model."""

    def test_shape_expression_eachof(self):
        """Test EachOf ShapeExpression."""
        from shapespresso.syntax.shexj import ShapeExpression, TripleConstraint, NodeConstraint

        expr = ShapeExpression(
            type="EachOf",
            expressions=[
                TripleConstraint(
                    type="TripleConstraint",
                    predicate="http://schema.org/name",
                    valueExpr=NodeConstraint(
                        type="NodeConstraint",
                        datatype="http://www.w3.org/2001/XMLSchema#string"
                    )
                ),
                TripleConstraint(
                    type="TripleConstraint",
                    predicate="http://schema.org/age",
                    valueExpr=NodeConstraint(
                        type="NodeConstraint",
                        datatype="http://www.w3.org/2001/XMLSchema#integer"
                    )
                )
            ]
        )
        assert expr.type == "EachOf"
        assert len(expr.expressions) == 2

    def test_shape_expression_oneof(self):
        """Test OneOf ShapeExpression."""
        from shapespresso.syntax.shexj import ShapeExpression, TripleConstraint, NodeConstraint

        expr = ShapeExpression(
            type="OneOf",
            expressions=[
                TripleConstraint(
                    type="TripleConstraint",
                    predicate="http://schema.org/email",
                    valueExpr=NodeConstraint(
                        type="NodeConstraint",
                        datatype="http://www.w3.org/2001/XMLSchema#string"
                    )
                ),
                TripleConstraint(
                    type="TripleConstraint",
                    predicate="http://schema.org/telephone",
                    valueExpr=NodeConstraint(
                        type="NodeConstraint",
                        datatype="http://www.w3.org/2001/XMLSchema#string"
                    )
                )
            ]
        )
        assert expr.type == "OneOf"
        assert len(expr.expressions) == 2


class TestShape:
    """Tests for Shape Pydantic model."""

    def test_shape_basic(self):
        """Test basic Shape."""
        from shapespresso.syntax.shexj import Shape, TripleConstraint, NodeConstraint

        shape = Shape(
            type="Shape",
            id="http://example.org/Person",
            expression=TripleConstraint(
                type="TripleConstraint",
                predicate="http://schema.org/name",
                valueExpr=NodeConstraint(
                    type="NodeConstraint",
                    datatype="http://www.w3.org/2001/XMLSchema#string"
                )
            )
        )
        assert shape.type == "Shape"
        assert shape.id == "http://example.org/Person"

    def test_shape_closed(self):
        """Test closed Shape."""
        from shapespresso.syntax.shexj import Shape, TripleConstraint, NodeConstraint

        shape = Shape(
            type="Shape",
            id="http://example.org/Person",
            closed=True,
            expression=TripleConstraint(
                type="TripleConstraint",
                predicate="http://schema.org/name",
                valueExpr=NodeConstraint(
                    type="NodeConstraint",
                    datatype="http://www.w3.org/2001/XMLSchema#string"
                )
            )
        )
        assert shape.closed is True

    def test_shape_with_extra(self):
        """Test Shape with EXTRA predicates."""
        from shapespresso.syntax.shexj import Shape, TripleConstraint, NodeConstraint

        shape = Shape(
            type="Shape",
            id="http://example.org/Person",
            extra=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
            expression=TripleConstraint(
                type="TripleConstraint",
                predicate="http://schema.org/name",
                valueExpr=NodeConstraint(
                    type="NodeConstraint",
                    datatype="http://www.w3.org/2001/XMLSchema#string"
                )
            )
        )
        assert shape.extra is not None
        assert len(shape.extra) == 1


class TestSchema:
    """Tests for Schema Pydantic model."""

    def test_schema_basic(self):
        """Test basic Schema."""
        from shapespresso.syntax.shexj import Schema, Shape, TripleConstraint, NodeConstraint

        schema = Schema(
            type="Schema",
            start="http://example.org/Person",
            shapes=[
                Shape(
                    type="Shape",
                    id="http://example.org/Person",
                    expression=TripleConstraint(
                        type="TripleConstraint",
                        predicate="http://schema.org/name",
                        valueExpr=NodeConstraint(
                            type="NodeConstraint",
                            datatype="http://www.w3.org/2001/XMLSchema#string"
                        )
                    )
                )
            ]
        )
        assert schema.type == "Schema"
        assert schema.start == "http://example.org/Person"
        assert len(schema.shapes) == 1
