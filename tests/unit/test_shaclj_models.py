"""Unit tests for SHACL Pydantic models."""

import pytest
from pydantic import ValidationError


class TestNodeConstraintSHACL:
    """Tests for NodeConstraintSHACL Pydantic model."""

    def test_node_constraint_shacl_with_class(self):
        """Test NodeConstraintSHACL with sh:class."""
        from shapespresso.syntax.shaclj import NodeConstraintSHACL

        nc = NodeConstraintSHACL(**{
            '@id': '_:b0',
            'sh:path': 'http://schema.org/knows',
            'sh:class': 'http://schema.org/Person',
            'sh:minCount': 0,
            'sh:maxCount': 10
        })
        assert nc.sh_class == 'http://schema.org/Person'
        assert nc.sh_datatype is None
        assert nc.sh_minCount == 0
        assert nc.sh_maxCount == 10

    def test_node_constraint_shacl_with_datatype(self):
        """Test NodeConstraintSHACL with sh:datatype."""
        from shapespresso.syntax.shaclj import NodeConstraintSHACL

        nc = NodeConstraintSHACL(**{
            '@id': '_:b0',
            'sh:path': 'http://schema.org/name',
            'sh:datatype': 'http://www.w3.org/2001/XMLSchema#string',
            'sh:minCount': 1,
            'sh:maxCount': 1
        })
        assert nc.sh_datatype == 'http://www.w3.org/2001/XMLSchema#string'
        assert nc.sh_class is None
        assert nc.sh_minCount == 1

    def test_node_constraint_shacl_mutual_exclusion(self):
        """Test that sh:class and sh:datatype are mutually exclusive."""
        from shapespresso.syntax.shaclj import NodeConstraintSHACL

        # Both set - should auto-fix by preferring sh:datatype (based on current implementation)
        nc = NodeConstraintSHACL(**{
            '@id': '_:b0',
            'sh:path': 'http://schema.org/name',
            'sh:class': 'http://schema.org/Person',
            'sh:datatype': 'http://www.w3.org/2001/XMLSchema#string',
            'sh:minCount': 1
        })
        # According to current implementation, sh:class should be None
        assert nc.sh_datatype == 'http://www.w3.org/2001/XMLSchema#string'
        assert nc.sh_class is None

    def test_node_constraint_shacl_neither_set(self):
        """Test that either sh:class or sh:datatype must be set."""
        from shapespresso.syntax.shaclj import NodeConstraintSHACL

        # Neither set should raise error
        with pytest.raises(ValidationError):
            NodeConstraintSHACL(**{
                '@id': '_:b0',
                'sh:path': 'http://schema.org/name',
                'sh:minCount': 1
            })

    def test_node_constraint_shacl_optional_cardinality(self):
        """Test NodeConstraintSHACL with optional cardinality (no minCount/maxCount)."""
        from shapespresso.syntax.shaclj import NodeConstraintSHACL

        nc = NodeConstraintSHACL(**{
            '@id': '_:b0',
            'sh:path': 'http://schema.org/name',
            'sh:datatype': 'http://www.w3.org/2001/XMLSchema#string'
        })
        assert nc.sh_minCount is None
        assert nc.sh_maxCount is None

    def test_node_constraint_shacl_alias_population(self):
        """Test that aliases work correctly for JSON-LD style keys."""
        from shapespresso.syntax.shaclj import NodeConstraintSHACL

        # Using the aliases directly
        nc = NodeConstraintSHACL(
            id_='_:b0',
            sh_path='http://schema.org/name',
            sh_datatype='http://www.w3.org/2001/XMLSchema#string',
            sh_minCount=1,
            sh_maxCount=1
        )
        assert nc.id_ == '_:b0'
        assert str(nc.sh_path) == 'http://schema.org/name'


class TestSHACLConstraintExtraction:
    """Tests for SHACL constraint extraction from JSON-LD."""

    def test_extract_path_from_jsonld(self, shacl_constraint_simple):
        """Test extracting sh:path from JSON-LD format."""
        path = shacl_constraint_simple.get('http://www.w3.org/ns/shacl#path', [{}])[0].get('@id')
        assert path == 'http://schema.org/name'

    def test_extract_datatype_from_jsonld(self, shacl_constraint_simple):
        """Test extracting sh:datatype from JSON-LD format."""
        datatype = shacl_constraint_simple.get('http://www.w3.org/ns/shacl#datatype', [{}])[0].get('@id')
        assert datatype == 'http://www.w3.org/2001/XMLSchema#string'

    def test_extract_cardinality_from_jsonld(self, shacl_constraint_simple):
        """Test extracting cardinality from JSON-LD format."""
        min_count = shacl_constraint_simple.get('http://www.w3.org/ns/shacl#minCount', [{}])[0].get('@value')
        max_count = shacl_constraint_simple.get('http://www.w3.org/ns/shacl#maxCount', [{}])[0].get('@value')
        assert min_count == 1
        assert max_count == 1

    def test_extract_sh_or_from_jsonld(self, shacl_constraint_with_sh_or):
        """Test extracting sh:or constraint from JSON-LD format."""
        sh_or = shacl_constraint_with_sh_or.get('http://www.w3.org/ns/shacl#or')
        assert sh_or is not None
        assert '@list' in sh_or[0]
        assert len(sh_or[0]['@list']) == 2


class TestSHACLSchemaStructure:
    """Tests for SHACL schema structure validation."""

    def test_parse_shacl_ttl_to_jsonld(self, minimal_shacl_schema):
        """Test parsing SHACL TTL to JSON-LD."""
        from rdflib import Graph
        import json

        graph = Graph()
        graph.parse(data=minimal_shacl_schema, format='turtle')
        jsonld = json.loads(graph.serialize(format='json-ld'))

        assert isinstance(jsonld, list)
        assert len(jsonld) > 0

    def test_shacl_nodeshape_structure(self, sample_shacl_jsonld):
        """Test SHACL NodeShape structure in JSON-LD."""
        node_shape = sample_shacl_jsonld[0]

        assert '@id' in node_shape
        assert 'http://www.w3.org/ns/shacl#NodeShape' in node_shape.get('@type', [])
        assert 'http://www.w3.org/ns/shacl#targetClass' in node_shape
        assert 'http://www.w3.org/ns/shacl#property' in node_shape

    def test_shacl_property_shape_structure(self, sample_shacl_jsonld):
        """Test SHACL PropertyShape structure in JSON-LD."""
        node_shape = sample_shacl_jsonld[0]
        properties = node_shape.get('http://www.w3.org/ns/shacl#property', [])

        assert len(properties) > 0

        prop = properties[0]
        assert 'http://www.w3.org/ns/shacl#path' in prop
        assert 'http://www.w3.org/ns/shacl#datatype' in prop or 'http://www.w3.org/ns/shacl#class' in prop
