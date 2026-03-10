"""Integration tests for similarity metrics (tree edit distance)."""

import pytest


class TestShapeNode:
    """Tests for ShapeNode tree structure."""

    def test_shape_node_creation(self):
        """Test basic ShapeNode creation."""
        from shapespresso.metrics.similarity import ShapeNode

        node = ShapeNode("Person")
        assert node.label == "Person"
        assert len(node.children) == 0

    def test_shape_node_add_kid(self):
        """Test adding children to ShapeNode."""
        from shapespresso.metrics.similarity import ShapeNode

        root = ShapeNode("Person")
        child = ShapeNode("name")
        root.add_kid(child)

        assert len(root.children) == 1
        assert root.children[0].label == "name"

    def test_shape_node_nested_structure(self):
        """Test building nested tree structure."""
        from shapespresso.metrics.similarity import ShapeNode

        # Build: Person -> name -> xsd:string -> {1,1}
        root = (
            ShapeNode("Person")
            .add_kid(
                ShapeNode("name")
                .add_kid(
                    ShapeNode("xsd:string")
                    .add_kid(ShapeNode("{1,1}"))
                )
            )
        )

        assert len(root.children) == 1
        assert root.children[0].label == "name"
        assert root.children[0].children[0].label == "xsd:string"
        assert root.children[0].children[0].children[0].label == "{1,1}"

    def test_shape_node_sort_children(self):
        """Test sorting children by label."""
        from shapespresso.metrics.similarity import ShapeNode

        root = ShapeNode("Person")
        root.add_kid(ShapeNode("z_prop"))
        root.add_kid(ShapeNode("a_prop"))
        root.add_kid(ShapeNode("m_prop"))

        root.sort_children(key=lambda n: n.label)

        assert root.children[0].label == "a_prop"
        assert root.children[1].label == "m_prop"
        assert root.children[2].label == "z_prop"


class TestTreeEditDistance:
    """Tests for tree edit distance computation."""

    def test_ted_identical_trees(self):
        """Test TED is 0 for identical trees."""
        from shapespresso.metrics.similarity import ShapeNode, compute_tree_edit_distance

        tree1 = ShapeNode("Person").add_kid(ShapeNode("name"))
        tree2 = ShapeNode("Person").add_kid(ShapeNode("name"))

        ted = compute_tree_edit_distance(tree1, tree2)
        assert ted == 0

    def test_ted_different_trees(self):
        """Test TED > 0 for different trees."""
        from shapespresso.metrics.similarity import ShapeNode, compute_tree_edit_distance

        tree1 = ShapeNode("Person").add_kid(ShapeNode("name"))
        tree2 = ShapeNode("Person").add_kid(ShapeNode("email"))

        ted = compute_tree_edit_distance(tree1, tree2)
        assert ted > 0

    def test_ted_empty_vs_nonempty(self):
        """Test TED between empty and non-empty trees."""
        from shapespresso.metrics.similarity import ShapeNode, compute_tree_edit_distance

        tree1 = ShapeNode("Person")
        tree2 = ShapeNode("Person").add_kid(ShapeNode("name"))

        ted = compute_tree_edit_distance(tree1, tree2)
        assert ted == 1  # One insertion needed


class TestShExTreeTransformation:
    """Tests for ShEx to tree transformation."""

    def test_transform_shex_to_tree_basic(self, sample_shexj):
        """Test transforming ShExJ to tree."""
        from shapespresso.metrics.similarity import transform_schema_to_tree

        tree = transform_schema_to_tree(sample_shexj, "http://example.org/Person")

        assert tree is not None
        assert tree.label == "http://example.org/Person"
        # Should have children for name and birthDate constraints
        assert len(tree.children) >= 1

    def test_transform_shex_to_tree_with_cardinality(self):
        """Test ShEx tree includes cardinality nodes."""
        from shapespresso.metrics.similarity import transform_schema_to_tree
        import json

        shexj = {
            "type": "Schema",
            "start": "http://example.org/Person",
            "shapes": [
                {
                    "type": "Shape",
                    "id": "http://example.org/Person",
                    "expression": {
                        "type": "EachOf",
                        "expressions": [
                            {
                                "type": "TripleConstraint",
                                "predicate": "http://schema.org/name",
                                "valueExpr": {
                                    "type": "NodeConstraint",
                                    "datatype": "http://www.w3.org/2001/XMLSchema#string"
                                },
                                "min": 1,
                                "max": 1
                            }
                        ]
                    }
                }
            ]
        }

        tree = transform_schema_to_tree(shexj, "http://example.org/Person")

        # Tree structure: Person -> predicate -> node_constraint -> cardinality
        assert tree is not None
        assert len(tree.children) == 1  # One constraint


class TestSHACLTreeTransformation:
    """Tests for SHACL to tree transformation."""

    def test_transform_shacl_to_tree_basic(self, minimal_shacl_schema):
        """Test transforming SHACL TTL to tree."""
        from shapespresso.metrics.similarity import transform_shacl_to_tree

        tree = transform_shacl_to_tree(minimal_shacl_schema)

        assert tree is not None
        # Should have the shape label as root
        assert "Person" in tree.label or "Shape" in tree.label or tree.label != "EmptyShape"

    def test_transform_shacl_to_tree_with_properties(self, shacl_with_cardinality):
        """Test SHACL tree includes property constraints."""
        from shapespresso.metrics.similarity import transform_shacl_to_tree

        tree = transform_shacl_to_tree(shacl_with_cardinality)

        assert tree is not None
        # Should have children for property shapes
        assert len(tree.children) >= 1

    def test_transform_shacl_to_tree_structure(self):
        """Test SHACL tree has correct structure: predicate -> constraint -> cardinality."""
        from shapespresso.metrics.similarity import transform_shacl_to_tree

        shacl_ttl = '''
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix ex: <http://example.org/> .

        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:datatype xsd:string ;
                sh:minCount 1 ;
                sh:maxCount 1
            ] .
        '''

        tree = transform_shacl_to_tree(shacl_ttl)

        assert tree is not None
        assert len(tree.children) == 1  # One property

        # Check tree depth: predicate -> node_constraint -> cardinality
        prop_node = tree.children[0]
        assert len(prop_node.children) == 1  # node constraint
        assert len(prop_node.children[0].children) == 1  # cardinality

    def test_transform_shacl_empty_schema(self):
        """Test handling of empty/invalid SHACL."""
        from shapespresso.metrics.similarity import transform_shacl_to_tree

        tree = transform_shacl_to_tree("invalid shacl content")

        # Should return empty shape node, not crash
        assert tree is not None
        assert tree.label == "EmptyShape"

    def test_transform_shacl_to_tree_with_class_constraint(self):
        """Test SHACL tree with sh:class constraint."""
        from shapespresso.metrics.similarity import transform_shacl_to_tree

        shacl_ttl = '''
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .

        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:knows ;
                sh:class ex:Person ;
                sh:minCount 0
            ] .
        '''

        tree = transform_shacl_to_tree(shacl_ttl)

        assert tree is not None
        assert len(tree.children) == 1

        # The node constraint should reference the class
        prop_node = tree.children[0]
        nc_node = prop_node.children[0]
        assert "Person" in nc_node.label or "ex:" in nc_node.label


class TestSHACLGraphTransformation:
    """Tests for SHACL to NetworkX graph transformation."""

    def test_transform_shacl_to_graph_basic(self, minimal_shacl_schema):
        """Test transforming SHACL TTL to NetworkX graph."""
        from shapespresso.metrics.similarity import transform_shacl_to_graph

        root_label, graph = transform_shacl_to_graph(minimal_shacl_schema)

        assert root_label is not None
        assert graph is not None
        assert len(graph.nodes()) >= 1

    def test_transform_shacl_to_graph_structure(self):
        """Test SHACL graph has correct structure."""
        from shapespresso.metrics.similarity import transform_shacl_to_graph

        shacl_ttl = '''
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix ex: <http://example.org/> .

        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:datatype xsd:string ;
                sh:minCount 1 ;
                sh:maxCount 1
            ] .
        '''

        root_label, graph = transform_shacl_to_graph(shacl_ttl)

        # Should have: root -> predicate -> constraint -> cardinality
        # That's 4 nodes for one property
        assert len(graph.nodes()) >= 4


class TestCrossFormatComparison:
    """Tests for comparing ShEx and SHACL schemas."""

    def test_similar_schemas_have_low_ted(self):
        """Test that semantically similar ShEx and SHACL schemas have low TED."""
        from shapespresso.metrics.similarity import (
            transform_schema_to_tree,
            transform_shacl_to_tree,
            compute_tree_edit_distance
        )

        # ShEx schema
        shexj = {
            "type": "Schema",
            "start": "http://example.org/Person",
            "shapes": [
                {
                    "type": "Shape",
                    "id": "http://example.org/Person",
                    "expression": {
                        "type": "EachOf",
                        "expressions": [
                            {
                                "type": "TripleConstraint",
                                "predicate": "http://example.org/name",
                                "valueExpr": {
                                    "type": "NodeConstraint",
                                    "datatype": "http://www.w3.org/2001/XMLSchema#string"
                                },
                                "min": 1,
                                "max": 1
                            }
                        ]
                    }
                }
            ]
        }

        # Equivalent SHACL schema
        shacl_ttl = '''
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix ex: <http://example.org/> .

        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:datatype xsd:string ;
                sh:minCount 1 ;
                sh:maxCount 1
            ] .
        '''

        shex_tree = transform_schema_to_tree(shexj, "http://example.org/Person")
        shacl_tree = transform_shacl_to_tree(shacl_ttl)

        # Both trees should have similar structure
        # The TED should be relatively low (differences mainly in label formatting)
        ted = compute_tree_edit_distance(shex_tree, shacl_tree)

        # TED should be small for similar schemas
        # Allow some difference due to label formatting (e.g., full URI vs prefixed)
        assert ted <= 10  # Reasonable threshold for similar schemas
