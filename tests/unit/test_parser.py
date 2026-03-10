"""Unit tests for ShExC/ShExJ parsing functionality."""

import json
import pytest


class TestShExCToShExJ:
    """Tests for shexc_to_shexj conversion."""

    def test_shexc_to_shexj_valid_schema(self, minimal_shex_schema):
        """Test conversion of valid ShExC schema to ShExJ."""
        from shapespresso.parser import shexc_to_shexj

        shexj_text, base, namespaces, comments = shexc_to_shexj(minimal_shex_schema)

        assert shexj_text is not None
        assert shexj_text != ""

        # Parse the JSON to verify structure
        shexj = json.loads(shexj_text)
        assert shexj.get("type") == "Schema"
        assert "shapes" in shexj

    def test_shexc_to_shexj_with_cardinality(self, shex_with_cardinality):
        """Test conversion of ShExC schema with various cardinality patterns."""
        from shapespresso.parser import shexc_to_shexj

        shexj_text, _, _, _ = shexc_to_shexj(shex_with_cardinality)

        assert shexj_text is not None
        shexj = json.loads(shexj_text)

        # Find the Person shape
        person_shape = None
        for shape in shexj.get("shapes", []):
            if "Person" in shape.get("id", ""):
                person_shape = shape
                break

        assert person_shape is not None
        assert "expression" in person_shape

    def test_shexc_to_shexj_with_value_sets(self, shex_with_value_sets):
        """Test conversion of ShExC schema with value sets."""
        from shapespresso.parser import shexc_to_shexj

        shexj_text, _, _, _ = shexc_to_shexj(shex_with_value_sets)

        assert shexj_text is not None
        shexj = json.loads(shexj_text)

        # Verify the schema has shapes
        assert "shapes" in shexj
        assert len(shexj["shapes"]) > 0

    def test_shexc_to_shexj_preserves_namespaces(self, minimal_shex_schema):
        """Test that namespace declarations are preserved."""
        from shapespresso.parser import shexc_to_shexj

        _, _, namespaces, _ = shexc_to_shexj(minimal_shex_schema)

        # Namespaces should be parsed
        assert namespaces is not None

    def test_shexc_to_shexj_invalid_syntax(self):
        """Test error handling for invalid ShExC syntax."""
        from shapespresso.parser import shexc_to_shexj

        invalid_shex = "THIS IS NOT VALID SHEX {{{}}}"
        shexj_text, _, _, _ = shexc_to_shexj(invalid_shex)

        # Should return empty string on parse failure
        assert shexj_text == ""

    def test_shexc_to_shexj_with_comments(self):
        """Test that comments are extracted."""
        from shapespresso.parser import shexc_to_shexj

        shex_with_comments = '''
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

# This is a comment about the Person shape
start = @<Person>

<Person> {
  schema:name xsd:string ;  # Name of the person
  schema:age xsd:integer ?   # Age is optional
}
'''
        _, _, _, comments = shexc_to_shexj(shex_with_comments)

        assert comments is not None
        assert len(comments) > 0


class TestShExJToShExC:
    """Tests for shexj_to_shexc conversion."""

    def test_shexj_to_shexc_basic(self, sample_shexj):
        """Test conversion of ShExJ back to ShExC."""
        from shapespresso.parser import shexj_to_shexc

        shexj_text = json.dumps(sample_shexj)
        shexc_text = shexj_to_shexc(shexj_text)

        assert shexc_text is not None
        assert shexc_text != ""
        # Should contain shape definition
        assert "Person" in shexc_text or "person" in shexc_text.lower()

    def test_shexj_to_shexc_roundtrip(self, minimal_shex_schema):
        """Test roundtrip conversion: ShExC -> ShExJ -> ShExC."""
        from shapespresso.parser import shexc_to_shexj, shexj_to_shexc

        # Convert to ShExJ
        shexj_text, base, namespaces, comments = shexc_to_shexj(minimal_shex_schema)
        assert shexj_text != ""

        # Convert back to ShExC
        shexc_result = shexj_to_shexc(shexj_text, base, namespaces, comments)
        assert shexc_result is not None

        # Should contain the shape name
        assert "Person" in shexc_result


class TestParserHelpers:
    """Tests for parser helper functions."""

    def test_position_start_line(self):
        """Test finding the start line position."""
        from shapespresso.parser.parser import position_start_line

        shex_text = """PREFIX schema: <http://schema.org/>

start = @<Person>

<Person> { }
"""
        pos = position_start_line(shex_text)
        assert pos >= 2  # Should be after the PREFIX line

    def test_position_start_line_no_start(self):
        """Test finding start line when 'start' keyword is missing."""
        from shapespresso.parser.parser import position_start_line

        shex_text = """PREFIX schema: <http://schema.org/>

<Person> { }
"""
        pos = position_start_line(shex_text)
        assert pos >= 2  # Should find the <Person> line

    def test_base_uri_parser_helper(self):
        """Test base URI extraction."""
        from shapespresso.parser.parser import base_uri_parser_helper

        shex_text = """BASE <http://example.org/>
PREFIX schema: <http://schema.org/>

start = @<Person>
"""
        base = base_uri_parser_helper(shex_text)
        assert base == "http://example.org/"

    def test_base_uri_parser_helper_no_base(self):
        """Test base URI extraction when no BASE is declared."""
        from shapespresso.parser.parser import base_uri_parser_helper

        shex_text = """PREFIX schema: <http://schema.org/>

start = @<Person>
"""
        base = base_uri_parser_helper(shex_text)
        assert base is None

    def test_remove_lines(self):
        """Test removal of specific lines."""
        from shapespresso.parser.parser import remove_lines

        text = "line1\nline2\nline3\nline4"
        result = remove_lines(text, [2, 4])
        assert result == "line1\nline3"
