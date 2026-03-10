"""Unit tests for SPARQL query utilities."""

import pytest
from unittest.mock import MagicMock, patch


class TestEndpointSparqlQuery:
    """Tests for endpoint_sparql_query function."""

    def test_select_query_success(self, mock_sparql_endpoint):
        """Test successful SELECT query."""
        from shapespresso.utils.query import endpoint_sparql_query

        # Setup mock response
        mock_sparql_endpoint.query.return_value.convert.return_value = {
            "results": {
                "bindings": [
                    {"subject": {"value": "http://example.org/Person1"}},
                    {"subject": {"value": "http://example.org/Person2"}}
                ]
            }
        }

        with patch('shapespresso.utils.query.SPARQLWrapper', return_value=mock_sparql_endpoint):
            results = endpoint_sparql_query(
                "SELECT ?subject WHERE { ?subject a <http://example.org/Person> }",
                "http://localhost:1234/api/endpoint/sparql"
            )

        assert len(results) == 2
        assert results[0]["subject"] == "http://example.org/Person1"
        assert results[1]["subject"] == "http://example.org/Person2"

    def test_ask_query_true(self, mock_sparql_endpoint):
        """Test ASK query returning true."""
        from shapespresso.utils.query import endpoint_sparql_query

        mock_sparql_endpoint.query.return_value.convert.return_value = {
            "boolean": True
        }

        with patch('shapespresso.utils.query.SPARQLWrapper', return_value=mock_sparql_endpoint):
            result = endpoint_sparql_query(
                "ASK { <http://example.org/Person1> a <http://example.org/Person> }",
                "http://localhost:1234/api/endpoint/sparql",
                mode="ask"
            )

        assert result is True

    def test_ask_query_false(self, mock_sparql_endpoint):
        """Test ASK query returning false."""
        from shapespresso.utils.query import endpoint_sparql_query

        mock_sparql_endpoint.query.return_value.convert.return_value = {
            "boolean": False
        }

        with patch('shapespresso.utils.query.SPARQLWrapper', return_value=mock_sparql_endpoint):
            result = endpoint_sparql_query(
                "ASK { <http://example.org/NonExistent> a <http://example.org/Person> }",
                "http://localhost:1234/api/endpoint/sparql",
                mode="ask"
            )

        assert result is False

    def test_query_empty_results(self, mock_sparql_endpoint):
        """Test query returning empty results."""
        from shapespresso.utils.query import endpoint_sparql_query

        mock_sparql_endpoint.query.return_value.convert.return_value = {
            "results": {
                "bindings": []
            }
        }

        with patch('shapespresso.utils.query.SPARQLWrapper', return_value=mock_sparql_endpoint):
            results = endpoint_sparql_query(
                "SELECT ?subject WHERE { ?subject a <http://example.org/NonExistent> }",
                "http://localhost:1234/api/endpoint/sparql"
            )

        assert results == []

    def test_query_multiple_variables(self, mock_sparql_endpoint):
        """Test query with multiple variables."""
        from shapespresso.utils.query import endpoint_sparql_query

        mock_sparql_endpoint.query.return_value.convert.return_value = {
            "results": {
                "bindings": [
                    {
                        "subject": {"value": "http://example.org/Person1"},
                        "predicate": {"value": "http://schema.org/name"},
                        "object": {"value": "John Doe"}
                    }
                ]
            }
        }

        with patch('shapespresso.utils.query.SPARQLWrapper', return_value=mock_sparql_endpoint):
            results = endpoint_sparql_query(
                "SELECT ?subject ?predicate ?object WHERE { ?subject ?predicate ?object }",
                "http://localhost:1234/api/endpoint/sparql"
            )

        assert len(results) == 1
        assert results[0]["subject"] == "http://example.org/Person1"
        assert results[0]["predicate"] == "http://schema.org/name"
        assert results[0]["object"] == "John Doe"

    def test_query_error_handling(self, mock_sparql_endpoint):
        """Test error handling for query failures."""
        from shapespresso.utils.query import endpoint_sparql_query
        from SPARQLWrapper.SPARQLExceptions import EndPointInternalError

        mock_sparql_endpoint.query.side_effect = EndPointInternalError()

        with patch('shapespresso.utils.query.SPARQLWrapper', return_value=mock_sparql_endpoint):
            results = endpoint_sparql_query(
                "SELECT ?subject WHERE { ?subject a <http://example.org/Person> }",
                "http://localhost:1234/api/endpoint/sparql"
            )

        assert results == []

    def test_wikidata_endpoint_user_agent(self):
        """Test that Wikidata endpoints use custom user agent."""
        from shapespresso.utils.query import endpoint_sparql_query

        with patch('shapespresso.utils.query.SPARQLWrapper') as mock_wrapper:
            mock_instance = MagicMock()
            mock_wrapper.return_value = mock_instance
            mock_instance.query.return_value.convert.return_value = {
                "results": {"bindings": []}
            }

            endpoint_sparql_query(
                "SELECT ?s WHERE { ?s ?p ?o }",
                "https://query.wikidata.org/sparql"
            )

            # Check that SPARQLWrapper was called with agent parameter
            call_args = mock_wrapper.call_args
            assert 'agent' in call_args.kwargs or len(call_args.args) > 1


class TestQueryHelpers:
    """Tests for query helper functions."""

    def test_query_cleans_indentation(self, mock_sparql_endpoint):
        """Test that queries are cleaned of extra indentation."""
        from shapespresso.utils.query import endpoint_sparql_query

        mock_sparql_endpoint.query.return_value.convert.return_value = {
            "results": {"bindings": []}
        }

        indented_query = """
            SELECT ?subject
            WHERE {
                ?subject a <http://example.org/Person>
            }
        """

        with patch('shapespresso.utils.query.SPARQLWrapper', return_value=mock_sparql_endpoint):
            results = endpoint_sparql_query(
                indented_query,
                "http://localhost:1234/api/endpoint/sparql"
            )

        # Should not raise an error due to indentation
        assert isinstance(results, list)
