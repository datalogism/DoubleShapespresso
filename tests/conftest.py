"""Shared pytest fixtures for shapespresso tests."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock


# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATASET_DIR = Path(__file__).parent.parent / "dataset"


# ============================================================================
# ShEx Fixtures
# ============================================================================

@pytest.fixture
def sample_shex_schema() -> str:
    """Load sample Scientist.shex from dataset/Shex/yagos/."""
    shex_path = DATASET_DIR / "Shex" / "yagos" / "Scientist.shex"
    if shex_path.exists():
        return shex_path.read_text()
    # Fallback minimal ShEx for testing
    return '''
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

start = @<Person>

<Person> {
  schema:name xsd:string ;
  schema:birthDate xsd:date ?
}
'''


@pytest.fixture
def minimal_shex_schema() -> str:
    """Minimal ShEx schema for basic tests."""
    return '''
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

start = @<Person>

<Person> {
  schema:name xsd:string ;
  schema:age xsd:integer ?
}
'''


@pytest.fixture
def shex_with_cardinality() -> str:
    """ShEx schema with various cardinality patterns."""
    return '''
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

start = @<Person>

<Person> {
  schema:name xsd:string ;          # exactly one (default)
  schema:email xsd:string * ;        # zero or more
  schema:phone xsd:string + ;        # one or more
  schema:nickname xsd:string ? ;     # zero or one
  schema:child @<Person> {0,5}       # between 0 and 5
}
'''


@pytest.fixture
def shex_with_value_sets() -> str:
    """ShEx schema with value sets."""
    return '''
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

start = @<Person>

<Person> {
  schema:gender [ schema:Male schema:Female schema:Other ] ;
  schema:name xsd:string
}
'''


# ============================================================================
# SHACL Fixtures
# ============================================================================

@pytest.fixture
def sample_shacl_schema() -> str:
    """Load sample SHACL from dataset/SHACL/dbpedia-v0/."""
    shacl_path = DATASET_DIR / "SHACL" / "dbpedia-v0" / "ScientistShapeTXT2KG_clean.ttl"
    if shacl_path.exists():
        return shacl_path.read_text()
    # Fallback minimal SHACL for testing
    return '''
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


@pytest.fixture
def minimal_shacl_schema() -> str:
    """Minimal SHACL schema for basic tests."""
    return '''
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://example.org/> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:name ;
        sh:datatype xsd:string ;
        sh:minCount 1
    ] .
'''


@pytest.fixture
def shacl_with_sh_or() -> str:
    """SHACL schema with sh:or constraint."""
    return '''
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://example.org/> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:identifier ;
        sh:or (
            [ sh:datatype xsd:string ]
            [ sh:datatype xsd:integer ]
        )
    ] .
'''


@pytest.fixture
def shacl_with_sh_and() -> str:
    """SHACL schema with sh:and constraint."""
    return '''
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://example.org/> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:age ;
        sh:and (
            [ sh:datatype xsd:integer ]
            [ sh:minInclusive 0 ]
            [ sh:maxInclusive 150 ]
        )
    ] .
'''


@pytest.fixture
def shacl_with_cardinality() -> str:
    """SHACL schema with various cardinality constraints."""
    return '''
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
    ] ,
    [
        sh:path ex:email ;
        sh:datatype xsd:string ;
        sh:minCount 0
    ] ,
    [
        sh:path ex:phone ;
        sh:datatype xsd:string ;
        sh:minCount 1
    ] .
'''


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_sparql_endpoint(mocker):
    """Mock SPARQLWrapper responses."""
    mock_wrapper = mocker.patch('shapespresso.utils.query.SPARQLWrapper')
    mock_instance = MagicMock()
    mock_wrapper.return_value = mock_instance

    # Default response
    mock_instance.query.return_value.convert.return_value = {
        "results": {
            "bindings": []
        }
    }

    return mock_instance


@pytest.fixture
def mock_llm_response(mocker):
    """Mock OpenAI/Claude API responses."""
    mock_openai = mocker.patch('openai.OpenAI')
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Default response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "{}"
    mock_client.chat.completions.create.return_value = mock_response

    return mock_client


# ============================================================================
# JSON-LD / ShExJ Fixtures
# ============================================================================

@pytest.fixture
def sample_shexj() -> dict:
    """Sample ShExJ JSON structure."""
    return {
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
                        },
                        {
                            "type": "TripleConstraint",
                            "predicate": "http://schema.org/birthDate",
                            "valueExpr": {
                                "type": "NodeConstraint",
                                "datatype": "http://www.w3.org/2001/XMLSchema#date"
                            },
                            "min": 0,
                            "max": 1
                        }
                    ]
                }
            }
        ]
    }


@pytest.fixture
def sample_shacl_jsonld() -> list:
    """Sample SHACL JSON-LD structure."""
    return [
        {
            "@id": "http://example.org/PersonShape",
            "@type": ["http://www.w3.org/ns/shacl#NodeShape"],
            "http://www.w3.org/ns/shacl#targetClass": [
                {"@id": "http://example.org/Person"}
            ],
            "http://www.w3.org/ns/shacl#property": [
                {
                    "http://www.w3.org/ns/shacl#path": [
                        {"@id": "http://example.org/name"}
                    ],
                    "http://www.w3.org/ns/shacl#datatype": [
                        {"@id": "http://www.w3.org/2001/XMLSchema#string"}
                    ],
                    "http://www.w3.org/ns/shacl#minCount": [
                        {"@value": 1}
                    ],
                    "http://www.w3.org/ns/shacl#maxCount": [
                        {"@value": 1}
                    ]
                }
            ]
        }
    ]


# ============================================================================
# Constraint Fixtures for Classification Tests
# ============================================================================

@pytest.fixture
def shex_constraint_simple() -> dict:
    """Simple ShEx triple constraint."""
    return {
        "type": "TripleConstraint",
        "predicate": "http://schema.org/name",
        "valueExpr": {
            "type": "NodeConstraint",
            "datatype": "http://www.w3.org/2001/XMLSchema#string"
        },
        "min": 1,
        "max": 1
    }


@pytest.fixture
def shacl_constraint_simple() -> dict:
    """Simple SHACL property constraint in JSON-LD format."""
    return {
        "@id": "_:b0",
        "http://www.w3.org/ns/shacl#path": [
            {"@id": "http://schema.org/name"}
        ],
        "http://www.w3.org/ns/shacl#datatype": [
            {"@id": "http://www.w3.org/2001/XMLSchema#string"}
        ],
        "http://www.w3.org/ns/shacl#minCount": [
            {"@value": 1}
        ],
        "http://www.w3.org/ns/shacl#maxCount": [
            {"@value": 1}
        ]
    }


@pytest.fixture
def shacl_constraint_with_sh_or() -> dict:
    """SHACL property constraint with sh:or in JSON-LD format."""
    return {
        "@id": "_:b0",
        "http://www.w3.org/ns/shacl#path": [
            {"@id": "http://example.org/identifier"}
        ],
        "http://www.w3.org/ns/shacl#or": [
            {
                "@list": [
                    {
                        "http://www.w3.org/ns/shacl#datatype": [
                            {"@id": "http://www.w3.org/2001/XMLSchema#string"}
                        ]
                    },
                    {
                        "http://www.w3.org/ns/shacl#datatype": [
                            {"@id": "http://www.w3.org/2001/XMLSchema#integer"}
                        ]
                    }
                ]
            }
        ]
    }
