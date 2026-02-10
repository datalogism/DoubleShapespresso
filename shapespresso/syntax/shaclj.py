"""SHACL Pydantic models for structured SHACL generation and validation.

This module provides Pydantic models for SHACL (Shapes Constraint Language)
that support both structured generation with LLMs and validation of SHACL shapes.
"""

from typing import Literal, Optional, Union, Any
from pydantic import AnyUrl, BaseModel, Field, model_validator
import warnings


class SHACLConstraint(BaseModel):
    """Base class for SHACL constraint components."""

    class Config:
        populate_by_name = True
        validate_assignment = True
        extra = 'allow'  # Allow additional fields for extensibility


class NodeKindConstraint(SHACLConstraint):
    """Constraint specifying the kind of RDF node (IRI, Literal, BlankNode)."""
    sh_nodeKind: Optional[str] = Field(
        default=None,
        alias='sh:nodeKind',
        description="One of sh:IRI, sh:Literal, sh:BlankNode, sh:BlankNodeOrIRI, sh:BlankNodeOrLiteral, sh:IRIOrLiteral"
    )


class DatatypeConstraint(SHACLConstraint):
    """Constraint specifying the datatype of a literal value."""
    sh_datatype: Optional[str] = Field(
        default=None,
        alias='sh:datatype',
        description="The XSD datatype URI (e.g., xsd:string, xsd:integer)"
    )


class ClassConstraint(SHACLConstraint):
    """Constraint specifying that values must be instances of a class."""
    sh_class: Optional[str] = Field(
        default=None,
        alias='sh:class',
        description="The class URI that values must be instances of"
    )


class ValueConstraint(SHACLConstraint):
    """Constraint specifying allowed values (sh:in) or pattern (sh:pattern)."""
    sh_in: Optional[list[str]] = Field(
        default=None,
        alias='sh:in',
        description="List of allowed values"
    )
    sh_pattern: Optional[str] = Field(
        default=None,
        alias='sh:pattern',
        description="Regular expression pattern that values must match"
    )
    sh_flags: Optional[str] = Field(
        default=None,
        alias='sh:flags',
        description="Regex flags for sh:pattern"
    )


class CardinalityConstraint(SHACLConstraint):
    """Constraint specifying cardinality (minCount, maxCount)."""
    sh_minCount: Optional[int] = Field(
        default=None,
        alias='sh:minCount',
        ge=0,
        description="Minimum number of values (default: 0)"
    )
    sh_maxCount: Optional[int] = Field(
        default=None,
        alias='sh:maxCount',
        ge=0,
        description="Maximum number of values (unbounded if not specified)"
    )


class StringConstraint(SHACLConstraint):
    """Constraints for string values (length, language)."""
    sh_minLength: Optional[int] = Field(
        default=None,
        alias='sh:minLength',
        ge=0
    )
    sh_maxLength: Optional[int] = Field(
        default=None,
        alias='sh:maxLength',
        ge=0
    )
    sh_languageIn: Optional[list[str]] = Field(
        default=None,
        alias='sh:languageIn',
        description="List of allowed language tags"
    )
    sh_uniqueLang: Optional[bool] = Field(
        default=None,
        alias='sh:uniqueLang',
        description="Whether each language tag can only appear once"
    )


class NumericConstraint(SHACLConstraint):
    """Constraints for numeric values."""
    sh_minInclusive: Optional[Union[int, float]] = Field(
        default=None,
        alias='sh:minInclusive'
    )
    sh_maxInclusive: Optional[Union[int, float]] = Field(
        default=None,
        alias='sh:maxInclusive'
    )
    sh_minExclusive: Optional[Union[int, float]] = Field(
        default=None,
        alias='sh:minExclusive'
    )
    sh_maxExclusive: Optional[Union[int, float]] = Field(
        default=None,
        alias='sh:maxExclusive'
    )


class LogicalConstraint(SHACLConstraint):
    """Logical constraint components (sh:or, sh:and, sh:not, sh:xone)."""
    sh_or: Optional[list[Any]] = Field(
        default=None,
        alias='sh:or',
        description="List of shapes, at least one must match"
    )
    sh_and: Optional[list[Any]] = Field(
        default=None,
        alias='sh:and',
        description="List of shapes, all must match"
    )
    sh_not: Optional[Any] = Field(
        default=None,
        alias='sh:not',
        description="Shape that must not match"
    )
    sh_xone: Optional[list[Any]] = Field(
        default=None,
        alias='sh:xone',
        description="List of shapes, exactly one must match"
    )


class PropertyShape(SHACLConstraint):
    """SHACL Property Shape defining constraints on a specific property.

    A PropertyShape describes constraints that apply to the values of a specific
    property for focus nodes that match the containing NodeShape.
    """
    id_: Optional[str] = Field(
        default=None,
        alias='@id',
        description="Optional identifier for the property shape"
    )
    sh_path: Optional[Union[str, dict]] = Field(
        alias='sh:path',
        description="The property path this shape constrains"
    )

    # Node constraints
    sh_class: Optional[str] = Field(
        default=None,
        alias='sh:class',
        description="Class that values must be instances of"
    )
    sh_datatype: Optional[str] = Field(
        default=None,
        alias='sh:datatype',
        description="Datatype of literal values"
    )
    sh_nodeKind: Optional[str] = Field(
        default=None,
        alias='sh:nodeKind',
        description="Kind of RDF node (sh:IRI, sh:Literal, etc.)"
    )

    # Cardinality
    sh_minCount: Optional[int] = Field(
        default=None,
        alias='sh:minCount',
        ge=0
    )
    sh_maxCount: Optional[int] = Field(
        default=None,
        alias='sh:maxCount',
        ge=0
    )

    # Value constraints
    sh_in: Optional[list[str]] = Field(
        default=None,
        alias='sh:in',
        description="Allowed values"
    )
    sh_pattern: Optional[str] = Field(
        default=None,
        alias='sh:pattern'
    )
    sh_flags: Optional[str] = Field(
        default=None,
        alias='sh:flags'
    )

    # String constraints
    sh_minLength: Optional[int] = Field(
        default=None,
        alias='sh:minLength',
        ge=0
    )
    sh_maxLength: Optional[int] = Field(
        default=None,
        alias='sh:maxLength',
        ge=0
    )
    sh_languageIn: Optional[list[str]] = Field(
        default=None,
        alias='sh:languageIn'
    )

    # Numeric constraints
    sh_minInclusive: Optional[Union[int, float]] = Field(
        default=None,
        alias='sh:minInclusive'
    )
    sh_maxInclusive: Optional[Union[int, float]] = Field(
        default=None,
        alias='sh:maxInclusive'
    )
    sh_minExclusive: Optional[Union[int, float]] = Field(
        default=None,
        alias='sh:minExclusive'
    )
    sh_maxExclusive: Optional[Union[int, float]] = Field(
        default=None,
        alias='sh:maxExclusive'
    )

    # Logical constraints
    sh_or: Optional[list[Any]] = Field(
        default=None,
        alias='sh:or'
    )
    sh_and: Optional[list[Any]] = Field(
        default=None,
        alias='sh:and'
    )
    sh_not: Optional[Any] = Field(
        default=None,
        alias='sh:not'
    )
    sh_xone: Optional[list[Any]] = Field(
        default=None,
        alias='sh:xone'
    )

    # Shape reference
    sh_node: Optional[str] = Field(
        default=None,
        alias='sh:node',
        description="Reference to another shape that values must conform to"
    )

    # Human-readable
    sh_name: Optional[str] = Field(
        default=None,
        alias='sh:name'
    )
    sh_description: Optional[str] = Field(
        default=None,
        alias='sh:description'
    )


class NodeShape(SHACLConstraint):
    """SHACL Node Shape defining constraints on a class or set of nodes.

    A NodeShape is the primary building block of SHACL schemas, defining
    constraints that apply to focus nodes matching a target.
    """
    id_: Optional[str] = Field(
        default=None,
        alias='@id',
        description="URI identifier for the node shape"
    )
    type_: Optional[Union[str, list[str]]] = Field(
        default=None,
        alias='@type',
        description="Should be sh:NodeShape"
    )

    # Targets
    sh_targetClass: Optional[Union[str, list[str]]] = Field(
        default=None,
        alias='sh:targetClass',
        description="Class whose instances are focus nodes"
    )
    sh_targetNode: Optional[Union[str, list[str]]] = Field(
        default=None,
        alias='sh:targetNode',
        description="Specific nodes that are focus nodes"
    )
    sh_targetSubjectsOf: Optional[Union[str, list[str]]] = Field(
        default=None,
        alias='sh:targetSubjectsOf',
        description="Focus nodes are subjects of triples with this predicate"
    )
    sh_targetObjectsOf: Optional[Union[str, list[str]]] = Field(
        default=None,
        alias='sh:targetObjectsOf',
        description="Focus nodes are objects of triples with this predicate"
    )

    # Property constraints
    sh_property: Optional[list[Union[PropertyShape, dict]]] = Field(
        default=None,
        alias='sh:property',
        description="Property shapes defining constraints on properties"
    )

    # Node-level constraints
    sh_class: Optional[str] = Field(
        default=None,
        alias='sh:class'
    )
    sh_nodeKind: Optional[str] = Field(
        default=None,
        alias='sh:nodeKind'
    )

    # Logical constraints at node level
    sh_or: Optional[list[Any]] = Field(
        default=None,
        alias='sh:or'
    )
    sh_and: Optional[list[Any]] = Field(
        default=None,
        alias='sh:and'
    )
    sh_not: Optional[Any] = Field(
        default=None,
        alias='sh:not'
    )
    sh_xone: Optional[list[Any]] = Field(
        default=None,
        alias='sh:xone'
    )

    # Closed shape
    sh_closed: Optional[bool] = Field(
        default=None,
        alias='sh:closed',
        description="Whether additional properties are disallowed"
    )
    sh_ignoredProperties: Optional[list[str]] = Field(
        default=None,
        alias='sh:ignoredProperties',
        description="Properties to ignore when sh:closed is true"
    )

    # Human-readable
    sh_name: Optional[str] = Field(
        default=None,
        alias='sh:name'
    )
    sh_description: Optional[str] = Field(
        default=None,
        alias='sh:description'
    )


class SHACLSchema(SHACLConstraint):
    """Complete SHACL schema containing multiple shapes.

    This is the top-level container for a SHACL shapes graph.
    """
    context: Optional[dict] = Field(
        default=None,
        alias='@context',
        description="JSON-LD context for prefix definitions"
    )
    graph: Optional[list[Union[NodeShape, dict]]] = Field(
        default=None,
        alias='@graph',
        description="List of shapes in the schema"
    )
    shapes: Optional[list[Union[NodeShape, dict]]] = Field(
        default=None,
        description="Alternative to @graph for listing shapes"
    )

    def get_shapes(self) -> list[Union[NodeShape, dict]]:
        """Return all shapes in the schema."""
        if self.graph:
            return self.graph
        if self.shapes:
            return self.shapes
        return []


# Legacy model for backward compatibility
class NodeConstraintSHACL(SHACLConstraint):
    """Legacy SHACL property constraint model for backward compatibility.

    Note: Prefer using PropertyShape for new code.
    """
    id_: Optional[str] = Field(default=None, alias='@id')
    sh_path: Optional[str] = Field(default=None, alias='sh:path')
    sh_minCount: Optional[int] = Field(default=None, alias='sh:minCount')
    sh_maxCount: Optional[int] = Field(default=None, alias='sh:maxCount')
    sh_class: Optional[str] = Field(default=None, alias='sh:class')
    sh_datatype: Optional[str] = Field(default=None, alias='sh:datatype')
    sh_nodeKind: Optional[str] = Field(default=None, alias='sh:nodeKind')
    sh_in: Optional[list[str]] = Field(default=None, alias='sh:in')
    sh_pattern: Optional[str] = Field(default=None, alias='sh:pattern')
    sh_or: Optional[list[Any]] = Field(default=None, alias='sh:or')
    sh_and: Optional[list[Any]] = Field(default=None, alias='sh:and')
    sh_not: Optional[Any] = Field(default=None, alias='sh:not')
    sh_node: Optional[str] = Field(default=None, alias='sh:node')

    @model_validator(mode='after')
    def node_constraint_types_validator(self):
        """Validate that at least one node constraint is provided."""
        # Check if we have at least one meaningful constraint
        has_node_constraint = any([
            self.sh_datatype,
            self.sh_class,
            self.sh_nodeKind,
            self.sh_in,
            self.sh_or,
            self.sh_and,
            self.sh_not,
            self.sh_node,
            self.sh_pattern
        ])

        # If we have a path but no constraint, that's still valid (just checking existence)
        if self.sh_path and not has_node_constraint:
            # This is allowed - just a property existence check
            pass

        # Both sh:class and sh:datatype set → warn and prefer sh:datatype
        if self.sh_datatype is not None and self.sh_class is not None:
            warnings.warn(
                "[NodeConstraintSHACL] Both 'sh:datatype' and 'sh:class' set; keeping 'sh:datatype'.",
                UserWarning
            )
            self.sh_class = None

        return self


# Helper functions for working with SHACL JSON-LD

def extract_property_shapes_from_jsonld(jsonld: list[dict]) -> list[dict]:
    """Extract property shapes from a SHACL JSON-LD document.

    Args:
        jsonld: JSON-LD representation of a SHACL schema

    Returns:
        List of property shape dictionaries
    """
    property_shapes = []

    for item in jsonld:
        # Check if this is a NodeShape
        item_types = item.get('@type', [])
        if isinstance(item_types, str):
            item_types = [item_types]

        if 'http://www.w3.org/ns/shacl#NodeShape' in item_types:
            # Extract property shapes
            props = item.get('http://www.w3.org/ns/shacl#property', [])
            property_shapes.extend(props)

    return property_shapes


def get_path_from_property_shape(prop: dict) -> Optional[str]:
    """Extract the sh:path from a property shape in JSON-LD format.

    Args:
        prop: Property shape dictionary in JSON-LD format

    Returns:
        The path URI or None if not found
    """
    path = prop.get('http://www.w3.org/ns/shacl#path')
    if path and isinstance(path, list) and len(path) > 0:
        return path[0].get('@id')
    return None


def has_sh_or(prop: dict) -> bool:
    """Check if a property shape contains sh:or constraint.

    Args:
        prop: Property shape dictionary in JSON-LD format

    Returns:
        True if sh:or is present
    """
    return 'http://www.w3.org/ns/shacl#or' in prop


def extract_sh_or_options(prop: dict) -> list[dict]:
    """Extract individual constraint options from sh:or.

    Args:
        prop: Property shape dictionary containing sh:or

    Returns:
        List of constraint dictionaries from the sh:or
    """
    sh_or = prop.get('http://www.w3.org/ns/shacl#or', [])
    if sh_or and isinstance(sh_or, list) and len(sh_or) > 0:
        # sh:or typically contains an @list
        or_list = sh_or[0].get('@list', [])
        return or_list
    return []
