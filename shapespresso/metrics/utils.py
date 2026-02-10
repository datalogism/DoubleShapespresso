"""Utility functions for metrics computation.

This module provides functions for extracting constraints from ShEx and SHACL
schemas, as well as helper functions for computing similarity metrics.
"""

import json
from typing import Optional

from loguru import logger
from rdflib import Graph, Namespace, RDF

from shapespresso.parser import shexc_to_shexj
from shapespresso.utils import prefix_substitute


# SHACL namespace
SH = Namespace("http://www.w3.org/ns/shacl#")

# SHACL property URIs
SH_PATH = 'http://www.w3.org/ns/shacl#path'
SH_MIN_COUNT = 'http://www.w3.org/ns/shacl#minCount'
SH_MAX_COUNT = 'http://www.w3.org/ns/shacl#maxCount'
SH_CLASS = 'http://www.w3.org/ns/shacl#class'
SH_DATATYPE = 'http://www.w3.org/ns/shacl#datatype'
SH_NODE_KIND = 'http://www.w3.org/ns/shacl#nodeKind'


def get_shapes_dict(shex_json: dict) -> dict:
    """Extract shapes dict from ShExJ json object.

    Args:
        shex_json: ShExJ json object

    Returns:
        shapes dict in {shape_id: shape}
    """
    shapes = dict()
    for shape in shex_json["shapes"]:
        if shape["id"] not in shapes:  # avoid overwriting duplicate content
            shapes[shape["id"]] = shape

    return shapes


def _is_valid_shacl_property_shape(constraint: dict) -> bool:
    """Check if a constraint dict is a valid SHACL property shape.

    A valid property shape must have at least sh:path.

    Args:
        constraint: Constraint dictionary in JSON-LD format

    Returns:
        True if valid property shape, False otherwise
    """
    return SH_PATH in constraint


def _get_jsonld_value(value) -> any:
    """Extract value from JSON-LD format.

    Args:
        value: JSON-LD value (list with @value/@id or direct value)

    Returns:
        Extracted value
    """
    if isinstance(value, list) and len(value) > 0:
        item = value[0]
        if isinstance(item, dict):
            return item.get('@value', item.get('@id'))
        return item
    return value


def summarize_shacl_constraint(constraint: dict) -> str:
    """Create a human-readable summary of a SHACL constraint.

    Useful for debugging constraint matching issues.

    Args:
        constraint: SHACL constraint in JSON-LD format

    Returns:
        Human-readable summary string
    """
    parts = []

    # Path
    path = _get_jsonld_value(constraint.get(SH_PATH))
    if path:
        parts.append(f"path={path}")

    # Cardinality
    min_count = _get_jsonld_value(constraint.get(SH_MIN_COUNT))
    max_count = _get_jsonld_value(constraint.get(SH_MAX_COUNT))
    if min_count is not None or max_count is not None:
        parts.append(f"card=[{min_count or 0},{max_count or '*'}]")

    # Class
    sh_class = _get_jsonld_value(constraint.get(SH_CLASS))
    if sh_class:
        parts.append(f"class={sh_class}")

    # Datatype
    datatype = _get_jsonld_value(constraint.get(SH_DATATYPE))
    if datatype:
        parts.append(f"datatype={datatype}")

    # NodeKind
    nodekind = _get_jsonld_value(constraint.get(SH_NODE_KIND))
    if nodekind:
        parts.append(f"nodeKind={nodekind}")

    if not parts:
        return f"<empty constraint: {list(constraint.keys())}>"

    return " | ".join(parts)


def _build_blank_node_index(shacl_json: list) -> dict:
    """Build an index of blank nodes for quick lookup.

    Args:
        shacl_json: List of JSON-LD items

    Returns:
        Dictionary mapping blank node IDs to their full objects
    """
    index = {}
    for item in shacl_json:
        if isinstance(item, dict):
            item_id = item.get('@id', '')
            if item_id.startswith('_:'):
                index[item_id] = item
    return index


def extract_shacl_constraints(
        shacl_text: str,
        shape_id: Optional[str] = None
) -> list:
    """Extract property constraints from a SHACL schema.

    This function parses a SHACL schema in Turtle format and extracts
    all property shapes from NodeShapes.

    Args:
        shacl_text: SHACL schema in Turtle format
        shape_id: Optional specific shape ID to extract constraints from

    Returns:
        List of property shape constraints in JSON-LD format
    """
    try:
        graph = Graph()
        graph.parse(data=shacl_text, format='turtle')
    except Exception as e:
        logger.error(f"Failed to parse SHACL text: {e}")
        return []

    # Serialize to JSON-LD for easier processing
    try:
        jsonld_str = graph.serialize(format="json-ld")
        shacl_json = json.loads(jsonld_str)
    except Exception as e:
        logger.error(f"Failed to serialize SHACL to JSON-LD: {e}")
        return []

    if not isinstance(shacl_json, list):
        shacl_json = [shacl_json]

    # Build index of blank nodes for efficient lookup
    blank_node_index = _build_blank_node_index(shacl_json)

    # Extract property shapes from all NodeShapes
    property_shapes = []

    for item in shacl_json:
        if not isinstance(item, dict):
            continue

        # Check if this is a NodeShape
        item_types = item.get('@type', [])
        if isinstance(item_types, str):
            item_types = [item_types]

        is_node_shape = any(
            t in item_types for t in [
                'http://www.w3.org/ns/shacl#NodeShape',
                str(SH.NodeShape)
            ]
        )

        if not is_node_shape:
            # Check if item has sh:property (might be a NodeShape without explicit type)
            if 'http://www.w3.org/ns/shacl#property' not in item:
                continue

        # Optionally filter by shape_id
        if shape_id:
            item_id = item.get('@id', '')
            if shape_id not in item_id:
                continue

        # Extract sh:property constraints
        sh_property = item.get('http://www.w3.org/ns/shacl#property', [])

        if isinstance(sh_property, dict):
            sh_property = [sh_property]

        for prop in sh_property:
            resolved_prop = None

            if isinstance(prop, dict):
                # Check if this is just a blank node reference (only has @id)
                # This happens when RDFLib serializes blank nodes separately
                if set(prop.keys()) == {'@id'}:
                    ref_id = prop['@id']
                    # Resolve the blank node reference using the index
                    resolved_prop = blank_node_index.get(ref_id)
                    if not resolved_prop:
                        logger.warning(f"Could not resolve blank node reference: {ref_id}")
                        continue
                elif '@id' in prop and len(prop) == 1:
                    # Handle case where @id is present but with different key structure
                    ref_id = prop['@id']
                    if ref_id.startswith('_:'):
                        resolved_prop = blank_node_index.get(ref_id)
                        if not resolved_prop:
                            logger.warning(f"Could not resolve blank node reference: {ref_id}")
                            continue
                    else:
                        # Named node reference
                        for other_item in shacl_json:
                            if isinstance(other_item, dict) and other_item.get('@id') == ref_id:
                                resolved_prop = other_item
                                break
                else:
                    # This is an inline property shape with actual constraint data
                    resolved_prop = prop

            elif isinstance(prop, str):
                # Property is a string reference - try to find it
                if prop.startswith('_:'):
                    resolved_prop = blank_node_index.get(prop)
                else:
                    for other_item in shacl_json:
                        if isinstance(other_item, dict) and other_item.get('@id') == prop:
                            resolved_prop = other_item
                            break

            # Validate and add the resolved property shape
            if resolved_prop:
                if _is_valid_shacl_property_shape(resolved_prop):
                    property_shapes.append(resolved_prop)
                else:
                    logger.debug(f"Skipping invalid property shape (no sh:path): {list(resolved_prop.keys())}")

    if not property_shapes:
        logger.warning("No property shapes found in SHACL schema")
    else:
        logger.debug(f"Extracted {len(property_shapes)} property shapes")
        # Log sample constraints for debugging
        for i, ps in enumerate(property_shapes[:3]):  # Log first 3
            logger.debug(f"  Sample {i+1}: {summarize_shacl_constraint(ps)}")

    return property_shapes


def extract_shacl_node_shapes(shacl_text: str) -> list:
    """Extract all NodeShapes from a SHACL schema.

    Args:
        shacl_text: SHACL schema in Turtle format

    Returns:
        List of NodeShape dictionaries in JSON-LD format
    """
    try:
        graph = Graph()
        graph.parse(data=shacl_text, format='turtle')
        jsonld_str = graph.serialize(format="json-ld")
        shacl_json = json.loads(jsonld_str)
    except Exception as e:
        logger.error(f"Failed to parse SHACL: {e}")
        return []

    if not isinstance(shacl_json, list):
        shacl_json = [shacl_json]

    node_shapes = []
    for item in shacl_json:
        if not isinstance(item, dict):
            continue

        item_types = item.get('@type', [])
        if isinstance(item_types, str):
            item_types = [item_types]

        if 'http://www.w3.org/ns/shacl#NodeShape' in item_types:
            node_shapes.append(item)

    return node_shapes


def get_shacl_target_class(node_shape: dict) -> Optional[str]:
    """Extract the target class from a SHACL NodeShape.

    Args:
        node_shape: NodeShape dictionary in JSON-LD format

    Returns:
        Target class URI or None
    """
    target_class = node_shape.get('http://www.w3.org/ns/shacl#targetClass')
    if target_class:
        if isinstance(target_class, list) and len(target_class) > 0:
            tc = target_class[0]
            if isinstance(tc, dict):
                return tc.get('@id')
            return tc
        elif isinstance(target_class, dict):
            return target_class.get('@id')
        elif isinstance(target_class, str):
            return target_class
    return None


def extract_shex_constraints(
        shexc_text: str,
        shape_id: Optional[str] = None
) -> list:
    """Extract constraints from a ShEx shape.

    Note:
        Only "EachOf" type expressions are fully supported.

    Args:
        shexc_text: ShExC text
        shape_id: ShExC shape ID (optional)

    Returns:
        List of constraints
    """
    # Transform ShExC text into ShExJ json object
    shexj_text, _, _, _ = shexc_to_shexj(shexc_text)
    if not shexj_text:
        logger.warning("Failed to load ShExC text")
        return []

    shex_json = json.loads(shexj_text)
    shapes = get_shapes_dict(shex_json)

    # Start shape
    if not shape_id:
        if "start" in shex_json:
            shape_id = shex_json["start"]
        else:
            shape_id = shex_json["shapes"][0]["id"]

    start_shape = shapes.get(shape_id)
    if not start_shape:
        logger.warning(f"Shape '{shape_id}' not found")
        return []

    # Extract constraints
    if "expression" not in start_shape:
        logger.warning(f"Failed to find 'expression' in ShExC shape {shape_id}")
        return []

    if "expressions" in start_shape["expression"]:
        expressions = start_shape["expression"]["expressions"]
        for expression in expressions:
            if expression.get("type") == "TripleConstraint" and "valueExpr" in expression:
                # Aggregate shape reference (into "valueExpr" value)
                if isinstance(expression["valueExpr"], str) and expression["valueExpr"] in shapes:
                    if expression["valueExpr"] != shape_id:  # avoid circular reference
                        expression["valueExpr"] = {
                            key: value for key, value in shapes[expression["valueExpr"]].items() if key != "id"
                        }
                    else:
                        expression["valueExpr"] = {
                            'type': 'NodeConstraint', 'nodeKind': 'iri'
                        }
        return expressions
    else:
        expression = start_shape["expression"]
        # Aggregate shape reference (into "valueExpr" value)
        if "valueExpr" in expression:
            if isinstance(expression["valueExpr"], str) and expression["valueExpr"] in shapes:
                if expression["valueExpr"] != shape_id:  # avoid circular reference
                    expression["valueExpr"] = {
                        key: value for key, value in shapes[expression["valueExpr"]].items() if key != "id"
                    }
                else:
                    expression["valueExpr"] = {
                        'type': 'NodeConstraint', 'nodeKind': 'iri'
                    }
        return [expression]


def get_predicate_node_label(constraint: dict) -> str:
    """Get node label of predicate.

    Args:
        constraint: triple constraint

    Returns:
        Constraint predicate node label
    """
    predicate_url = constraint.get("predicate")
    if predicate_url:
        node_label = prefix_substitute(predicate_url)
        return node_label
    else:
        return ""


def get_node_constraint_node_label(constraint: dict, shapes: dict, start_shape_id: str) -> str:
    """Get node label of node constraint.

    Note:
        Only two layers of referenced shapes are considered.

    Args:
        constraint: triple constraint
        shapes: shapes dict
        start_shape_id: start shape ID (avoid circular reference)

    Returns:
        Node constraint node label
    """
    value_expr = constraint.get("valueExpr")
    if not value_expr:
        return "."

    if isinstance(value_expr, str):  # shape reference
        if value_expr == start_shape_id:
            return start_shape_id
        elif value_expr in shapes:
            referenced_shape = shapes[value_expr]
            if "expression" in referenced_shape:
                if "predicate" in referenced_shape["expression"]:
                    ref_shape_predicate = prefix_substitute(referenced_shape["expression"]["predicate"])
                    ref_shape_value_expr = referenced_shape["expression"].get("valueExpr", ".")
                    if isinstance(ref_shape_value_expr, str):
                        node_label = f"{ref_shape_predicate} {ref_shape_value_expr}".strip()
                    else:
                        values = ref_shape_value_expr.get("values", [""])
                        values = [prefix_substitute(value) for value in values]
                        values_label = " ".join(sorted(values))
                        node_label = f"{ref_shape_predicate} {values_label}".strip()
                else:  # referenced shape has multiple constraints
                    ref_shape_expressions = referenced_shape["expression"].get("expressions", [])
                    if ref_shape_expressions:
                        # Assume that the first one is 'instance of' constraint
                        ref_shape_predicate = prefix_substitute(
                            referenced_shape["expression"]["expressions"][0]["predicate"]
                        )
                        ref_shape_value_expr = referenced_shape["expression"]["expressions"][0].get("valueExpr", {})
                        values = ref_shape_value_expr.get("values", [""])
                        values = [prefix_substitute(value) for value in values]
                        values = " ".join(sorted(values))
                        node_label = f"{ref_shape_predicate} {values}".strip()
                    else:
                        logger.warning(f"Failed to parse referenced shape {referenced_shape}")
                        node_label = "."
            else:
                logger.warning(f"Failed to find 'expression' in shape {referenced_shape}")
                node_label = "."
        else:  # label of valueExpr is not in shapes
            node_label = value_expr

    elif isinstance(value_expr, dict):  # node constraint
        if "values" in value_expr:
            values = value_expr["values"]
            if isinstance(values[0], dict) and "stem" in values[0]:  # IriStem, etc.
                node_label = f"{values[0]['type']}: {values[0]['stem']}"
            else:  # value set
                values = [prefix_substitute(value) for value in values]
                node_label = " ".join(sorted(values)).strip()
        elif "datatype" in value_expr:  # datatype
            node_label = prefix_substitute(value_expr["datatype"])
        elif "nodeKind" in value_expr:  # nodeKind
            node_label = value_expr["nodeKind"]
        else:
            logger.warning(f"Failed to parse value_expr {value_expr} in schema graph!")
            node_label = "."
    else:
        logger.warning(f"Failed to parse value_expr {value_expr} in schema graph!")
        node_label = "."

    return node_label


def get_cardinality_node_label(constraint: dict) -> str:
    """Get node label of cardinality.

    Args:
        constraint: triple constraint

    Returns:
        Cardinality node label
    """
    min_value = constraint.get("min", 1)
    max_value = constraint.get("max", 1)

    if min_value == 0 and max_value == -1:
        node_label = "*"
    elif min_value == 1 and max_value == -1:
        node_label = "+"
    elif min_value == 0 and max_value == 1:
        node_label = "?"
    else:
        node_label = f"{{{min_value}, {max_value}}}"

    return node_label


# SHACL-specific utility functions

def get_shacl_predicate_node_label(constraint: dict) -> str:
    """Get node label of predicate from SHACL property shape.

    Args:
        constraint: SHACL property shape in JSON-LD format

    Returns:
        Predicate node label
    """
    path = constraint.get('http://www.w3.org/ns/shacl#path')
    if path and isinstance(path, list) and len(path) > 0:
        path_id = path[0].get('@id')
        if path_id:
            return prefix_substitute(path_id)
    return ""


def get_shacl_node_constraint_label(constraint: dict) -> str:
    """Get node label for SHACL node constraint.

    Args:
        constraint: SHACL property shape in JSON-LD format

    Returns:
        Node constraint label
    """
    # sh:class
    sh_class = constraint.get('http://www.w3.org/ns/shacl#class')
    if sh_class:
        if isinstance(sh_class, list) and len(sh_class) > 0:
            class_id = sh_class[0].get('@id')
            if class_id:
                return prefix_substitute(class_id)

    # sh:datatype
    sh_datatype = constraint.get('http://www.w3.org/ns/shacl#datatype')
    if sh_datatype:
        if isinstance(sh_datatype, list) and len(sh_datatype) > 0:
            dt_id = sh_datatype[0].get('@id')
            if dt_id:
                return prefix_substitute(dt_id)

    # sh:nodeKind
    sh_nodekind = constraint.get('http://www.w3.org/ns/shacl#nodeKind')
    if sh_nodekind:
        if isinstance(sh_nodekind, list) and len(sh_nodekind) > 0:
            nk_id = sh_nodekind[0].get('@id')
            if nk_id:
                return nk_id.split('#')[-1]  # e.g., "IRI" from "http://...#IRI"

    # sh:or
    sh_or = constraint.get('http://www.w3.org/ns/shacl#or')
    if sh_or:
        return "sh:or"

    return "."


def get_shacl_cardinality_label(constraint: dict) -> str:
    """Get cardinality label for SHACL property shape.

    Args:
        constraint: SHACL property shape in JSON-LD format

    Returns:
        Cardinality label
    """
    min_count = 0
    max_count = -1  # unbounded

    sh_min = constraint.get('http://www.w3.org/ns/shacl#minCount')
    if sh_min and isinstance(sh_min, list) and len(sh_min) > 0:
        min_count = sh_min[0].get('@value', 0)

    sh_max = constraint.get('http://www.w3.org/ns/shacl#maxCount')
    if sh_max and isinstance(sh_max, list) and len(sh_max) > 0:
        max_count = sh_max[0].get('@value', -1)

    if min_count == 0 and max_count == -1:
        return "*"
    elif min_count == 1 and max_count == -1:
        return "+"
    elif min_count == 0 and max_count == 1:
        return "?"
    elif max_count == -1:
        return f"{{{min_count},*}}"
    else:
        return f"{{{min_count},{max_count}}}"
