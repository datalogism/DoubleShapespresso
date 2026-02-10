"""Classification metrics for evaluating ShEx and SHACL shapes.

This module provides functions for comparing predicted constraints against
ground truth constraints using various matching strategies.
"""

import math
import os
from typing import Optional, Any

from loguru import logger
from itertools import product
from pathlib import Path

from shapespresso.metrics.utils import extract_shex_constraints, extract_shacl_constraints
from shapespresso.utils import prefix_substitute, endpoint_sparql_query
from shapespresso.pipeline import prefix_replace

# SHACL namespace constants
SH_PATH = 'http://www.w3.org/ns/shacl#path'
SH_MIN_COUNT = 'http://www.w3.org/ns/shacl#minCount'
SH_MAX_COUNT = 'http://www.w3.org/ns/shacl#maxCount'
SH_CLASS = 'http://www.w3.org/ns/shacl#class'
SH_DATATYPE = 'http://www.w3.org/ns/shacl#datatype'
SH_NODE_KIND = 'http://www.w3.org/ns/shacl#nodeKind'
SH_OR = 'http://www.w3.org/ns/shacl#or'
SH_AND = 'http://www.w3.org/ns/shacl#and'
SH_NOT = 'http://www.w3.org/ns/shacl#not'
SH_IN = 'http://www.w3.org/ns/shacl#in'
SH_PATTERN = 'http://www.w3.org/ns/shacl#pattern'
SH_NODE = 'http://www.w3.org/ns/shacl#node'


def _get_shacl_path(constraint: dict) -> Optional[str]:
    """Extract the sh:path from a SHACL constraint.

    Handles both direct path and path within sh:or/sh:and.

    Args:
        constraint: SHACL constraint dictionary in JSON-LD format

    Returns:
        The path URI or None if not found
    """
    # Direct path
    path = constraint.get(SH_PATH)
    if path and isinstance(path, list) and len(path) > 0:
        return path[0].get('@id')

    # Path might be outside the logical constraint
    return None


def _has_logical_constraint(constraint: dict) -> bool:
    """Check if constraint contains sh:or, sh:and, or sh:not."""
    return any(k in constraint for k in [SH_OR, SH_AND, SH_NOT])


def _extract_sh_or_options(constraint: dict) -> list[dict]:
    """Extract individual options from sh:or constraint.

    Args:
        constraint: SHACL constraint containing sh:or

    Returns:
        List of constraint option dictionaries
    """
    sh_or = constraint.get(SH_OR, [])
    if sh_or and isinstance(sh_or, list) and len(sh_or) > 0:
        # sh:or contains an @list in JSON-LD
        or_item = sh_or[0]
        if isinstance(or_item, dict) and '@list' in or_item:
            return or_item['@list']
        # Sometimes it's just a list of items
        return sh_or
    return []


def _extract_sh_and_options(constraint: dict) -> list[dict]:
    """Extract individual options from sh:and constraint.

    Args:
        constraint: SHACL constraint containing sh:and

    Returns:
        List of constraint option dictionaries
    """
    sh_and = constraint.get(SH_AND, [])
    if sh_and and isinstance(sh_and, list) and len(sh_and) > 0:
        and_item = sh_and[0]
        if isinstance(and_item, dict) and '@list' in and_item:
            return and_item['@list']
        return sh_and
    return []


def _get_shacl_value_from_jsonld(value: Any) -> Any:
    """Extract the actual value from JSON-LD formatted value.

    Args:
        value: JSON-LD value (could be list with @value or @id)

    Returns:
        The extracted value
    """
    if isinstance(value, list) and len(value) > 0:
        item = value[0]
        if isinstance(item, dict):
            return item.get('@value', item.get('@id'))
        return item
    return value


def predicate_match(y_true: dict, y_pred: dict, syntax: str) -> bool:
    """Exact predicate match.

    Args:
        y_true: ground truth constraint
        y_pred: predicted constraint
        syntax: "ShEx" or "SHACL"

    Returns:
        True if predicates of constraints match, otherwise False
    """
    if syntax == "SHACL":
        y_true_predicate = _get_shacl_path(y_true)
        y_pred_predicate = _get_shacl_path(y_pred)

        # Handle sh:or case - check if path is present in either constraint
        if y_true_predicate is None and _has_logical_constraint(y_true):
            # For logical constraints, we need to compare the overall constraint
            # The path should be at the same level or inherited
            logger.debug(f"Logical constraint detected in ground truth: {list(y_true.keys())}")

        if y_pred_predicate is None and _has_logical_constraint(y_pred):
            logger.debug(f"Logical constraint detected in prediction: {list(y_pred.keys())}")

        if y_true_predicate and y_pred_predicate:
            return y_true_predicate == y_pred_predicate
        elif y_true_predicate is None and y_pred_predicate is None:
            # Both might be logical constraints - consider them for matching
            # at a higher level
            return False
        else:
            return False
    else:
        # ShEx handling
        y_true_predicate = y_true.get("predicate")
        y_pred_predicate = y_pred.get("predicate")

        if y_true_predicate:
            return y_true_predicate == y_pred_predicate
        else:
            return False


def _get_shacl_cardinality(constraint: dict) -> tuple[int, float]:
    """Extract minCount and maxCount from SHACL constraint.

    Args:
        constraint: SHACL constraint dictionary

    Returns:
        Tuple of (minCount, maxCount) where maxCount can be math.inf
    """
    min_count = 0
    max_count = math.inf

    if SH_MIN_COUNT in constraint:
        min_val = _get_shacl_value_from_jsonld(constraint[SH_MIN_COUNT])
        if min_val is not None:
            min_count = int(min_val)

    if SH_MAX_COUNT in constraint:
        max_val = _get_shacl_value_from_jsonld(constraint[SH_MAX_COUNT])
        if max_val is not None:
            max_count = int(max_val)

    return min_count, max_count


def cardinality_match(y_true: dict, y_pred: dict, syntax: str) -> bool:
    """Exact cardinality match.

    Args:
        y_true: ground truth constraint
        y_pred: predicted constraint
        syntax: "ShEx" or "SHACL"

    Returns:
        True if cardinality match, otherwise False
    """
    if syntax == "SHACL":
        y_true_min, y_true_max = _get_shacl_cardinality(y_true)
        y_pred_min, y_pred_max = _get_shacl_cardinality(y_pred)
    else:
        y_true_min = y_true.get("min", 1)
        y_true_max = y_true.get("max", 1)
        y_pred_min = y_pred.get("min", 1)
        y_pred_max = y_pred.get("max", 1)

    return y_true_min == y_pred_min and y_true_max == y_pred_max


def _get_shacl_node_constraint(constraint: dict) -> dict:
    """Extract the node constraint (class, datatype, nodeKind, etc.) from SHACL.

    Handles sh:or by returning a normalized representation.

    Args:
        constraint: SHACL constraint dictionary

    Returns:
        Normalized node constraint dictionary
    """
    result = {}

    # Direct constraints
    if SH_CLASS in constraint:
        result['class'] = _get_shacl_value_from_jsonld(constraint[SH_CLASS])
    if SH_DATATYPE in constraint:
        result['datatype'] = _get_shacl_value_from_jsonld(constraint[SH_DATATYPE])
    if SH_NODE_KIND in constraint:
        result['nodeKind'] = _get_shacl_value_from_jsonld(constraint[SH_NODE_KIND])
    if SH_IN in constraint:
        # sh:in is a list of allowed values
        in_values = constraint[SH_IN]
        if isinstance(in_values, list) and len(in_values) > 0:
            in_item = in_values[0]
            if isinstance(in_item, dict) and '@list' in in_item:
                result['in'] = [_get_shacl_value_from_jsonld([v]) for v in in_item['@list']]
            else:
                result['in'] = [_get_shacl_value_from_jsonld([v]) for v in in_values]
    if SH_PATTERN in constraint:
        result['pattern'] = _get_shacl_value_from_jsonld(constraint[SH_PATTERN])
    if SH_NODE in constraint:
        result['node'] = _get_shacl_value_from_jsonld(constraint[SH_NODE])

    # Handle sh:or - store as list of possible constraints
    if SH_OR in constraint:
        or_options = _extract_sh_or_options(constraint)
        result['or'] = [_get_shacl_node_constraint(opt) for opt in or_options]

    # Handle sh:and - all constraints must be satisfied
    if SH_AND in constraint:
        and_options = _extract_sh_and_options(constraint)
        result['and'] = [_get_shacl_node_constraint(opt) for opt in and_options]

    return result


def node_constraint_match(y_true: dict, y_pred: dict, syntax: str) -> bool:
    """Exact node constraint match.

    Args:
        y_true: ground truth constraint
        y_pred: predicted constraint
        syntax: "ShEx" or "SHACL"

    Returns:
        True if node constraints match, otherwise False
    """
    if syntax == "SHACL":
        # Extract normalized node constraints
        y_true_nc = _get_shacl_node_constraint(y_true)
        y_pred_nc = _get_shacl_node_constraint(y_pred)

        # Handle sh:or matching
        if 'or' in y_true_nc and 'or' in y_pred_nc:
            # Both have sh:or - check if the options match
            true_options = y_true_nc['or']
            pred_options = y_pred_nc['or']

            # Check if all options in true are present in pred
            if len(true_options) != len(pred_options):
                return False

            # Try to match each option
            matched = set()
            for true_opt in true_options:
                for i, pred_opt in enumerate(pred_options):
                    if i not in matched and true_opt == pred_opt:
                        matched.add(i)
                        break
            return len(matched) == len(true_options)

        elif 'or' in y_true_nc or 'or' in y_pred_nc:
            # Only one has sh:or - check if the single constraint matches any option
            if 'or' in y_true_nc:
                # pred should match at least one option in true's sh:or
                for true_opt in y_true_nc['or']:
                    if true_opt == y_pred_nc:
                        return True
                return False
            else:
                # true should match at least one option in pred's sh:or
                for pred_opt in y_pred_nc['or']:
                    if y_true_nc == pred_opt:
                        return True
                return False

        # Direct comparison (no sh:or)
        return y_true_nc == y_pred_nc

    else:
        # ShEx handling
        y_true_value_expr = y_true.get("valueExpr")
        y_pred_value_expr = y_pred.get("valueExpr")

        if y_true_value_expr and y_true_value_expr == y_pred_value_expr:
            return True
        else:
            return False


def exact_constraint_match(y_true: dict, y_pred: dict, syntax: str) -> bool:
    """Exact match of all constraint components.

    Args:
        y_true: ground truth constraint
        y_pred: predicted constraint
        syntax: "ShEx" or "SHACL"

    Returns:
        True if all elements in both constraints are equal, otherwise False
    """
    return (predicate_match(y_true, y_pred, syntax) and
            node_constraint_match(y_true, y_pred, syntax) and
            cardinality_match(y_true, y_pred, syntax))


def ask_subclass_of(dataset: str, true_class: str, pred_class: str, endpoint_url: str = 'http://localhost:1234/api/endpoint/sparql') -> bool:
    """Ask if ground truth class is subclass of predicted class.

    Args:
        dataset: name of dataset
        true_class: ground truth class url
        pred_class: predicted class url
        endpoint_url: SPARQL endpoint URL

    Returns:
        True if ground truth class is subclass of predicted class, otherwise False
    """
    subclass_of_prop = "wdt:P279" if dataset == "wes" else "rdfs:subClassOf"
    query = f"ASK {{ {prefix_substitute(true_class)} {subclass_of_prop}* {prefix_substitute(pred_class)} }}"
    result = endpoint_sparql_query(query, endpoint_url=endpoint_url, mode="ask")
    return bool(result)


def approximate_class_match(
        dataset: str,
        true_classes: list[str],
        pred_classes: list[str],
        value_type_const_classes: list[str] = None,
        endpoint_url: str = 'http://localhost:1234/api/endpoint/sparql'
) -> bool:
    """Approximate class matching.

    Args:
        dataset: name of dataset
        true_classes: ground truth classes
        pred_classes: predicted classes
        value_type_const_classes: list of value type constraint classes (optional)
        endpoint_url: SPARQL endpoint URL

    Returns:
        True if one of ground truth classes is subclass of one of predicted classes
    """
    if value_type_const_classes:
        class_pairs = list(product(true_classes, pred_classes)) + list(product(value_type_const_classes, pred_classes))
    else:
        class_pairs = list(product(true_classes, pred_classes))
    for class_pair in class_pairs:
        if ask_subclass_of(dataset, class_pair[0], class_pair[1], endpoint_url=endpoint_url):
            return True
    return False


def approximate_class_constraint_match(
        dataset: str,
        y_true: dict,
        y_pred: dict,
        syntax: str,
        value_type_const_classes: list[str] = None,
        endpoint_url: str = 'http://localhost:1234/api/endpoint/sparql'
) -> bool:
    """Approximate class constraint match.

    Args:
        dataset: name of dataset
        y_true: ground truth constraint
        y_pred: predicted constraint
        syntax: "ShEx" or "SHACL"
        value_type_const_classes: list of value type constraint classes (optional)
        endpoint_url: SPARQL endpoint URL

    Returns:
        True if one of ground truth classes is subclass of one of predicted classes
    """
    if syntax == "SHACL":
        # Extract classes from SHACL constraints
        true_nc = _get_shacl_node_constraint(y_true)
        pred_nc = _get_shacl_node_constraint(y_pred)
        true_classes = []
        pred_classes = []

        if 'class' in true_nc:
            true_classes.append(true_nc['class'])
        if 'class' in pred_nc:
            pred_classes.append(pred_nc['class'])

        # Handle sh:or
        if 'or' in true_nc:
            for opt in true_nc['or']:
                if 'class' in opt:
                    true_classes.append(opt['class'])
        if 'or' in pred_nc:
            for opt in pred_nc['or']:
                if 'class' in opt:
                    pred_classes.append(opt['class'])
        if true_classes and pred_classes:
            return approximate_class_match(dataset, true_classes, pred_classes, value_type_const_classes, endpoint_url=endpoint_url)
        return False

    else:
        # ShEx handling
        y_true_value_expr = y_true.get("valueExpr")
        y_pred_value_expr = y_pred.get("valueExpr")

        if y_true_value_expr and y_true_value_expr == y_pred_value_expr:
            return True
        else:
            if isinstance(y_pred_value_expr, str):
                return False
            elif y_true_value_expr is None or y_pred_value_expr is None:
                return False
            elif y_true_value_expr.get("type") != y_pred_value_expr.get("type"):
                return False
            else:
                # value set
                if y_true_value_expr.get("type") == "NodeConstraint":
                    true_classes = y_true_value_expr.get("values")
                    pred_classes = y_pred_value_expr.get("values")
                    if true_classes and pred_classes:
                        pred_classes = [item for item in pred_classes if isinstance(item, str)]
                        return approximate_class_match(dataset, true_classes, pred_classes, value_type_const_classes, endpoint_url=endpoint_url)
                    else:
                        return False
                # shape reference
                elif y_true_value_expr.get("type") == "Shape":
                    true_classes = y_true_value_expr.get("expression", {}).get("valueExpr", {}).get("values", [])
                    pred_classes = y_pred_value_expr.get("expression", {}).get("valueExpr", {}).get("values", [])
                    if true_classes and pred_classes:
                        return approximate_class_match(dataset, true_classes, pred_classes, value_type_const_classes, endpoint_url=endpoint_url)
                    else:
                        return False

    return False


def _normalize_datatype(datatype: str) -> str:
    """Normalize datatype URIs for comparison.

    Args:
        datatype: datatype URI

    Returns:
        Normalized datatype URI
    """
    # langString -> string
    if datatype == "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString":
        return "http://www.w3.org/2001/XMLSchema#string"
    # float -> decimal
    if datatype == "http://www.w3.org/2001/XMLSchema#float":
        return "http://www.w3.org/2001/XMLSchema#decimal"
    return datatype


def get_constraint_datatype(constraint: dict, syntax: str) -> dict | str:
    """Extract datatype information from constraint.

    Args:
        constraint: constraint dictionary
        syntax: "ShEx" or "SHACL"

    Returns:
        Datatype information dictionary
    """
    if syntax == "SHACL":
        nc = _get_shacl_node_constraint(constraint)

        if 'datatype' in nc:
            normalized = _normalize_datatype(nc['datatype'])
            return {"type": "NodeConstraint", "datatype": normalized}

        if 'nodeKind' in nc:
            return {"type": "NodeConstraint", "nodeKind": nc['nodeKind']}

        if 'class' in nc:
            return {"type": "NodeConstraint", "nodeKind": "iri"}

        # Handle sh:or
        if 'or' in nc:
            # For sh:or, check if all options have the same datatype
            datatypes = []
            for opt in nc['or']:
                if 'datatype' in opt:
                    datatypes.append(_normalize_datatype(opt['datatype']))
            if datatypes:
                return {"type": "NodeConstraint", "or_datatypes": datatypes}

        return {"type": "NodeConstraint", "nodeKind": "iri"}

    else:
        # ShEx handling
        if "valueExpr" in constraint:
            if "datatype" in constraint["valueExpr"]:
                normalized = _normalize_datatype(constraint["valueExpr"]["datatype"])
                return {"type": "NodeConstraint", "datatype": normalized}
            elif "nodeKind" in constraint["valueExpr"]:
                return constraint["valueExpr"]
            else:
                return {"type": "NodeConstraint", "nodeKind": "iri"}
        else:
            return "."


def datatype_match(y_true: dict, y_pred: dict, syntax: str) -> bool:
    """Datatype match.

    Args:
        y_true: ground truth constraint
        y_pred: predicted constraint
        syntax: "ShEx" or "SHACL"

    Returns:
        True if datatype of constraints match, otherwise False
    """
    y_true_datatype = get_constraint_datatype(y_true, syntax)
    y_pred_datatype = get_constraint_datatype(y_pred, syntax)

    return y_true_datatype == y_pred_datatype


def loosened_cardinality_match(y_true: dict, y_pred: dict, syntax: str) -> bool:
    """Relax the evaluation by allowing broader matches on cardinality.

    Args:
        y_true: ground truth constraint
        y_pred: predicted constraint
        syntax: "ShEx" or "SHACL"

    Returns:
        True if y_pred_min <= y_true_min <= y_true_max <= y_pred_max
        For rejected properties (0,0), exact match is required
    """
    if syntax == "SHACL":
        y_true_min, y_true_max = _get_shacl_cardinality(y_true)
        y_pred_min, y_pred_max = _get_shacl_cardinality(y_pred)
    else:
        y_true_min = y_true.get("min", 1)
        y_true_max = y_true.get("max", 1) if y_true.get("max", 1) != -1 else math.inf
        y_pred_min = y_pred.get("min", 1)
        y_pred_max = y_pred.get("max", 1) if y_pred.get("max", 1) != -1 else math.inf

    # Rejected property should be rejected as well
    if y_true_min == 0 and y_true_max == 0:
        return y_pred_min == 0 and y_pred_max == 0
    else:
        return y_pred_min <= y_true_min and y_true_max <= y_pred_max


def constraint_match(
        dataset: str,
        syntax: str,
        y_true: dict,
        y_pred: dict,
        node_constraint_matching_level: str = "exact",
        cardinality_matching_level: str = "exact",
        value_type_const_classes: list[str] = None,
        endpoint_url: str = 'http://localhost:1234/api/endpoint/sparql'
) -> bool:
    """Constraint match at a given matching level.

    Args:
        dataset: name of dataset
        syntax: "ShEx" or "SHACL"
        y_true: ground truth constraint
        y_pred: predicted constraint
        node_constraint_matching_level: "exact", "approximate", or "datatype"
        cardinality_matching_level: "exact" or "loosened"
        value_type_const_classes: list of value type constraint classes (optional)
        endpoint_url: SPARQL endpoint URL

    Returns:
        True if constraint match at the given matching level
    """
    if not predicate_match(y_true, y_pred, syntax):
        return False

    # Node constraint matching
    if node_constraint_matching_level == "exact":
        node_constraint_matching = node_constraint_match(y_true, y_pred, syntax)
    elif node_constraint_matching_level == "approximate":
        node_constraint_matching = approximate_class_constraint_match(
            dataset, y_true, y_pred, syntax, value_type_const_classes, endpoint_url=endpoint_url
        )
    elif node_constraint_matching_level == "datatype":
        node_constraint_matching = datatype_match(y_true, y_pred, syntax)
    else:
        raise NotImplementedError("'node_constraint_matching_level' must be 'exact', 'approximate', or 'datatype'")

    # Cardinality matching
    if cardinality_matching_level == "exact":
        cardinality_matching = cardinality_match(y_true, y_pred, syntax)
    elif cardinality_matching_level == "loosened":
        cardinality_matching = loosened_cardinality_match(y_true, y_pred, syntax)
    else:
        raise NotImplementedError("'cardinality_matching_level' must be 'exact' or 'loosened'")

    return node_constraint_matching and cardinality_matching


def count_true_positives(
        dataset: str,
        syntax: str,
        y_true: list[dict],
        y_pred: list[dict],
        node_constraint_matching_level: str = "exact",
        cardinality_matching_level: str = "exact",
        value_type_constraints: dict = None,
        endpoint_url: str = 'http://localhost:1234/api/endpoint/sparql'
) -> tuple[int, list[dict]]:
    """Count the number of true positives (matched constraints).

    Args:
        dataset: name of dataset
        syntax: "ShEx" or "SHACL"
        y_true: ground truth constraints
        y_pred: predicted constraints
        node_constraint_matching_level: node constraint matching level
        cardinality_matching_level: cardinality matching level
        value_type_constraints: value type constraints (optional)
        endpoint_url: SPARQL endpoint URL

    Returns:
        Tuple of (number of true positives, list of true positives)
    """
    true_positives = []

    logger.debug(f"Comparing {len(y_true)} ground truth vs {len(y_pred)} predicted constraints")
    for true_constraint in y_true:
        for pred_constraint in y_pred:
            value_type_const_classes = None

            if value_type_constraints:

                # Extract predicate ID for value type constraint lookup
                if syntax == "SHACL":
                    pred_path = _get_shacl_path(pred_constraint)
                    if pred_path:
                        predicate_id = pred_path.split("/")[-1]
                        vtc = value_type_constraints.get(predicate_id, {})
                        value_type_const_classes = [item["url"] for item in vtc.get("value_type_constraint", [])]
                else:
                    predicate_id = pred_constraint.get("predicate", "").split("/")[-1]
                    vtc = value_type_constraints.get(predicate_id, {})
                    value_type_const_classes = [item["url"] for item in vtc.get("value_type_constraint", [])]

            if constraint_match(
                    dataset=dataset,
                    syntax=syntax,
                    y_true=true_constraint,
                    y_pred=pred_constraint,
                    node_constraint_matching_level=node_constraint_matching_level,
                    cardinality_matching_level=cardinality_matching_level,
                    value_type_const_classes=value_type_const_classes,
                    endpoint_url=endpoint_url
            ):
                true_positives.append(true_constraint)
                break  # Avoid counting same true constraint multiple times

    return len(true_positives), true_positives


def evaluate(
        dataset: str,
        syntax: str,
        class_urls: list[str],
        class_labels: list[str],
        ground_truth_dir: str | Path,
        predicted_dir: str | Path,
        node_constraint_matching_level: str = "exact",
        cardinality_matching_level: str = "exact",
        value_type_constraints: dict = None,
        endpoint_url: str = 'http://localhost:1234/api/endpoint/sparql'
) -> tuple[float, float, float]:
    """Evaluate function based on classification metrics.

    Args:
        dataset: name of dataset
        syntax: "ShEx" or "SHACL"
        class_urls: list of class urls to evaluate
        class_labels: list of class labels
        ground_truth_dir: path to ground truth schema directory
        predicted_dir: path to predicted schema directory
        node_constraint_matching_level: "exact", "approximate", or "datatype"
        cardinality_matching_level: "exact" or "loosened"
        value_type_constraints: value type constraints (optional)
        endpoint_url: SPARQL endpoint URL for approximate matching

    Returns:
        Tuple of (macro_precision, macro_recall, macro_f1_score)
    """
    precisions, recalls, f1_scores = [], [], []
    total_true_positives, total_true, total_pred = 0, 0, 0

    for class_url, class_label in zip(class_urls, class_labels):
        class_id = class_url.split("/")[-1]
        if dataset == "wes":
            shape_id = "".join([word.capitalize() for word in class_label.split()])
        else:
            shape_id = class_label
        logger.info(f"Evaluating shape '{shape_id}' in class '{class_id}'")

        if syntax == "SHACL":
            # Extract ground truth constraints
            true_shacl_path = os.path.join(ground_truth_dir, f"{class_id}.ttl")
            if not os.path.exists(true_shacl_path):
                logger.warning(f"Ground truth file '{true_shacl_path}' does not exist")
                continue
            true_shacl_text = Path(true_shacl_path).read_text(encoding="utf-8")
            true_constraints = extract_shacl_constraints(shacl_text=true_shacl_text)
            logger.debug(f"Extracted {len(true_constraints)} ground truth constraints")

            # Extract predicted constraints
            pred_shacl_path = os.path.join(predicted_dir, f"{class_id}.ttl")
            if not os.path.exists(pred_shacl_path):
                logger.warning(f"Predicted file '{pred_shacl_path}' does not exist")
                continue
            pred_shacl_text = Path(pred_shacl_path).read_text(encoding="utf-8")

            pred_constraints = extract_shacl_constraints(shacl_text=pred_shacl_text)
            logger.debug(f"Extracted {len(pred_constraints)} predicted constraints")
            if not pred_constraints:
                logger.warning(f"Unable to parse SHACL file '{pred_shacl_path}'")
                continue

        else:
            # Extract ground truth constraints
            true_shex_path = os.path.join(ground_truth_dir, f"{class_id}.shex")
            if not os.path.exists(true_shex_path):
                logger.warning(f"Ground truth file '{true_shex_path}' does not exist")
                continue
            true_shexc_text = Path(true_shex_path).read_text()
            true_constraints = extract_shex_constraints(shexc_text=true_shexc_text)

            # Extract predicted constraints
            pred_shex_path = os.path.join(predicted_dir, f"{class_id}.shex")
            if not os.path.exists(pred_shex_path):
                logger.warning(f"Predicted file '{pred_shex_path}' does not exist")
                continue
            pred_shexc_text = Path(pred_shex_path).read_text()
            pred_constraints = extract_shex_constraints(shexc_text=pred_shexc_text)
            if not pred_constraints:
                logger.warning(f"Unable to parse ShEx file '{pred_shex_path}'")
                continue

        # Count true positives
        num_true, num_pred = len(true_constraints), len(pred_constraints)
        logger.debug(f"Ground truth: {num_true} constraints, Predicted: {num_pred} constraints")
        num_true_positives, true_positives = count_true_positives(
            dataset=dataset,
            syntax=syntax,
            y_true=true_constraints,
            y_pred=pred_constraints,
            node_constraint_matching_level=node_constraint_matching_level,
            cardinality_matching_level=cardinality_matching_level,
            value_type_constraints=value_type_constraints,
            endpoint_url=endpoint_url
        )
        logger.debug(f"True positives: {num_true_positives}")
        logger.debug(true_positives)
        total_true_positives += num_true_positives
        total_true += num_true
        total_pred += num_pred

        # Calculate precision
        precision = num_true_positives / num_pred if num_pred > 0 else 0.0
        precisions.append(precision)

        # Calculate recall
        recall = num_true_positives / num_true if num_true > 0 else 0.0
        recalls.append(recall)

        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        f1_scores.append(f1_score)

        logger.info(f"True positives: {num_true_positives}, Precision: {precision:.3f}, "
                    f"Recall: {recall:.3f}, F1 Score: {f1_score:.3f}")

    if not precisions:
        logger.warning("No schemas were evaluated")
        return 0.0, 0.0, 0.0

    # Calculate macro scores
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    logger.info(f"Macro ({node_constraint_matching_level}, {cardinality_matching_level}) "
                f"scores calculated for {len(precisions)} classes:")
    logger.info(f"Precision: {macro_precision:.3f} & Recall: {macro_recall:.3f} & F1: {macro_f1_score:.3f}")

    return macro_precision, macro_recall, macro_f1_score
