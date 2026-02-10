import json
import os

from loguru import logger
from pathlib import Path

import networkx as nx

import matplotlib
from matplotlib import pyplot as plt

from shapespresso.metrics.utils import (
    get_shapes_dict,
    get_predicate_node_label,
    get_node_constraint_node_label,
    get_cardinality_node_label,
    # SHACL-specific utilities
    extract_shacl_constraints,
    extract_shacl_node_shapes,
    get_shacl_target_class,
    get_shacl_predicate_node_label,
    get_shacl_node_constraint_label,
    get_shacl_cardinality_label
)
from shapespresso.parser import shexc_to_shexj

from zss import simple_distance


def node_match(node_1, node_2):
    """ node match function for networkx.graph_edit_distance
    """
    if node_1["label"] == node_2["label"]:
        return 0
    else:
        return 1


def edge_match(edge_1, edge_2):
    """ edge match function for networkx.graph_edit_distance
    """
    if edge_1["label"] == edge_2["label"]:
        return 0
    else:
        return 1


def transform_schema_to_graph(schema: dict):
    """ transform schema to NetworkX DiGraph rooted in start shape ID

    Args:
        schema (dict): schema in ShExJ json object

    Returns:
        NetworkX DiGraph rooted in start shape ID
    """
    start_shape_id = schema['start']  # root node
    schema_graph = nx.DiGraph()
    schema_graph.add_node(start_shape_id, label=start_shape_id)

    shapes = get_shapes_dict(schema)
    start_shape = shapes[start_shape_id]

    if "expression" in start_shape:
        if "expressions" in start_shape["expression"]:
            for triple_constraint in start_shape["expression"]["expressions"]:
                # predicate node
                predicate_node = get_predicate_node_label(triple_constraint)
                schema_graph.add_node(predicate_node, label=predicate_node)
                schema_graph.add_edge(start_shape_id, predicate_node, label=f"{start_shape_id} {predicate_node}")
                # node_constraint node
                node_constraint_node = get_node_constraint_node_label(triple_constraint, shapes, start_shape_id)
                schema_graph.add_node(node_constraint_node, label=node_constraint_node)
                schema_graph.add_edge(predicate_node, node_constraint_node,
                                      label=f"{predicate_node} {node_constraint_node}")
                # cardinality node
                cardinality_node = get_cardinality_node_label(triple_constraint)
                schema_graph.add_node(cardinality_node, label=cardinality_node)
                schema_graph.add_edge(node_constraint_node, cardinality_node,
                                      label=f"{node_constraint_node} {cardinality_node}")

            return start_shape_id, schema_graph
        else:
            logger.warning(f"Failed to find expressions in {start_shape['expression']}")
            return start_shape_id, schema_graph
    else:
        logger.warning(f"Failed to find expression in {start_shape}")
        return start_shape_id, schema_graph


class ShapeNode(object):
    """
    custom tree format
    """

    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children else []

    def __str__(self):
        paths = []
        self.collect_paths(current_path=[], paths=paths)
        return "\n".join(" -> ".join(map(str, path)) for path in paths)

    def __len__(self):
        return len(self.children)

    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.label

    def collect_paths(self, current_path, paths):
        current_path.append(self.label)
        if not self.children:  # leaf node
            paths.append(list(current_path))
        else:
            for child in self.children:
                child.collect_paths(current_path, paths)
        current_path.pop()  # backtrack

    def add_kid(self, node, before=False):
        if before:
            self.children.insert(0, node)
        else:
            self.children.append(node)
        return self

    def sort_children(self, key=None, reverse=False):
        # sort this node's children
        self.children.sort(key=key, reverse=reverse)
        # recursively sort their children
        for child in self.children:
            child.sort_children(key=key, reverse=reverse)


def transform_schema_to_tree(schema: dict, shape_id: str):
    """ transform schema to ShapeNode (i.e., tree) rooted in start shape ID

    Args:
        schema (dict): schema in ShExJ json object
        shape_id (str): start shape ID

    Returns:
        ShapeNode rooted in start shape ID
    """
    shapes = get_shapes_dict(schema)
    try:
        start_shape = shapes[shape_id]
    except KeyError:
        shape_id = schema["shapes"][0]["id"]
        start_shape = shapes[shape_id]

    start_node = ShapeNode(shape_id)

    if "expression" in start_shape:
        if "expressions" in start_shape["expression"]:
            for triple_constraint in start_shape["expression"]["expressions"]:
                # predicate node
                predicate_node = get_predicate_node_label(triple_constraint)
                if not predicate_node:  # TODO: e.g., "type": "OneOf"
                    continue
                # node_constraint node
                node_constraint_node = get_node_constraint_node_label(triple_constraint, shapes, shape_id)
                # cardinality node
                cardinality_node = get_cardinality_node_label(triple_constraint)

                # build schema tree
                constraint_node = (
                    ShapeNode(predicate_node)
                    .add_kid(ShapeNode(node_constraint_node)
                             .add_kid(ShapeNode(cardinality_node))
                             )
                )
                start_node.add_kid(constraint_node)

            return start_node
        else:
            logger.warning(f"Failed to find expressions in {start_shape['expression']}")
            return start_node
    else:
        logger.warning(f"Failed to find expression in {start_shape}")
        return start_node


def transform_shacl_to_graph(shacl_text: str, shape_id: str = None):
    """Transform SHACL schema to NetworkX DiGraph.

    Args:
        shacl_text: SHACL schema in Turtle format
        shape_id: Optional specific shape ID to transform

    Returns:
        Tuple of (root_node_id, NetworkX DiGraph)
    """
    node_shapes = extract_shacl_node_shapes(shacl_text)
    if not node_shapes:
        logger.warning("No NodeShapes found in SHACL schema")
        return None, nx.DiGraph()

    # Find the target shape
    target_shape = None
    if shape_id:
        for ns in node_shapes:
            ns_id = ns.get('@id', '')
            if shape_id in ns_id:
                target_shape = ns
                break
    if not target_shape:
        target_shape = node_shapes[0]

    root_id = target_shape.get('@id', 'UnknownShape')
    target_class = get_shacl_target_class(target_shape)
    root_label = target_class.split('/')[-1] if target_class else root_id.split('/')[-1]

    schema_graph = nx.DiGraph()
    schema_graph.add_node(root_label, label=root_label)

    # Extract property constraints
    property_shapes = extract_shacl_constraints(shacl_text, shape_id)

    for prop in property_shapes:
        # Predicate node
        predicate_label = get_shacl_predicate_node_label(prop)
        if not predicate_label:
            continue

        schema_graph.add_node(predicate_label, label=predicate_label)
        schema_graph.add_edge(root_label, predicate_label, label=f"{root_label} {predicate_label}")

        # Node constraint node
        nc_label = get_shacl_node_constraint_label(prop)
        schema_graph.add_node(nc_label, label=nc_label)
        schema_graph.add_edge(predicate_label, nc_label, label=f"{predicate_label} {nc_label}")

        # Cardinality node
        card_label = get_shacl_cardinality_label(prop)
        schema_graph.add_node(card_label, label=card_label)
        schema_graph.add_edge(nc_label, card_label, label=f"{nc_label} {card_label}")

    return root_label, schema_graph


def transform_shacl_to_tree(shacl_text: str, shape_id: str = None) -> ShapeNode:
    """Transform SHACL schema to ShapeNode tree for tree edit distance computation.

    Args:
        shacl_text: SHACL schema in Turtle format
        shape_id: Optional specific shape ID to transform

    Returns:
        ShapeNode tree rooted at the target class
    """
    node_shapes = extract_shacl_node_shapes(shacl_text)
    if not node_shapes:
        logger.warning("No NodeShapes found in SHACL schema")
        return ShapeNode("EmptyShape")

    # Find the target shape
    target_shape = None
    if shape_id:
        for ns in node_shapes:
            ns_id = ns.get('@id', '')
            target_class = get_shacl_target_class(ns)
            # Match by shape ID or target class
            if shape_id in ns_id or (target_class and shape_id in target_class):
                target_shape = ns
                break
    if not target_shape:
        target_shape = node_shapes[0]

    # Determine root label
    target_class = get_shacl_target_class(target_shape)
    if target_class:
        root_label = target_class.split('/')[-1]
    else:
        root_label = target_shape.get('@id', 'UnknownShape').split('/')[-1]

    root_node = ShapeNode(root_label)

    # Extract property constraints
    property_shapes = extract_shacl_constraints(shacl_text, shape_id)

    for prop in property_shapes:
        # Predicate node
        predicate_label = get_shacl_predicate_node_label(prop)
        if not predicate_label:
            continue

        # Node constraint node
        nc_label = get_shacl_node_constraint_label(prop)

        # Cardinality node
        card_label = get_shacl_cardinality_label(prop)

        # Build constraint subtree: predicate -> node_constraint -> cardinality
        constraint_node = (
            ShapeNode(predicate_label)
            .add_kid(ShapeNode(nc_label)
                     .add_kid(ShapeNode(card_label))
                     )
        )
        root_node.add_kid(constraint_node)

    return root_node


def plot_schema_graph(schema_graph):
    """ plot schema graph in tree layout
    """
    matplotlib.use('TkAgg')

    plt.figure(figsize=(20, 14))

    pos = nx.nx_agraph.graphviz_layout(schema_graph, prog='dot')

    nx.draw_networkx_nodes(schema_graph, pos, node_color='skyblue', node_size=1200, alpha=0.9)
    nx.draw_networkx_edges(schema_graph, pos, arrows=True, arrowstyle='->', arrowsize=10, edge_color='gray')
    nx.draw_networkx_labels(schema_graph, pos, font_size=8, font_family='sans-serif')

    plt.axis('off')
    plt.title("Shape Schema Graph", fontsize=14)
    plt.show()


def compute_graph_edit_distance(graph_1, graph_2, roots=None) -> float:
    """ compute graph edit distance by networkx.graph_edit_distance()

    Args:
        graph_1 (networkx.DiGraph): schema graph
        graph_2 (networkx.DiGraph): schema graph
        roots (2-tuple): tuple of root nodes

    Returns:
        ged (float): graph edit distance
    """
    ged = nx.graph_edit_distance(
        G1=graph_1, G2=graph_2,
        node_subst_cost=node_match, edge_subst_cost=edge_match,
        roots=roots, timeout=60
    )

    return ged


def compute_tree_edit_distance(tree_1, tree_2) -> float:
    """ compute tree edit distance by Zhang-Shasha algorithm

    Args:
        tree_1 (ShapeNode): schema tree
        tree_2 (ShapeNode): schema tree

    Returns:
        ted (float): tree edit distance
    """
    # sort nodes due to the nature of ordered labeled trees
    tree_1.sort_children(key=lambda node: (node.label not in [c.label for c in tree_2.children], node.label))
    tree_2.sort_children(key=lambda node: (node.label not in [c.label for c in tree_1.children], node.label))

    ted = simple_distance(tree_1, tree_2)

    return ted


def evaluate_ted(
        dataset: str,
        syntax: str,
        class_urls: list[str],
        class_labels: list[str],
        ground_truth_dir: str | Path,
        predicted_dir: str | Path
) -> float:
    """Evaluate schemas based on tree edit distance similarity metrics.

    Supports both ShEx and SHACL syntax.

    Args:
        dataset: Name of dataset (wes, yagos, dbpedia)
        syntax: Schema syntax ("ShEx" or "SHACL")
        class_urls: List of class URLs to evaluate
        class_labels: List of class labels
        ground_truth_dir: Path to ground truth schema directory
        predicted_dir: Path to predicted schema directory

    Returns:
        Average tree edit distance
    """
    teds, normalized_teds = list(), list()

    # Determine file extension based on syntax
    file_ext = ".shex" if syntax == "ShEx" else ".ttl"

    for class_url, class_label in zip(class_urls[:], class_labels[:]):
        class_id = class_url.split("/")[-1]
        if dataset == "wes":
            shape_id = "".join([word.capitalize() for word in class_label.split()])
        else:
            shape_id = class_label
        logger.info(f"Evaluating shape '{shape_id}' in class '{class_id}' (syntax: {syntax})")

        # Ground truth path
        true_path = os.path.join(ground_truth_dir, f"{class_id}{file_ext}")
        if not os.path.exists(true_path):
            # Try alternative naming for SHACL (e.g., ShapeNameShapeTXT2KG_clean.ttl)
            if syntax == "SHACL":
                alt_patterns = [
                    f"{shape_id}ShapeTXT2KG_clean.ttl",
                    f"{shape_id}Shape.ttl",
                    f"{shape_id}.ttl",
                    f"{class_id}Shape.ttl"
                ]
                for pattern in alt_patterns:
                    alt_path = os.path.join(ground_truth_dir, pattern)
                    if os.path.exists(alt_path):
                        true_path = alt_path
                        break
        if not os.path.exists(true_path):
            logger.warning(f"Ground truth file not found for '{class_id}'")
            continue

        # Predicted path
        pred_path = os.path.join(predicted_dir, f"{class_id}{file_ext}")
        if not os.path.exists(pred_path):
            # Try alternative naming for SHACL
            if syntax == "SHACL":
                alt_patterns = [
                    f"{shape_id}.ttl",
                    f"{shape_id}Shape.ttl",
                    f"{class_id}.ttl"
                ]
                for pattern in alt_patterns:
                    alt_path = os.path.join(predicted_dir, pattern)
                    if os.path.exists(alt_path):
                        pred_path = alt_path
                        break
        if not os.path.exists(pred_path):
            logger.warning(f"Predicted file '{pred_path}' does not exist!")
            continue

        # Transform schemas to trees based on syntax
        if syntax == "ShEx":
            # ShEx processing
            true_shexc_text = Path(true_path).read_text()
            true_shexj_text, _, _, _ = shexc_to_shexj(true_shexc_text)
            if not true_shexj_text:
                logger.warning(f"Failed to parse ground truth ShEx for '{class_id}'")
                continue
            true_shexj_json = json.loads(true_shexj_text)
            true_schema_tree = transform_schema_to_tree(schema=true_shexj_json, shape_id=shape_id)

            pred_shexc_text = Path(pred_path).read_text()
            pred_shexj_text, _, _, _ = shexc_to_shexj(pred_shexc_text)
            if not pred_shexj_text:
                logger.warning(f"Failed to parse predicted ShEx for '{class_id}'")
                continue
            pred_shexj_json = json.loads(pred_shexj_text)
            pred_schema_tree = transform_schema_to_tree(schema=pred_shexj_json, shape_id=shape_id)

        else:
            # SHACL processing
            true_shacl_text = Path(true_path).read_text()
            true_schema_tree = transform_shacl_to_tree(shacl_text=true_shacl_text, shape_id=shape_id)

            pred_shacl_text = Path(pred_path).read_text()
            pred_schema_tree = transform_shacl_to_tree(shacl_text=pred_shacl_text, shape_id=shape_id)

        # Compute tree edit distance
        ted = compute_tree_edit_distance(true_schema_tree, pred_schema_tree)

        # Normalize by ground truth tree size (each constraint has 3 nodes: predicate, node_constraint, cardinality)
        tree_size = len(true_schema_tree) if len(true_schema_tree) > 0 else 1
        normalized_ted = ted / (3 * tree_size)

        logger.info(
            f"Class: {class_id} | TED: {ted} | Ground Truth Tree Size: {tree_size} | Normalized TED: {normalized_ted:.3f}"
        )
        teds.append(ted)
        normalized_teds.append(normalized_ted)

    if not teds:
        logger.warning("No schemas could be evaluated")
        return 0.0

    avg_ted = sum(teds) / len(teds)
    avg_normalized_ted = sum(normalized_teds) / len(normalized_teds)
    logger.info(f"Average TED (over {len(teds)} schemas): {avg_ted:.3f}")
    logger.info(f"Normalized Average TED (over {len(normalized_teds)} schemas): {avg_normalized_ted:.3f}")

    return avg_ted


def evaluate_shacl_ted(
        class_urls: list[str],
        class_labels: list[str],
        ground_truth_dir: str | Path,
        predicted_dir: str | Path,
        dataset: str = "dbpedia"
) -> tuple[float, float]:
    """Convenience function to evaluate SHACL schemas using tree edit distance.

    Args:
        class_urls: List of class URLs to evaluate
        class_labels: List of class labels
        ground_truth_dir: Path to ground truth SHACL directory
        predicted_dir: Path to predicted SHACL directory
        dataset: Dataset name (default: dbpedia)

    Returns:
        Tuple of (average TED, average normalized TED)
    """
    teds, normalized_teds = [], []

    for class_url, class_label in zip(class_urls, class_labels):
        class_id = class_url.split("/")[-1]
        shape_id = class_label

        logger.info(f"Evaluating SHACL shape '{shape_id}' for class '{class_id}'")

        # Find ground truth file
        true_path = None
        possible_true_paths = [
            os.path.join(ground_truth_dir, f"{shape_id}ShapeTXT2KG_clean.ttl"),
            os.path.join(ground_truth_dir, f"{shape_id}Shape.ttl"),
            os.path.join(ground_truth_dir, f"{shape_id}.ttl"),
            os.path.join(ground_truth_dir, f"{class_id}.ttl"),
        ]
        for p in possible_true_paths:
            if os.path.exists(p):
                true_path = p
                break

        if not true_path:
            logger.warning(f"Ground truth SHACL not found for '{class_id}'")
            continue

        # Find predicted file
        pred_path = None
        possible_pred_paths = [
            os.path.join(predicted_dir, f"{shape_id}.ttl"),
            os.path.join(predicted_dir, f"{class_id}.ttl"),
            os.path.join(predicted_dir, f"{shape_id}Shape.ttl"),
        ]
        for p in possible_pred_paths:
            if os.path.exists(p):
                pred_path = p
                break

        if not pred_path:
            logger.warning(f"Predicted SHACL not found for '{class_id}'")
            continue

        # Transform to trees
        true_shacl_text = Path(true_path).read_text()
        true_tree = transform_shacl_to_tree(true_shacl_text, shape_id)

        pred_shacl_text = Path(pred_path).read_text()
        pred_tree = transform_shacl_to_tree(pred_shacl_text, shape_id)

        # Compute TED
        ted = compute_tree_edit_distance(true_tree, pred_tree)
        tree_size = len(true_tree) if len(true_tree) > 0 else 1
        normalized_ted = ted / (3 * tree_size)

        logger.info(f"Class: {class_id} | TED: {ted} | Normalized: {normalized_ted:.3f}")
        teds.append(ted)
        normalized_teds.append(normalized_ted)

    if not teds:
        logger.warning("No SHACL schemas could be evaluated")
        return 0.0, 0.0

    avg_ted = sum(teds) / len(teds)
    avg_normalized_ted = sum(normalized_teds) / len(normalized_teds)

    logger.info(f"SHACL Average TED: {avg_ted:.3f}")
    logger.info(f"SHACL Average Normalized TED: {avg_normalized_ted:.3f}")

    return avg_ted, avg_normalized_ted
