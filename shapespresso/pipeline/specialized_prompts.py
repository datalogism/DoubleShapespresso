import json
import random
import tomllib
import sys
from loguru import logger
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from shapespresso.pipeline import (
    concat_object_values,
    query_triple_examples,
    query_property_list,
    query_property_information,
    query_instances_predicate_count
)
from shapespresso.utils import endpoint_sparql_query, prefix_substitute
from shapespresso.parser import shexc_to_shexj

from rdflib import Graph, URIRef, Literal, BNode,Namespace
from rdflib.namespace import RDF

def getShapeType(shape_file, syntax):
    if syntax == "SHACL":

        shacl_g = Graph()
        shacl_g.parse(shape_file)
        get_types = """
            SELECT DISTINCT ?target_class
            WHERE {
                ?a sh:targetClass ?target_class
            }"""
        qres = shacl_g.query(get_types)
        return [str(row[0]) for row in qres][0]
    else:
        raise NotImplementedError


def getPropertylist(shape_file, syntax):
    if syntax == "SHACL":

        shacl_g = Graph()
        shacl_g.parse(shape_file)
        get_prop = """
        SELECT DISTINCT ?target_prop ?datatype
        WHERE {
            ?a sh:path ?target_prop.
        }"""
        qres = shacl_g.query(get_prop)
        return [str(row[0]) for row in qres]

    else:
        raise NotImplementedError


def getPropertyCaard(idProp, shape_file, syntax):
    if syntax == "SHACL":

        shacl_g = Graph()
        shacl_g.parse(shape_file)
        get_prop = """
        SELECT ?min ?max
        WHERE {
            ?s ?p ?o.
            ?o sh:path <""" + idProp + """>.
            OPTIONAL { ?o sh:minCount ?min }
            OPTIONAL { ?o sh:maxCount ?max }
        }"""
        qres = shacl_g.query(get_prop)
        res = {"min": "0", "max": "N"}
        for row in qres:
            if (row[0] is not None):
                res["min"] = str(row[0])
            if (row[1] is not None):
                res["max"] = str(row[1])
        return res

    else:
        raise NotImplementedError


def getPropertyType(idProp, shape_file, syntax):
    if syntax == "SHACL":

        shacl_g = Graph()
        shacl_g.parse(shape_file)
        get_prop = """
        SELECT ?dt ?obj
        WHERE {
            ?s ?p ?o.
            ?o sh:path <""" + idProp + """>.
            OPTIONAL { ?o sh:class  ?obj }
            OPTIONAL { ?o  sh:datatype ?dt }
        }"""
        qres = shacl_g.query(get_prop)
        res = {"data_type": None, "obj_type": None}
        for row in qres:
            if (row[0] is not None):
                res["data_type"] = str(row[0])
            if (row[1] is not None):
                res["obj_type"] = str(row[1])
        return res

    else:
        raise NotImplementedError

def _extract_shex_perfect_input(shex_file_path: str) -> tuple[list[str], dict, dict]:
    """Extract property list, cardinalities, and types from a ShEx file.

    Args:
        shex_file_path: Path to the ground truth .shex file

    Returns:
        Tuple of (property_list, cardinality_dict, type_dict)
    """
    shexc_text = Path(shex_file_path).read_text(encoding="utf-8")
    shexj_text, _, _, _ = shexc_to_shexj(shexc_text)
    if not shexj_text:
        logger.warning(f"Failed to parse ShEx file: {shex_file_path}")
        return [], {}, {}

    shex_json = json.loads(shexj_text)

    # Get start shape
    if "start" in shex_json:
        shape_id = shex_json["start"]
    else:
        shape_id = shex_json["shapes"][0]["id"]

    shapes = {s["id"]: s for s in shex_json.get("shapes", [])}
    start_shape = shapes.get(shape_id, {})

    if "expression" not in start_shape:
        logger.warning(f"No expression in start shape '{shape_id}'")
        return [], {}, {}

    # Collect triple constraints
    expression = start_shape["expression"]
    if "expressions" in expression:
        constraints = expression["expressions"]
    else:
        constraints = [expression]

    property_list = []
    caard_dict = {}
    type_dict = {}

    for tc in constraints:
        if tc.get("type") != "TripleConstraint":
            continue
        predicate = tc.get("predicate", "")
        if not predicate:
            continue

        property_list.append(predicate)

        # Cardinality (ShEx defaults: min=1, max=1)
        min_val = tc.get("min", 1)
        max_val = tc.get("max", 1)
        caard_dict[predicate] = {
            "min": str(min_val),
            "max": "N" if max_val == -1 else str(max_val)
        }

        # Value expression / type
        value_expr = tc.get("valueExpr")
        if value_expr is None:
            type_dict[predicate] = {"type": "unconstrained"}
        elif isinstance(value_expr, str):
            # Shape reference
            type_dict[predicate] = {"type": "shape_reference", "shape": value_expr}
        elif isinstance(value_expr, dict):
            if "datatype" in value_expr:
                type_dict[predicate] = {"data_type": value_expr["datatype"], "obj_type": None}
            elif "nodeKind" in value_expr:
                type_dict[predicate] = {"data_type": None, "obj_type": value_expr["nodeKind"]}
            elif "values" in value_expr:
                type_dict[predicate] = {"data_type": None, "obj_type": "value_set",
                                        "values": value_expr["values"]}
            else:
                type_dict[predicate] = {"data_type": None, "obj_type": None}
        else:
            type_dict[predicate] = {"data_type": None, "obj_type": None}

    return property_list, caard_dict, type_dict


def construct_perfect_input_prompt(
        class_uri: str,
        class_label: str,
        instance_of_uri: str,
        dataset: str,
        syntax: str,
        endpoint_url: str,
        mode: str,
        few_shot: bool = True,
        few_shot_example_path: str = None,
        graph_info_path: str = None,
        information_types: list[str] = None,
        num_instances: int = 3,
        num_class_distribution: int = 3,
        threshold: int = 5,
        sort_by: str = 'predicate_count',
        answer_keys: list[str] = None,
        load_prompt_path: str = None,
        save_prompt_path: str = None
) -> list[dict] | list[list[dict]]:
    if syntax == "SHACL":
        system_content = (
            "You are a skilled knowledge engineer with deep expertise in writing SHACL (Shapes "
            "Constraint Language) constraints. Carefully analyze the provided few-shot examples to "
            "understand the end-to-end generation process. Generate precise, well-structured "
            "SHACL constraints based on given example items and their related triples."
        )
        prompt = [
            {"role": "system", "content": system_content}
        ]
        property_list = getPropertylist(few_shot_example_path, syntax)
        caard_dict = {}
        type_dict = {}
        for property in property_list:
            caard_dict[property] = getPropertyCaard(property, few_shot_example_path, syntax)
            type_dict[property] = getPropertyType(property, few_shot_example_path, syntax)

        local_prompt = {
            "role": "user",
            "content": (
                f"Based on the information, generate the SHACL shape for the class '{class_uri} ({class_label})'. "
                f"The property list to use is this one :{str(property_list)}.\n"
                f"The cardinalities of these properties are listed here :{str(caard_dict)}.\n"
                f"The types of these properties are listed here :{str(type_dict)}.\n"
            )
        }
        prompt.append(local_prompt)

    elif syntax == "ShEx":
        system_content = (
            "You are a skilled knowledge engineer with deep expertise in writing ShEx (Shape "
            "Expressions) schemas. Carefully analyze the provided information to understand the "
            "end-to-end generation process. Generate precise, well-structured ShEx schemas "
            "based on the given property constraints."
        )
        prompt = [
            {"role": "system", "content": system_content}
        ]
        property_list, caard_dict, type_dict = _extract_shex_perfect_input(few_shot_example_path)

        local_prompt = {
            "role": "user",
            "content": (
                f"Based on the information, generate the ShEx schema for the class '{class_uri} ({class_label})'. "
                f"The property list to use is this one :{str(property_list)}.\n"
                f"The cardinalities of these properties are listed here :{str(caard_dict)}.\n"
                f"The types of these properties are listed here :{str(type_dict)}.\n"
            )
        }
        prompt.append(local_prompt)

    else:
        raise NotImplementedError(f"Syntax '{syntax}' not supported in construct_perfect_input_prompt")

    if save_prompt_path:
        logger.info(f"Save prompts to {save_prompt_path}")
        Path(save_prompt_path).write_text(json.dumps(prompt, ensure_ascii=False, indent=2))

    return prompt
