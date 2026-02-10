import json
import re
import sys

from loguru import logger
from pathlib import Path
from rdflib import Graph
from shapespresso.parser import shexj_to_shexc
from shapespresso.syntax import Cardinality, NodeConstraint,NodeConstraintSHACL
from shapespresso.pipeline import (
    query_property_list,
    query_datatype,
    construct_prompt,
    construct_cardinality_prompt,
    construct_node_constraint_prompt,
    construct_perfect_input_prompt,
    utils as fct
)
def agentic_generation_workflow(
        model,
        class_uri: str,
        class_label: str,
        instance_of_uri: str,
        syntax: str,
        dataset: str,
        endpoint_url: str,
        mode: str,
        output_dir: str | Path,
        few_shot: bool,
        few_shot_example_path: str = None,
        num_instances: int = 5,
        sort_by: str = 'entity_id',
        graph_info_path: str = None,
        load_prompt_path: str = None,
):
    """
    Run ShEx generation workflow in either local or triples mode

    Args:
        model: model used for ShEx generation
        class_uri (str): URI of the class
        class_label (str): label of the class
        instance_of_uri (str): property used to represent 'instance of'
        dataset (str): name of the dataset
        endpoint_url (str): endpoint URL
        mode (str): mode of prompt engineering, either 'local' or 'triples'
        output_dir (str): directory path where outputs (e.g., logs, results) will be saved
        few_shot (bool): whether to include few-shot examples in the prompt
        few_shot_example_path (str): path to the few-shot example file
        num_instances (int): number of instances to return, default is 5
        sort_by (str): sort criteria for instance selection, default is 'entity_id'
        graph_info_path (str): path to the graph information file
        load_prompt_path (str): path to load previously saved prompts

    Returns:

    """
    logger.info(
        f"Generate ShEx with arguments: "
        f"model={model.model_name}, "
        f"class_url={class_uri}, "
        f"class_label={class_label}, "
        f"instance_of_uri={instance_of_uri}, "
        f"dataset={dataset}, "
        f"mode={mode}, "
        f"endpoint_url={endpoint_url}, "
        f"output_dir={output_dir}, "
        f"few_shot={few_shot}, "
        f"few_shot_example_path={few_shot_example_path}, "
        f"num_instances={num_instances}, "
        f"sort_by={sort_by}, "
        f"graph_info_path={graph_info_path}, "
        f"load_prompt_path={load_prompt_path}"
    )

    prompt = construct_perfect_input_prompt(
        class_uri=class_uri,
        class_label=class_label,
        instance_of_uri=instance_of_uri,
        dataset=dataset,
        syntax=syntax,
        endpoint_url=endpoint_url,
        mode=mode,
        few_shot=few_shot,
        few_shot_example_path=few_shot_example_path,
        graph_info_path=graph_info_path,
        num_instances=num_instances,
        sort_by=sort_by,
        load_prompt_path=load_prompt_path,
    )

    response = model.model_response(prompt)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / f"{class_uri.split('/')[-1]}.txt"
    response_path.write_text(response, encoding='utf-8')



def local_generation_workflow(
        model,
        class_uri: str,
        class_label: str,
        instance_of_uri: str,
        dataset: str,
        endpoint_url: str,
        syntax:str,
        mode: str,
        output_dir: str | Path,
        few_shot: bool,
        few_shot_example_path: str = None,
        num_instances: int = 5,
        sort_by: str = 'entity_id',
        graph_info_path: str = None,
        load_prompt_path: str = None,
):
    """
    Run ShEx generation workflow in either local or triples mode

    Args:
        model: model used for ShEx generation
        class_uri (str): URI of the class
        class_label (str): label of the class
        instance_of_uri (str): property used to represent 'instance of'
        dataset (str): name of the dataset
        endpoint_url (str): endpoint URL
        mode (str): mode of prompt engineering, either 'local' or 'triples'
        output_dir (str): directory path where outputs (e.g., logs, results) will be saved
        few_shot (bool): whether to include few-shot examples in the prompt
        few_shot_example_path (str): path to the few-shot example file
        num_instances (int): number of instances to return, default is 5
        sort_by (str): sort criteria for instance selection, default is 'entity_id'
        graph_info_path (str): path to the graph information file
        load_prompt_path (str): path to load previously saved prompts

    Returns:

    """
    logger.info(
        f"Generate shape with arguments: "
        f"model={model.model_name}, "
        f"class_url={class_uri}, "
        f"class_label={class_label}, "
        f"instance_of_uri={instance_of_uri}, "
        f"syntax={syntax}, "
        f"dataset={dataset}, "
        f"mode={mode}, "
        f"endpoint_url={endpoint_url}, "
        f"output_dir={output_dir}, "
        f"few_shot={few_shot}, "
        f"few_shot_example_path={few_shot_example_path}, "
        f"num_instances={num_instances}, "
        f"sort_by={sort_by}, "
        f"graph_info_path={graph_info_path}, "
        f"load_prompt_path={load_prompt_path}"
    )

    prompt = construct_prompt(
        class_uri=class_uri,
        class_label=class_label,
        instance_of_uri=instance_of_uri,
        dataset=dataset,
        syntax=syntax,
        endpoint_url=endpoint_url,
        mode=mode,
        few_shot=few_shot,
        few_shot_example_path=few_shot_example_path,
        graph_info_path=graph_info_path,
        num_instances=num_instances,
        sort_by=sort_by,
        load_prompt_path=load_prompt_path,
    )

    response = model.model_response(prompt)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / f"{class_uri.split('/')[-1]}.txt"
    response_path.write_text(response, encoding='utf-8')

    if(syntax=="SHACL"):
        try:
            shacl_text = re.findall(r"```turtle(.*?)```", response, re.DOTALL)
            if not shacl_text:
                shacl_text = re.findall(r"```(.*?)```", response, re.DOTALL)
            shacl_text = shacl_text[0].strip()
        except IndexError:
            shacl_text = response

        output_path = Path(output_dir) / f"{class_uri.split('/')[-1]}.ttl"
        logger.info(f"Writing SHACL schema to {output_path}")
        output_path.write_text(shacl_text, encoding='utf-8')
    else:
        try:
            shexc_text = re.findall(r"```shex(.*?)```", response, re.DOTALL)
            if not shexc_text:
                shexc_text = re.findall(r"```(.*?)```", response, re.DOTALL)
            shexc_text = shexc_text[0].strip()
        except IndexError:
            shexc_text = response

        output_path = Path(output_dir) / f"{class_uri.split('/')[-1]}.shex"

        logger.info(f"Writing ShEx schema to {output_path}")
        output_path.write_text(shexc_text, encoding='utf-8')


def global_generation_workflow(
        model,
        class_uri: str,
        class_label: str,
        syntax: str,
        instance_of_uri: str,
        dataset: str,
        endpoint_url: str,
        output_dir: str | Path,
        few_shot: bool = True,
        few_shot_example_path: str = None,
        property_list_path: str = None,
        num_instances: int = 5,
        num_class_distribution: int = 3,
        threshold: int = 5,
        sort_by: str = 'entity_id',
        graph_info_path: str = None,
):
    """
    Run ShEx generation workflow in global mode

    Args:
        model: model used for ShEx generation
        class_uri (str): URI of the class
        class_label (str): label of the class
        instance_of_uri (str): property used to represent 'instance of'
        dataset (str): name of the dataset
        endpoint_url (str): endpoint URL
        output_dir (str): directory path where outputs (e.g., logs, results) will be saved
        few_shot (bool): whether to include few-shot examples in the prompt
        few_shot_example_path (str): path to the few-shot example file
        property_list_path (str): path to the property list file
        num_instances (int): number of instances to return, default is 5
        num_class_distribution (int): number of classes to return in object class distribution, default is 3
        threshold (int): threshold for filtering common properties, default is 5
        sort_by (str): sort criteria for instance selection, default is 'entity_id'
        graph_info_path (str): path to the graph information file

    Returns:

    """
    logger.info(
        f"Generate shape with arguments: "
        f"model={model.model_name}, "
        f"class_url={class_uri}, "
        f"class_label={class_label}, "
        f"syntax={syntax}, "
        f"instance_of_uri={instance_of_uri}, "
        f"dataset={dataset}, "
        f"endpoint_url={endpoint_url}, "
        f"output_dir={output_dir}, "
        f"few_shot={few_shot}, "
        f"few_shot_example_path={few_shot_example_path}, "
        f"property_list_path={property_list_path}, "
        f"num_instances={num_instances}, "
        f"num_class_distribution={num_class_distribution}, "
        f"threshold={threshold}, "
        f"sort_by={sort_by}, "
        f"graph_info_path={graph_info_path}"
    )

    # property list
    if property_list_path:
        predicates = json.loads(Path(property_list_path).read_text())[class_uri]
        if dataset == "wes":
            predicates = [item for item in predicates if item["count"] >= 5]
    else:
        predicates = query_property_list(
            class_uri=class_uri,
            dataset=dataset,
            endpoint_url=endpoint_url,
            instance_of_uri=instance_of_uri,
            threshold=threshold,
        )
    predicate_uris = [item["predicate"] for item in predicates]
    logger.info(f"{len(predicate_uris)} predicates retrieved for class {class_uri} ({class_label})")

    triple_constraints, value_shapes = [], []

    for predicate_uri in predicate_uris:
        # predict cardinality
        prompt = construct_cardinality_prompt(
            class_uri=class_uri,
            class_label=class_label,
            predicate_uri=predicate_uri,
            syntax=syntax,
            dataset=dataset,
            instance_of_uri=instance_of_uri,
            endpoint_url=endpoint_url,
            few_shot=few_shot,
            few_shot_example_path=few_shot_example_path,
            num_instances=num_instances,
            graph_info_path=graph_info_path,
        )
        response = model.structured_response(
            prompt=prompt,
            response_model=Cardinality,
        )
        min_value, max_value = response.min, response.max
        logger.info(f"Predicted cardinality for ({class_uri} {predicate_uri}): min={min_value}, max={max_value}")

        # generate node constraint
        if max_value == 0:
            continue

        datatype = query_datatype(
            class_uri=class_uri,
            predicate_uri=predicate_uri,
            instance_of_uri=instance_of_uri,
            endpoint_url=endpoint_url
        )
        start_shape_id = "".join(
            [word.capitalize() for word in class_label.split()]) if dataset == "wes" else class_label
        if syntax == "SHACL":
            current_id=fct.prefix_replace(str(class_uri))+"_"+fct.prefix_replace(str(predicate_uri))
            if datatype != "IRI":
                    if(max_value!=-1):
                        triple_constraint = {
                            "@id": "_:"+current_id,
                            "sh:path": predicate_uri,
                            "sh:datatype": datatype,
                            "sh:minCount": min_value,
                            "sh:maxCount": max_value
                        }
                    else:
                        triple_constraint = {
                            "@id": "_:"+current_id,
                            "sh:path": predicate_uri,
                            "sh:datatype": datatype,
                            "sh:minCount": min_value
                        }
                    triple_constraints.append(triple_constraint)

            else:
                # Generate node constraint for IRI values using LLM
                prompt = construct_node_constraint_prompt(
                    class_uri=class_uri,
                    class_label=class_label,
                    predicate_uri=predicate_uri,
                    dataset=dataset,
                    syntax=syntax,
                    instance_of_uri=instance_of_uri,
                    endpoint_url=endpoint_url,
                    few_shot=few_shot,
                    few_shot_example_path=few_shot_example_path,
                    num_instances=num_instances,
                    num_class_distribution=num_class_distribution,
                    graph_info_path=graph_info_path,
                )
                response = model.structured_response(prompt=prompt, response_model=NodeConstraintSHACL)

                # Build base constraint with cardinality
                triple_constraint = {
                    "@id": "_:" + current_id,
                    "sh:path": predicate_uri,
                    "sh:minCount": min_value,
                }
                if max_value != -1:
                    triple_constraint["sh:maxCount"] = max_value

                # Determine node constraint type: sh:or > sh:class > sh:IRI fallback
                if response.sh_or is not None and len(response.sh_or) > 0:
                    or_items = []
                    for item in response.sh_or:
                        if isinstance(item, dict):
                            cls = item.get("sh:class") or item.get("sh_class")
                            if cls:
                                or_items.append({"sh:class": cls})
                        elif isinstance(item, str):
                            or_items.append({"sh:class": item})
                    if or_items:
                        triple_constraint["sh:or"] = or_items
                    else:
                        triple_constraint["sh:datatype"] = "sh:IRI"
                elif response.sh_class is not None:
                    triple_constraint["sh:class"] = response.sh_class
                else:
                    triple_constraint["sh:datatype"] = "sh:IRI"

                triple_constraints.append(triple_constraint)

        else:
            if datatype != "IRI":
                    triple_constraint = {
                        "type": "TripleConstraint",
                        "predicate": predicate_uri,
                        "valueExpr": {
                            "type": "NodeConstraint",
                            "datatype": datatype
                        },
                        "min": min_value,
                        "max": max_value
                    }
                    triple_constraints.append(triple_constraint)
            else:
                prompt = construct_node_constraint_prompt(
                    class_uri=class_uri,
                    class_label=class_label,
                    predicate_uri=predicate_uri,
                    dataset=dataset,
                    syntax= syntax,
                    instance_of_uri=instance_of_uri,
                    endpoint_url=endpoint_url,
                    few_shot=few_shot,
                    few_shot_example_path=few_shot_example_path,
                    num_instances=num_instances,
                    num_class_distribution=num_class_distribution,
                    graph_info_path=graph_info_path,
                )
                response = model.structured_response(
                    prompt=prompt,
                    response_model=NodeConstraint

                )

                if response.type == "value_shape":
                    try:
                        value_shape_name = response.name.replace(" ", "")
                    except AttributeError:
                        value_shape_name = predicate_uri.split("/")[-1]
                    triple_constraint = {
                        "type": "TripleConstraint",
                        "predicate": predicate_uri,
                        "valueExpr": value_shape_name,
                        "min": min_value,
                        "max": max_value
                    }
                    value_shape = {
                        "type": "Shape",
                        "id": value_shape_name,
                        "extra": [instance_of_uri],
                        "expression": {
                            "type": "TripleConstraint",
                            "predicate": instance_of_uri,
                            "valueExpr": {
                                "type": "NodeConstraint",
                                "values": response.values
                            }
                        }
                    }
                    triple_constraints.append(triple_constraint)
                    value_shapes.append(value_shape)

                elif response.type == "values_constraint":
                    triple_constraint = {
                        "type": "TripleConstraint",
                        "predicate": predicate_uri,
                        "valueExpr": {
                            "type": "NodeConstraint",
                            "values": response.values
                        },
                        "min": min_value,
                        "max": max_value
                    }
                    triple_constraints.append(triple_constraint)

                else:
                    triple_constraint = {
                        "type": "TripleConstraint",
                        "predicate": predicate_uri,
                        "valueExpr": {
                            "type": "NodeConstraint",
                            "nodeKind": "iri"
                        },
                        "min": min_value,
                        "max": max_value
                    }
                    triple_constraints.append(triple_constraint)

    start_shape_id = "".join(
        [word.capitalize() for word in class_label.split()]) if dataset == "wes" else class_label
    if syntax=="SHACL":
        jsonld_shape = {
            "@context": {
                "sh": "http://www.w3.org/ns/shacl#",
                "dbo": "http://dbpedia.org/ontology/",
                "dbr": "http://dbpedia.org/resource/",
                "foaf": "http://xmlns.com/foaf/0.1/",
                "geo": "http://www.opengis.net/ont/geosparql#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "schema": "http://schema.org/",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
                "yago": "http://yago-knowledge.org/resource/",
                "wd": "http://www.wikidata.org/entity/",
                "wdt": "http://www.wikidata.org/prop/direct/",
                "dcat": "http://www.w3.org/ns/dcat#",
                "sh:targetClass": {"@type": "@id"},
                "sh:path": {"@type": "@id"},
                "sh:class": {"@type": "@id"},
                "sh:datatype": {"@type": "@id"},
                "sh:nodeKind": {"@type": "@id"},
                "sh:or": {"@container": "@list"},
            },
            "@id": start_shape_id,
            "@type": "sh:NodeShape",
            "sh:targetClass": class_uri,
            "sh:property": triple_constraints
        }
        jsonld_txt=json.dumps(jsonld_shape)
        g = Graph()
        g.parse(data=jsonld_txt, format='json-ld')

        output_path1 = Path(output_dir) / f"{class_uri.split('/')[-1]}.json"
        logger.info(f"Writing Json-ld Shape to {output_path1}")
        with open(output_path1, 'w', encoding='utf-8') as f:
            json.dump(jsonld_shape, f)

        output_path2 = Path(output_dir) / f"{class_uri.split('/')[-1]}.ttl"
        logger.info(f"Writing Turtle Shape to {output_path2}")
        g.serialize(destination=output_path2,format='turtle')

    else:
        # format ShEx
        # TODO: include more advanced ShEx syntax features
        start_shape = {
            "type": "Shape",
            "id": start_shape_id,
            "extra": [instance_of_uri],
            "expression": {
                "type": "EachOf",
                "expressions": triple_constraints
            }
        }
        shexj_json = {
            "type": "Schema",
            "start": start_shape_id,
            "shapes": [start_shape, *value_shapes],
        }
        shexj_text = json.dumps(shexj_json)
        shexc_text = shexj_to_shexc(
            shexj_text=shexj_text,
        )

        output_path = Path(output_dir) / f"{class_uri.split('/')[-1]}.shex"
        logger.info(f"Writing ShEx schema to {output_path}")
        output_path.write_text(shexc_text, encoding='utf-8')
