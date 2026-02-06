#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 17:06:13 2025

@author: cringwal
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 17:29:43 2025

@author: cringwal
"""
import ast
from os import listdir
from os.path import isfile, join
from SPARQLWrapper import JSON, SPARQLWrapper
import time
import csv
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import RDF
from unidecode import unidecode
import random
import urllib
import urllib.parse
import re
import json
import tomlkit

import json
import pandas as pd

import json

prefix_map = None
with open("/user/cringwal/home/Desktop/KCLvisit/shapespresso-main/resources/prefix_map.json") as f:
    prefix_map = json.load(f)

sparql_dbpedia = "http://localhost:8080/sparql"


def camel_case_split(name):
    last_part = name.split("/")[-1]
    split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', last_part)).split()
    return split


def extractPredNL(predicate):
    if ("#" in predicate):
        prop_name = predicate.split("#")[-1]
    else:
        prop_name = predicate.split("/")[-1]
    return prop_name


def getCardStat(idEnt, idProp):
    query = "PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX dcat: <http://www.w3.org/ns/dcat#> SELECT ?card (COUNT( ?card) as ?nb) WHERE {  SELECT ?s (COUNT(DISTINCT ?o) as ?card)  WHERE { ?s a <" + idEnt + ">. ?s <" + idProp + "> ?o.} GROUP BY ?s } ORDER BY DESC(?nb) "
    sparql = SPARQLWrapper(sparql_dbpedia)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    return {row['card']["value"]: row["nb"]["value"] for row in qres["results"]["bindings"]}


def prefix_replace(data):
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    str_data = str(data)
    if ("<" in str_data):
        for url in prefix_map.keys():
            prefix = prefix_map[url]
            str_data = str_data.replace("<" + url, prefix + ":")
        str_data = str_data.replace(">", "")
    for url in prefix_map.keys():
        prefix = prefix_map[url]
        str_data = str_data.replace(url, prefix + ":")
    return str_data


def getShapeType(shacl_g):
    get_types = """
        SELECT DISTINCT ?target_class
        WHERE {
            ?a sh:targetClass ?target_class
        }"""
    qres = shacl_g.query(get_types)
    return [str(row[0]) for row in qres][0]


def getNbClassDB(idEnt):
    query = '''select (COUNT(distinct ?s) as ?nb) where {?s a <''' + str(idEnt) + '''>} '''
    sparql = SPARQLWrapper(sparql_dbpedia)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    return qres["results"]["bindings"][0]["nb"]["value"]


def getRevelantPropDB(idEnt):
    query = '''select ?p (COUNT(distinct ?a) as ?nb) where {?a a <''' + str(
        idEnt) + '''>. ?a ?p ?o FILTER (!regex(?p, "http://dbpedia.org/property/.*"))} GROUP BY ?p ORDER BY DESC (?nb) '''
    print(query)
    sparql = SPARQLWrapper(sparql_dbpedia)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    return {row['p']["value"]: row["nb"]["value"] for row in qres["results"]["bindings"] if int(row["nb"]["value"]) > 5}


def get_bestExamplesValuesProp(idClass, idProp, n):
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?subject ?o1 WHERE {
{
SELECT ?subject  (COUNT( DISTINCT ?predicate) as ?nb)  

WHERE {
 ?subject a <""" + idClass + """>. 
 ?subject ?predicate ?object .
  FILTER (!regex(?predicate, "http://dbpedia.org/property/.*")).
  FILTER (?predicate NOT IN(<http://dbpedia.org/ontology/abstract>,
  <http://dbpedia.org/ontology/thumbnail>,
  <http://dbpedia.org/ontology/wikiPageExternalLink>,
  <http://dbpedia.org/ontology/wikiPageID>,
  <http://dbpedia.org/ontology/wikiPageLength>,
  <http://dbpedia.org/ontology/wikiPageRevisionID>,
  <http://dbpedia.org/ontology/wikiPageWikiLink>,
  <http://purl.org/dc/terms/subject>,
  <http://purl.org/linguistics/gold/hypernym>,
  <http://www.w3.org/2000/01/rdf-schema#comment>,
  <http://www.w3.org/2000/01/rdf-schema#label>,
  <http://www.w3.org/2002/07/owl#sameAs>,
  <http://www.w3.org/ns/prov#wasDerivedFrom>,
  <http://xmlns.com/foaf/0.1/depiction>,
  <http://xmlns.com/foaf/0.1/isPrimaryTopicOf>,
  <http://www.w3.org/2000/01/rdf-schema#seeAlso>,
  <http://www.w3.org/2002/07/owl#differentFrom>,
  <http://dbpedia.org/ontology/wikiPageInterLanguageLink>,
  <http://dbpedia.org/ontology/wikiPageRedirects>,
<http://schema.org/sameAs>)).                 
FILTER EXISTS {
 ?subject <""" + idProp + """> ?o.  
 }              
}
ORDER BY DESC( ?nb) LIMIT """ + str(n) + """
}. 
?subject <""" + idProp + """> ?o1.
} ORDER BY RAND()  LIMIT 10
"""
    print(query)
    sparql = SPARQLWrapper(sparql_dbpedia)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    # print(qres)
    # return qres
    return [str(row["subject"]["value"]) + " " + idProp + " " + str(row["o1"]["value"]) for row in
            qres["results"]["bindings"]]


def get_NPropertiesRealised(idClass, idProp):
    query1 = "PREFIX dbo: <http://dbpedia.org/ontology/>  select (COUNT(DISTINCT ?s) as ?nb )  where {  ?s a <" + idClass + ">. ?s <" + idProp + "> ?o } "
    print(query1)
    sparql = SPARQLWrapper(sparql_dbpedia)
    sparql.setQuery(query1)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    nb = int(qres["results"]["bindings"][0]["nb"]["value"])
    return nb


def get_DTropertiesRealised(idClass, idProp):
    query = """SELECT DISTINCT
      (datatype(?o) AS ?datatype)
    WHERE {
     ?s a <""" + idClass + """>. 
      ?s <""" + idProp + """> ?o .
      }"""
    sparql = SPARQLWrapper(sparql_dbpedia)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    print(qres)
    if (len(qres["results"]["bindings"][0].keys()) == 0):
        return "IRI"
    else:
        return [str(row["datatype"]["value"]) for row in qres["results"]["bindings"] if len(row.keys()) > 0]


def get_shexj(idProp, shape):
    query = """
    PREFIX sh: <http://www.w3.org/ns/shacl#> 
    PREFIX owl: <http://www.w3.org/2002/07/owl#> 
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
    SELECT
        ?p ?val
    WHERE {
     ?b sh:path  <""" + idProp + """>.
     ?b ?p ?val.
    }"""
    print(query)
    qres = shape.query(query)
    data = {str(row[0]): str(row[1]) for row in qres}
    prop_name = idProp.split("/")[-1]
    response = {"triple_constraint": {"predicate": idProp, "min": 0, "max": -1, "type": "TripleConstraint"}}

    if ("http://www.w3.org/ns/shacl#minCount" in data.keys()):
        response["triple_constraint"]["min"] = int(float(data["http://www.w3.org/ns/shacl#minCount"]))
    if ("http://www.w3.org/ns/shacl#maxCount" in data.keys()):
        response["triple_constraint"]["max"] = int(float(data["http://www.w3.org/ns/shacl#maxCount"]))
    if ('http://www.w3.org/ns/shacl#class' in data.keys()):
        class_name = data['http://www.w3.org/ns/shacl#class'].split("/")[-1]
        exp_ref = class_name.title() + prop_name.title()
        response["triple_constraint"]["valueExpr"] = exp_ref
        response["triple_constraint"]["value_shape"] = {}
        response["triple_constraint"]["value_shape"]["type"] = "Shape"
        response["triple_constraint"]["value_shape"]["id"] = exp_ref
        response["triple_constraint"]["value_shape"]["extra"] = ['http://www.w3.org/1999/02/22-rdf-syntax-ns#type']
        response["triple_constraint"]["value_shape"]["predicate"] = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        response["triple_constraint"]["value_shape"]["values"] = [data['http://www.w3.org/ns/shacl#class']]
    if ("http://www.w3.org/ns/shacl#datatype" in data.keys()):
        response["triple_constraint"]["valueExpr"] = {"type": "NodeConstraint",
                                                      "datatype": data["http://www.w3.org/ns/shacl#datatype"]}

    return response


def get_microJsonLD(idProp, shape, idClass):
    query = """
    PREFIX sh: <http://www.w3.org/ns/shacl#> 
    PREFIX owl: <http://www.w3.org/2002/07/owl#> 
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
    SELECT
        ?p ?val
    WHERE {
     ?b sh:path  <""" + idProp + """>.
     ?b ?p ?val.
    }"""
    print(query)
    qres = shape.query(query)
    data = {str(row[0]): str(row[1]) for row in qres}
    print(data)
    prop_name = idProp.split("/")[-1]
    # if('http://www.w3.org/ns/shacl#class' in data.keys()):
    class_name = extractPredNL(idClass)

    exp_ref = class_name + extractPredNL(prop_name)
    response = {
        "@id": "_:" + exp_ref,
        "sh:path": idProp,
        "sh:minCount": 0,
        "sh:maxCount": -1
    }

    if ("http://www.w3.org/ns/shacl#minCount" in data.keys()):
        response["sh:minCount"] = int(float(data["http://www.w3.org/ns/shacl#minCount"]))
    if ("http://www.w3.org/ns/shacl#maxCount" in data.keys()):
        response["sh:maxCount"] = int(float(data["http://www.w3.org/ns/shacl#maxCount"]))

    if ('http://www.w3.org/ns/shacl#class' in data.keys()):
        response["sh:class"] = data['http://www.w3.org/ns/shacl#class']
    elif ("http://www.w3.org/ns/shacl#datatype" in data.keys()):
        response["sh:datatype"] = data["http://www.w3.org/ns/shacl#datatype"]
    else:
        response["sh:datatype"] = "sh:IRI"
    return response


shape_file = "/user/cringwal/home/Desktop/KCLvisit/testConvertDBpedia/Scientist.ttl"

n_shot = 5

doc: tomlkit.TOMLDocument = tomlkit.document()

shape = Graph()
shape.parse(shape_file)
idClass = getShapeType(shape)

nb_class = getNbClassDB(idClass)
relevant_prop = getRevelantPropDB(idClass)
idx = 0
exluded_prop = ["http://dbpedia.org/ontology/abstract", "http://dbpedia.org/ontology/thumbnail",
                "http://dbpedia.org/ontology/wikiPageExternalLink", "http://dbpedia.org/ontology/wikiPageID",
                "http://dbpedia.org/ontology/wikiPageLength", "http://dbpedia.org/ontology/wikiPageRevisionID",
                "http://dbpedia.org/ontology/wikiPageWikiLink", "http://purl.org/dc/terms/subject",
                "http://purl.org/linguistics/gold/hypernym", "http://www.w3.org/2000/01/rdf-schema#comment",
                "http://www.w3.org/2000/01/rdf-schema#label", "http://www.w3.org/2002/07/owl#sameAs",
                "http://www.w3.org/ns/prov#wasDerivedFrom", "http://xmlns.com/foaf/0.1/depiction",
                "http://xmlns.com/foaf/0.1/isPrimaryTopicOf", "http://www.w3.org/2000/01/rdf-schema#seeAlso",
                "http://www.w3.org/2002/07/owl#differentFrom", "http://dbpedia.org/ontology/wikiPageInterLanguageLink",
                "http://dbpedia.org/ontology/wikiPageRedirects", "http://schema.org/sameAs"]
for prop in relevant_prop.keys():

    if (prop not in exluded_prop):
        doc.add("example_" + str(idx) + ".class_uri", idClass)
        doc.add("example_" + str(idx) + ".predicate_uri", prop)
        # res=get_NPropertiesRealised(idClass,prop,n_shot)
        examples_str = prefix_replace(get_bestExamplesValuesProp(idClass, prop, n_shot))
        examples_list = ast.literal_eval(examples_str)
        doc.add("example_" + str(idx) + ".triple_examples", examples_list)
        nb_prop = get_NPropertiesRealised(idClass, prop)
        ratio_prop = float(nb_prop) / float(nb_class) * 100
        freq_str = str(ratio_prop) + "% instance(s) in the class use the predicate"
        doc.add("example_" + str(idx) + ".frequency", freq_str)

        distrib_dict = getCardStat(idClass, prop)
        card_text = ""
        for card in distrib_dict.keys():
            nb_card = distrib_dict[card]
            ratio_card = float(nb_card) / float(nb_class) * 100
            if (ratio_card > 0.1):
                card_text = card_text + " " + str(
                    ratio_card) + "% instances in the class have " + card + "object(s) when using the property,\n"
        doc.add("example_" + str(idx) + ".cardinality_distribution", card_text)
        dt_realize = get_DTropertiesRealised(idClass, prop)
        doc.add("example_" + str(idx) + ".datatype_of_objects", str(dt_realize))
        constr_ = get_microJsonLD(prop, shape, idClass)
        doc.add("answer_" + str(idx) + ".shaclj", str(constr_))
        idx = idx + 1

with open("/user/cringwal/home/Desktop/KCLvisit/shapespresso-main/resources/dbpedia_global_few_shot_examples.toml",
          'w') as toml_file:
    toml_file.write(tomlkit.dumps(doc))

toml_file.close()