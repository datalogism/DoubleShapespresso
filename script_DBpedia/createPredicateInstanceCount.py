#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 18:00:14 2025

@author: cringwal
"""


from os import listdir
from os.path import isfile, join
from SPARQLWrapper import JSON, SPARQLWrapper
import time
import csv
from rdflib import Graph, URIRef, Literal, BNode,Namespace
from rdflib.namespace import RDF
from unidecode import unidecode
import random
import urllib
import urllib.parse
import re
import json


sparql_dbpedia="http://dbpedia.org/sparql"
mypath="/user/cringwal/home/Desktop/KCLvisit/testConvertDBpedia"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    
def getShapeType(shacl_g):
    get_types = """
        SELECT DISTINCT ?target_class
        WHERE {
            ?a sh:targetClass ?target_class
        }"""
    qres = shacl_g.query(get_types)
    return [ str(row[0])for row in qres][0]
def getShapePropList(shacl_g):
    get_prop = """
    SELECT DISTINCT ?target_prop 
    WHERE {
        ?a sh:path ?target_prop;
    }"""
    qres = shacl_g.query(get_prop)
    return [ str(row[0])for row in qres]

def getNbProp_ClassDB(idEnt,idProp):
    query='''select (COUNT(distinct ?a) as ?nb) where {?a a <'''+str(idEnt)+'''>. ?a <'''+str(idProp)+'''> ?o .} '''
    sparql = SPARQLWrapper(sparql_dbpedia)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    if len( qres["results"]["bindings"])>0:
        return qres["results"]["bindings"][0]["nb"]["value"] 
    else:
        return 0
    
field = ["class_uri", "class_label","dataset","count"]
result={}
for shape_file in onlyfiles:
    
    if("data_stat" not in shape_file):
        shape = Graph()
       
        shape.parse(mypath+"/"+shape_file)
        idClass=getShapeType(shape)
        result[idClass]=[]
        prop_list=getShapePropList(shape)
        for idProp in prop_list:
            nb=getNbProp_ClassDB(idClass,idProp)
            result[idClass].append({"subject":idProp,"count":str(nb)})
import json
with open('/user/cringwal/home/Desktop/KCLvisit/dbpedia_predicate_count_instances.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)