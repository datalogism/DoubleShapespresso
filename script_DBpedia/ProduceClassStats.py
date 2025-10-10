#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 16:59:13 2025

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
def camel_case_split(name):
    last_part=name.split("/")[-1]
    split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', last_part)).split()
    return split


def getShapeProp(shacl_g):
    
    get_prop = """
    SELECT DISTINCT ?target_prop
    WHERE {
        ?a sh:path ?target_prop
    }"""
    qres = shacl_g.query(get_prop)
    return [ str(row[0]) for row in qres]

def getLabel(idEnt):
    query='''PREFIX  rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX   dbo:  <http://dbpedia.org/ontology/>
    PREFIX   bif:  <bif:>
    
    SELECT DISTINCT 
                    ?itemLabel 
     WHERE
       {  
        <'''+str(idEnt)+'''>    rdfs:label    ?itemLabel .
         FILTER (lang(?itemLabel) = 'en')
       }
   LIMIT 1'''
    sparql = SPARQLWrapper(sparql_dbpedia)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    if len( qres["results"]["bindings"])>0:
        return  [row['itemLabel']["value"] for row in qres["results"]["bindings"] ][0]
    else:
        return camel_case_split(idEnt)
def getNbClassDB(idEnt):
    query='''select (COUNT(distinct ?s) as ?nb) where {?s a <'''+str(idEnt)+'''>} '''
    sparql = SPARQLWrapper(sparql_dbpedia)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    return qres["results"]["bindings"][0]["nb"]["value"]     
def getShapeType(shacl_g):
    get_types = """
        SELECT DISTINCT ?target_class
        WHERE {
            ?a sh:targetClass ?target_class
        }"""
    qres = shacl_g.query(get_types)
    return [ str(row[0])for row in qres][0]
    
mypath="/user/cringwal/home/Desktop/KCLvisit/txt2kg_clean"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
                
with open('/user/cringwal/home/Desktop/KCLvisit/dbpedia.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"')
    field = ["class_uri", "class_label","dataset","count"]
    writer.writerow(field)    
    for shape_file in onlyfiles:
        
        if("data_stat" not in shape_file):
            shape = Graph()
           
            shape.parse(mypath+"/"+shape_file)
            idClass=getShapeType(shape)
            print("====>",idClass)
            print("STEP1")
            nb_class=getNbClassDB(idClass)
            label=getLabel(idClass)
            writer.writerow([idClass,label,"dbpedia",nb_class])    