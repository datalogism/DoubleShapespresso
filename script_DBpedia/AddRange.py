#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 15:26:59 2025

@author: cringwal
"""

import csv
import pandas as pd 
from rdflib import Graph, Literal, RDF, URIRef, BNode
from rdflib.namespace import FOAF, RDF, SH
#Text2KGDbpediaClasses=["MusicalWork"]

onto_g = Graph()
onto_g.parse("https://mappings.dbpedia.org/server/ontology/dbpedia.owl")
shape_content=onto_g.serialize(format='turtle')
    
file="/user/cringwal/home/Desktop/KCLvisit/Annotation2 - Feuille 1.csv"
df = pd.read_csv(file)
prop_list=list(set(df["prop"]))

prop_dict={}
for prop in prop_list:
    q = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX 	rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
    
        SELECT ?o
        WHERE {
            <"""+prop+"""> rdfs:domain ?o
        }
    """
    res=onto_g.query(q)
    domain=[str(r[0])  for r in res]
    q = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX 	rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
    
        SELECT ?o
        WHERE {
            <"""+prop+"""> rdfs:range ?o
        }
    """
    res=onto_g.query(q)
    range0=[str(r[0])  for r in res]
    
    prop_dict[prop]={"domain":domain,"range":range0,"datatype":""}
    
df["domain"]=None

for index, row in df.iterrows():
    if(row["prop"] in prop_dict.keys()):
        df.loc[index, 'range']= str(prop_dict[row["prop"]]["range"])
        df.loc[index, 'domain'] = str(prop_dict[row["prop"]]["domain"])

df.to_csv("/user/cringwal/home/Desktop/KCLvisit/ShapeAnnotation_withRangeDomain.csv", sep='\t', quotechar='"', encoding='utf-8')
    
    
    