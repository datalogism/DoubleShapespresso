#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 15:26:59 2025

@author: cringwal
"""
import ast
import csv
import pandas as pd 
from rdflib import Graph, Literal, RDF, URIRef, BNode, XSD
from rdflib.namespace import FOAF, RDF, SH
#Text2KGDbpediaClasses=["MusicalWork"]
    
file="/user/cringwal/home/Desktop/KCLvisit/ShapeAnnotation_withRangeDomain.csv"

df = pd.read_csv(file, sep='\t', quotechar='"')
df=df.sort_values(by='class', ascending=False)
previous_class=""
sub_g=None
class_name=""
for index, row in df.iterrows():
    #print(row["prop"])
    class_name0=row["class"].split("/")[-1]
    print("================>",class_name0)
    if(previous_class!=class_name0):
        if(sub_g!=None):
             print("SAVE")
             shape_content=sub_g.serialize(format='turtle')
             path_w="/user/cringwal/home/Desktop/KCLvisit/testConvertDBpedia/"+class_name+".ttl"
             with open(path_w, mode='w') as f:
                 f.write(shape_content)
        
        class_name=row["class"].split("/")[-1]
        sub_g=Graph()
        uri_="http://shaclshapes.org/"+class_name+"Shape"
        uri_subj = URIRef(uri_)
        sub_g.add((uri_subj,RDF.type,SH.NodeShape))
        sub_g.add((uri_subj,SH.targetClass,URIRef(row["class"])))
        
    previous_class=class_name
    bnode = BNode()
    keep=str(row["to_keep"]).upper()
    if( keep!="?" and keep!="NAN"):
        if(str(int(float(keep)))=="1" ):
            #print("OK")
            sub_g.add((uri_subj,SH.property,bnode))
            sub_g.add((bnode,SH.path,URIRef(row["prop"])))
            min_card=str(row["card_min"]).upper()
            max_card=str(row["card_max"]).upper()
            
            if(str(min_card)!=str(0.0) and min_card!="NONE" and min_card!="NAN" and min_card!="?" ):
                min_c= Literal(int(float(min_card)), datatype=XSD.integer)
                sub_g.add((bnode,URIRef("http://www.w3.org/ns/shacl#minCount"),min_c))
            if(max_card!="N" and max_card!="NONE"  and max_card!="NAN" and max_card!="?" ):
                max_c= Literal(int(float(max_card)), datatype=XSD.integer)
                sub_g.add((bnode,URIRef("http://www.w3.org/ns/shacl#maxCount"),max_c))
                
            if(str(row["range"]).upper() != "NAN" or str(row["range"]).upper()!="NONE"):
                range_ = ast.literal_eval(str(row["range"]))
                if(len(range_)==1):
                    range_uri = URIRef(range_[0])
                    if("dbpedia"in range_[0]):
                        sub_g.add((bnode,URIRef("http://www.w3.org/ns/shacl#class"),range_uri))
                    else:
                        sub_g.add((bnode,SH.datatype,range_uri))

print("SAVE")
shape_content=sub_g.serialize(format='turtle')
path_w="/user/cringwal/home/Desktop/KCLvisit/testConvertDBpedia/"+class_name+".ttl"
with open(path_w, mode='w') as f:
    f.write(shape_content)
