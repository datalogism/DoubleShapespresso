#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 16:40:15 2025

@author: cringwal
"""

from tomlkit import parse,dumps
from pathlib import Path
import tomllib
import json

file="/user/cringwal/home/Desktop/KCLvisit/shapespresso-main/resources/yagos_global_few_shot_examples.toml"
content = parse(Path(file).read_text())
for i in range(len(content) // 2):
    print("=================")
    example = content[f"answer_{i}"]
    constraints=json.loads(example["shexj"])
    class_name=content[f"example_{i}"]['class_uri'].split("/")[-1]
    if("#" in  content[f"example_{i}"]['predicate_uri']):
        prop_name=content[f"example_{i}"]['predicate_uri'].split("#")[-1]    
    else:
        prop_name=content[f"example_{i}"]['predicate_uri'].split("/")[-1]    
    constraints=json.loads(example["shexj"])
    exp_ref=class_name.title()+prop_name.title()
    tempo={
      "@id":"_:"+exp_ref,
      "sh:path":content[f"example_{i}"]['predicate_uri'],
      "sh:minCount":0,
      "sh:maxCount":-1
    }
    if(len(constraints.keys())==0):
        tempo["sh:datatype"]= "sh:IRI"
    else:
        if("min" in constraints['triple_constraint'].keys()):
            tempo["sh:minCount"]=int(float(constraints['triple_constraint']["min"]))
        if("max" in constraints['triple_constraint'].keys()):
            tempo["sh:maxCount"]=int(float(constraints['triple_constraint']["max"]))
            
        if(type( constraints['triple_constraint']["valueExpr"])==str):
            tempo["sh:class"]=constraints['triple_constraint']['valueExpr']
        elif('value_shape' in constraints['triple_constraint'].keys()):
            if(len(constraints['triple_constraint']['value_shape']["values"])==1):
                tempo["sh:class"]=constraints['triple_constraint']['value_shape']["values"][0]
            else:
                tempo["sh:class"]= { "sh:or":    constraints['triple_constraint']['value_shape']["values"]   } 
        elif("valueExpr" in constraints['triple_constraint'].keys() and "datatype" in constraints['triple_constraint']["valueExpr"].keys()):
            tempo["sh:datatype"]=constraints['triple_constraint']["valueExpr"]["datatype"]
        else:
          tempo["sh:datatype"]= "sh:IRI"
          
    content[f"answer_{i}"]["shaclj"]=json.dumps(tempo)
    print("=================")
    print(tempo)


file="/user/cringwal/home/Desktop/KCLvisit/shapespresso-main/resources/yagos_global_few_shot_examples2.toml"
with open(file, 'w') as toml_file:
    toml_file.write(dumps(content))

toml_file.close()