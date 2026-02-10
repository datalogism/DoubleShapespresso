import re
import json
from pathlib import Path
prefix_map = json.loads(Path("resources/prefix_map.json").read_text())
def prefix_replace(data):
    #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    str_data=str(data)
    if("<" in str_data):
        for url in prefix_map.keys():
            prefix=prefix_map[url]
            str_data = str_data.replace("<"+url,prefix+":")
        str_data = str_data.replace(">","")
    for url in prefix_map.keys():
        prefix=prefix_map[url]
        str_data = str_data.replace(url,prefix+":")
    return str_data
def camel_case_split(name):
    last_part=name.split("/")[-1]
    split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', last_part)).split()
    return split

def extractPredNL(predicate):
    if ("#" in predicate):
        prop_name = predicate.split("#")[-1]
    else:
        prop_name = predicate.split("/")[-1]
    return camel_case_split(prop_name)


def shortPred(predicate):
    if ("#" in predicate):
        prop_name = predicate.split("#")[-1]
    else:
        prop_name = predicate.split("/")[-1]
    return prop_name

def unflatten_toml(d):
    out = {}
    for key, value in d.items():
        parts = key.split('.')
        current = out
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return out

