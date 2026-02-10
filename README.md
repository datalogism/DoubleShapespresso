# DoubleShapespresso ☕️☕️

This repository contains the code, data, and instructions to extend the experiments from 
our EMNLP 2025 paper "Schema Generation for Large Knowledge Graphs Using Large Language Models"
[[arXiv](https://arxiv.org/abs/2506.04512)].

Extension to Text2KGBench micro-ontologies :
- DBpedia / SHACL 
- WD / Shex
Translation of Yago to SHACL.

## CHECKLIST

- [x] YAGO Shacl translation + validation of every shape using SHACLplayground
- [x] First draft Dbpedia Shacl annotation  based on Text2KGBench micro-ontologies
- [ ] First draft Wikidata annotation  based on Text2KGBench micro-ontologies
- [x] Developing scripts for DBpedia statistics needed for global settings
- [x] Adapting the prompts to local/triples/global settings adapted to DBpedia/SHACL
- [x] Generate prompt for Yago / SHACL 
- [ ] Generate prompt for DBpedia / SHACL (triples no finished)
- [x] Adapting generation for local / triple settings
- [ ] Adapting generation for global setting
- [ ] Adapting the evalution to SHACL

## DBpedia endpoint
The endpoint used here is [Corese](https://github.com/Wimmics/corese), a Software platform for the Semantic Web of Linked Data. 
The software could be deployed locally via a  [JAR software](https://github.com/Wimmics/corese/releases/download/release-4.5.0/corese-server-4.5.0.jar) and the persistancy of the KB is allowed by the tdbloader of Jena (available in this directory).

### Starting from scratch: Data base initialization
1- First download CORESE jar file 
2- Download the last version of [Jena](https://jena.apache.org/download/index.cgi) and locate the tdbloader script module
3- Download the datadump gathering all the interesting data: [https://databus.dbpedia.org/](https://databus.dbpedia.org/)
4- Load the data in CORESE using the **tdbloader** script as :
```
bash tdbloader --loc TDBLOADER_DIR FILES_DIR
```
4- configure the config.properties path in consequence

5- run the Corese server via :
```
java -Xmx10g -jar corese-server-4.5.0.jar -init "config.properties"
```
6- The KB endpoint is now accesible via  'http://localhost:8080/sparql'

## Shape Filtering and Rerun Options

Three optional arguments allow selective processing and idempotency:

| Argument | Description |
|----------|-------------|
| `--shape_name` | Process only the specified shape (e.g., `Airport`, `Q4220917`). Accepts with or without file extension. |
| `--shape_dir` | Directory of shape files (`.ttl`/`.shex`). Only CSV shapes with a matching file in this directory will be processed. |
| `--force` | Force rerun even if output already exists. By default, shapes with existing non-empty output are skipped. |

**Examples:**
```bash
# Process a single shape
python main.py --task prompt --dataset dbpedia --mode local --shape_name Airport ...

# Process only shapes that have ground truth in a directory
python main.py --task prompt --dataset dbpedia --mode local --shape_dir dataset/SHACL/dbpedia-v1/ ...

# Force rerun of all shapes (ignore existing outputs)
python main.py --task prompt --dataset dbpedia --mode local --force ...
```

## Prompt generation
### YAGO SHACL
#### local
```text
python main.py--task prompt --dataset yagos --output_dir output/prompts/shacl/local/yago/entity_id/5 --mode local --syntax SHACL --num_instances 5 --sort_by predicate_count --few_shot --few_shot_example_path dataset/SHACL/yago/Scientist.ttl --save_log --endpoint_url "http://localhost:1234/api/endpoint/sparql" --graph_info_path resources/yagos_predicate_count_instances.json
```

#### global
```text
python main.py--task prompt --dataset yagos --output_dir output/prompts/shacl/global/yago/entity_id/5 --mode global --syntax SHACL --num_instances 5 --sort_by entity_id --few_shot --few_shot_example_path resources/yagos_global_few_shot_examples.toml --save_log --graph_info_path resources/yagos_predicate_count_instances.json
```

#### triples
```text
python main.py--task prompt --dataset yagos --output_dir output/prompts/shacl/triples/yago/entity_id/5 --mode triples --syntax SHACL --num_instances 5 --sort_by entity_id --few_shot --few_shot_example_path dataset/SHACL/yago/Airline.ttl --save_log --endpoint_url "http://localhost:1234/api/endpoint/sparql" --graph_info_path resources/yagos_predicate_count_instances.json
```

### DBPedia SHACL
#### local
```text
python main.py--task prompt --dataset dbpedia --output_dir output/prompts/shacl/local/dbpedia/entity_id/5 --mode local --syntax SHACL --num_instances 5 --sort_by predicate_count --few_shot --few_shot_example_path dataset/SHACL/dbpedia-1/Scientist.ttl --save_log --endpoint_url "http://localhost:8080/sparql" --graph_info_path resources/dbpedia_predicate_count_instances.json
```

#### global
```text
python main.py--task prompt --dataset dbpedia --output_dir output/prompts/global/dbpedia/entity_id/5 --mode global --syntax SHACL --num_instances 1 --sort_by predicate_count --few_shot --few_shot_example_path dataset/SHACL/dbpedia-0/AirportShapeTXT2KG_clean.ttl --save_log --endpoint_url "http://localhost:8080/sparql"
```
#### triples
```text
python main.py--task prompt --dataset dbpedia --output_dir output/prompts/shacl/triples/dbpedia/entity_id/5 --mode triples --syntax SHACL --num_instances 5 --sort_by predicate_count --few_shot --few_shot_example_path dataset/SHACL/dbpedia-1/Scientist.ttl --save_log --endpoint_url "http://localhost:8080/sparql" --graph_info_path resources/dbpedia_predicate_count_instances.json
```

