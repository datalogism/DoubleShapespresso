# CLAUDE.md

## Project Overview
Describe the project in 2–5 sentences.

- Purpose: Shapespresso 
- Target users: Knowledge scientist
- Core functionality: enable the creation of shape in SHACL or Shex
- Tech stack: LLM + RDF + SHACL + Shex + SPARQL endpoint 
- Current status: still in dev

---

## Instructions for Claude

### TO DO
Globally the objective of this development phase is to extent the current shapespresso to the DBpedia endpoint using RDF data and DBpedia specific Shacl shapes
Which include:
- SHACL extension
    - Prompts (local & triples)
- Global setting with structured generation
    - Property filtering (aka cardinality prediction)
    - Node constraints generation
    - Scripts merge (merge JSON or other structured output)
- Global setting without structured generation
    - Property filtering (aka cardinality prediction)
    - Node constraints generation
    - Scripts merge (using LLMs)
- Evaluation metrics extension to SHACL shapes and graphs
    - Classification metrics
    - Similarity metrics

### Primary Goals
- Help develop, maintain, and improve this project.
- Prioritize correctness, readability, and maintainability.
- Follow project conventions and constraints listed below.

### When Writing Code
- Prefer simple, readable solutions over clever ones.
- Follow existing patterns in the repository.
- Avoid introducing new dependencies unless necessary.
- Add comments when logic is not obvious.
- Include error handling where appropriate.

---

## Coding Standards

### General
- Keep functions small and focused.
- Use clear and descriptive naming.
- Avoid duplication (DRY principle).
- Write modular and reusable components.

### Formatting
- Follow existing linting / formatting rules.
- If none exist:
  - Use consistent indentation.
  - Keep line length reasonable (~80–120 chars).
  - Use meaningful whitespace.

---

## Language-Specific Guidelines
(Add or remove sections depending on your project)

### Python
- Follow PEP 8.
- Prefer type hints.
- Use docstrings for public functions.

---

## Testing Requirements
- Write tests for new functionality.
- Do not break existing tests.
- Prefer unit tests over large integration tests unless necessary.
- Mock external services when possible.

---

## Documentation Rules
- Update README when behavior changes.
- Document public APIs.
- Add examples for complex features.

---

## Security Guidelines
- Never expose secrets or API keys.
- Validate and sanitize user input.
- Follow least-privilege principles.
- Highlight potential security risks.

---

## Performance Guidelines
- Avoid premature optimization.
- Flag obvious performance bottlenecks.
- Consider scalability when designing new features.

---

## Allowed Actions
Claude may:

✅ Refactor code  
✅ Suggest improvements  
✅ Write tests  
✅ Improve documentation  
✅ Explain code  
✅ Suggest performance optimizations  

---

## Disallowed Actions
Claude must NOT:

❌ Delete large sections of code without explanation  
❌ Introduce breaking changes silently  
❌ Add dependencies without justification  
❌ Modify environment or deployment configs without warning  

---

## Preferred Response Style
- Be concise but clear.
- Explain reasoning for significant changes.
- Provide examples when helpful.
- Ask clarifying questions if requirements are unclear.

---

## Project Context
Schemas play a vital role in ensuring data qual-
ity and supporting usability in the Semantic
Web and natural language processing. Traditionally, their creation demands substantial in-
volvement from knowledge engineers and domain experts. Leveraging the impressive ca-
pabilities of large language models (LLMs) in tasks like ontology engineering, we explore
schema generation using LLMs. To bridge the resource gap, we introduce two datasets:
YAGO Schema and Wikidata EntitySchema, along with novel evaluation metrics. The LLM-
based pipelines utilize local and global information from knowledge graphs (KGs) to generate
schemas in Shape Expressions (ShEx/SHACL). Experiments demonstrate LLMs’ strong potential in
producing high-quality ShEx schemas, paving the way for scalable, automated schema gener-
ation for large KGs. Furthermore, our benchmark introduces a new challenge for structured
generation, pushing the limits of LLMs on syntactically rich formalisms.

### Architecture Notes
The system use a Knowledge Base endpoint (SPARQL)
The system use LLM
The system is designed to generate SHACL/Shex Shapes
The system is designed to evaluate the produced shapes


### General description of the project
```text
conda create -n shapespresso python=3.11
conda activate shapespresso
pip install -r requirements.txt
```

**Note:** It is recommended to have a locally running or stable endpoint to avoid potential timeout errors.
If you do not have one, you can set it up easily using [qEndpoint](https://github.com/the-qa-company/qEndpoint).

**Setting up Wikidata Endpoint**
```text
docker run -p 1234:1234 --name qendpoint-wikidata qacompany/qendpoint-wikidata
```
**Setting up YAGO Endpoint**
```text
docker run -p 1234:1234 --name qendpoint-yago qacompany/qendpoint
```
After setting up the YAGO endpoint, upload the YAGO 4.5 triples from [here](https://yago-knowledge.org/downloads/yago-4-5).

The endpoint URL will then be accessible at: [`http://localhost:1234/api/endpoint/sparql`](http://localhost:1234/api/endpoint/sparql).

## Running Experiments

**Example 1: Generate Prompts (Local Setting, WES Dataset)**
```text
python main.py --task prompt \
               --dataset wes \
               --output_dir output/prompts/local/wes/entity_id/5 \
               --mode local \
               --num_instances 5 \
               --sort_by entity_id \
               --few_shot \
               --few_shot_example_path dataset/wes/Q4220917.shex \
               --save_log
```

**Example 2: Generate Schema (Global Setting, YAGOS Dataset)**
```text
python main.py --task generate \
               --model_name gpt-4o-mini \
               --dataset wes \
               --mode local \
               --output_dir output/prompts/local/gpt-4o-mini/wes/entity_id/5 \
               --prompts_dir output/prompts/local/wes/entity_id/5 \
               --num_instances 5 \
               --sort_by entity_id \
               --few_shot \
               --few_shot_example_path resources/wes_global_few_shot_examples.toml \
               --graph_info_path resources/wikidata_property_information.json \
               --save_log
```

**Example 3: Evaluate (Classification Metrics, Exact Matching)**
```text
python evaluate.py --dataset wes \
                   --ground_truth_dir dataset/wes \
                   --predictions_dir output/results/local/gpt-4o-mini/wes/entity_id/5 \
                   --node_constraint_matching_level exact \
                   --cardinality_matching_level exact \
                   --classification
```

**Example 4: Evaluate (Similarity Metrics)**
```text
python evaluate.py --dataset wes \
                   --ground_truth_dir dataset/wes \
                   --predictions_dir output/results/local/gpt-4o-mini/wes/entity_id/5 \
                   --similarity
```

## Resources

List of few-shot example files and graph information files:

| `mode`  | `dataset` |            `few_shot_example_path`            |               `graph_info_path`                |
|:-------:|:---------:|:---------------------------------------------:|:----------------------------------------------:|
|  local  |    WES    |           dataset/wes/Q4220917.shex           |  resources/wes_predicate_count_instances.json  |
|  local  |   YAGOS   |         dataset/yagos/Scientist.shex          | resources/yagos_predicate_count_instances.json |
| global  |    WES    |  resources/wes_global_few_shot_examples.toml  |  resources/wikidata_property_information.json  |
| global  |   YAGOS   | resources/yagos_global_few_shot_examples.toml |                       /                        |
| triples |    WES    |           dataset/wes/Q4220917.shex           |  resources/wes_predicate_count_instances.json  |
| triples |   YAGOS   |         dataset/yagos/Scientist.shex          | resources/yagos_predicate_count_instances.json |



---

## Contribution Workflow
1. Understand the task.
2. Check existing patterns.
3. Implement solution.
4. Add tests.
5. Update documentation.
6. Verify lint / build / tests pass.

---
