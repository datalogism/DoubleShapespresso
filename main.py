import argparse

from loguru import logger
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd

from shapespresso.pipeline import construct_prompt, construct_perfect_input_prompt
from shapespresso.pipeline import OpenAIModel, ClaudeModel, OllamaModel, LiquiAIModel
from shapespresso.pipeline import local_generation_workflow, global_generation_workflow, agentic_generation_workflow


def get_expected_output_path(task: str, syntax: str, output_dir: str, class_id: str) -> Path:
    """Determine the expected output file path for a given task/syntax/class_id."""
    output_dir = Path(output_dir)
    if task in ("prompt", "test_prompt"):
        return output_dir / f"{class_id}.json"
    elif task in ("generate", "test_generate"):
        ext = ".ttl" if syntax == "SHACL" else ".shex"
        return output_dir / f"{class_id}{ext}"
    else:
        return output_dir / f"{class_id}.json"


def resolve_shape_list(
    class_uris: List[str],
    class_labels: List[str],
    shape_name: Optional[str] = None,
    shape_dir: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Filter the class list based on --shape_name and --shape_dir.

    Args:
        class_uris: List of class URIs from the CSV.
        class_labels: List of class labels from the CSV.
        shape_name: If provided, keep only the shape matching this name.
        shape_dir: If provided, keep only shapes with a file in this directory.

    Returns:
        Filtered list of (class_uri, class_label) pairs.
    """
    # Build unique pairs preserving order (dict dedup like the original code)
    pairs = list(dict(zip(class_uris, class_labels)).items())

    # Filter by --shape_name
    if shape_name:
        # Strip extension if provided (e.g., "Airport.ttl" -> "Airport")
        name = Path(shape_name).stem
        pairs = [(uri, label) for uri, label in pairs if uri.split("/")[-1] == name]
        if not pairs:
            logger.warning(f"No shape found matching name '{name}' in the dataset CSV")

    # Filter by --shape_dir
    if shape_dir:
        shape_dir_path = Path(shape_dir)
        if not shape_dir_path.is_dir():
            logger.warning(f"Shape directory '{shape_dir}' does not exist")
            return []
        # Collect basenames (without extension) of .ttl and .shex files
        dir_basenames = set()
        for f in shape_dir_path.iterdir():
            if f.suffix in (".ttl", ".shex"):
                dir_basenames.add(f.stem)
        pairs = [(uri, label) for uri, label in pairs if uri.split("/")[-1] in dir_basenames]
        if not pairs:
            logger.warning(f"No shapes from CSV match files in '{shape_dir}'")

    return pairs


def main():
    parser = argparse.ArgumentParser()

    # general options
    parser.add_argument(
        '--task',
        type=str,
        choices=["prompt", "generate","test_prompt","test_generate"],
        required=True,
        help="Task to perform: 'prompt' to construct prompts, 'generate' to run ShEx generation"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=False,
        help="Name of the model (e.g., 'gpt-4', 'claude-3', 'llama3')"
    )
    parser.add_argument(
        '--endpoint_url',
        type=str,
        default='http://localhost:1234/api/endpoint/sparql',
        required=False,
        help="Endpoint URL"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=["wes", "yagos", "dbpedia"],
        required=True,
        help="Dataset to use: 'wes' (Wikidata EntitySchema) or 'yagos'"
    )
    parser.add_argument(
        '--syntax',
        type=str,
        default='ShEx',
        required=False,
        choices=["ShEx","SHACL"],
        help="Mode of prompt engineering"
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=["local", "global", "triples","agentic"],
        help="Mode of prompt engineering"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        help="Directory path where outputs (e.g., logs, results) will be saved"
    )
    parser.add_argument(
        '--prompts_dir',
        type=str,
        required=False,
        help="Prompts saving/loading directory"
    )
    parser.add_argument(
        '--save_log',
        action='store_true',
        help="If set, saves logs to the specified output directory"
    )

    # few-shot options
    parser.add_argument(
        '--few_shot',
        action='store_true',
        help="Whether to include few-shot examples in the prompt"
    )
    parser.add_argument(
        '--few_shot_example_path',
        type=str,
        help="Path to the few-shot example file"
    )

    # local generation options
    parser.add_argument(
        '--num_instances',
        type=int,
        default=3,
        required=False,
        help="Number of instances to return (default: 3)"
    )
    parser.add_argument(
        '--sort_by',
        type=str,
        default="entity_id",
        choices=["predicate_count", "entity_id", "edit_frequency"],
        help="Sorting criteria for instance selection (default: entity_id)"
    )

    # global generation options
    parser.add_argument(
        '--num_class_distribution',
        type=int,
        default=3,
        required=False,
        help="Number of classes to return in object class distribution (default: 3)"
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=5,
        required=False,
        help="Threshold for filtering common properties (default: 5)"
    )
    parser.add_argument(
        '--graph_info_path',
        type=str,
        default=None,
        required=False,
        help="Path to the graph information file (e.g., wes_predicate_count_instances.json, wikidata_property_information.json)"
    )
    parser.add_argument(
        '--property_list_path',
        type=str,
        default=None,
        required=False,
        help="Path to the property list file"
    )
    parser.add_argument(
        '--information_types',
        nargs='+',
        default=None,
        required=False,
        help="Types of information to include in global generation prompts"
    )
    parser.add_argument(
        '--answer_keys',
        nargs='+',
        help="List of keys to include from answers"
    )

    # shape filtering and rerun options
    parser.add_argument(
        '--shape_name',
        type=str,
        default=None,
        required=False,
        help="Process only this shape (e.g., 'Airport', 'Q4220917'). Accepts with or without extension."
    )
    parser.add_argument(
        '--shape_dir',
        type=str,
        default=None,
        required=False,
        help="Directory of shape files; only CSV shapes with a matching file (.ttl/.shex) will be processed."
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force rerun even if output already exists."
    )

    args = parser.parse_args()

    if args.save_log:
        args.output_dir = Path(args.output_dir or Path.cwd())
        log_dir = args.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%m-%d-%H-%M')
        log_file = log_dir / f"{args.model_name}-{args.dataset}-{timestamp}.log"
        logger.add(log_file)

    # load class list
    dataset_df = pd.read_csv(f'dataset/{args.dataset}.csv')
    class_uris = dataset_df['class_uri'].tolist()
    class_labels = dataset_df['class_label'].tolist()

    if args.dataset == "wes":
        instance_of_uri = "http://www.wikidata.org/prop/direct/P31"
    else:
        instance_of_uri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    # Resolve shape list based on filters
    shapes = resolve_shape_list(
        class_uris, class_labels,
        shape_name=args.shape_name,
        shape_dir=args.shape_dir,
    )

    # Summary logging
    total_shapes = len(shapes)
    if total_shapes == 0:
        logger.warning("No shapes to process after filtering. Exiting.")
        return

    skipped = 0
    to_process = []
    for class_uri, class_label in shapes:
        class_id = class_uri.split("/")[-1]
        if not args.force and args.output_dir:
            output_path = get_expected_output_path(args.task, args.syntax, args.output_dir, class_id)
            if output_path.exists() and output_path.stat().st_size > 0:
                skipped += 1
                continue
        to_process.append((class_uri, class_label))

    shape_names = [uri.split("/")[-1] for uri, _ in shapes]
    logger.info(f"Total shapes found: {total_shapes}")
    logger.info(f"Shapes to process: {len(to_process)}, skipped (already exist): {skipped}")
    logger.info(f"Shape list: {shape_names}")

    if args.task == "prompt":
        for class_uri, class_label in to_process:
            class_id = class_uri.split("/")[-1]
            logger.info(f"Processing shape '{class_id}' ({class_label})")
            save_prompt_path = f"{args.output_dir}/{class_id}.json"
            prompt = construct_prompt(
                class_uri=class_uri,
                class_label=class_label,
                instance_of_uri=instance_of_uri,
                dataset=args.dataset,
                syntax=args.syntax,
                endpoint_url=args.endpoint_url,
                mode=args.mode,
                few_shot=args.few_shot,
                few_shot_example_path=args.few_shot_example_path,
                graph_info_path=args.graph_info_path,
                information_types=args.information_types,
                num_instances=args.num_instances,
                num_class_distribution=args.num_class_distribution,
                threshold=args.threshold,
                sort_by=args.sort_by,
                answer_keys=args.answer_keys,
                save_prompt_path=save_prompt_path,
            )
    elif args.task == "test_prompt":
        for class_uri, class_label in to_process:
            class_id = class_uri.split("/")[-1]
            logger.info(f"Processing shape '{class_id}' ({class_label})")
            save_prompt_path = f"{args.output_dir}/{class_id}.json"
            prompt = construct_perfect_input_prompt(
                class_uri=class_uri,
                class_label=class_label,
                instance_of_uri=instance_of_uri,
                dataset=args.dataset,
                syntax=args.syntax,
                endpoint_url=args.endpoint_url,
                mode=args.mode,
                few_shot=args.few_shot,
                few_shot_example_path=args.few_shot_example_path,
                graph_info_path=args.graph_info_path,
                information_types=args.information_types,
                num_instances=args.num_instances,
                num_class_distribution=args.num_class_distribution,
                threshold=args.threshold,
                sort_by=args.sort_by,
                answer_keys=args.answer_keys,
                save_prompt_path=save_prompt_path,
            )
    elif args.task == "generate":
        if "llama" in args.model_name.lower() or "qwen" in args.model_name.lower():
            model = OllamaModel(model_name=args.model_name)
        elif "deepseek" in args.model_name or "gpt" in args.model_name:
            model = OpenAIModel(model_name=args.model_name)
        elif "claude" in args.model_name:
            model = ClaudeModel(model_name=args.model_name)
        elif "slm" in args.model_name:
            model = LiquiAIModel(model_name="LiquidAI/LFM2-350M")
        else:
            raise NotImplementedError(f"Model {args.model_name} not implemented")
        logger.info(f"Running ShEx generation using model '{args.model_name}'")

        for class_uri, class_label in to_process:
            class_id = class_uri.split("/")[-1]
            logger.info(f"Processing shape '{class_id}' ({class_label})")
            if args.mode == "agentic":
                agentic_generation_workflow(
                    model=model,
                    class_uri=class_uri,
                    class_label=class_label,
                    instance_of_uri=instance_of_uri,
                    dataset=args.dataset,
                    syntax=args.syntax,
                    endpoint_url=args.endpoint_url,
                    mode=args.mode,
                    output_dir=args.output_dir,
                    few_shot=args.few_shot,
                    few_shot_example_path=args.few_shot_example_path,
                    num_instances=args.num_instances,
                    sort_by=args.sort_by,
                    graph_info_path=args.graph_info_path,
                    load_prompt_path=f"{args.prompts_dir}/{class_id}.json"
                )
            elif args.mode in ["local", "triples"]:
                local_generation_workflow(
                    model=model,
                    class_uri=class_uri,
                    class_label=class_label,
                    instance_of_uri=instance_of_uri,
                    dataset=args.dataset,
                    syntax=args.syntax,
                    endpoint_url=args.endpoint_url,
                    mode=args.mode,
                    output_dir=args.output_dir,
                    few_shot=args.few_shot,
                    few_shot_example_path=args.few_shot_example_path,
                    num_instances=args.num_instances,
                    sort_by=args.sort_by,
                    graph_info_path=args.graph_info_path,
                    load_prompt_path=f"{args.prompts_dir}/{class_id}.json"
                )
            elif args.mode == "global":
                global_generation_workflow(
                    model=model,
                    class_uri=class_uri,
                    class_label=class_label,
                    instance_of_uri=instance_of_uri,
                    dataset=args.dataset,
                    syntax=args.syntax,
                    endpoint_url=args.endpoint_url,
                    output_dir=args.output_dir,
                    few_shot=args.few_shot,
                    few_shot_example_path=args.few_shot_example_path,
                    property_list_path=args.property_list_path,
                    num_instances=args.num_instances,
                    num_class_distribution=args.num_class_distribution,
                    threshold=args.threshold,
                    sort_by=args.sort_by,
                    graph_info_path=args.graph_info_path
                )
            else:
                raise NotImplementedError(f"Mode {args.mode} not implemented!")
    else:
        raise NotImplementedError(f"Task '{args.task}' not implemented!")


if __name__ == "__main__":
    main()
