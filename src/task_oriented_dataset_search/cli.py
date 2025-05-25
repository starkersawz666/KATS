import json
import os
from pathlib import Path
import click
from task_oriented_dataset_search.embedding.embedder import SentenceTransformerEmbedder
from task_oriented_dataset_search.embedding.pipeline import EmbeddingPipeline
from task_oriented_dataset_search.extraction.client import OpenAIClient
from task_oriented_dataset_search.extraction.extractor import StandardExtractor
from task_oriented_dataset_search.extraction.file_extractor import extract_file
from task_oriented_dataset_search.graph.builder import GraphBuilder
from task_oriented_dataset_search.graph.dataset_merger import DatasetMerger
from task_oriented_dataset_search.graph.task_merger import TaskMerger
from task_oriented_dataset_search.importer.db_importer import TinyDBImporter
from task_oriented_dataset_search.pipeline import PipelineConfig, TodsEngine
from task_oriented_dataset_search.preprocessing.processor import preprocess
from task_oriented_dataset_search.utils.cache import CacheManager


def _get_config_from_ctx(ctx) -> PipelineConfig:
    params = {k: v for k, v in ctx.params.items() if v is not None}
    if "api_key" not in params and "OPENAI_API_KEY" not in os.environ:
        raise click.UsageError(
            "Missing option '--api-key' or environment variable 'OPENAI_API_KEY'."
        )
    return PipelineConfig(**params)


def _get_engine_from_ctx(ctx) -> TodsEngine:
    config = _get_config_from_ctx(ctx)
    return TodsEngine(config=config)


pipeline_options = [
    click.option(
        "--api-key",
        envvar="OPENAI_API_KEY",
        help="OpenAI API key. Can also be set via OPENAI_API_KEY env var.",
    ),
    click.option(
        "--cache-root",
        default=".cache",
        show_default=True,
        type=click.Path(),
        help="Root directory for caching.",
    ),
    click.option(
        "--preprocess-workers",
        default=1,
        show_default=True,
        type=int,
        help="Number of workers for preprocessing.",
    ),
    click.option(
        "--extract-workers",
        default=1,
        show_default=True,
        type=int,
        help="Number of workers for extraction.",
    ),
    click.option("--api-base", default=None, help="Custom OpenAI API base URL."),
    click.option(
        "--model",
        default="gpt-4o-mini",
        show_default=True,
        help="LLM model for extraction.",
    ),
    click.option(
        "--qa-api-key",
        envvar="QA_OPENAI_API_KEY",
        help="OpenAI API key for QA (defaults to api-key).",
    ),
    click.option(
        "--qa-api-base", default=None, help="Custom OpenAI API base URL for QA."
    ),
    click.option(
        "--qa-model", default="gpt-4o", show_default=True, help="LLM model for QA."
    ),
    click.option(
        "--db-path",
        default=None,
        type=click.Path(),
        help="Path to TinyDB database file.",
    ),
    click.option(
        "--faiss-tasks-index-path",
        default=None,
        type=click.Path(),
        help="Path to Faiss index for tasks.",
    ),
    click.option(
        "--faiss-datasets-index-path",
        default=None,
        type=click.Path(),
        help="Path to Faiss index for datasets.",
    ),
    click.option(
        "--task-parquet-path",
        default=None,
        type=click.Path(),
        help="Path to Parquet file for tasks.",
    ),
    click.option(
        "--dataset-parquet-path",
        default=None,
        type=click.Path(),
        help="Path to Parquet file for datasets.",
    ),
    click.option(
        "--graph-path",
        default=None,
        type=click.Path(),
        help="Path to the original knowledge graph.",
    ),
    click.option(
        "--graph-processed-path",
        default=None,
        type=click.Path(),
        help="Path to the processed knowledge graph.",
    ),
    click.option(
        "--graph-tasks-path",
        default=None,
        type=click.Path(),
        help="Path to the tasks similarity graph.",
    ),
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@click.group()
@add_options(pipeline_options)
@click.pass_context
def cli(ctx, **kwargs):
    ctx.ensure_object(dict)
    ctx.params = {k: v for k, v in kwargs.items() if v is not None}


@cli.command(name="preprocess")
@click.argument("path", type=click.Path(exists=True))
def preprocess_cmd(path):
    click.echo(f"Preprocessing: {path}")
    if os.path.isfile(path):
        doc = preprocess(path)
        click.echo(
            f"Processed: {doc.metadata['path']} -> .cache/preprocessing/{CacheManager().fingerprint_file(path)}.txt"
        )
    else:
        # TODO: Directory preprocessing
        click.echo("Directory preprocessing not yet implemented in CLI.")


@cli.command(name="extract")
@click.argument("fingerprint", type=str)
@click.pass_context
def extract_cmd(ctx, fingerprint):
    config = _get_config_from_ctx(ctx)
    cache = CacheManager(config.cache_root)
    pre_dir = os.path.join(cache.cache_root, "preprocessing")
    txt_path = os.path.join(pre_dir, f"{fingerprint}.txt")

    if not os.path.exists(txt_path):
        click.echo(f"Error: File not found: {txt_path}", err=True)
        return

    click.echo(f"Extracting from: {txt_path}")
    client = OpenAIClient(
        api_key=config.api_key, model=config.model, base_url=config.api_base
    )
    extractor = StandardExtractor(client)
    result = extract_file(txt_path, extractor, cache)
    click.echo(
        f"Extraction complete. Result saved to: .cache/extraction/{fingerprint}.json"
    )


@cli.command(name="import")
@click.argument("fingerprint", type=str)
@click.pass_context
def import_db_cmd(ctx, fingerprint):
    config = _get_config_from_ctx(ctx)
    cache = CacheManager(config.cache_root)
    json_path = Path(
        os.path.join(cache.cache_root, "extraction", f"{fingerprint}.json")
    )
    if not json_path.exists():
        click.echo(f"Error: File not found: {json_path}", err=True)
        return
    click.echo(f"Importing: {json_path}")
    importer = TinyDBImporter(db_path=config.db_path)
    importer.import_file(json_path)
    click.echo(f"Import complete. DB updated: {config.db_path}")


@cli.command(name="embed")
@click.pass_context
def embed_cmd(ctx):
    config = _get_config_from_ctx(ctx)
    click.echo("Starting Embedding Pipeline...")
    embedder = SentenceTransformerEmbedder()
    pipeline = EmbeddingPipeline(
        embedder,
        task_index_path=config.faiss_tasks_index_path,
        dataset_index_path=config.faiss_datasets_index_path,
        task_parquet_path=config.task_parquet_path,
        dataset_parquet_path=config.dataset_parquet_path,
    )
    pipeline.embed_all(db_path=config.db_path)
    click.echo("Embedding complete.")


@cli.command(name="build-kg")
@click.pass_context
def build_kg_cmd(ctx):
    config = _get_config_from_ctx(ctx)
    click.echo("Building basic Knowledge Graph...")
    graph_builder = GraphBuilder(db_path=config.db_path, graph_path=config.graph_path)
    graph_builder.build_basic_graph()
    graph_builder.save_graph()
    click.echo(f"Basic KG built and saved to: {config.graph_path}")


@cli.command(name="merge-task")
@click.pass_context
def merge_task_cmd(ctx):
    config = _get_config_from_ctx(ctx)
    click.echo("Merging tasks...")
    task_merger = TaskMerger(
        db_path=config.db_path,
        graph_path=config.graph_path,
        graph_processed_path=config.graph_processed_path,
        task_faiss_path=config.faiss_tasks_index_path,
        task_parquet_path=config.task_parquet_path,
        strong_similarity_threshold=config.strong_similarity_threshold,
        keyword_overlap_threshold=config.keyword_overlap_threshold,
        weak_similarity_threshold=config.weak_similarity_threshold,
        max_merge=config.task_max_merge,
    )
    task_merger.merge_tasks()
    task_merger.save_graph()
    click.echo(f"Tasks merged. Graph saved to: {config.graph_processed_path}")


@cli.command(name="merge-dataset")
@click.pass_context
def merge_dataset_cmd(ctx):
    config = _get_config_from_ctx(ctx)
    click.echo("Merging datasets...")
    llm_client = OpenAIClient(
        api_key=config.api_key,
        model=config.model,
        base_url=config.api_base,
        temperature=config.temperature,
    )
    cache = CacheManager(config.cache_root)
    dataset_merger = DatasetMerger(
        db_path=config.db_path,
        graph_path=config.graph_processed_path,
        graph_processed_path=config.graph_processed_path,
        dataset_faiss_path=config.faiss_datasets_index_path,
        dataset_parquet_path=config.dataset_parquet_path,
        llm_client=llm_client,
        cache_manager=cache,
        similarity_threshold=config.dataset_merge_similarity_threshold,
        k_neighbors=config.dataset_merge_k_neighbors,
        llm_retries=config.llm_retries,
        llm_retry_delay=config.llm_retry_delay,
    )
    dataset_merger.merge_datasets()
    dataset_merger.save_graph()
    click.echo(f"Datasets merged. Graph saved to: {config.graph_processed_path}")


@cli.command(name="separate-task")
@click.pass_context
def separate_task_cmd(ctx):
    config = _get_config_from_ctx(ctx)
    click.echo("Creating task similarity graph...")
    merged_graph_builder = GraphBuilder(
        db_path=config.db_path,
        graph_path=config.graph_processed_path,
        save_path=config.graph_tasks_path,
    )
    merged_graph_builder.build_and_save_task_similarity_graph()
    click.echo(f"Task similarity graph saved to: {config.graph_tasks_path}")


@cli.command(name="build")
@click.option(
    "--input-folder",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Input folder containing documents to process.",
)
@click.pass_context
def build_cmd(ctx, input_folder):
    engine = _get_engine_from_ctx(ctx)
    engine.cfg.input_folder = input_folder
    click.echo(f"Starting FULL build process from: {input_folder}")
    engine.build()
    click.echo("Full build process complete.")


@cli.command(name="search")
@click.argument("task_description", type=str)
@click.option(
    "--top-k", default=5, show_default=True, help="Number of datasets to return."
)
@click.pass_context
def search_cmd(ctx, task_description, top_k):
    """Search for datasets based on a task description."""
    engine = _get_engine_from_ctx(ctx)
    click.echo(f"Searching for task: '{task_description}' (Top K={top_k})")
    results = engine.search(
        task_description,
        top_k_datasets=top_k,
    )
    if results:
        click.echo("--- Search Results ---")
        click.echo(json.dumps(results, indent=2))
        click.echo("----------------------")
    else:
        click.echo("No relevant datasets found.")


@cli.command(name="qa")
@click.argument("task_description", type=str)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    help="Number of datasets to consider for the answer.",
)
@click.pass_context
def qa_cmd(ctx, task_description, top_k):
    """Ask a question and get a natural language answer about datasets."""
    engine = _get_engine_from_ctx(ctx)
    click.echo(f"Generating answer for task: '{task_description}'")
    answer = engine.qa(
        task_description,
        top_k_datasets=top_k,
    )
    click.echo("--- Answer ---")
    click.echo(answer)
    click.echo("--------------")


if __name__ == "__main__":
    cli()
