import json
import os
from pathlib import Path
import click
from task_oriented_dataset_search.extraction.client import OpenAIClient
from task_oriented_dataset_search.extraction.extractor import StandardExtractor
from task_oriented_dataset_search.extraction.file_extractor import extract_file
from task_oriented_dataset_search.importer.db_importer import TinyDBImporter
from task_oriented_dataset_search.preprocessing.processor import preprocess
from task_oriented_dataset_search.utils.cache import CacheManager


@click.group()
def cli():
    pass


@cli.command(name="preprocess")
@click.argument("path", type=click.Path(exists=True))
def preprocess_cmd(path):
    doc = preprocess(path)
    click.echo(f"Source: {doc.metadata['path']}")
    # click.echo(doc.text[:500])


if __name__ == "__main__":
    cli()


@cli.command(name="extract")
@click.argument("fingerprint", type=str)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    prompt=True,
    hide_input=True,
    help="Your OpenAI API key",
)
@click.option(
    "--api-base",
    default=None,
    help="Custom OpenAI API base URL (e.g. https://your-proxy/v1)",
)
@click.option(
    "--model", default="gpt-4o-mini", show_default=True, help="Which LLM model to use"
)
def extract_cmd(fingerprint, api_key, api_base, model):
    cache = CacheManager()
    pre_dir = os.path.join(cache.cache_root, "preprocessing")
    txt_path = os.path.join(pre_dir, f"{fingerprint}.txt")
    if not os.path.exists(txt_path):
        click.echo(f"Error: File not found: {txt_path}", err=True)
        return

    client = OpenAIClient(api_key=api_key, model=model, base_url=api_base)
    extractor = StandardExtractor(client)
    result = extract_file(txt_path, extractor, cache)


@cli.command(name="import")
@click.argument("fingerprint", type=str)
def import_db_cmd(fingerprint):
    importer = TinyDBImporter()
    cache = CacheManager()
    importer.import_file(
        Path(os.path.join(cache.cache_root, "extraction", f"{fingerprint}.json"))
    )
    click.echo("Import completed.")
