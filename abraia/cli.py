"""Command-line interface for the Abraia SDK."""

import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import click
from tqdm import tqdm

from . import APIError, Abraia, __version__, config
from .utils.concurrency import (
    DEFAULT_MAX_REMOTE_WORKERS,
    bounded_map,
)


abraia = Abraia()


def process_map(
    task, values, *iterables, desc=""
):
    """Apply a task concurrently, bounded by the finite inputs."""
    inputs = (values,) + iterables
    lengths = [len(value) for value in inputs if hasattr(value, "__len__")]
    if not lengths:
        raise ValueError("At least one process_map input must have a length")
    total = lengths[0]
    if any(length != total for length in lengths[1:]):
        raise ValueError("All process_map iterables must have the same length")
    prepared = [
        value if hasattr(value, "__len__") else itertools.islice(value, total)
        for value in inputs
    ]
    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_REMOTE_WORKERS) as executor:
        results = []
        with tqdm(total=total, desc=desc) as progress:
            for result in bounded_map(
                executor,
                lambda args: task(*args),
                zip(*prepared),
                max_pending=DEFAULT_MAX_REMOTE_WORKERS * 2,
            ):
                progress.set_postfix_str(str(result))
                progress.update(1)
                results.append(result)
        return results


def download_file(path, folder):
    """Download a remote file through the SDK's shared cache."""
    remote_info = path if isinstance(path, dict) else None
    remote_path = remote_info.get("path") if remote_info else path
    dest = os.path.join(folder, os.path.basename(remote_path))
    return abraia.download_file(remote_path, dest, remote_info=remote_info)


def input_files(src):
    src = os.path.join(src, "**/*") if os.path.isdir(src) else src
    return [path for path in glob(src, recursive=True) if os.path.isfile(path)]


@click.group("abraia")
@click.version_option(__version__)
def cli():
    """Abraia CLI tool."""


@cli.command()
def configure():
    """Configure the Abraia API key."""
    click.echo(
        "Go to ["
        + click.style("https://abraia.me/editor/", fg="green")
        + "] to get your API key\n"
    )
    try:
        _abraia_id, abraia_key = config.load()
        abraia_key = click.prompt("Abraia Key", default=abraia_key)
        config.load_auth(abraia_key)
        config.save(abraia_key)
    except (OSError, ValueError) as error:
        raise click.ClickException(
            f"Unable to save Abraia credentials: {error}"
        ) from error


def format_output(files, folders=None):
    folders = folders or []
    output = "\n".join(
        "{:>28}  {}/".format("", click.style(folder["name"], fg="blue", bold=True))
        for folder in folders
    ) + "\n"
    output += "\n".join(
        "{}  {:>7}  {}".format(file["date"], file["size"], file["name"])
        for file in files
    )
    return output + f"\ntotal {len(files)}"


def _run_remote(operation):
    try:
        return operation()
    except click.ClickException:
        raise
    except Exception as error:
        raise click.ClickException(str(error)) from error


@cli.command("list")
@click.argument("folder", required=False, default="")
def list_files_command(folder):
    """List files in Abraia."""
    _run_remote(lambda: click.echo(format_output(*abraia.list_files(folder))))


@cli.command()
@click.argument("src", type=click.Path(exists=True))
@click.argument("folder", required=False, default="")
def upload(src, folder):
    """Upload files to Abraia."""
    _run_remote(
        lambda: process_map(
            abraia.upload_file,
            input_files(src),
            itertools.repeat(folder),
            desc="Uploading",
        )
    )


@cli.command()
@click.argument("path")
@click.argument("folder", required=False, default="")
def download(path, folder):
    """Download files from Abraia."""
    def operation():
        files = abraia.list_files(path)[0]
        return process_map(
            download_file,
            files,
            itertools.repeat(folder),
            desc="Downloading",
        )

    _run_remote(operation)


@cli.command()
@click.argument("path")
def remove(path):
    """Remove files from Abraia."""
    def operation():
        files = abraia.list_files(path)[0]
        click.echo(format_output(files))
        if files and click.confirm("Are you sure you want to remove the files?"):
            return process_map(
                abraia.remove_file,
                [file["path"] for file in files],
                desc="Removing",
            )
        return None

    _run_remote(operation)


@cli.command()
@click.option("--remove", help="Remove file metadata", is_flag=True)
@click.argument("path")
def metadata(path, remove):
    """Load or remove file metadata."""
    def operation():
        if remove:
            abraia.remove_metadata(path)
            click.echo(f"Removed metadata for {path}")
            return
        click.echo(json.dumps(abraia.load_metadata(path), indent=2))

    _run_remote(operation)


@cli.command()
@click.argument("project")
@click.argument("query", required=False, default="man with red shirt")
def search(project, query):
    """Search a project using text or an image path."""
    from .demo import search_images

    _run_remote(lambda: search_images(project, query=query))


@cli.command()
@click.argument("project")
@click.argument("demo_or_source", required=False, default="")
@click.argument("src", required=False, default=None)
@click.option(
    "--accelerator",
    type=click.Choice(["auto", "onnx", "cpu", "gpu", "hailo"]),
    default="auto",
    show_default=True,
    help="Inference accelerator; auto prefers a compatible Hailo device.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    help="Save rendered video frames to this local file.",
)
def run(project, demo_or_source, src, accelerator, output=None):
    """Run a project's saved pipeline."""
    def operation():
        if project == "demo":
            if demo_or_source == "faces":
                from .demo import track_faces

                track_faces(src, accelerator=accelerator, output=output)
            else:
                from .demo import monitor_objects

                monitor_objects(
                    src,
                    demo_or_source or "detect",
                    accelerator=accelerator,
                    output=output,
                )
            return

        from .runtime import Pipeline
        from .runtime.sources import infer_source_type, normalize_source_type

        source_override = src if src is not None else demo_or_source or None
        try:
            pipeline_config = abraia.load_json(f"{project}/pipeline.json")
        except APIError as error:
            if error.code != 404:
                raise
            pipeline_config = None
        if pipeline_config is None:
            click.echo(
                f"No saved pipeline found in project '{project}'. "
                "Add a pipeline.json file to run inference."
            )
            return

        if source_override is not None:
            source_config = pipeline_config.setdefault("source", {})
            source_config["src"] = source_override
            inferred_type = infer_source_type(source_override)
            if (
                inferred_type
                and normalize_source_type(source_config.get("type")) != "genicam"
            ):
                source_config["type"] = inferred_type
        if output is not None:
            pipeline_config.setdefault("display", {})["dest"] = output
        with Pipeline.from_dict(
            pipeline_config,
            accelerator=accelerator,
        ) as pipeline:
            pipeline.run()

    _run_remote(operation)


def main():
    """Run the Abraia CLI."""
    return cli()


__all__ = ["cli", "main"]


if __name__ == "__main__":
    main()
