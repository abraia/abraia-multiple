"""Command-line interface for the Abraia SDK."""

import itertools
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

import click
from click.shell_completion import CompletionItem
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
@click.option(
    "--completion",
    "install_completion",
    is_flag=True,
    help="Set up shell completion instead of configuring API credentials.",
)
@click.option(
    "--shell",
    "completion_shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    help="Shell to configure; defaults to the shell in SHELL.",
)
def configure(install_completion=False, completion_shell=None):
    """Configure Abraia credentials or shell completion."""
    if completion_shell and not install_completion:
        raise click.UsageError("--shell can only be used with --completion.")
    if install_completion:
        _install_shell_completion(completion_shell)
        return

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


def _install_shell_completion(shell=None):
    """Add Abraia completion to a Bash, Zsh, or Fish configuration."""
    shell = shell or Path(os.environ.get("SHELL", "")).name
    if shell not in ("bash", "zsh", "fish"):
        raise click.ClickException(
            "Could not detect your shell. Choose bash, zsh, or fish."
        )
    if shell == "fish":
        from click.shell_completion import FishComplete

        path = Path.home() / ".config/fish/completions/abraia.fish"
        content = FishComplete(
            cli, {}, "abraia", "_ABRAIA_COMPLETE"
        ).source().rstrip() + "\n"
    else:
        path = Path.home() / f".{shell}rc"
        content = f'eval "$(_ABRAIA_COMPLETE={shell}_source abraia)"\n'

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if shell == "fish":
            if existing == content:
                click.echo(f"Abraia completion is already installed in {path}.")
                return
            raise click.ClickException(
                f"Refusing to overwrite the existing completion file: {path}"
            )
        if content.rstrip() in existing.splitlines():
            click.echo(f"Abraia completion is already installed in {path}.")
            return
        if existing and not existing.endswith("\n"):
            existing += "\n"
    else:
        existing = ""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing + content, encoding="utf-8")
    click.echo(f"Added Abraia completion to {path}. Open a new shell to use it.")


@cli.group()
def discover():
    """Discover cameras and video streams available to this machine."""


@discover.command("cameras")
def discover_cameras_command():
    """Find local camera devices and their numeric source indexes."""
    from .runtime import discover_cameras

    cameras = _run_operation(discover_cameras)
    if not cameras:
        click.echo("No local cameras found.")
        return
    for camera in cameras:
        click.echo(f"{camera['source']}  {camera['label']}")
    click.echo("\nRun one with: abraia run demo detect SOURCE")


@discover.command("streams")
def discover_streams_command():
    """Find RTSP streams advertised by ONVIF cameras on the local network."""
    from .runtime import discover_streams

    streams = _run_operation(discover_streams)
    if not streams:
        click.echo("No ONVIF video streams found.")
        return
    for stream in streams:
        click.echo(stream)
    click.echo("\nRun one with: abraia run demo detect STREAM_URL")


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


def _run_operation(operation):
    try:
        return operation()
    except click.ClickException:
        raise
    except Exception as error:
        raise click.ClickException(str(error)) from error


def _available_demos():
    from .demo import CAMERA_PIPELINES, PIPELINES

    return tuple(dict.fromkeys(
        tuple(PIPELINES)
        + ("faces",)
        + tuple(CAMERA_PIPELINES)
    ))


def _show_demos():
    click.echo("Available demos: " + ", ".join(_available_demos()))


def _completion_items(values, incomplete):
    prefix = (incomplete or "").lower()
    return [
        CompletionItem(value)
        for value in values
        if value.lower().startswith(prefix)
    ]


def _complete_run_project(ctx, param, incomplete):
    del ctx, param
    return _completion_items(("demo",), incomplete)


def _complete_run_demo(ctx, param, incomplete):
    del param
    if ctx.params.get("project") != "demo":
        return []
    return _completion_items(("list",) + _available_demos(), incomplete)


def _complete_demo_model(ctx, param, incomplete):
    del param
    project = ctx.params.get("project")
    demo_name = ctx.params.get("demo_or_source")
    if project is None and ctx.args:
        project = ctx.args[0]
    if not demo_name and project == "demo" and len(ctx.args) > 1:
        demo_name = ctx.args[1]
    if project != "demo":
        return []
    if not demo_name or demo_name == "list":
        return []
    from .demo import demo_model_options

    return _completion_items(demo_model_options(demo_name), incomplete)


def _run_demo(demo_name, src=None, accelerator="auto", output=None, model=None):
    """Run one of the built-in demo pipelines."""
    demo_name = demo_name or "detect"
    if demo_name not in _available_demos():
        available = ", ".join(_available_demos())
        raise click.ClickException(
            f"Unknown demo '{demo_name}'. Available demos: {available}"
        )
    if demo_name == "pi-camera":
        if model is not None:
            raise click.ClickException("The pi-camera demo does not use a model.")
        from .demo import run_camera_pipeline

        return run_camera_pipeline(
            src=src,
            accelerator=accelerator,
            output=output,
        )
    if demo_name == "faces":
        if model is not None:
            raise click.ClickException(
                "The faces demo does not support model variants."
            )
        from .demo import track_faces

        return track_faces(src, accelerator=accelerator, output=output)

    from .demo import monitor_objects

    options = {"accelerator": accelerator, "output": output}
    if model is not None:
        options["model"] = model
    return monitor_objects(src, demo_name, **options)


@cli.command("list")
@click.argument("folder", required=False, default="")
def list_files_command(folder):
    """List files in Abraia."""
    _run_operation(lambda: click.echo(format_output(*abraia.list_files(folder))))


@cli.command()
@click.argument("src", type=click.Path(exists=True))
@click.argument("folder", required=False, default="")
def upload(src, folder):
    """Upload files to Abraia."""
    _run_operation(
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

    _run_operation(operation)


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

    _run_operation(operation)


@cli.command()
@click.argument("project")
@click.argument("query", required=False, default="man with red shirt")
def search(project, query):
    """Search a project using text or an image path."""
    from .demo import search_images

    _run_operation(lambda: search_images(project, query=query))


@cli.command()
@click.argument(
    "project",
    metavar="PROJECT",
    shell_complete=_complete_run_project,
)
@click.argument(
    "demo_or_source",
    required=False,
    default="",
    metavar="[DEMO_OR_SOURCE]",
    shell_complete=_complete_run_demo,
)
@click.argument(
    "src",
    required=False,
    default=None,
    metavar="[SOURCE]",
    type=click.Path(exists=False),
)
@click.option(
    "--model",
    "demo_model",
    shell_complete=_complete_demo_model,
    help="Use a compatible built-in model variant for a demo.",
)
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
def run(
    project,
    demo_or_source,
    src,
    accelerator,
    output=None,
    demo_model=None,
):
    """Run a built-in demo or a saved project's pipeline.

    Use ``abraia run demo NAME [SOURCE]`` for demos or
    ``abraia run PROJECT [SOURCE]`` for saved pipelines.
    """
    def operation():
        if project == "demo":
            if demo_or_source == "list":
                _show_demos()
                return
            return _run_demo(
                demo_or_source or "detect",
                src,
                accelerator=accelerator,
                output=output,
                model=demo_model,
            )

        if demo_model is not None:
            raise click.ClickException(
                "--model can only be used with 'abraia run demo'."
            )

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

    _run_operation(operation)


def main():
    """Run the Abraia CLI."""
    return cli()


__all__ = ["cli", "main"]


if __name__ == "__main__":
    main()
