"""Command-line interface for the Abraia SDK."""

import itertools
import json
import os
import re
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


def list_files(folder):
    return abraia.list_files(folder)


def upload_file(file, folder):
    return abraia.upload_file(file, folder)


def download_file(path, folder):
    """Download a remote file through the SDK's shared cache."""
    remote_info = path if isinstance(path, dict) else None
    remote_path = remote_info.get("path") if remote_info else path
    dest = os.path.join(folder, os.path.basename(remote_path))
    os.makedirs(folder, exist_ok=True)
    abraia.download_file(remote_path, dest, remote_info=remote_info)
    return dest


def remove_file(path):
    return abraia.remove_file(path)


def _error_message(error):
    return str(getattr(error, "message", None) or error)


def echo_error(error):
    """Raise a Click-formatted error for a command failure."""
    raise click.ClickException(_error_message(error)) from error


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


@cli.command()
@click.argument("project")
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    help="Delete recommended remote files and update annotations.",
)
@click.option(
    "--yes",
    "assume_yes",
    is_flag=True,
    help="Skip confirmation when applying destructive changes.",
)
@click.option(
    "--apply-outliers",
    is_flag=True,
    help="Also delete outliers; review them carefully first.",
)
def curate(
    project,
    apply_changes,
    assume_yes,
    apply_outliers,
):
    """Analyze an image dataset and optionally apply recommendations."""
    from .dataset import load_dataset

    def operation():
        confirm = None
        if apply_changes and not assume_yes:
            confirm = lambda paths: click.confirm(
                f"Delete {len(paths)} recommended remote file(s)?"
            )
        report = load_dataset(project).curate(
            apply=apply_changes,
            analyze_options={"remove_outliers": apply_outliers},
            confirm=confirm,
        )
        click.echo(json.dumps(report, indent=2))

    _run_remote(operation)


@cli.group("files")
def cli_files():
    """Manage files on the cloud storage."""


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
        echo_error(error)


@cli_files.command("list")
@click.argument("folder", required=False, default="")
def list_files_command(folder):
    """List files in Abraia."""
    _run_remote(lambda: click.echo(format_output(*list_files(folder))))


@cli_files.command()
@click.argument("src", type=click.Path(exists=True))
@click.argument("folder", required=False, default="")
def upload(src, folder):
    """Upload files to Abraia."""
    _run_remote(
        lambda: process_map(
            upload_file,
            input_files(src),
            itertools.repeat(folder),
            desc="Uploading",
        )
    )


@cli_files.command()
@click.argument("path")
@click.argument("folder", required=False, default="")
def download(path, folder):
    """Download files from Abraia."""
    def operation():
        files = list_files(path)[0]
        return process_map(
            download_file,
            files,
            itertools.repeat(folder),
            desc="Downloading",
        )

    _run_remote(operation)


@cli_files.command()
@click.argument("path")
def remove(path):
    """Remove files from Abraia."""
    def operation():
        files = list_files(path)[0]
        click.echo(format_output(files))
        if files and click.confirm("Are you sure you want to remove the files?"):
            return process_map(
                remove_file,
                [file["path"] for file in files],
                desc="Removing",
            )
        return None

    _run_remote(operation)


@cli_files.command()
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


@cli.command("list")
def list_datasets_command():
    """List available datasets."""
    from .dataset import list_datasets

    _run_remote(lambda: click.echo(json.dumps(list_datasets(), indent=2)))


def convert_to_jpg(src):
    from .utils.image import load_image, save_image

    image = load_image(src)
    save_image(image, f"{os.path.splitext(src)[0]}.jpg")
    os.remove(src)


def anonymize(src):
    from .editing import anonymize_image
    from .utils.image import load_image, save_image

    save_image(anonymize_image(load_image(src)), src)


def upscale(src, threshold):
    from .editing import upscale_image
    from .utils.image import load_image, save_image

    image = load_image(src)
    if max(image.shape) < threshold:
        save_image(upscale_image(image), src)


def process_dataset(project, anonymize_images=False, upscale_threshold=0):
    from .utils.filesystem import get_type

    files = input_files(project)
    heics = [file for file in files if get_type(file) == "image/heic"]
    if heics:
        process_map(convert_to_jpg, heics, desc="Converting images")
        files = input_files(project)
    if anonymize_images:
        images = [file for file in files if get_type(file).startswith("image")]
        process_map(anonymize, images, desc="Anonymizing images")
    if upscale_threshold > 0:
        images = [file for file in files if get_type(file).startswith("image")]
        process_map(
            upscale,
            images,
            itertools.repeat(upscale_threshold),
            desc="Upscaling images",
        )
    return files


@cli.command()
@click.argument("project")
@click.argument("query", required=False, default="")
@click.option("--anonymize", help="Anonymize images (blur faces and plates)", is_flag=True)
@click.option("--upscale", help="Upscale images smaller than threshold", type=int, default=0)
def create(project, query, anonymize=False, upscale=0):
    """Create or update a dataset."""
    from .dataset.search import search_images
    from .dataset import load_dataset

    def operation():
        dataset = load_dataset(project)
        files = []
        if query:
            search_images(query, f"{project}/", limit=100)
        if os.path.exists(project):
            files = process_dataset(
                project,
                anonymize_images=anonymize,
                upscale_threshold=upscale,
            )
        process_map(upload_file, files, itertools.repeat(project + "/"), desc="Uploading images")
        dataset.save()

    _run_remote(operation)


@cli.command()
@click.argument("project")
@click.argument("label")
@click.option("--segment", help="Segment objects from boxes", is_flag=True)
def annotate(project, label, segment=False):
    """Annotate a dataset using Grounding Dino."""
    from .dataset import load_dataset

    _run_remote(lambda: load_dataset(project).annotate(label, segment=segment))


@cli.command()
@click.argument("project")
@click.argument("epochs", type=int, required=False, default=None)
@click.option(
    "--backend",
    type=click.Choice(
        ["ultralytics", "rtmdet", "libreyolo"], case_sensitive=False
    ),
    default="ultralytics",
    show_default=True,
    help="Training backend; LibreYOLO supports detection only.",
)
@click.option(
    "--imgsz",
    type=int,
    help="Training image size in pixels.",
)
@click.option(
    "--model-size",
    type=click.Choice(["small", "medium", "large"], case_sensitive=False),
    default="small",
    show_default=True,
    help="Model size to train.",
)
@click.option(
    "--publish-common",
    is_flag=True,
    help="Publish a LibreYOLO ONNX model to multiple/models/ after training.",
)
def train(project, epochs, imgsz, model_size, backend, publish_common):
    """Train a model on the specified dataset."""
    from .training import Training

    def operation():
        click.echo("Loading dataset...")
        messages = {
            "Loading model": "Training model...",
            "Validating model": "Evaluating metrics...",
            "Exporting model": "Saving model...",
        }
        reported = set()

        def report_progress(event):
            stage = event.get("stage")
            if stage in messages and stage not in reported:
                click.echo(messages[stage])
                reported.add(stage)

        result = Training(abraia).train(
            project,
            epochs,
            report_progress,
            lambda: False,
            model_size=model_size,
            image_size=imgsz,
            backend=backend,
            publish_common=publish_common,
            analyze_thresholds=False,
        )
        click.echo(json.dumps(result["stats"], indent=2))

    _run_remote(operation)


@cli.command("compile")
@click.argument("project")
@click.option(
    "--device",
    type=click.Choice(["hailo8", "hailo8l", "hailo10h", "hailo15h", "hailo15l"]),
    help="Hailo target architecture.",
    default="hailo8",
    show_default=True,
)
@click.option(
    "--conf",
    type=click.FloatRange(0, 1),
    help="Confidence threshold for the compiled model.",
)
@click.option(
    "--iou",
    type=click.FloatRange(0, 1),
    help="IoU threshold for NMS in the compiled model.",
)
def compile_model_command(project, device, conf, iou):
    """Compile a trained model to a Hailo HEF bundle."""
    from .training import Training

    def operation():
        training = Training(abraia)
        records = training.list_models(project)
        records = [
            record for record in records
            if record.get("artifacts", {}).get("checkpoint")
        ]
        if not records:
            raise click.ClickException(
                "No cloud training checkpoint is available for this project. "
                "Train and save a model before compiling it."
            )
        def version(record):
            match = re.search(r"_v(\d+)\.onnx$", record["name"])
            return int(match.group(1)) if match else 0

        record = max(records, key=version)
        model_version = version(record) or None
        click.echo("Compiling model...")
        result = training.compile(
            device,
            project,
            record,
            conf=conf,
            iou=iou,
        )
        click.echo(json.dumps(result, indent=2))

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
@click.argument("classes", required=False, default="")
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
def run(project, classes, src, accelerator, output=None):
    """Run a project's saved pipeline or its most accurate model."""
    if project == "demo":
        if classes == "faces":
            from .demo import track_faces

            track_faces(src, accelerator=accelerator, output=output)
        else:
            from .demo import monitor_objects

            monitor_objects(
                src,
                classes or "detect",
                accelerator=accelerator,
                output=output,
            )
        return
    def operation():
        from .demo import resolve_pipeline
        from .runtime import Pipeline
        from .sources import infer_source_type
        from .training.artifacts import list_model_records
        from .training.model_data import normalize_custom_models

        source_override = src if src is not None else (
            classes if classes not in ("", "faces") else None
        )
        try:
            pipeline_config = abraia.load_json(f"{project}/pipeline.json")
        except APIError as error:
            if error.code != 404:
                raise
            pipeline_config = None
        if pipeline_config is None:
            models = normalize_custom_models(
                (
                    record
                    for record in list_model_records(
                        project, client=abraia, sort_by_accuracy=True
                    )
                    if str(record["name"]).lower().endswith(".onnx")
                ),
                project=project,
                userid=abraia.userid,
            )
            if not models:
                click.echo(f"No trained model found in project '{project}'")
                return
            model = models[0]
            pipeline_config = resolve_pipeline("detect", accelerator="onnx")
            model_config = pipeline_config["steps"][0]["model"]
            model_config.update({
                "kind": model["kind"],
                "task": model["task"],
                "uri": model["uri"],
            })
            click.echo(f"No saved pipeline found; using {model['name']}")

        if source_override is not None:
            source_config = pipeline_config.setdefault("source", {})
            source_config["src"] = source_override
            inferred_type = infer_source_type(source_override)
            if inferred_type:
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
