"""Command-line interface for the Abraia SDK."""

import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import click
from tqdm import tqdm

from . import APIError, Abraia, __version__, config


abraia = Abraia()


def process_map(task, *values, desc="", max_workers=3):
    """Apply a task concurrently while reporting progress.

    Network operations use threads so workers share no copied client process
    state. Iterators such as ``itertools.repeat`` are bounded by the first
    finite input length.
    """
    if not values:
        return []
    lengths = [len(value) for value in values if hasattr(value, "__len__")]
    if not lengths:
        raise ValueError("At least one process_map input must have a length")
    total = lengths[0]
    if any(length != total for length in lengths[1:]):
        raise ValueError("All process_map iterables must have the same length")
    prepared = [
        value if hasattr(value, "__len__") else itertools.islice(value, total)
        for value in values
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = []
        with tqdm(total=total, desc=desc) as progress:
            for result in executor.map(task, *prepared):
                progress.set_postfix_str(str(result))
                progress.update(1)
                results.append(result)
        return results


def process_media(src, callback):
    """Apply a callback to an image or each frame of a video source."""
    from .runtime import Video
    from .utils import get_type, load_image, show_image

    if get_type(str(src)).startswith("image"):
        show_image(callback(load_image(src)))
        return
    with Video(src) as video:
        for frame in video:
            video.show(callback(frame))


def list_files(folder):
    return abraia.list_files(folder)


def upload_file(file, folder):
    return abraia.upload_file(file, folder)


def download_file(path, folder):
    dest = os.path.join(folder, os.path.basename(path))
    return abraia.download_file(path, dest)


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
    "--work-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Keep FastDup's intermediate files in this directory.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    help="Delete recommended remote files and update annotations.",
)
@click.option(
    "--apply-outliers",
    is_flag=True,
    help="Also delete outliers; review them carefully first.",
)
@click.option(
    "--duplicate-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.96,
    show_default=True,
)
@click.option(
    "--blur-threshold",
    type=float,
    default=None,
    help="Absolute FastDup blur score threshold; lower scores are blurrier.",
)
@click.option(
    "--blur-percentile",
    type=click.FloatRange(0.0, 1.0, min_open=True),
    default=0.05,
    show_default=True,
    help="Flag the lowest fraction of blur scores.",
)
def curate(
    project,
    work_dir,
    apply_changes,
    apply_outliers,
    duplicate_threshold,
    blur_threshold,
    blur_percentile,
):
    """Analyze and optionally prune an image dataset with FastDup."""
    from .training import FastdupAnalyzer, curate_dataset

    def operation():
        report = curate_dataset(
            project,
            work_dir=work_dir,
            apply=apply_changes,
            analyzer=FastdupAnalyzer(
                duplicate_threshold=duplicate_threshold,
                blur_threshold=blur_threshold,
                blur_percentile=blur_percentile,
                remove_outliers=apply_outliers,
            ),
        )
        click.echo(json.dumps(report.to_dict(), indent=2))

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
            [file["path"] for file in files],
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
        click.echo(json.dumps(abraia.load_metadata(path), indent=2))

    _run_remote(operation)


@cli.command("list")
def list_datasets_command():
    """List available datasets."""
    from .training import list_datasets

    _run_remote(lambda: click.echo(json.dumps(list_datasets(), indent=2)))


def convert_to_jpg(src):
    from .utils import load_image, save_image

    image = load_image(src)
    save_image(image, f"{os.path.splitext(src)[0]}.jpg")
    os.remove(src)


def anonymize(src):
    from .editing import anonymize_image
    from .utils import load_image, save_image

    save_image(anonymize_image(load_image(src)), src)


def upscale(src, threshold):
    from .editing import upscale_image
    from .utils import load_image, save_image

    image = load_image(src)
    if max(image.shape) < threshold:
        save_image(upscale_image(image), src)


def process_dataset(project, anonymize_images=False, upscale_threshold=0):
    from .utils import get_type

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
    from .training import load_dataset, search_images

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
@click.argument("label", type=str, required=False, default="")
@click.option("--segment", help="Segment objects from boxes", is_flag=True)
def annotate(project, label, segment=False):
    """Annotate a dataset using Grounding Dino."""
    if not label:
        raise click.UsageError("A label is required for annotation")
    from .training import load_dataset

    _run_remote(lambda: load_dataset(project).annotate(label, segment=segment))


@cli.command()
@click.argument("project")
@click.argument("epochs", type=int, required=False, default=None)
def train(project, epochs):
    """Train a model on the specified dataset."""
    from .training import ModelTrainer, load_dataset, prepare_dataset

    def operation():
        click.echo("Loading dataset...")
        dataset = load_dataset(project)
        prepare_dataset(dataset)
        click.echo("Training model...")
        training_session = ModelTrainer(project, dataset.task, dataset.classes)
        training_session.train(epochs)
        click.echo("Evaluating metrics...")
        click.echo(json.dumps(training_session.test(), indent=2))
        click.echo("Saving model...")
        training_session.save()

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
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="The trained Ultralytics .pt checkpoint to compile.",
)
@click.option(
    "--calibration-data",
    type=click.Path(exists=True),
    default=None,
    help="Representative dataset YAML or directory for INT8 calibration.",
)
@click.option("--fraction", type=float, default=None, help="Calibration data fraction.")
@click.option("--imgsz", type=int, default=None, help="Fixed Hailo input size.")
@click.option("--conf", type=float, default=None, help="Hailo NMS confidence threshold.")
@click.option("--iou", type=float, default=None, help="Hailo NMS IoU threshold.")
@click.option("--version", type=int, default=None, help="Existing Abraia model version.")
def compile_model_command(
    project,
    device,
    checkpoint,
    calibration_data,
    fraction,
    imgsz,
    conf,
    iou,
    version,
):
    """Compile a trained Ultralytics model to a Hailo HEF bundle."""
    from .training import ModelTrainer, load_dataset, prepare_dataset

    def operation():
        click.echo("Loading dataset...")
        dataset = load_dataset(project)
        prepare_dataset(dataset)
        click.echo("Compiling model...")
        result = ModelTrainer(
            project,
            dataset.task,
            dataset.classes,
            checkpoint=checkpoint,
        ).compile(
            device=device,
            version=version,
            calibration_data=calibration_data,
            fraction=fraction,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
        click.echo(json.dumps(result, indent=2))

    _run_remote(operation)


@cli.command()
@click.argument("project")
@click.argument("classes", required=False, default="")
@click.argument("src", required=False, default=None)
@click.option(
    "--accelerator",
    type=click.Choice(["auto", "onnx", "cpu", "gpu", "hailo"]),
    default="auto",
    show_default=True,
    help="Accelerator for demos; auto prefers a compatible Hailo device.",
)
def run(project, classes, src, accelerator):
    """Run a demo or a custom trained model."""
    if project == "demo":
        if classes == "faces":
            from .demo import track_faces

            track_faces(src, accelerator=accelerator)
        else:
            from .demo import monitor_objects

            monitor_objects(src, classes or "detect", accelerator=accelerator)
        return
    if project == "hailo":
        raise click.UsageError(
            "The separate Hailo demo mode was removed; use "
            "`run demo <name> --accelerator hailo`."
        )
    if classes == "search":
        from .demo import search_images

        search_images(project, query=src or "man with red shirt")
        return

    from .inference.models.detection import Model
    from .training import list_models
    from .utils import render_results

    models = list_models(project)
    if not models:
        click.echo(f"No trained model found in project '{project}'")
        return
    model = Model(f"{abraia.userid}/{project}/{models[0]}")

    def callback(image):
        return render_results(image, model.run(image))

    source = src if src is not None else (
        classes if classes not in ("", "faces", "search") else 0
    )
    process_media(source, callback)


def main():
    """Run the Abraia CLI."""
    return cli()


__all__ = [name for name in globals() if not name.startswith("_")]


if __name__ == "__main__":
    main()
