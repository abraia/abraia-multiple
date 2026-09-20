"""Remote image search and upload helpers."""

import io
import itertools
import os
import re
import tempfile
import urllib

import filetype
import requests
from PIL import Image
from tqdm import tqdm

from ..utils import HEADERS
from .core import _resolve_client


GOOGLE_BASE_URL = "https://www.google.com/search?q="
GOOGLE_PICTURE_ID = (
    "&biw=1536&bih=674&tbm=isch&sxsrf="
    "ACYBGNSXXpS6YmAKUiLKKBs6xWb4uUY5gA:1581168823770"
    "&source=lnms&sa=X&ved=0ahUKEwioj8jwiMLnAhW9AhAIHbXTBMMQ_AUI3QUoAQ"
)
BING_BASE_URL = "https://www.bing.com/images/async?q="


def dhash(img, hash_size=8):
    """Return a compact difference hash for an image."""
    import numpy as np

    img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(img, dtype=float)
    diff = pixels[:, 1:] > pixels[:, :-1]
    decimal_value = 0
    for bit in diff.flatten():
        decimal_value = (decimal_value << 1) | int(bit)
    return f"{decimal_value:0{hash_size * hash_size // 4}x}"


def convert_to_jpg(src, save_output, max_size=1920):
    """Normalize a remote image into a deduplicated JPEG upload."""
    image = Image.open(src).convert("RGB")
    image.thumbnail([max_size, max_size], Image.LANCZOS)
    filename = f"{dhash(image)}.jpg"
    image.save(os.path.join(save_output, filename))
    return filename


def download_page(url):
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    return response.text


def save_image_file(
    link,
    upload_folder,
    existing_filenames=None,
    timeout=10,
    max_size=1920,
    client=None,
):
    """Download one image, normalize it, and upload it when new."""
    client = _resolve_client(client)
    response = requests.get(
        link,
        headers=HEADERS,
        allow_redirects=True,
        timeout=timeout,
    )
    response.raise_for_status()
    kind = filetype.guess(response.content)
    if not kind or not kind.mime.startswith("image"):
        raise ValueError("Invalid image, not saving")

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = convert_to_jpg(io.BytesIO(response.content), temp_dir, max_size)
        if existing_filenames is not None and filename in existing_filenames:
            return False, filename
        client.upload_file(os.path.join(temp_dir, filename), upload_folder)
        return True, filename


def scan_bing_page(html):
    """Yield image URLs embedded in one Bing response."""
    for link in re.findall(r"murl&quot;:&quot;(.*?)&quot;", html):
        yield link.replace(" ", "%20")


def search_bing(query, limit=50):
    """Yield image URLs from Bing result pages."""
    for page_counter in range(100):
        request_url = (
            BING_BASE_URL
            + urllib.parse.quote_plus(query)
            + f"&first={page_counter}&count={limit}&adlt=off"
        )
        for link in scan_bing_page(download_page(request_url)):
            yield link


def scan_google_page(html, extensions=None, timer=5000):
    extensions = extensions or {".jpg", ".jpeg", ".webp"}
    """Yield supported image URLs embedded in a Google response."""
    scanner_counter = -1
    scanner = html.find
    for _ in range(timer):
        new_line = scanner('"https://', scanner_counter + 1)
        scanner_counter = scanner('"', new_line + 1)
        buffer_end = scanner("\\", new_line + 1, scanner_counter)
        last_line = buffer_end if buffer_end != -1 else scanner_counter
        link = html[new_line + 1:last_line]
        if any(extension in link for extension in extensions):
            yield link.replace(" ", "%20")


def search_google(query):
    """Yield image URLs from Google image search."""
    request_url = GOOGLE_BASE_URL + urllib.parse.quote_plus(query) + GOOGLE_PICTURE_ID
    for link in scan_google_page(download_page(request_url)):
        yield link


def search_images(query, save_output, limit=100, callback=None, client=None):
    """Search and download images from Google and Bing."""
    seen = set()
    download_count = 0
    client = _resolve_client(client)
    try:
        files = client.list_files(save_output)[0]
        existing_filenames = {file_data["name"] for file_data in files}
    except Exception:
        existing_filenames = set()

    links = [search_google(query), search_bing(query)]
    ends = [False] * len(links)
    progress = tqdm(total=limit, desc="Downloading images") if callback is None else None
    try:
        for link_index in itertools.cycle(range(len(links))):
            try:
                link = next(links[link_index])
            except StopIteration:
                ends[link_index] = True
                if set(ends) == {True}:
                    break
                continue
            if link in seen:
                continue
            seen.add(link)
            if download_count >= limit:
                break
            try:
                uploaded, filename = save_image_file(
                    link,
                    save_output,
                    existing_filenames=existing_filenames,
                    client=client,
                )
            except Exception:
                continue
            if uploaded:
                existing_filenames.add(filename)
                download_count += 1
                if callback:
                    callback({
                        "current": download_count,
                        "total": limit,
                        "filename": filename,
                    })
                elif progress:
                    progress.set_description(f"Downloaded {filename}")
                    progress.update(1)
    finally:
        if progress:
            progress.close()
    return client.list_files(save_output)[0]


__all__ = [
    "convert_to_jpg",
    "dhash",
    "download_page",
    "save_image_file",
    "scan_bing_page",
    "scan_google_page",
    "search_bing",
    "search_google",
    "search_images",
]
