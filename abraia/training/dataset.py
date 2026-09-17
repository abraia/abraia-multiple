import os
import sys
import re
import io
import json
import math
import urllib
import requests
import filetype
import itertools
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from PIL import Image

from ..client import Abraia
from ..utils import HEADERS, load_image, load_url, url_path
from .core import DatasetBase

abraia = Abraia()
if sys.platform == 'darwin':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

GOOGLE_BASE_URL = 'https://www.google.com/search?q='
GOOGLE_PICTURE_ID = '''&biw=1536&bih=674&tbm=isch&sxsrf=ACYBGNSXXpS6YmAKUiLKKBs6xWb4uUY5gA:1581168823770&source=lnms&sa=X&ved=0ahUKEwioj8jwiMLnAhW9AhAIHbXTBMMQ_AUI3QUoAQ'''

BING_BASE_URL = 'https://www.bing.com/images/async?q='


def dhash(img, hash_size=8):
    import numpy as np
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(img, dtype=float)
    diff = pixels[:, 1:] > pixels[:, :-1]
    hash_bits = diff.flatten()
    decimal_value = 0
    for bit in hash_bits:
        decimal_value = (decimal_value << 1) | int(bit)
    return f"{decimal_value:0{hash_size * hash_size // 4}x}"


def convert_to_jpg(src, save_output, max_size=1920):
    im = Image.open(src).convert('RGB')
    im.thumbnail([max_size, max_size], Image.LANCZOS)
    phash = dhash(im)
    filename = phash + '.jpg'
    im.save(os.path.join(save_output, filename))
    return filename


def download_page(url):
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.text


def save_image_file(
    link,
    upload_folder,
    existing_filenames=None,
    timeout=10,
    max_size=1920,
    client=None,
):
    client = abraia if client is None else client
    resp = requests.get(link, headers=HEADERS, allow_redirects=True, timeout=timeout)
    resp.raise_for_status()
    kind = filetype.guess(resp.content)
    if kind and kind.mime.startswith('image'):
        d = io.BytesIO(resp.content)
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = convert_to_jpg(d, temp_dir, max_size)
            if existing_filenames is None or filename not in existing_filenames:
                local_path = os.path.join(temp_dir, filename)
                client.upload_file(local_path, upload_folder)
                return True, filename
            return False, filename
    else:
        raise ValueError(f'Invalid image, not saving')


def scan_bing_page(html):
    links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
    for link in links:
        link = link.replace(" ", "%20")
        yield link


def search_bing(query, limit=50):
    for page_counter in range(100):
        request_url = BING_BASE_URL + urllib.parse.quote_plus(query) \
                        + '&first=' + str(page_counter) + '&count=' + str(limit) + '&adlt=off'
        html = download_page(request_url)
        for link in scan_bing_page(html):
            yield link


def scan_google_page(html, extensions={'.jpg', '.jpeg', '.webp'}, timer=5000):
    """Scans for pictures to download based on the keywords"""
    SCANNER_COUNTER = -1
    scanner = html.find
    for _ in range(timer):
        new_line = scanner('"https://', SCANNER_COUNTER + 1)  # How Many New lines
        SCANNER_COUNTER = scanner('"', new_line + 1)  # Ends of line
        buffor = scanner('\\', new_line + 1, SCANNER_COUNTER)
        last_line = buffor if buffor != -1 else SCANNER_COUNTER
        link = html[new_line + 1:last_line]
        if any(extension in link for extension in extensions):
            link = link.replace(" ", "%20")
            yield link


def search_google(query):
    request_url = GOOGLE_BASE_URL + urllib.parse.quote_plus(query) + GOOGLE_PICTURE_ID
    html = download_page(request_url)
    for link in scan_google_page(html):
        yield link


def search_images(query, save_output, limit=100, callback=None, client=None):
    """Search and download images from Google and Bing."""
    seen = set()
    download_count = 0
    client = abraia if client is None else client
    try:
        files = client.list_files(save_output)[0]
        existing_filenames = {f['name'] for f in files}
    except Exception:
        existing_filenames = set()

    links = [search_google(query), search_bing(query)]
    ends = [False] * len(links)
    
    pbar = tqdm(total=limit, desc="Downloading images") if callback is None else None
    
    for id in itertools.cycle(range(len(links))):
        try:
            link = next(links[id])
            if link not in seen:
                seen.add(link)
                if download_count < limit:
                    try:
                        uploaded, filename = save_image_file(
                            link,
                            save_output,
                            existing_filenames=existing_filenames,
                            client=client,
                        )
                        if uploaded:
                            existing_filenames.add(filename)
                            download_count += 1
                            if callback:
                                callback({'current': download_count, 'total': limit, 'filename': filename})
                            elif pbar:
                                pbar.set_description(f"Downloaded {filename}")
                                pbar.update(1)
                    except Exception:
                        pass
                else:
                    break
        except StopIteration:
            ends[id] = True
            if set(ends) == {True}:
                break
    
    if pbar:
        pbar.close()
        
    return client.list_files(save_output)[0]


def download_file(path, folder, client=None):
    client = abraia if client is None else client
    dest = os.path.join(folder, os.path.basename(path))
    if not os.path.exists(dest):
        client.download_file(path, dest)
    return dest


def list_datasets(client=None):
    client = abraia if client is None else client
    folders = client.list_files()[1]
    def has_annotations(folder):
        try:
            return client.check_file(f"{folder['name']}/annotations.json")
        except Exception:
            return False

    if not folders:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(folders))) as executor:
        valid = executor.map(has_annotations, folders)
        return [folder['name'] for folder, is_valid in zip(folders, valid) if is_valid]


def list_models(project, client=None):
    client = abraia if client is None else client
    files = client.list_files(f"{project}/")[0]
    return [f['name'] for f in files if f['name'].endswith('.onnx')]


class Annotator:
    def __init__(self, model="IDEA-Research/grounding-dino-tiny", segment=False):
        if sys.platform == 'darwin':
            # Prevent native thread-pool contention when annotation follows
            # local PyTorch/Ultralytics training in the same process.
            import torch
            torch.set_num_threads(1)
        from transformers import pipeline
        self.pipe = pipeline(task="zero-shot-object-detection", model=model)
        self.segment_enabled = segment
        if self.segment_enabled:
            from abraia.inference.models.sam import SAM
            self.sam = SAM()

    def detect(self, img, classes, threshold=0.3):
        classes = [label.lower().strip() for label in classes]
        labels = [f"{label}." if not label.endswith('.') else label for label in classes]
        results = self.pipe(Image.fromarray(img), candidate_labels=labels, threshold=threshold)
        objects = []
        for result in results:
            score = result["score"]
            if score > threshold:
                label = result["label"].rpartition('.')[0]
                xmin, ymin, xmax, ymax = result['box'].values()
                objects.append({"label": label, "score": score, "box": [xmin, ymin, xmax - xmin, ymax - ymin]})
        return objects

    def segment(self, img, objects):
        from abraia.inference.postprocess.masks import mask_to_polygon

        self.sam.encode(img)
        for result in objects:
            x, y, w, h = result['box']
            mask = self.sam.predict(img, prompt=json.dumps([{"type": "rectangle", "data": [x, y, x+w, y+h]}]))
            height, width = mask.shape[:2]
            left = max(0, min(width, math.floor(x)))
            top = max(0, min(height, math.floor(y)))
            right = max(left, min(width, math.ceil(x + w)))
            bottom = max(top, min(height, math.ceil(y + h)))
            result['polygon'] = mask_to_polygon(
                mask[top:bottom, left:right], (left, top)
            )
        return objects

def annotate_image(image_data, classes, segment=False, annotator=None,
                   include_empty=False):
    """Load and annotate one image row.

    Keeping image transport and annotation in one primitive prevents the SDK,
    batch helper, and Studio service from drifting apart.
    """
    annotator = annotator or Annotator(segment=segment)
    url, filename = image_data["url"], image_data["name"]
    img = load_image(load_url(url))
    objects = annotator.detect(img, classes)
    if objects and segment:
        try:
            objects = annotator.segment(img, objects)
        except Exception:
            objects = None
    if not objects and not include_empty:
        return None
    return {"url": url, "filename": filename, "objects": objects}
class Dataset(DatasetBase):

    def __init__(self, project, client=None):
        super().__init__(project)
        self.client = abraia if client is None else client

    def load(self, validate=True):
        if validate:
            available_projects = (
                list_datasets()
                if self.client is abraia
                else list_datasets(self.client)
            )
            if self.project not in available_projects:
                self.annotations = []
                self.images = []
                self.classes = []
                self.task = ""
                self._update_annotated()
                return self
        with ThreadPoolExecutor(max_workers=2) as executor:
            annotations_future = executor.submit(self._load_annotations, self.project)
            images_future = executor.submit(self._list_images, self.project)
            self.annotations = annotations_future.result()
            self.images = images_future.result()
        self.classes, self.task = self._process_annotations(self.annotations)
        self._update_annotated()
        return self
    
    def _load_annotations(self, project):
        annotations = self.client.load_json(f"{project}/annotations.json")
        for annotation in annotations:
            annotation['path'] = f"{project}/{annotation['filename']}"
            annotation['url'] = url_path(f"{self.client.userid}/{annotation['path']}")
        return annotations

    def _list_images(self, project):
        files = self.client.list_files(f"{project}/")[0]
        files = [f for f in files if f['type'] in ['image/jpeg', 'image/png']]
        for data in files:
            data['url'] = url_path(f"{self.client.userid}/{data['path']}")
        return files

    def annotate(self, label, segment=False, callback=None):
        annotated_filenames = {a['filename'] for a in self.annotations}
        images = [img for img in self.images if img['name'] not in annotated_filenames]
        annotator = Annotator(segment=segment)
        
        pbar = tqdm(images) if callback is None else None
        iterable = pbar if pbar else images
        for i, row in enumerate(iterable):
            if pbar:
                pbar.set_description(f"Annotating {row['name']}")
            filename = row['name']
            annotation = annotate_image(
                row,
                [label],
                segment=segment,
                annotator=annotator,
                include_empty=True,
            )
            self.annotations.append(annotation)
            self.save()
            if callback:
                callback({'current': i + 1, 'total': len(images), 'filename': filename})
        if pbar:
            pbar.close()
        self._update_annotated()
        return self.annotations

    def save(self):
        self.client.save_json(f"{self.project}/annotations.json", self.annotations)
        self._update_annotated()


def load_dataset(project, validate=True, client=None):
    return Dataset(project, client=client).load(validate=validate)
