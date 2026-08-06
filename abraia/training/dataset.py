import os
import re
import io
import json
import urllib
import requests
import filetype
import itertools

from tqdm import tqdm
from PIL import Image

from abraia.inference.ops import mask_to_polygon

from ..client import Abraia
from ..utils import HEADERS, load_image, load_url, list_dir, url_path
from .ops import train_test_split
from ..inference.sam import SAM


abraia = Abraia()

GOOGLE_BASE_URL = 'https://www.google.com/search?q='
GOOGLE_PICTURE_ID = '''&biw=1536&bih=674&tbm=isch&sxsrf=ACYBGNSXXpS6YmAKUiLKKBs6xWb4uUY5gA:1581168823770&source=lnms&sa=X&ved=0ahUKEwioj8jwiMLnAhW9AhAIHbXTBMMQ_AUI3QUoAQ'''

BING_BASE_URL = 'https://www.bing.com/images/async?q='


def convert_to_jpg(src, save_output, max_size=1920):
    import imagehash
    im = Image.open(src).convert('RGB')
    im.thumbnail([max_size, max_size], Image.LANCZOS)
    phash = str(imagehash.phash(im))
    filename = phash + '.jpg'
    im.save(os.path.join(save_output, filename))
    return filename


def download_page(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text


def save_image_file(link, upload_folder, existing_filenames=None, timeout=10, max_size=1920):
    resp = requests.get(link, headers=HEADERS, allow_redirects=True, timeout=timeout)
    kind = filetype.guess(resp.content)
    if kind and kind.mime.startswith('image'):
        d = io.BytesIO(resp.content)
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = convert_to_jpg(d, temp_dir, max_size)
            if existing_filenames is None or filename not in existing_filenames:
                local_path = os.path.join(temp_dir, filename)
                abraia.upload_file(local_path, upload_folder)
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


def search_images(query, save_output, limit=100, callback=None):
    """Search and download images from Google and Bing."""
    seen = set()
    download_count = 0
    try:
        files = abraia.list_files(save_output)[0]
        existing_filenames = {f['name'] for f in files}
    except:
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
                        uploaded, filename = save_image_file(link, save_output, existing_filenames=existing_filenames)
                        if uploaded:
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
        
    return abraia.list_files(save_output)[0]


def download_file(path, folder):
    dest = os.path.join(folder, os.path.basename(path))
    if not os.path.exists(dest):
        abraia.download_file(path, dest)
    return dest


def list_datasets():
    folders = abraia.list_files()[1]
    return [folder['name'] for folder in folders if abraia.check_file(f"{folder['name']}/annotations.json")]


def list_models(project):
    files = abraia.list_files(f"{project}/")[0]
    return [f['name'] for f in files if f['name'].endswith('.onnx')]


class Annotator:
    def __init__(self, model="IDEA-Research/grounding-dino-tiny", segment=False):
        from transformers import pipeline
        self.pipe = pipeline(task="zero-shot-object-detection", model=model)
        self.segment_enabled = segment
        if self.segment_enabled:
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
        self.sam.encode(img)
        for result in objects:
            x, y, w, h = result['box']
            mask = self.sam.predict(img, prompt=json.dumps([{"type": "rectangle", "data": [x, y, x+w, y+h]}]))
            result['polygon'] = mask_to_polygon(mask[y:y+h, x:x+w], (x, y))
        return objects

    def annotate(self, img, label, threshold=0.3):
        objects = self.detect(img, [label], threshold=threshold)
        if objects and self.segment_enabled:
            try:
                objects = self.segment(img, objects)
            except:
                return None
        return objects


class Dataset:
    def __init__(self, project):
        self.project = project
        self.annotations = []
        self.classes = []
        self.task = ''
        self.images = []

    def load(self):
        if self.project in list_datasets():
            self.annotations = self._load_annotations(self.project)
            self.classes, self.task = self._process_annotations(self.annotations)
            self.images = self._list_images(self.project)
        return self
    
    def _load_annotations(self, project):
        annotations = abraia.load_json(f"{project}/annotations.json")
        for annotation in annotations:
            annotation['path'] = f"{project}/{annotation['filename']}"
            annotation['url'] = url_path(f"{abraia.userid}/{annotation['path']}")
        return annotations

    def _process_annotations(self, annotations):
        labels = set()
        classify, detect, segment = False, False, False
        for annotation in annotations:
            for obj in annotation.get('objects', []):
                label = obj.get('label')
                if label:
                    labels.add(label)
                    classify = True
                if 'polygon' in obj:
                    segment = True
                elif 'box' in obj:
                    detect = True
        return list(labels), 'segment' if segment else 'detect' if detect else 'classify' if classify else ''

    def _list_images(self, project):
        files = abraia.list_files(f"{project}/")[0]
        files = [f for f in files if f['type'] in ['image/jpeg', 'image/png']]
        for data in files:
            data['url'] = url_path(f"{abraia.userid}/{data['path']}")
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
            url, filename = row['url'], row['name']
            img = load_image(load_url(url))
            objects = annotator.annotate(img, label)
            annotation = {'url': url, 'filename': filename, 'objects': objects}
            self.annotations.append(annotation)
            self.save()
            if callback:
                callback({'current': i + 1, 'total': len(images), 'filename': filename})
        if pbar:
            pbar.close()
        return self.annotations

    def save(self):
        abraia.save_json(f"{self.project}/annotations.json", self.annotations)

    def split(self):
        # TODO: Split dataset by classes to avoid class imbalance
        backgrounds = [annotation for annotation in self.annotations if not annotation.get('objects')]
        annotations = [annotation for annotation in self.annotations if annotation.get('objects')]
        train, test = train_test_split(annotations, test_size=0.3)
        val, test = train_test_split(test, test_size=0.5)
        train.extend(backgrounds)
        return train, val, test

        
def load_dataset(project):
    return Dataset(project).load()
