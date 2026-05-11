import os
import re
import io
import urllib
import requests
import filetype
import itertools

from tqdm import tqdm
from PIL import Image
from io import BytesIO

from ..client import Abraia
from ..utils import HEADERS, load_image, load_url, list_dir, url_path
from ..inference.detect import segment_objects


abraia = Abraia()

GOOGLE_BASE_URL = 'https://www.google.com/search?q='
GOOGLE_PICTURE_ID = '''&biw=1536&bih=674&tbm=isch&sxsrf=ACYBGNSXXpS6YmAKUiLKKBs6xWb4uUY5gA:1581168823770&source=lnms&sa=X&ved=0ahUKEwioj8jwiMLnAhW9AhAIHbXTBMMQ_AUI3QUoAQ'''

BING_BASE_URL = 'https://www.bing.com/images/async?q='


def convert_to_jpg(src, output_dir, max_size=1920):
    import imagehash
    im = Image.open(src).convert('RGB')
    im.thumbnail([max_size, max_size], Image.LANCZOS)
    phash = str(imagehash.phash(im))
    im.save(os.path.join(output_dir, phash + '.jpg'))


def download_page(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text


def save_image_file(link, output_dir, timeout=10, max_size=1920):
    resp = requests.get(link, headers=HEADERS, allow_redirects=True, timeout=timeout)
    kind = filetype.guess(resp.content)
    if kind and kind.mime.startswith('image'):
        d = io.BytesIO(resp.content)
        convert_to_jpg(d, output_dir, max_size)
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


def download(query, limit=100, output_dir='dataset', verbose=True):
    seen = set()
    download_count = 0
    os.makedirs(output_dir, exist_ok=True)
    links = [search_google(query), search_bing(query)]
    ends = [False] * len(links)
    for id in itertools.cycle(range(len(links))):
        try:
            link = next(links[id])
            if link not in seen:
                seen.add(link)
                if download_count < limit:
                    try:
                        save_image_file(link, output_dir)
                        download_count += 1
                        if verbose:
                            print(f"[%] Downloaded Image #{download_count} from {link}")
                    except Exception as e:
                        print(f"[!] Error getting {link}: {e}")
                else:
                    break
        except StopIteration:
            ends[id] = True
            if set(ends) == {True}:
                break


def search_images(query, limit=100, output_dir='dataset', verbose=True):
    """Search and download images from Google and Bing."""
    download(query, limit=limit, output_dir=output_dir,  verbose=verbose)
    return list_dir(output_dir)


def download_file(path, folder):
    dest = os.path.join(folder, os.path.basename(path))
    if not os.path.exists(dest):
        abraia.download_file(path, dest)
    return dest


def detect_ollama(img, label, model='gemma4:e2b'):
    import ollama
    
    im = Image.fromarray(img)
    with BytesIO() as buffer:
        im.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()

    prompt = f"Detect {label} and provide the bounding box coordinates [ymin, xmin, ymax, xmax]."
    response = ollama.generate(model=model, prompt=prompt, images=[image_bytes])

    # Gemma4 returns coordinates in 0-1000 scale: [ymin, xmin, ymax, xmax]
    matches = re.findall(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', response['response'])
    
    height, width = img.shape[:2]
    objects = []
    for match in matches:
        ymin, xmin, ymax, xmax = map(float, match)
        # Assuming 0-1000 normalized coordinates from moondream
        xmin, ymin, xmax, ymax = round(xmin * width / 1000), round(ymin * height / 1000), round(xmax * width / 1000), round(ymax * height / 1000)
        objects.append({'label': label, 'score': 1.0, 'box': [xmin, ymin, xmax - xmin, ymax - ymin]})
    return objects


def detect_dino(img, classes, threshold=0.3):
    """Detect objects in an image using Grounding Dino."""
    from transformers import pipeline

    classes = [label.lower().strip() for label in classes]
    labels = [f"{label}." if not label.endswith('.') else label for label in classes]
    pipe = pipeline(task="zero-shot-object-detection", model="IDEA-Research/grounding-dino-tiny")
    results = pipe(Image.fromarray(img), candidate_labels=labels, threshold=threshold)

    objects = []
    for result in results:
        score = result["score"]
        if score > threshold:
            label = result["label"].rpartition('.')[0]
            xmin, ymin, xmax, ymax = result['box'].values()
            objects.append({"label": label, "score": score, "box": [xmin, ymin, xmax - xmin, ymax - ymin]})
    return objects


def annotate_images(images, classes, segment=False):
    """Annotate a dataset using Grounding Dino."""
    annotations = []
    for row in tqdm(images):
        url, filename = row['url'], row['name']
        img = load_image(load_url(url))
        objects = detect_dino(img, classes)
        if objects:
            if segment:
                try:
                    objects = segment_objects(img, objects)
                except:
                    continue
            annotation = {'url': url, 'filename': filename, 'objects': objects}
            annotations.append(annotation)
    return annotations


def list_datasets():
    folders = abraia.list_files()[1]
    return [folder['name'] for folder in folders if abraia.check_file(f"{folder['name']}/annotations.json")]


def list_models(project):
    files = abraia.list_files(f"{project}/")[0]
    return [f['name'] for f in files if f['name'].endswith('.onnx')]


def load_annotations(project):
    annotations = abraia.load_json(f"{project}/annotations.json")
    for annotation in annotations:
        annotation['path'] = f"{project}/{annotation['filename']}"
        annotation['url'] = url_path(f"{abraia.userid}/{annotation['path']}")
    return annotations


def load_labels(annotations):
    labels = []
    for annotation in annotations:
        for object in annotation.get('objects', []):
            label = object.get('label')
            if label and label not in labels:
                labels.append(label)
    return list(set(labels))


def load_task(annotations):
    classify, detect, segment = False, False, False
    for annotation in annotations:
        for object in annotation.get('objects', []):
            if 'polygon' in object:
                segment = True
            elif 'box' in object:
                detect = True
            elif 'label' in object:
                classify = True
    return 'segment' if segment else 'detect' if detect else 'classify' if classify else ''


def list_images(project):
    files = abraia.list_files(f"{project}/")[0]
    files = [f for f in files if f['type'] in ['image/jpeg', 'image/png']]
    for data in files:
        data['url'] = url_path(f"{abraia.userid}/{data['path']}")
    return files


def save_annotations(project, annotations):
    abraia.save_json(f"{project}/annotations.json", annotations)


class Dataset:
    def __init__(self, project):
        self.project = project
        self.annotations = []
        self.classes = []
        self.task = ''
        self.images = []

    def load(self):
        if self.project in list_datasets():
            self.annotations = load_annotations(self.project)
            self.classes = load_labels(self.annotations)
            self.task = load_task(self.annotations)
            self.images = list_images(self.project)
        return self
    
    def annotate(self, label, segment=False):
        annotated_filenames = {a['filename'] for a in self.annotations}
        images = [img for img in self.images if img['name'] not in annotated_filenames]
        new_annotations = annotate_images(images, [label], segment=segment)
        self.annotations.extend(new_annotations)
        return self.annotations

    def save(self, annotations=None):
        if annotations is not None:
            self.annotations = annotations
        save_annotations(self.project, self.annotations)

    def split(self):
        from sklearn.model_selection import train_test_split
        # TODO: Split dataset by classes to avoid class imbalance
        backgrounds = [annotation for annotation in self.annotations if not annotation.get('objects')]
        annotations = [annotation for annotation in self.annotations if annotation.get('objects')]
        train, test = train_test_split(annotations, test_size=0.3)
        val, test = train_test_split(test, test_size=0.5)
        train.extend(backgrounds)
        return train, val, test

        
def load_dataset(project):
    return Dataset(project).load()
