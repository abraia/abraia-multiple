import os
import re
import io
import urllib
import requests
import filetype
import itertools
import imagehash

from tqdm import tqdm
from PIL import Image
from transformers import pipeline
from ..utils import HEADERS, load_image, load_url
from . import list_datasets, list_images, load_annotations, save_annotations

GOOGLE_BASE_URL = 'https://www.google.com/search?q='
GOOGLE_PICTURE_ID = '''&biw=1536&bih=674&tbm=isch&sxsrf=ACYBGNSXXpS6YmAKUiLKKBs6xWb4uUY5gA:1581168823770&source=lnms&sa=X&ved=0ahUKEwioj8jwiMLnAhW9AhAIHbXTBMMQ_AUI3QUoAQ'''

BING_BASE_URL = 'https://www.bing.com/images/async?q='


def download_page(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text


def save_image(link, output_dir, timeout=10, max_size=1920):
    resp = requests.get(link, headers=HEADERS, allow_redirects=True, timeout=timeout)
    kind = filetype.guess(resp.content)
    if kind and kind.mime.startswith('image'):
        d = io.BytesIO(resp.content)
        # d.seek(0)
        im = Image.open(d).convert('RGB')
        im.thumbnail([max_size, max_size])
        phash = str(imagehash.phash(im))
        im.save(os.path.join(output_dir, phash + '.jpg'))
    else:
        raise ValueError(f'Invalid image, not saving')


def get_filter(shorthand):
        if shorthand == "line" or shorthand == "linedrawing":
            return "+filterui:photo-linedrawing"
        elif shorthand == "photo":
            return "+filterui:photo-photo"
        elif shorthand == "clipart":
            return "+filterui:photo-clipart"
        elif shorthand == "gif" or shorthand == "animatedgif":
            return "+filterui:photo-animatedgif"
        elif shorthand == "transparent":
            return "+filterui:photo-transparent"
        else:
            return ""
        

def scan_bing_page(html):
    links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
    for link in links:
        link = link.replace(" ", "%20")
        yield link

        
def search_bing(query, limit=50, adult='off', filter=''):
    for page_counter in range(100):
        # Parse the page source and download pics
        request_url = BING_BASE_URL + urllib.parse.quote_plus(query) \
                        + '&first=' + str(page_counter) + '&count=' + str(limit) \
                        + '&adlt=' + adult + '&qft=' + get_filter(filter)
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
    links = [search_google(query), search_bing(query, adult='off', filter='')]
    ends = [False] * len(links)
    for id in itertools.cycle(range(len(links))):
        try:
            link = next(links[id])
            if link not in seen:
                seen.add(link)
                if download_count < limit:
                    try:
                        save_image(link, output_dir)
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
    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    return files


# As the Grounding DINO model was trained with a "." after each text, we'll do the same here.
def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    return result if result.endswith('.') else result + '.'


def format_results(results, threshold=0.6):
    r = []
    for result in results:
        score = result["score"]
        if score > threshold:
            label = result["label"].rpartition('.')[0]
            xmin, ymin, xmax, ymax = result['box'].values()
            r.append({"label": label, "score": score, "box": [xmin, ymin, xmax - xmin, ymax - ymin]})
    return r


def annotate_image(pipe, img, classes, threshold=0.3):
    im = Image.fromarray(img)
    labels = [preprocess_caption(txt) for txt in classes]
    objects = format_results(pipe(im, candidate_labels=labels, threshold=threshold))
    return objects


def annotate_images(images, classes):
    """Annotate a dataset using Grounding Dino."""
    pipe = pipeline(task="zero-shot-object-detection", model="IDEA-Research/grounding-dino-tiny")
    #pipe = pipeline(task="zero-shot-object-detection", model="google/owlv2-base-patch16-ensemble")
    annotations = []
    for row in tqdm(images):
        url, filename = row['url'], row['name']
        img = load_image(load_url(url))
        objects = annotate_image(pipe, img, classes)
        if objects:
            annotation = {'url': url, 'filename': filename, 'objects': objects}
            annotations.append(annotation)
    return annotations
