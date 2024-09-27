import os
import json
import tempfile
import requests

from PIL import Image, ImageDraw, ImageFont

tempdir = tempfile.gettempdir()


def temporal_src(path):
    dest = os.path.join(tempdir, path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    return dest


def download_file(url):
    dest = temporal_src(os.path.basename(url))
    if not os.path.exists(dest):
        r = requests.get(url, allow_redirects=True)
        open(dest, 'wb').write(r.content)
    return dest


def load_json(src):
    with open(src, 'r') as file:
        return json.load(file)


def load_image(src):
    return Image.open(src)


def get_color(idx):
    colors = ['#D0021B', '#F5A623', '#F8E71C', '#8B572A', '#7ED321',
    '#417505', '#BD10E0', '#9013FE', '#4A90E2', '#50E3C2', '#B8E986',
    '#000000', '#545454', '#737373', '#A6A6A6', '#D9D9D9', '#FFFFFF']
    return colors[idx % (len(colors) - 1)]


def hex_to_rgb(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def render_results(img, results):
    draw = ImageDraw.Draw(img, "RGBA")
    for result in results:
        label = result.get('label')
        prob = result.get('confidence')
        color = hex_to_rgb(result.get('color', '#009BFF'))
        x, y, w, h = result.get('box', [0, 0, 0, 0])
        if result.get('polygon'):
            draw.polygon(result['polygon'], fill=(color[0], color[1], color[2], 50), outline=color, width=2)
        elif result.get('box'):
            draw.rectangle([(x, y), (x + w, y + h)], fill=(color[0], color[1], color[2], 50), outline=color, width=2)
        if (label):
            text = f"{label} {round(100 * prob, 1)}%"
            font = ImageFont.load_default()
            y = max(y - 11, 0)
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x, y), text, font=font)
    return img
