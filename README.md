[![PyPI](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
[![Build Status](https://travis-ci.org/abraia/abraia-python.svg)](https://travis-ci.org/abraia/abraia-python)
[![Coverage Status](https://coveralls.io/repos/github/abraia/abraia-python/badge.svg)](https://coveralls.io/github/abraia/abraia-python)
![Analytics](https://ga-beacon.appspot.com/UA-108018608-1/github/python?pixel)

# Abraia API client

Python client for the [Abraia API](https://abraia.me/docs/api/). Batch optimize images for web with no quality damage based on perception-driven
technology.

* Optimal image compression with our perceptual adjustment to preserve the quality and maximize the compression.
* Smart crop and resize images with our saliency and aesthetic based model which balances between content and aesthetics.

Automatically crop, resize, and compress images for web.

## Installation

The Abraia python client and CLI works in Windows, Mac, and Linux with Python 2 and 3 (python>=2.6.5), and can be installed and upgraded with a simple command:

```sh
python -m pip install -U abraia
```

Moreover, you have to configure your [ABRAIA_KEY](https://abraia.me/console/settings) as an environment variable:

```sh
export ABRAIA_KEY=api_key
```

On Windows you need to use set instead of export:

```sh
set ABRAIA_KEY=api_key
```

NOTE: To persist the configuration use your system options to set your ABRAIA_KEY environment variable and avoid to run the previous command every time you start a terminal/console session.

## Usage

The Abraia fluent API is the easiest way to resize and compress images with Python. You just need to define the source of the image, the transformation operation, and the sink for the resultant image.

For instance, to optimize a batch of JPEG images limiting the maximum size to 2000 pixels width:

```python
from glob import glob
from abraia import Abraia

abraia = Abraia()

paths = glob('images/*.jpg')
for path in paths:
    abraia.from_file(path).resize(width=2000).to_file(path+'o')
```

### Compress images

To compress an image you just need to specify the input and output paths for the image:

```python
abraia.from_file('images/lion.jpg').to_file('images/lion_o.jpg')
abraia.from_file('images/jaguar.png').to_file('images/jaguar_o.png')
```

You can also compress and image directly from an url:

```python
abraia.from_url('https://api.abraia.me/files/demo/birds.jpg').to_file('images/birds_o.jpg')
```

![Image compressed from url](https://github.com/abraia/abraia-python/raw/master/images/birds_o.jpg)

### Resize images

To resize an image you just need to specify the `width` or the `height` of the image:

```python
abraia.from_file('images/lion.jpg').resize(width=500).to_file('images/lion_500.jpg')
```

![Image lion resized](https://github.com/abraia/abraia-python/raw/master/images/lion_500.jpg)

You can also [automatically crop and resize](https://abraia.me/docs/smart-cropping) and image to change the aspect ratio specifying both `width` and `height` size parameters:

```python
abraia.from_file('images/lion.jpg').resize(width=333, height=333).to_file('images/lion_333x333.jpg')
```

![Image lion smart cropped](https://github.com/abraia/abraia-python/raw/master/images/lion_333x333.jpg)

### Convert images

To convert images to a web format (JPEG, PNG, WebP, GIF) or between these formats you just need to change the filename extension for the destination file:

```python
abraia.from_file('images/jaguar.png').to_file('jaguar.webp')
abraia.from_file('images/jaguar.png').to_file('jaguar.jpg')
```

![PNG Jaguar optimized](https://github.com/abraia/abraia-python/raw/master/images/jaguar_o.png)
![WEBP Jaguar optimized](https://github.com/abraia/abraia-python/raw/master/images/jaguar.webp)
![JPEG Jaguar optimized](https://github.com/abraia/abraia-python/raw/master/images/jaguar.jpg)

*Optimized PNG (16.1KB) vs optimized WebP (6.5KB) vs optimized JPEG (14.4KB)*

### Watermark images

Using templates images can be easily edited and consistently branded. You just need to [create a template in the web editor](https://abraia.me/console/editor) to watermark your images.

```python
abraia.from_file('images/lion.jpg').process({'action': 'test.atn'}).resize(width=333).to_file('images/lion_brand.jpg')
```

![Branded lion](https://github.com/abraia/abraia-python/raw/master/images/branded.jpg)

As a result you get a perfectly branded and optimized image ready to be used on your website, ecommerce, marketplace, or social media.

## License

This software is licensed under the MIT License. [View the license](LICENSE).
