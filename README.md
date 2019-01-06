[![PyPI](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
[![Build Status](https://travis-ci.org/abraia/abraia-python.svg)](https://travis-ci.org/abraia/abraia-python)
[![Coverage Status](https://coveralls.io/repos/github/abraia/abraia-python/badge.svg)](https://coveralls.io/github/abraia/abraia-python)

# Abraia API client for Python

Python client for the [Abraia](https://abraia.me) API, used to smartly
transform and optimize (compress) images on-line. Read more at
[https://abraia.me/docs](https://abraia.me/docs).

Optimize images for Web with no quality damage based on perception-driven
technology.

* Optimal image compression with our perceptual adjustment to preserve the
quality and maximize the compression.
* Smart crop and resize images with our saliency and aesthetic based model
which balances between content and aesthetics.

```
abraia optimize --width 800 --height 400 https://images.pexels.com/photos/700948/pexels-photo-700948.jpeg images/skater.jpg
```

![Optimized and smart cropped skater](https://github.com/abraia/abraia-python/raw/master/images/skater.jpg)

The example takes a 10.1MB [image by Willian Was from Pexels](https://www.pexels.com/photo/f-s-flip-700948/)
with a size of 4865x3321 pixels and automatically generates a 94.4kB header of
800x400 pixels, cropping, resizing, and optimizing the image to directly be
used on Web.

## Installation

The Abraia python client works in Windows, Mac, and Linux with Python 2 and 3
(python>=2.6.5).

Install the API client and the CLI with a simple command:

```sh
pip install -U abraia
```

Verify that the abraia CLI is correctly installed:

```sh
abraia --version
```

Finally, configure your API Keys using the command bellow:

```sh
abraia configure
```

You can [create a free account](https://abraia.me/login) to get your API Keys.

## Command line interface

With the CLI tool you can optimize and resize images by batches.

You can easily compress a folder of images with a simple command:

```sh
abraia optimize images
```

![Batch output](https://github.com/abraia/abraia-python/raw/master/images/batch_output.png)

To resize an image you just need to specify the `width` or the `height` of
the image:

```sh
abraia optimize --width 500 images/lion.jpg images/resized.jpg
```

![Resized lion](https://github.com/abraia/abraia-python/raw/master/images/resized.jpg)

To [automatically crop and resize](https://abraia.me/docs/smart-cropping)
specify both the `width` and `height` size parameters:

```sh
abraia optimize --width 333 --height 333 images/lion.jpg images/cropped.jpg
```

![Smart cropped lion](https://github.com/abraia/abraia-python/raw/master/images/cropped.jpg)

To filter and image specify some of the [available filters](https://abraia.me/docs/image/filters):

```sh
abraia optimize --width 333 --height 333 --filter desaturate images/lion.jpg images/filtered.jpg
```

![Filtered lion](https://github.com/abraia/abraia-python/raw/master/images/filtered.jpg)

Moreover, images can be converted from one format to another changing the
filename extension for the destination file.

```sh
abraia optimize images/jaguar.png images/jaguar.jpg
```

## Fluent API

Abraia fluent API is the easiest way to compress and transform images with
python. You just need to define the source of the image, the transformation
operation, and the sink for the resultant image.

```python
import abraia

abraia.from_file('images/bird.jpeg').resize(
    width=375, height=375).to_file('images/bird_375x375.jpg')

abraia.from_url('https://api.abraia.me/files/demo/birds.jpg').resize(
    width=750).to_file('images/birds_750.jpg')

abraia.from_store('demo/birds.jpg').resize(
    width=375, height=375).to_file('images/birds_375x375.jpg')
```

![Smart croppend bird](https://github.com/abraia/abraia-python/raw/master/images/bird_375x375.jpeg)
![Smart cropped birds](https://github.com/abraia/abraia-python/raw/master/images/birds_375x375.jpg)

*Smart cropped image examples*

All the operation parameters are automatically chosen to provide the best
results balancing quality and file size for a perfectly responsive website.

PNG and WebP images can significantly optimized also.

```python
abraia.from_file('images/jaguar.png').to_file('jaguar_o.jpg')
abraia.from_file('images/jaguar.png').to_file('jaguar.jpg')
```

![PNG Jaguar original](https://github.com/abraia/abraia-python/raw/master/images/jaguar.png)
![PNG Jaguar optimized](https://github.com/abraia/abraia-python/raw/master/images/jaguar_o.png)
![JPEG Jaguar optimized](https://github.com/abraia/abraia-python/raw/master/images/jaguar.jpg)

*Original PNG (45KB) vs optimized PNG (15.8KB) vs optimized JPEG (14.1KB)*

## License

This software is licensed under the MIT License. [View the license](LICENSE).
