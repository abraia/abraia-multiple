[![Build Status](https://github.com/abraia/abraia-python/actions/workflows/build.yml/badge.svg)](https://github.com/abraia/abraia-python/actions/workflows/build.yml)
[![Python Package](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
[![Coverage Status](https://coveralls.io/repos/github/abraia/abraia-python/badge.svg)](https://coveralls.io/github/abraia/abraia-python)
![Analytics](https://ga-beacon.appspot.com/UA-108018608-1/github/python?pixel)

# Abraia API client

Automatically crop, resize, convert, and compress images for web.

> No more complex ImageMagick parametrizations. Simple convert and resize your master images from the command line and get perfectly optimized images for web and mobile apps.

Batch resize and [optimize images](https://abraia.me/docs/image-optimization/) with no quality damage based on perception-driven technology.

* Automatically crop and resize images to an specific size with smart cropping technology (our saliency and aesthetic based model balances between content and aesthetics).
* Convert SVGs to PNG or WebP images preserving the transparent background, or add a color to generate JPEG or WebP images.
* Automatically optimize image compression with our perceptual adjustment to preserve the quality and maximize the compression.

Choose an specific size, select the background color, and convert and optimize images for web (JPEG, PNG, WebP).

## Abraia command line

The Abraia CLI tool provides a simple way to bulk resize, convert, and optimize your images and photos for web. Enabling the conversion from different input formats to get images in the right formats to be used in the web - JPEG, WebP, or PNG -. Moreover, it supports a number of transformations that can be applied to image batches. So you can easily convert your images to be directly published on the web.

For instance, you can optimize all the images in a folder with the simple command bellow:

```sh
abraia convert images
```

![batch resize command](https://github.com/abraia/abraia-python/raw/master/images/batch-resize-command.gif)

### Installation

The Abraia CLI is a multiplatform tool (Windows, Mac, Linux) based on Python (Python 2.6.5 or higher), that can be be installed with a simple command:

```sh
python -m pip install -U abraia
```

If you are on Windows install [Python](https://www.python.org/downloads/) first, otherwise open a terminal or command line and write the previous command to install or upgrade the Abraia CLI.

The first time you run Abraia CLI you need to configure your API key, just write the command bellow and paste your key.

```sh
abraia configure
```

Now, you are ready to bulk resize and convert your images for web.

### Resize images

To compress an image you just need to specify the input and output paths for the image:

```sh
abraia convert images/birds.jpg images/birds_o.jpg
```

![Image compressed from url](https://github.com/abraia/abraia-python/raw/master/images/birds_o.jpg)

To resize and optimize and image maintaining the aspect ratio is enough to specify the `width` or the `height` of the new image:

```sh
abraia convert --width 500 images/usain-bolt.jpeg images/usaint-bolt_500.jpeg
```

![Usain Bolt resized](https://github.com/abraia/abraia-python/raw/master/images/usaint-bolt_500.jpeg)

You can also automatically change the aspect ratio specifying both `width` and `height` parameters and setting the resize `mode` (pad, crop, thumb):

```sh
abraia convert --width 333 --height 333 --mode pad images/lion.jpg images/lion_333x333.jpg
abraia convert --width 333 --height 333 images/lion.jpg images/lion_333x333.jpg
```

![Image lion smart cropped](https://github.com/abraia/abraia-python/raw/master/images/lion_333x333_pad.jpg)
![Image lion smart cropped](https://github.com/abraia/abraia-python/raw/master/images/lion_333x333.jpg)

So, you can automatically resize all the images in a specific folder preserving the aspect ration of each image just specifying the target `width` or `height`:

```sh
abraia convert --width 300 [path] [dest]
```

Or, automatically pad or crop all the images contained in the folder specifying both `width` and `height`:

```sh
abraia convert --width 300 --height 300 --mode crop [path] [dest]
```

![beauty casual resized](https://github.com/abraia/abraia-python/raw/master/images/beauty-casual_333x500.jpg)
![beauty casual smart cropped](https://github.com/abraia/abraia-python/raw/master/images/beauty-casual_500x500.jpg)

### Convert images

The JPEG image format is still the most common format to publish photos on the web. However, converting images to WebP provides a significant improvement for web publishing.

To convert images to a web format (JPEG, PNG, WebP) or between these formats you just need to change the filename extension for the destination file:

```sh
abraia convert garlic.jpg garlic.webp
```

<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/garlic.jpg" alt="garlic jpeg">
<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/garlic.webp" alt="garlic webp">

In addition, you can also convert SVG and PSD files. For instance, converting a SVG to PNG is so simple as to type the command bellow:

```sh
abraia convert bat.svg bat.png
```

<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/bat.svg" alt="bat svg">
<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/bat.png" alt="bat png">

> The SVG vector image is rendered in a Chrome instance to provide maximum fidelity, and preserving the transparent background.

Moreover, you can easily convert a PSD file (the layered image file used in Adobe Photoshop for saving data) flattening all the visible layers with a command like bellow:

```sh
abraia convert strawberry.psd strawberry.jpg
abraia convert strawberry.psd strawberry.png
```

<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/strawberry.jpg" alt="white background strawberry">
<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/strawberry.png" alt="transparent strawberry">

> When the PSD file is converted to JPEG a white background is added automatically, because the JPEG format does not support transparency. Instead, using the PNG or the WebP format you can preserve the transparent background.

Or, convert a batch of Photoshop files with a simple command. Just copy your PSD files to a folder, for instance the `photoshop` folder, and convert all the files in that folder.

```sh
abraia convert photoshop
```

You can also take web from the command line just specifying and url to get the capture.

```sh
abraia convert https://abraia.me screenshot.jpg
```

### Watermark images

Using templates images can be easily edited and consistently branded. You just need to [create a template in the web editor](https://abraia.me/console/editor) to watermark your images.

```sh
abraia convert --width 333 --action 'test.atn' lion.jpg lion_brand.jpg
```

![Branded lion](https://github.com/abraia/abraia-python/raw/master/images/branded.jpg)

As a result you get a perfectly branded and optimized image ready to be used on your website, ecommerce, marketplace, or social media.

## Abraia python API

The Abraia python API provides and easy way to automate image transformations. For instance, to optimize a batch of JPEG images limiting the maximum size to 2000 pixels width:

```python
from glob import glob
from abraia import Abraia

abraia = Abraia()

paths = glob('images/*.jpg')
for path in paths:
    res = abraia.upload(path)
    abraia.transform(res['path'], path+'o', {'width': 2000})
```

You can directly start from your browser with one of the notebooks available. Just click on notebook link bellow:

* Getting started [![Getting started](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-python/blob/master/notebooks/started.ipynb)

## Configuration

Installed the package, you have to configure your [ABRAIA KEY](https://abraia.me/console/settings) as environment variable:

```sh
export ABRAIA_KEY=api_key
```

On Windows you need to use `set` instead of `export`:

```sh
set ABRAIA_KEY=api_key
```

NOTE: To persist the configuration use your system options to set your ABRAIA_KEY environment variable and avoid to run the previous command every time you start a terminal/console session.

## Usage

Abraia provides a direct interface to directly load and save images. You can easily load the image data and the file metadata, or save a new image.

```python
from abraia import Abraia

abraia = Abraia()
f = abraia.load('test.jpg')
meta = abraia.metadata('test.jpg')
abraia.save('test.jpg', f)
```

You can directly visualize the image using Matplotlib.

```python
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

img = np.asarray(Image.open(f))

plt.figure()
plt.title('Image')
plt.imshow(img)
plt.axis('off')
plt.show()
```

### List files

Retrieve information about all the files stored in a cloud folder.

```python
folder = ''
files, folders = abraia.list(folder)
```

Return the list of `files` and `folders` on the specified cloud `folder`.

### Upload files

Upload a local (`src`) or a remote (`url`) file to the cloud.

```python
src = 'test.png'
path = 'test/test.png'
abraia.upload(src, path)
```

This creates the resource on the cloud and returns the file data, when the process is finished.

```python
url = 'http://upload.wikimedia.org/wikipedia/commons/1/13/Usain_Bolt_16082009_Berlin.JPG'
abraia.upload(url, 'usain.jpg')
```

### Download files

Retrieve an stored file.

```python
path = 'test/birds.jpg'
dest = 'birds.jpg'
abraia.download(path, dest)
```

### Delete files

Delete a stored resource specified by its `path`.

```python
abraia.delete(path)
```

### Transform images

Transform and optimize images. The service will automatically choose every compression parameter to provide the best result based on the perceived analysis of the original image.

```python
path = 'test/birds.jpg'
dest = 'birds_o.jpg'
params = {'width': 300, 'height': 300, 'mode': 'pad'}
abraia.transform(path, dest, params)
```

Parameter | Description
----------|------------
width | Image width (original width by default)
height | Image height (original height by default)
mode | Resize and crop mode: crop, face, thumb, resize (smart crop by default)
background | Change background color in padded mode (white by default)
action | Path to the action file to be used as template
format | Set the image format: jpeg, png, gif, webp (original format by default)
quality | Set the image quality (auto by default)

#### Legacy filters

Example | Parameters | Description
--------|------------|------------
<img src="https://github.com/abraia/abraia-python/raw/master/images/building-wall-house-architecture_500.jpg" alt="building wall house architecture original" /> | | Original building wall house image.
<img src="https://github.com/abraia/abraia-python/raw/master/images/building-wall-house-architecture_500_cbalance.jpg" alt="house color balanced" /> | <code>f=cbalance</code> | Applies a simplest color balance.
<img src="https://github.com/abraia/abraia-python/raw/master/images/building-wall-house-architecture_500_ibalance.jpg" alt="house intensity balanced" /> | <code>f=ibalance</code> | Applies a simplest intensity balance.
<img src="https://github.com/abraia/abraia-python/raw/master/images/building-wall-house-architecture_500_sharpen.jpg" alt="house sharpen" /> | <code>f=sharpen</code> | Applies a sharpen filter to the image.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500.jpg" alt="beach bungalow original" /> | | Original beach bungalow image
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_blur.jpg" alt="beach blur filter" /> | <code>f=blur</code> | Description: Applies a Gaussian blur filter to the image.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_pixelate.jpg" alt="beach pixelate filter" /> | <code>f=pixelate</code> | Applies a pixelizer filter to the image.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_grayscale.jpg" alt="beach grayscale filter" /> | <code>f=grayscale</code> | Converts the image to grayscale.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_desaturate.jpg" alt="beach desaturate filter" /> | <code>f=desaturate</code> | Desaturates the image.</p>
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_brighten.jpg" alt="beach brighten filter" /> | <code>f=brighten</code> | Applies a brighten effect to the image.</p>
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_contrast.jpg" alt="beach contrast filter" /> | <code>f=contrast</code> | Applies a contrast effect to the image.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_sepia.jpg" alt="beach sepia filter" /> | <code>f=sepia</code> | Applies a sepia effect.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_sunlight.jpg" alt="beach sunlight filter" /> | <code>f=sunlight</code> | Applies a sunlight effect to the image.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_lumo.jpg" alt="beach lumo filter" /> | <code>f=lumo</code> | Applies a lumo effect to the image.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_country.jpg" alt="beach country filter" /> | <code>f=country</code> | Applies a country effect to the image.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_sketch.jpg" alt="beach sketch filter" /> | <code>f=sketch</code> | Applies a sketch effect to the image.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_crossprocess.jpg" alt="beach crossprocess filter" /> | <code>f=crossprocess</code> | Applies the crossprocess film effect filter.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_velviaesque.jpg" alt="beach velviaesque filter" /> | <code>f=velviaesque</code> | Applies the velviaesque film effect filter.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_proviaesque.jpg" alt="beach proviaesque filter" /> | <code>f=proviaesque</code> | Applies the proviaesque film effect filter.
<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500_portraesque.jpg" alt="beach portraesque filter" /> | <code>f=portraesque</code> | Applies the portraesque film effect filter.
<img src="https://github.com/abraia/abraia-python/raw/master/images/pexels-photo-289224_500_blur-faces.jpeg" alt="anonymized couple picture" /> | <code>atn=blur-faces</code> | Anonymize pictures using Abraia's face detection feature.

## License

This software is licensed under the MIT License. [View the license](LICENSE).
