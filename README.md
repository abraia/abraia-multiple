[![PyPI](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
[![Build Status](https://travis-ci.org/abraia/abraia-python.svg)](https://travis-ci.org/abraia/abraia-python)
[![Coverage Status](https://coveralls.io/repos/github/abraia/abraia-python/badge.svg)](https://coveralls.io/github/abraia/abraia-python)
![Analytics](https://ga-beacon.appspot.com/UA-108018608-1/github/python?pixel)

# Abraia API client

Automatically crop, resize, convert, and compress images for web.

> No more complex ImageMagick parametrizations. Simple convert and resize your master images from the command line and get perfectly optimized images for web and mobile apps.

Batch resize and optimize images with no quality damage based on perception-driven technology.

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

To resize and optimize and image maintaining the aspect ratio is enough to specify the `width` or the `height` of the new image:

```sh
abraia convert --width 500 images/usain-bolt.jpeg images/usaint-bolt_500.jpeg
```

![Usain Bolt resized](https://github.com/abraia/abraia-python/raw/master/images/usaint-bolt_500.jpeg)

You can also automatically change the aspect ratio specifying both `width` and `height` parameters and setting the resize `mode` (pad, crop, thumb):

```sh
abraia convert --width 333 --height 333 images/lion.jpg images/lion_333x333.jpg
abraia convert --width 333 --height 333 --mode pad images/lion.jpg images/lion_333x333.jpg
```

![Image lion resized](https://github.com/abraia/abraia-python/raw/master/images/lion_500.jpg)
![Image lion smart cropped](https://github.com/abraia/abraia-python/raw/master/images/lion_333x333.jpg)
![Image lion smart cropped](https://github.com/abraia/abraia-python/raw/master/images/lion_333x333_pad.jpg)

### Convert images

To convert images to a web format (JPEG, PNG, WebP, GIF) or between these formats you just need to change the filename extension for the destination file:

```sh
abraia convert images/jaguar.png jaguar.webp
abraia convert images/jaguar.png jaguar.jpg
```

![PNG Jaguar optimized](https://github.com/abraia/abraia-python/raw/master/images/jaguar_o.png)
![WEBP Jaguar optimized](https://github.com/abraia/abraia-python/raw/master/images/jaguar.webp)
![JPEG Jaguar optimized](https://github.com/abraia/abraia-python/raw/master/images/jaguar.jpg)

*Optimized PNG (16.1KB) vs optimized WebP (6.5KB) vs optimized JPEG (14.4KB)*

#### Convert SVG to PNG

Converting a SVG image to PNG, now is so simple as to type the command bellow:

```sh
abraia convert bat.svg bat.png
```

<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/bat.svg" alt="bat svg">

<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/bat.png" alt="bat png">

The SVG vector image is rendered in a Chrome instance to provide maximum fidelity, and preserving the transparent background.

#### Convert image to WebP

The JPEG image format is still the most common format to publish photos on the web. However, converting images to WebP provides a significant improvement for web publishing. To convert an image to WebP just write a simple command like bellow:

```sh
abraia convert garlic.jpg garlic.webp
```

<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/garlic.jpg" alt="garlic jpeg">

<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/garlic.webp" alt="garlic webp">

The same can be used to convert a GIF animation to WebP and save several bytes.

```sh
abraia convert bob-sponge.gif bob-sponge.webp
```

<img src="https://github.com/abraia/abraia-python/raw/master/images/bob-sponge.gif" alt="bob sponge gif">

<img src="https://github.com/abraia/abraia-python/raw/master/images/bob-sponge.webp" alt="bob sponge webp">

#### Convert PSD to JPG<!-- ## Convert PSD to SVG -->

A .PSD file is a layered image file used in Adobe PhotoShop for saving data. You can easily convert then, and get the result of flattening all the visible layers with a command like bellow:

```sh
abraia convert strawberry.psd strawberry.jpg
```

The previous command just convert a PSD file to JPEG, automatically adding a white background, in this case, because the JPEG format does not support transparency. Instead, using the PNG format you can preserve the transparent background.

```sh
abraia convert strawberry.psd strawberry.png
```

<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/strawberry.jpg" alt="white background strawberry">

<img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/strawberry.png" alt="transparent strawberry">

<!--         
  Take web screenshots from the command line just specifying and ur and get the capture.
  
  Convert a batch of Photoshop files with a simple command.

  Just copy your PSD files to a folder, for instance the `photoshop` folder, and convert all the files in that folder.

  ```sh
  abraia convert photoshop
  ```
-->

### Resize and compress images for web

To compress an image you just need to specify the input and output paths for the image:

```sh
abraia convert images/birds.jpg images/birds_o.jpg
abraia convert images/jaguar.png images/jaguar_o.png
```

![Image compressed from url](https://github.com/abraia/abraia-python/raw/master/images/birds_o.jpg)

To resize your images just specify the target `width` or `height`. So, to get a set of images with a fixed width of 300px preserving the aspect ratio of each image:</p>

```sh
abraia convert --width 300 [path] [dest]
```

To automatically crop all your images contained in a folder using our smart cropping technology, you just need to specify both `width` and `height` of the output image.

```sh
abraia convert --width 300 --height 300 [path] [dest]
```

<img src="https://github.com/abraia/abraia-python/raw/master/images/beauty-casual_333x500.jpg" alt="beauty casual resized">

<img src="https://github.com/abraia/abraia-python/raw/master/images/beauty-casual_500x500.jpg" alt="beauty casual smart cropped">

### Watermark images

Using templates images can be easily edited and consistently branded. You just need to [create a template in the web editor](https://abraia.me/console/editor) to watermark your images.

```sh
abraia convert --width 333 --action 'test.atn' lion.jpg lion_brand.jpg
```

![Branded lion](https://github.com/abraia/abraia-python/raw/master/images/branded.jpg)

As a result you get a perfectly branded and optimized image ready to be used on your website, ecommerce, marketplace, or social media.

## Abraia python API

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

### Configuration

To use the API, you have to configure your [ABRAIA_KEY](https://abraia.me/console/settings) as an environment variable:

```sh
export ABRAIA_KEY=api_key
```

On Windows you need to use set instead of export:

```sh
set ABRAIA_KEY=api_key
```

NOTE: To persist the configuration use your system options to set your ABRAIA_KEY environment variable and avoid to run the previous command every time you start a terminal/console session.

### Files storage API

#### List files

Retrieve information about all the files stored in a cloud folder.

```python
folder = ''
files, folders = abraia.list(folder)
```

Return the list of `files` and `folders` on the specified cloud `folder`.

The above command returns JSON structured like this:

```json
{
   "files": [
      {
         "date": 1513537440.0,
         "md5": "9f1f07e9884c4b11b048614561eebd89",
         "name": "random_97.jpg",
         "size": 112594,
         "source": "demo/random_97.jpg",
         "thumbnail": "demo/tb_random_97.jpg"
      },
      {
         "date": 1513533727.0,
         "md5": "7ac57ee27b4474c7109e3c643948a9de",
         "name": "random_96.jpg",
         "size": 70232,
         "source": "demo/random_96.jpg",
         "thumbnail": "demo/tb_random_96.jpg"
      }
   ],
   "folders": [
      {
         "name": "videos",
         "source":"demo/videos/"
      }
   ]
}
```

#### Upload file

Upload a local (`src`) or a remote (`url`) file to the cloud.

```python
src = 'test.png'
path = 'test/test.png'
abraia.upload(src, path)
```

This creates the resource on the cloud and returns the file data, when the process is finished.

It is also possible to upload a remote URL. For instance, an image of Usain Bolt from wikimedia.

```python
url = 'http://upload.wikimedia.org/wikipedia/commons/1/13/Usain_Bolt_16082009_Berlin.JPG'
abraia.upload(url, 'usain.jpg')
```

<img src="https://github.com/abraia/abraia-python/raw/master/images/usaint-bolt_qauto.jpeg" alt="Usain Bolt from Wikimedia Commons" />

#### Download file

Retrieve an stored file.

```python
path = 'test/bird.jpg'
dest = 'bird.jpg'
abraia.download(path, dest)
```

#### Delete file

Delete a stored resource specified by its `path`.

```python
abraia.delete(path)
```

#### Move file

Move a stored file from an `old_path` to a `new_path`.

```python
abraia.move_file(old_path, new_path)
```

### Image optimization API

The image optimization API provides powerful algorithms to achieve the best quality results resizing and [optimizing images](https://abraia.me/docs/image-optimization/).

To retrieve an optimized image is as simple as using the image path and the `quality` parameter set to 'auto'. The service will automatically choose and optimize every compression parameter to provide the best result based on the perceptive analysis of the original image.

Parameter | Description
----------|------------
format | Set the image format: jpeg, png, gif, webp (original format by default)
quality | Set the image quality (auto by default)
width | Image width (original width by default)
height | Image height (original height by default)
mode | Resize and crop mode: crop, face, thumb, resize (smart crop by default)
background | Change background color in padded mode (white by default)

#### Image transformations

Image | Parameters | Description
------|------------|------------
<img src="https://github.com/abraia/abraia-python/raw/master/images/cornflower-ladybug-siebenpunkt-blue_300.jpg" alt="flowers resize width" /> | <code>width=300</code> | Resizes the image maintaining the aspect ratio.
<img src="https://github.com/abraia/abraia-python/raw/master/images/cornflower-ladybug-siebenpunkt-blue_x192.jpg" alt="flowers resize height" /> | <code>height=192</code> | Resizes the image maintaining the aspect ratio.
<img src="https://github.com/abraia/abraia-python/raw/master/images/cornflower-ladybug-siebenpunkt-blue_300x192.jpg" alt="flowers forced crop" /> | <code>width=300&height=192&mode=crop</code> | Forces the crop of the image when the aspect ratio is the original one.
<img src="https://github.com/abraia/abraia-python/raw/master/images/cornflower-ladybug-siebenpunkt-blue_300x300.jpg" alt="flowers smart crop" /> | <code>width=300&height=300</code> | Smartly crops and resize the image to adopt the new aspect ratio.
<img src="https://github.com/abraia/abraia-python/raw/master/images/cornflower-ladybug-siebenpunkt-blue_300x300_pad.jpg" alt="flowers forces resize" /> | <code>width=300&height=300&mode=pad</code> | Resize the image adding some pad to adopt the new aspect ratio.
<img src="https://github.com/abraia/abraia-python/raw/master/images/cornflower-ladybug-siebenpunkt-blue_300.jpg" alt="flowers quality" /> | <code>width=300&quality=50</code> | Sets the quality of the delivered jpg or webp image to 50 in the range (1, 100) - 1 is the lowest quality and 100 is the highest.

#### Enhancement filters

<img src="https://github.com/abraia/abraia-python/raw/master/images/building-wall-house-architecture_500.jpg" alt="building wall house architecture original" />
<center><i>Original building wall house image</i></center><br>

Example | Parameters | Description
--------|------------|------------
<img src="https://github.com/abraia/abraia-python/raw/master/images/building-wall-house-architecture_500_cbalance.jpg" alt="house color balanced" /> | <code>f=cbalance</code> | Applies a simplest color balance.
<img src="https://github.com/abraia/abraia-python/raw/master/images/building-wall-house-architecture_500_ibalance.jpg" alt="house intensity balanced" /> | <code>f=ibalance</code> | Applies a simplest intensity balance.
<img src="https://github.com/abraia/abraia-python/raw/master/images/building-wall-house-architecture_500_sharpen.jpg" alt="house sharpen" /> | <code>f=sharpen</code> | Applies a sharpen filter to the image.

#### Filter effects

<img src="https://github.com/abraia/abraia-python/raw/master/images/beach-bungalow-caribbean-jetty_500.jpg" alt="beach bungalow original" />
<center><i>Original beach bungalow image</i></center><br>

Example | Parameters | Description
--------|------------|------------
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

#### Action filters

Actions are an experimental feature to provide a powerful content-based edition tool. They are going to be developed to enable smart actions like adaptive watermarking. For instance, changing the text color based on the background color, or using the negative space to place the watermark.

<img src="https://github.com/abraia/abraia-python/raw/master/images/pexels-photo-289224_500_blur-faces.jpeg" alt="anonymized couple picture" />
<p class="has-text-centered">Parameters: <code>atn=blur-faces</code></p>
<p class="has-text-centered">Description: Anonymize pictures using Abraia's face detection feature.</p>

## License

This software is licensed under the MIT License. [View the license](LICENSE).
