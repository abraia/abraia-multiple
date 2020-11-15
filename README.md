[![PyPI](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
[![Build Status](https://travis-ci.org/abraia/abraia-python.svg)](https://travis-ci.org/abraia/abraia-python)
[![Coverage Status](https://coveralls.io/repos/github/abraia/abraia-python/badge.svg)](https://coveralls.io/github/abraia/abraia-python)
![Analytics](https://ga-beacon.appspot.com/UA-108018608-1/github/python?pixel)

# Abraia API client

Batch optimize images for web with no quality damage based on perception-driven technology.

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

## REST API

### Introduction

There are three main endpoints to handle the API in `https://api.abraia.me`:

* The `files` endpoint (`https://api.abraia.me/files/`) handles files.
* The `images` endpoint (`https://api.abraia.me/images/`) handles image transformations.
* The `videos` endpoint (`https://api.abraia.me/videos/`) handles video transformations.

### Authentication

Authentication is performed with API Keys via HTTP Basic Auth. Provide your API Key as the basic auth username value and the API Secret as the password value. You can manage your API keys in the [console](https://abraia.me/console).

<article class="message is-danger">
  <div class="message-body">
    Do not share your secret API keys in publicly accessible areas such GitHub,
    client-side code, and so forth.
  </div>
</article>

All API requests must be made over HTTPS. Calls made over plain HTTP will fail. API requests without authentication will also fail.

Example request:

```sh
curl -u apiKey:apiSecret https://api.abraia.me/files/
```

curl uses the -u flag to pass basic auth credentials (user:password).

To test request you must use your account API keys, replacing `apiKey` and `apiSecret` with your actual API keys. Curl uses the `-u` flag to pass basic
auth credentials (apiKey:apiSecret).

### Files storage api

#### List files

Retrieve information about all the files stored in a cloud folder using the following URL:

	GET /files/{userid}/

This endpoint retrieves all the file data as a JSON structure with a list of `files` and `folders`.

##### Example

```sh
curl -u apiKey:apiSecret https://api.abraia.me/files/demo/
```

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

Upload a local or a remote file using the following URL:

   POST /files/{userid}/{path}

This URL creates the resource on the cloud and returns the signed URL to upload the file.

A JSON object specifies the `name` and `type` of the file or the remote `url`.

##### Example

```sh
curl -u apiKey:apiSecret -X POST -d '{"name": "tiger.jpg", "type": "image/jpeg"}' https://api.abraia.me/files/userid/
```

The above command returns JSON structured with the upload URL like this:

```json
{
   "uploadURL": "https://s3.eu-west-1.amazonaws.com/store.abraia.me/demo/tiger.jpg?AWSAccessKeyId=ASIAWYD4NR3V2Q3QGRHM&Content-Type=image%2Fjpeg&Expires=1539444387&Signature=PtC917pbwI33Yy%2BHLvCRUquW3Z4%3D&x-amz-security-token=FQoGZXIvYXdzEPn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDAydZFATwU34O1CCNCLvAaQESCzuGMJjBdvp6A5rLvlfGN%2BlvdqUoTm52fwBH0Pjaf3Pq7bD2gNXk34RvbRCrG1GKMyq6nEDiMuzReJdl%2F4R7uMLhr2es%2FcvvHqYqopgsYpIwQRqVfEOCi9uxuLwOmbiuDl137QkTG9LcxnI4qc%2BswakShQZkdZ1RtkJrWluWfl0qN6LVMvDVIhFb74%2BWojy8asgw%2BvtNRV%2FAv9aaCPUjzNGmsmd%2FuLU9Vc%2Fsj0CSOns%2BD3%2BQmFKPGz5D6P4ihWqxfqhzm4i23Oq93XfiGqgrA%2FKMPNP06zfL%2F3qVpJLHZRhnkJRv27jObGVa0GoKJ2WiN4F"
}
```

This `uploadURL` needs to be used to upload the file data.

```sh
curl -H 'Content-Type: image/jpeg' -T tiger.jpg 'ttps://s3.eu-west-1.amazonaws.com/store.abraia.me/demo/tiger.jpg?AWSAccessKeyId=ASIAWYD4NR3V2Q3QGRHM&Content-Type=image%2Fjpeg&Expires=1539444387&Signature=PtC917pbwI33Yy%2BHLvCRUquW3Z4%3D&x-amz-security-token=FQoGZXIvYXdzEPn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDAydZFATwU34O1CCNCLvAaQESCzuGMJjBdvp6A5rLvlfGN%2BlvdqUoTm52fwBH0Pjaf3Pq7bD2gNXk34RvbRCrG1GKMyq6nEDiMuzReJdl%2F4R7uMLhr2es%2FcvvHqYqopgsYpIwQRqVfEOCi9uxuLwOmbiuDl137QkTG9LcxnI4qc%2BswakShQZkdZ1RtkJrWluWfl0qN6LVMvDVIhFb74%2BWojy8asgw%2BvtNRV%2FAv9aaCPUjzNGmsmd%2FuLU9Vc%2Fsj0CSOns%2BD3%2BQmFKPGz5D6P4ihWqxfqhzm4i23Oq93XfiGqgrA%2FKMPNP06zfL%2F3qVpJLHZRhnkJRv27jObGVa0GoKJ2WiN4F'
```

It is also possible to upload a remote URL. For instance, an image of Usain Bolt from wikimedia.

<a href="http://upload.wikimedia.org/wikipedia/commons/1/13/Usain_Bolt_16082009_Berlin.JPG">
  <img src="https://store.abraia.me/demo/public/api/usaint-bolt_qauto.jpeg" alt="Usain Bolt from Wikimedia Commons" />
</a>

```sh
curl -u apiKey:apiSecret -X POST -d '{"url": "http://upload.wikimedia.org/wikipedia/commons/1/13/Usain_Bolt_16082009_Berlin.JPG"}' https://api.abraia.me/files/userid/
```

#### Download file

Retrieve an stored file using the following URL:

   GET /files/{userid}/{path}

Replace `{userid}` and `{path}` with your user id and file path. The API call will return an 307 status code with the signed URL redirection to download the file data.

##### Example

```sh
curl -u apiKey:apiSecret -L https://api.abraia.me/files/demo/bird.jpg -o bird.jpg
```

#### Delete file

Delete a stored resource specified by its `path`.

	DELETE /files/{userid}/{path}

#### Move file

Move a stored file from an `oldPath` to a `newPath` using the following URL:

   POST /files/{newPath}

This URL creates the new resource from the one specified in the JSON object with the `store` parameter.

##### Example

```sh
curl -u apiKey:apiSecret -X POST -d '{"store": "demo/bird.jpg"}' https://api.abraia.me/files/demo/test/bird.jpg
```

The above command returns a response as bellow:

```json
{
  "file": {
    "name": "bird.jpg",
    "source": "demo/test/bird.jpg"
  }
}
```

### Image compression api

The image compression API provides powerful algorithms to achieve the best quality results resizing and [optimizing images](timizing images](/docs/image-optimization/), in order to adapt them to the graphic design of your website or mobile application.

To retrieve an optimized image is as simple as using the following URL:

   GET /images/{userid}/{path}?q=auto

Replace `{userid}` and `{path}` with the image path and filename, and the API call will transform the image on-the-fly to return the optimized result. We chose and optimize every compression parameter to provide the best result based on the perceptive analysis of the original image.

#### Query parameters

Parameter | Description
----------|------------
format | Set the image format: jpeg, png, gif, webp (original format by default)
quality | Set the image quality (auto by default)
width | Image width (original width by default)
height | Image height (original height by default)
mode | Resize and crop mode: crop, face, thumb, resize (smart crop by default)
background | Change background color in padded mode (white by default)

#### Examples

To optimize and resize an image maintaining the aspect ratio is enough to specify the width (`w`) or the height (`h`) of the new image. `w=500` sets the width of the image to 500 pixels and maintains the aspect ratio of the image adapting the height.

```sh
curl -u apiKey:apiSecret https://api.abraia.me/images/demo/Usain_Bolt_16082009_Berlin.JPG?w=500 -o UsainBolt.jpg
```

<img src="https://store.abraia.me/demo/public/api/usaint-bolt_500.jpeg" alt="Usain Bolt from Wikimedia Commons Resized to width 500" />

To adapt the image to the required aspect ratio is enough to specify the width and height parameters. `w=500&h=500` sets the width and the height to 500 pixels changing the aspect ratio to 1:1, automatically selecting the best cropping area.

```sh
curl -u apiKey:apiSecret https://api.abraia.me/images/demo/Usain_Bolt_16082009_Berlin.JPG?w=500&h=500 -o UsainBolt.jpg
```

<img src="https://store.abraia.me/demo/public/api/usaint-bolt_500x500.jpeg" alt="Usain Bolt from Wikimedia Commons Smartly Cropped to 500x500" />

##### Image transformations

<div class="columns is-multiline">
  <div class="column is-full-mobile is-half-tablet is-one-third-desktop has-text-centered">
    <img src="https://store.abraia.me/demo/public/api/cornflower-ladybug-siebenpunkt-blue_300.jpg" alt="flowers resize width" />
    <p class="has-text-centered">Parameters: <code>w=300</code></p>
    <p>Description: Resizes the image maintaining the aspect ratio.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-one-third-desktop has-text-centered">
    <img src="https://store.abraia.me/demo/public/api/cornflower-ladybug-siebenpunkt-blue_x192.jpg" alt="flowers resize height" />
    <p class="has-text-centered">Parameters: <code>h=192</code></p>
    <p>Description: Resizes the image maintaining the aspect ratio.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-one-third-desktop has-text-centered">
    <img src="https://store.abraia.me/demo/public/api/cornflower-ladybug-siebenpunkt-blue_300x192.jpg" alt="flowers forced crop" />
    <p class="has-text-centered">Parameters: <code>w=300&h=192&m=crop</code></p>
    <p>Description: Forces the crop of the image when the aspect ratio is the original one.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-one-third-desktop has-text-centered">
    <img src="https://store.abraia.me/demo/public/api/cornflower-ladybug-siebenpunkt-blue_300x300.jpg" alt="flowers smart crop" />
    <p class="has-text-centered">Parameters: <code>w=300&h=300</code></p>
    <p>Description: Smartly crops and resize the image to adopt the new aspect ratio.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-one-third-desktop has-text-centered">
    <img src="https://store.abraia.me/demo/public/api/cornflower-ladybug-siebenpunkt-blue_300x300_resize.jpg" alt="flowers forces resize" />
    <p class="has-text-centered">Parameters: <code>w=300&h=300&m=resize</code></p>
    <p>Description: Forces the resize of the image when the aspect ratio is different from the one in the original image.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-one-third-desktop has-text-centered">
    <img src="https://store.abraia.me/demo/public/api/cornflower-ladybug-siebenpunkt-blue_300x300_ar.jpg" alt="flowers resize aspect ratio" />
    <p class="has-text-centered">Parameters: <code>w=300&ar=1:1</code> or <code>h=300&ar=1</code></p>
    <p>Description: Smartly crops and resize the image to adopt the new aspect ratio.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-one-third-desktop has-text-centered">
    <img src="https://store.abraia.me/demo/public/api/cornflower-ladybug-siebenpunkt-blue_300_ar15.jpg" alt="flowers resize aspect ratio" />
    <p class="has-text-centered">Parameters: <code>w=300&ar=1.5</code></p>
    <p>Description: Smartly crops and resize the image to adopt the new aspect ratio.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-one-third-desktop has-text-centered">
    <img src="https://store.abraia.me/demo/public/api/cornflower-ladybug-siebenpunkt-blue_150_ar15_dpr2.jpg" alt="flowers resize dpr" />
    <p class="has-text-centered">Parameters: <code>w=150&ar=1.5&dpr=2</code></p>
    <p>Description: Adapts the image for retina displays, adopting the specified device pixel ratio (1, 2, 3).</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-one-third-desktop has-text-centered">
    <img src="https://store.abraia.me/demo/public/api/cornflower-ladybug-siebenpunkt-blue_300_ar15.jpg" alt="flowers quality" />
    <p class="has-text-centered">Parameters: <code>w=300&ar=1.5&q=50</code></p>
    <p>Description: Sets the quality of the delivered jpg or webp image to 50 in the range (1, 100) - 1 is the lowest quality and 100 is the highest.</p>
  </div>
</div>

##### Enhancement filters

<div class="columns is-multiline">
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/building-wall-house-architecture_500.jpg" alt="building wall house architecture original" />
    <p class="has-text-centered"><i>Original building wall house image</i></p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/building-wall-house-architecture_500_cbalance.jpg" alt="house color balanced" />
    <p>Parameters: <code>f=cbalance</code></p>
    <p>Description: Applies a simplest color balance..</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/building-wall-house-architecture_500_ibalance.jpg" alt="house intensity balanced" />
    <p>Parameters: <code>f=ibalance</code></p>
    <p>Description: Applies a simplest intensity balance.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/building-wall-house-architecture_500_sharpen.jpg" alt="house sharpen" />
    <p>Parameters: <code>f=sharpen</code></p>
    <p>Description: Applies a sharpen filter to the image.</p>
  </div>
</div>

##### Filter effects

<img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500.jpg" alt="beach bungalow original" />
<center><i>Original beach bungalow image</i></center><br>

<div class="columns is-multiline">
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_blur.jpg" alt="beach blur filter" />
    <p>Parameters: <code>f=blur</code></p>
    <p>Description: Applies a Gaussian blur filter to the image.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_pixelate.jpg" alt="beach pixelate filter" />
    <p>Parameters: <code>f=pixelate</code></p>
    <p>Description: Applies a pixelizer filter to the image.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_grayscale.jpg" alt="beach grayscale filter" />
    <p>Parameters: <code>f=grayscale</code></p>
    <p>Description: Converts the image to grayscale.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_desaturate.jpg" alt="beach desaturate filter" />
    <p>Parameters: <code>f=desaturate</code></p>
    <p>Description: Desaturates the image.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_brighten.jpg" alt="beach brighten filter" />
    <p>Parameters: <code>f=brighten</code></p>
    <p>Description: Applies a brighten effect to the image.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_contrast.jpg" alt="beach contrast filter" />
    <p>Parameters: <code>f=contrast</code></p>
    <p>Description: Applies a contrast effect to the image.</p>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_sepia.jpg" alt="beach sepia filter" />
    <p>Parameters: <code>f=sepia</code></p>
    <p>Description: Applies a sepia effect.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_sunlight.jpg" alt="beach sunlight filter" />
    <p>Parameters: <code>f=sunlight</code></p>
    <p>Description: Applies a sunlight effect to the image.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_lumo.jpg" alt="beach lumo filter" />
    <p>Parameters: <code>f=lumo</code></p>
    <p>Description: Applies a lumo effect to the image.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_country.jpg" alt="beach country filter" />
    <p>Parameters: <code>f=country</code></p>
    <p>Description: Applies a country effect to the image.</p>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_cartoonify.jpg" alt="beach cartoonify filter" />
    <p>Parameters: <code>f=cartoonify</code></p>
    <p>Description: Applies a cartoonify effect to the image.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_sketch.jpg" alt="beach sketch filter" />
    <p>Parameters: <code>f=sketch</code></p>
    <p>Description: Applies a sketch effect to the image.</p>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_crossprocess.jpg" alt="beach crossprocess filter" />
    <p>Parameters: <code>f=crossprocess</code></p>
    <p>Description: Applies the crossprocess film effect filter.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_velviaesque.jpg" alt="beach velviaesque filter" />
    <p>Parameters: <code>f=velviaesque</code></p>
    <p>Description: Applies the velviaesque film effect filter.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
    <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_proviaesque.jpg" alt="beach proviaesque filter" />
    <p>Parameters: <code>f=proviaesque</code></p>
    <p>Description: Applies the proviaesque film effect filter.</p><br>
  </div>
  <div class="column is-full-mobile is-half-tablet is-half-desktop">
   <img src="https://store.abraia.me/demo/public/api/beach-bungalow-caribbean-jetty_500_portraesque.jpg" alt="beach portraesque filter" />
    <p>Parameters: <code>f=portraesque</code></p>
    <p>Description: Applies the portraesque film effect filter.</p>
  </div>
</div>

##### Action filters

Actions are an experimental feature to provide a powerful content-based edition tool. They are going to be developed to enable smart actions like adaptive watermarking. For instance, changing the text color based on the background color, or using the negative space to place the watermark.

<img src="https://store.abraia.me/demo/public/api/pexels-photo-289224_500_blur-faces.jpeg" alt="anonymized couple picture" />
<p class="has-text-centered">Parameters: <code>atn=blur-faces</code></p>
<p class="has-text-centered">Description: Anonymize pictures using Abraia's face detection feature.</p>

### Video optimization api

The video optimization API provides powerful algorithms to achiveve the best quality results to transcode and [optimizate videos](timizate videos](/docs/video-optimization).

Run a video transcoding task using the following URL:

   GET /videos/{userid}/{path}

Replace `{userid}` and `{path}` with the video path. The API call returns the
path where the video will be placed.

#### Query parameters

Parameter | Description
----------|------------
format | Set video format codec: mp4, webm, gif, jpeg, h264, vp9, h265, hls (mp4 by default)
quality | Set video quality CRF (23 by default)
width | Video width (original width by default)
height | Video height (original height by default)
mode | Resize and crop mode: thumb, pad, crop, blur (thumb by default)
background | Change background color in padded mode (white by default)
overlay | Path to the image to be overlayed (none by default)
mute | Remove audio when is true (false by default)
subtitles | Path to the srt subtitles file (none by default)
frame | Time in seconds to the frame from start (none by default)
from | Start time in seconds (0 by default)
to | End time in seconds (duration by default)

#### Example

```sh
curl -u apiKey:apiSecret https://api.abraia.me/videos/demo/videos/zara.mp4?format=mp4&width=600
```

The above command returns JSON structured like this:

```json
{
   "path": "demo/videos/zara/zara_600.mp4"
}
```

### Errors

Abraia uses conventional HTTP response codes to indicate the success or failure of an API request. In general, codes in the 2xx range indicate success, codes in the 4xx range indicate an error that failed given the information (e.g., a required parameter was omitted, a not found resource, etc.), and codes in the 5xx range indicate an error with Abraia's servers (these are rare).

#### HTTP status code summary

Code | Status | Description
-----|--------|------------
200 | OK. Successful | Everything worked as expected.
201 | Created | A new resource was successfully created.
307 | Temporal Redirect | The resource requested has been temporarily moved to the URL given by the Location headers.
400 | Bad Request | The request was unacceptable, often due to missing a required parameter.
401 | Unauthorized | No valid API key provided.
402 | Payment Required | No credits available.
403 | Not allowed | The service is not available for your account.
404 | Not Found | The resource requested does not exist.
500 | Server Error | Something went wrong on the service.

## License

This software is licensed under the MIT License. [View the license](LICENSE).
