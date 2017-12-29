[![Build Status](https://travis-ci.org/abraia/abraia-python.svg)](https://travis-ci.org/abraia/abraia-python)
[![Coverage Status](https://coveralls.io/repos/github/abraia/abraia-python/badge.svg?branch=develop)](https://coveralls.io/github/abraia/abraia-python?branch=develop)

# Abraia API client for Python

Python client for the Abraia API, used for [Abraia](https://abraia.me) to transform
and optimize (compress) images on-line intelligently. Read more at [https://abraia.me/docs](https://abraia.me/docs).

## Installation

Install the API client:

```
pip install -U abraia
```

## Usage

```python
import abraia

abraia.from_file('images/lion.jpg').resize(
  width=600, height=600).to_file('images/lion_600x600.jpg')
abraia.from_url('https://abraia.me/images/random.jpg').resize(
  width=600, height=400).to_file('images/random_600x400.jpg')
```

## License

This software is licensed under the MIT License. [View the license](LICENSE).
