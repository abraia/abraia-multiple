[![Build Status](https://travis-ci.org/abraia/abraia-python.svg)](https://travis-ci.org/abraia/abraia-python)

# Abraia API client for Python

Python client for the Abraia API, used for [Abraia](https://abraia.me) to transform
and optimize (compress) images on-line intelligently. Read more at [https://abraia.me/docs](https://abraia.me/docs).

## Installation

Install the API client:

```
pip install abraia
```

## Usage

```python
import abraia

abraia.from_file('original.jpg').resize(width=600, height=600).to_file('resized.jpg')
```

## License

This software is licensed under the MIT License. [View the license](LICENSE).
