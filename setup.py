import pathlib
import pkg_resources

from setuptools import setup, find_packages

from abraia import __version__

with open('README.md') as f:
    long_description = f.read()

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [str(requirement) for requirement
        in pkg_resources.parse_requirements(requirements_txt)]
    
extras_require = {
    'multiple': ['spectral>=0.23.1', 'scipy>=1.14.1', 'tifffile>=2020.9.4'],
    'dev': ['opencv-python>=4.10.0.84', 'ultralytics>=8.3.59', 'onnx>=1.18.0', 'onnxsim>=0.4.36'],
}

setup(
    name='abraia',
    version=__version__,
    description='Abraia Python SDK',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/abraia/abraia-multiple',
    author='Jorge Rodriguez Araujo',
    author_email='jorge@abraiasoftware.com',
    license='MIT',
    zip_safe=False,
    packages=find_packages(exclude=['tests']),
    package_data={'': ['*.jpg', '*.gz']},
    include_package_data=True,
    tests_require=['pytest'],
    setup_requires=['setuptools>=38.6.0', 'pytest-runner'],
    scripts=['scripts/abraia', 'scripts/abraia.bat'],
    install_requires=install_requires,
    extras_require=extras_require
)
