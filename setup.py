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
    'dev': ['ultralytics>=8.4.97,<9', 'onnx>=1.16.0', 'transformers>=4.57.1'],
    'curation': ['fastdup'],
    # The Hailo Dataflow Compiler is distributed separately by Hailo and
    # cannot be declared as a normal PyPI dependency.
    'hailo': ['ultralytics>=8.4.97,<9', 'PyYAML>=6.0'],
    'grounding-dino': ['transformers>=4.57.1'],
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
    packages=find_packages(exclude=['tests', 'tests.*', 'miscode', 'miscode.*']),
    package_data={
        '': ['*.jpg'],
    },
    include_package_data=True,
    tests_require=['pytest'],
    setup_requires=['setuptools>=38.6.0'],
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'abraia=abraia.cli:main',
        ],
    },
)
