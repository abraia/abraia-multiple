import pathlib
import pkg_resources

from setuptools import setup, find_packages

from abraia import __version__

with open('README.md') as f:
    long_description = f.read()

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [str(requirement) for requirement
        in pkg_resources.parse_requirements(requirements_txt)]
    
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
        'abraia.runtime': ['*.json'],
    },
    include_package_data=True,
    tests_require=['pytest'],
    setup_requires=['setuptools>=38.6.0'],
    install_requires=install_requires,
    extras_require={
        'genicam': ['harvesters>=1.4.3'],
    },
    entry_points={
        'console_scripts': [
            'abraia=abraia.cli:main',
        ],
    },
)
