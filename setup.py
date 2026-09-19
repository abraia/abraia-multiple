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
    'multiple': ['spectral>=0.23.1', 'scipy>=1.14.1', 'tifffile>=2024.8.30'],
    'analysis': ['scikit-learn>=1.3.0', 'joblib>=1.3.0'],
    'gis': ['folium>=0.14.0', 'shapely>=2.0.0'],
    'dev': ['tifffile>=2024.8.30', 'ultralytics==8.3.230', 'onnx>=1.16.0', 'transformers>=4.57.1'],
    'grounding-dino': ['transformers>=4.57.1'],
    'studio': [
        'PySide6>=6.5',
        'spectral>=0.23.1',
        'tifffile>=2024.8.30',
        'scikit-learn>=1.3.0',
        'joblib>=1.3.0',
    ],
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
        '': ['*.jpg', '*.gz'],
        'multiple': [
            'models/*.onnx',
            'models/*.json',
            'models/*.txt',
            'models/clip/*.gz',
        ],
        'studio': ['assets/mdi/*.svg', 'assets/mdi/README.md'],
    },
    include_package_data=True,
    tests_require=['pytest'],
    setup_requires=['setuptools>=38.6.0'],
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'abraia=abraia.cli:main',
            'abraia-studio=studio.app:main',
        ],
    },
)
