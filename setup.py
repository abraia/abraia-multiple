import pathlib
import pkg_resources

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [str(requirement) for requirement
        in pkg_resources.parse_requirements(requirements_txt)]
    
extras_require = {
    'multiple': ['spectral>=0.23.1', 'scipy>=1.14.1', 'tifffile>=2020.9.4'],
    'dev': ['opencv-python>=4.7.0.72', 'scipy>=1.14.1']
}

setup(
    name='abraia',
    version='0.19.1',
    description='Abraia Python SDK',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/abraia/abraia-multiple',
    author='Jorge Rodriguez Araujo',
    author_email='jorge@abraiasoftware.com',
    license='MIT',
    zip_safe=False,
    packages=find_packages(exclude=['tests']),
    package_data={'': ['*.jpg']},
    include_package_data=True,
    tests_require=['pytest'],
    setup_requires=['setuptools>=38.6.0', 'pytest-runner'],
    scripts=['scripts/abraia', 'scripts/abraia.bat'],
    install_requires=install_requires,
    extras_require=extras_require
)
