from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='abraia',
    version='0.6.6',
    description='Abraia Python SDK',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/abraia/abraia-python',
    author='Jorge Rodriguez Araujo',
    author_email='jorge@abraiasoftware.com',
    license='MIT',
    zip_safe=False,
    packages=['abraia'],
    tests_require=['pytest'],
    setup_requires=['setuptools>=38.6.0', 'pytest-runner'],
    scripts=['scripts/abraia', 'scripts/abraia.bat'],
    install_requires=['requests', 'click', 'tqdm', 'future']
)
