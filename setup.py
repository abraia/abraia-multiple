from setuptools import setup

setup(name='abraia',
      version='0.2.9',
      description='Abraia Python SDK',
      url='https://github.com/abraia/abraia-python',
      author='Jorge Rodriguez Araujo',
      author_email='jorge@abraiasoftware.com',
      license='MIT',
      zip_safe=False,
      packages=['abraia'],
      tests_require=['pytest'],
      setup_requires=['pytest-runner'],
      scripts=['scripts/abraia', 'scripts/abraia.bat'],
      install_requires=['requests', 'tqdm', 'future'])
