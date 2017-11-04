from setuptools import setup

setup(name='abraia',
      version='0.1.2',
      description='Abraia Python SDK',
      url='https://github.com/abraia/abraia-python',
      author='Jorge Rodriguez Araujo',
      author_email='jorge@abraiasoftware.com',
      license='MIT',
      packages=['abraia'],
      zip_safe=False,
      tests_require=['pytest'],
      setup_requires=['pytest-runner'],
      install_requires=['requests'])
