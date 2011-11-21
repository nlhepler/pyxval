#!/usr/bin/env python

import sys

from os.path import abspath, join, split
from setuptools import setup

sys.path.insert(0, join(split(abspath(__file__))[0], 'lib'))
from pyxval import __version__ as _pyxval_version

setup(name='pyxval',
      version=_pyxval_version,
      description='Machine learning utilities',
      author='N Lance Hepler',
      author_email='nlhepler@gmail.com',
      url='http://github.com/nlhepler/pyxval',
      license='GNU GPL version 3',
      packages=['pyxval', 'pyxval._tests'],
      package_dir={
            'pyxval': 'lib/pyxval',
            'pyxval._tests': 'lib/pyxval/_tests'
     },
     )
