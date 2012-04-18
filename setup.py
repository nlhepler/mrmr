#!/usr/bin/env python

import sys

from os.path import abspath, join, split
from setuptools import setup

sys.path.insert(0, join(split(abspath(__file__))[0], 'lib'))
from mrmr import __version__ as _mrmr_version

setup(name='mrmr',
      version=_mrmr_version,
      description='minimum redundancy maximum relevance feature selection',
      author='N Lance Hepler',
      author_email='nlhepler@gmail.com',
      url='http://github.com/nlhepler/mrmr',
      license='GNU GPL version 2',
      packages=['mrmr', 'mrmr._kde'],
      package_dir={
            'mrmr': 'lib/mrmr',
            'mrmr._kde': 'lib/mrmr/_kde'
      },
      scripts=['scripts/mrmr'],
      requires=['fakemp', 'numpy']
     )
