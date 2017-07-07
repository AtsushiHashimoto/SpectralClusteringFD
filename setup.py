#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages, Extension
from spectral_clustering_fd import __author__, __version__, __license__

setup(
        name             = 'spectral_clustering_fd',
        version          = __version__,
        description      = '.',
        license          = __license__,
        author           = __author__,
        author_email     = 'ahasimoto@mm.media.kyoto-u.ac.jp',
        url              = 'https://github.com/AtsushiHashimoto/spectral_clustering_fd.git',
        keywords         = 'spectral clustering, frequent direction, matrix sketch',
        packages         = find_packages(),
	include_package_data = True,
        install_requires = ['numpy','sklearn','scipy','frequent_direction'],
        )
