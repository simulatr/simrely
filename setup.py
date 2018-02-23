# -*- coding: utf-8 -*-
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

How to make a new release:

Update version information
> someeditor simrel/version.py

Generate package files
> python setup.py sdist
> python setup.py bdist_wheel

Upload package to PyPI
> twine upload dist/*

Tag the new release
> git tag -a vX.Y.Z -m "Tag release X.Y.Z"
> git push origin --tags
"""
from setuptools import setup


setup()
