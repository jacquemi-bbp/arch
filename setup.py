#!/usr/bin/env python
"""
Automated Recognition and Classification of Histological layers.
Machine learning for histological annotation and quantification of cortical layers pipeline
"""
import importlib

from setuptools import find_packages, setup

spec = importlib.util.spec_from_file_location("layer_recognition.version", "layer_recognition/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    classifiers=[
        "Programming Language :: Python :: 3.11",
    ],
    description="Machine learning for histological annotation and quantification of cortical layers pipeline",
    author="Blue Brain Project, EPFL",
    version=VERSION,
    install_requires=requirements,
    packages=find_packages(),
    name="layer-recognition",
    entry_points={"console_scripts": ["pylayer_recognition=layer_recognition.app.__main__:app"]},
)
