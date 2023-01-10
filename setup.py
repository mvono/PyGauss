from setuptools import setup
import os
import sys

path_requirements = 'requirements.txt'
list_packages = [
    'pygauss',
]

with open(path_requirements) as f:
    required = f.read().splitlines()

setup(
    name='pygauss',
    version='0.1.0',
    packages=list_packages,
    python_requires='>=3.6, <4',
    install_requires=required,
)
