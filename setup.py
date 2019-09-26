#!/usr/bin/env python 3.6
import os
import io
import re

from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('README.md') as f:
    long_description = f.read()

VERSION = find_version('hparams', '__init__.py')

setup_info = dict(
    # Metadata
    name='hparams',
    version=VERSION,
    author='Michael Petrochuk',
    author_email='petrochukm@gmail.com',
    url='',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    install_requires=['typeguard'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='hyperparameters hparams configurable',
    python_requires='>=3.5',

    # Package info
    packages=find_packages(exclude=['tests']),
    zip_safe=True)

setup(**setup_info)
