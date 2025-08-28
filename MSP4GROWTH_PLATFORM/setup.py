#!/usr/bin/env python3

import os
from setuptools import setup, find_packages

# Read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='msp4growth',
    version='1.0.0',
    description='Marine Spatial Planning for Growth Platform',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='MSP4GROWTH Team',
    author_email='info@msp4growth.org',
    url='https://github.com/msp4growth',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'msp4growth-api=msp4growth.api.main:app',
        ],
    },
    include_package_data=True,
    package_data={
        'msp4growth': ['data/*.geojson'],
    },
)