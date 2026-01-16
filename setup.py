#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Code for paper XXX",
    author="Rumaisa Chowdhury",
    author_email="r.chowdhury.0137@gmail.com",
    url="https://github.com/rumncoke-ai/Wildfire-Risk-Assement-For-Satellite-Computing",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
