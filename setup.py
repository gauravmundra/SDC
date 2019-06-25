#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:47:11 2019
@author: Gaurav Mundra
"""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='SDCompanion',
    version='1.0',
    packages=find_packages(),
    install_requires=required,
    url='',
    license='',
    author='Gaurav Mundra',
    author_email='gauravm.67@gmail.com',
    description='text classification app',
    long_description="This is service delivery companion app for IDeaS"
)
