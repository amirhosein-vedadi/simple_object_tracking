# setup.py
from setuptools import setup, find_packages

setup(
    name="simple_tracker",
    version="0.1",
    author="Amirhosein Vedadi",
    author_email="amirhsein.vedadi@gmail.com",
    description="simple object tracker in python",
    packages=find_packages(), 
    install_requires=[
        line.strip() for line in open('requirements.txt').readlines()
    ],
)