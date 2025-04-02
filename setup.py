from setuptools import setup, find_packages

setup(
    name="gym_race",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "matplotlib"
    ],
) 