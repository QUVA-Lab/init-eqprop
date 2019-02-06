from setuptools import setup, find_packages

setup(
    name = "init-eqprop",
    author = "peter",
    version = 0,
    install_requires = ['numpy', 'matplotlib', 'torch', 'torchvision', 'artemis-ml'],
    dependency_links = (),
    scripts = [],
    packages=find_packages(),
    )
