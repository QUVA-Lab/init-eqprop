from setuptools import setup, find_packages

setup(
    name = "init-eqprop",
    author = "peter",
    version = 0,
    install_requires = ['numpy', 'matplotlib', 'torch', 'torchvision', 'plato', 'artemis'],
    dependency_links = [
        "http://github.com/petered/plato/tarball/ongoing_changes_2#egg=plato",
        "http://github.com/quva-lab/artemis/tarball/peter#egg=artemis",
    ],
    scripts = [],
    packages=find_packages(),
    )
