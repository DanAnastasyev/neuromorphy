from setuptools import find_packages, setup

setup(
    name="neuromorphy",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.12.1',
        'dawg >= 0.7.8',
        'tensorflow >= 1.9',
        'gensim >= 3.4.0',
        'attr >= 0.3.1',
    ],
)