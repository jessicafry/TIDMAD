from setuptools import setup

setup(
    name='environment_setup',
    version='0.1',
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'torch'  # PyTorch is listed as 'torch' in PyPI
    ],
)