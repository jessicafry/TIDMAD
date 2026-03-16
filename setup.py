from setuptools import setup

setup(
    name='environment_setup',
    version='0.1',
    install_requires=[
        'numpy >= 2.0',
        'scipy',
        'tqdm',
        'torch >= 2.6',  # PyTorch is listed as 'torch' in PyPI
        'h5py',
        'jaxlib',
        'jax',
        'iminuit',
        'matplotlib'
    ],
)
