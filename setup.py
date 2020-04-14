from setuptools import setup

setup(
    name='svp',
    version='0.1.0',
    author='Cody Coleman',
    author_email='cody@cs.stanford.edu',
    packages=['svp'],
    install_requires=[
        'typing_extensions',
        'click~=7.0',
        'tqdm~=4.26.0',
        'torch~=1.4.0',
        'torchvision~=0.5.0',
        'scipy~=1.4.1',
        'pandas~=1.0.3'
    ]
)
