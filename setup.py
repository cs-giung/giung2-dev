import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'giung2'))

from setuptools import find_packages
from setuptools import setup
from version import __version__

setup(
    name='giung2',
    version=__version__,
    description='giung2',
    author='cs-giung',
    author_email='giung@kaist.ac.kr',
    url='http://github.com/cs-giung/giung2-dev',
    license='MIT',
    packages=find_packages(),
    package_dir={'giung2': 'giung2'},
    install_requires=[],
    extras_require={
        'jax': ['jax', 'jaxlib', 'flax',],
        'tensorflow': ['tensorflow',],
        'torch': ['torch', 'torchvision',],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='machine learning',
)
