"""
Basic set up file
"""

from setuptools import setup, find_packages

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name="battlesim",
    version="0.3.5",
    description="A python package for simulating battles and visualizing them in animation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gregparkes/battlesim",
    author="Gregory Parkes",
    author_email="g.m.parkes@soton.ac.uk",
    license="GPL-3.0",
    packages=find_packages(),
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.11.0",
        "scipy>=1.3",
        "pandas>=0.25.1",
        "matplotlib>=3.1.1",
        "numba>=0.45"
    ],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Framework :: IPython",
        "Framework :: Jupyter",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
