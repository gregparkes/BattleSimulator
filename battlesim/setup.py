"""
Basic set up file
"""

from setuptools import setup, find_packages

setup(
	name="battlesim",
	version="0.3.1",
	description="A python package for simulating battles and visualizing them in animation",
	url="https://github.com/gregparkes/battlesim",
	author="Gregory Parkes",
	author_email="g.m.parkes@soton.ac.uk",
	license="MIT",
	packages=find_packages(),
	zip_safe=False,
	install_requires=[
		"numpy","pandas","matplotlib","itertools","numba","pytest"
	],
	classifiers=[
		"Natural Language :: English",
		"Programming Language :: Python :: 3.7",
		"License :: OSI Approved :: MIT License",
		"Intended Audience :: Science/Research",
		"Framework :: IPython", "Framework :: Jupyter",
		"Development Status :: 1 - Planning"
	],
)

