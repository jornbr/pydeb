# pydeb

pydeb is a Python packages for building and analyzing Dynamic Energy Budget models.
It supports [phylogenetic inference of DEB models for new species](https://deb.bolding-bruggeman.com), simulation of
growth, reproduction and survival, and Bayesian inference with the [Adaptive Metropolis algorithm](https://projecteuclid.org/euclid.bj/1080222083). It is designed for high performance and can analyze 100,000 parameter sets in under 2 minutes on most workstations.

## Installation

`python setup.py install`

In the above, `python` is the name of the your Python interpreter.
Replace it if necessary. For instance, some systems have Python 3
installed as `python3`. If you want to install pydeb for this distribution,
use `python3 setup.py install`.

## Usage

pydeb comes with a number of [Jupyter Notebook](https://jupyter.org) examples.
To use these, run `jupyter notebook` from the `examples` subdirectory.