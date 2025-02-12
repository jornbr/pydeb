# pydeb

pydeb is a Python package for building and analyzing Dynamic Energy Budget
models. It supports [phylogenetic inference of DEB models for new species](https://deb.bolding-bruggeman.com),
simulation of growth, reproduction and survival, and Bayesian inference with
the [Adaptive Metropolis algorithm](https://projecteuclid.org/euclid.bj/1080222083).
It is designed for high performance and can analyze 100,000 parameter sets in
under 2 minutes on most workstations.

To get an impression of the capabilities of pydeb, check out [the Debber webservice](https://deb.bolding-bruggeman.com/).
Debber builds and simulates DEB models for any species of interest and uses
pydeb underneath for all computations.

## Installation

To use pydeb, you need a Python distribution with [NumPy](https://numpy.org).
To run the examples, you additionally need [jupyter](https://jupyter.org) and
[plotly](https://plotly.com/python/).

If you are starting from scratch, we recommend you use a Python distribution
such as [Anaconda](https://docs.anaconda.com/anaconda/) or
[Miniconda](https://docs.anaconda.com/miniconda/).

### Anaconda

If you have Anaconda or Miniconda, you install pydeb and its dependencies with:

`conda install -c conda-forge pydeb plotly jupyterlab`

### From source

Alternatively, you can install pydeb from source with

`python -m pip install <PYDEB_DIR> --user`

Notes:
* `python` is the name of the your Python interpreter.
Replace it if necessary. For instance, some systems have Python 3
installed as `python3`. If you want to install pydeb for this distribution, use `python3 -m pip install <PYDEB_DIR> --user`.
* If you run the above from the pydeb top-level directory (the one that contains this README file), `<PYDEB_DIR>` would be `.`
* Installation from source requires a C compiler, which may need to be installed separately on Windows as described [here](https://wiki.python.org/moin/WindowsCompilers).
* The above installs in your [user-specific site packages directory](https://www.python.org/dev/peps/pep-0370/). To install system-wide, omit `--user`. But this may require administrator/sudo permissions.

## Usage

pydeb comes with a number of [Jupyter Notebook](https://jupyter.org) examples.
To use these, run `jupyter notebook` from the `examples` subdirectory.