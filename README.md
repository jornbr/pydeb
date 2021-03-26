# pydeb

pydeb is a Python package for building and analyzing Dynamic Energy Budget models.
It supports [phylogenetic inference of DEB models for new species](https://deb.bolding-bruggeman.com), simulation of
growth, reproduction and survival, and Bayesian inference with the [Adaptive Metropolis algorithm](https://projecteuclid.org/euclid.bj/1080222083). It is designed for high performance and can analyze 100,000 parameter sets in under 2 minutes on most workstations.

## Installation

You need a Python distribution with [NumPy](https://numpy.org) and [Cython](https://cython.org).
To run the examples, you additionally need [jupyter](https://jupyter.org) and [plotly](https://plotly.com/python/).
If you are starting from scratch, we recommend you install a Python distribution such as [Anaconda](https://www.anaconda.com), which comes with most of these packages preinstalled.  You can then install any remaining ones with `conda`, e.g., `conda install plotly`.

If you have the prerequites described above, install pydeb with:

`python -m pip install <PYDEB_DIR> --user`

Notes:
* `python` is the name of the your Python interpreter.
Replace it if necessary. For instance, some systems have Python 3
installed as `python3`. If you want to install pydeb for this distribution, use `python3 -m pip install <PYDEB_DIR> --user`.
* If you run the above from the pydeb top-level directory (the one that contains this README file), `<PYDEB_DIR>` would be `.`
* The above installs in your [user-specific site packages directory](https://www.python.org/dev/peps/pep-0370/). To install system-wide, omit `--user`. But this may require administrator/sudo permissions.

## Usage

pydeb comes with a number of [Jupyter Notebook](https://jupyter.org) examples.
To use these, run `jupyter notebook` from the `examples` subdirectory.