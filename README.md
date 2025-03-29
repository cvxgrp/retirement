# Retirement funding

This code repository accompanies the paper [XXX](YYY).


## Poetry

We assume you share the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
make install
```

to replicate the virtual environment we have defined in [pyproject.toml](pyproject.toml)
and locked in [poetry.lock](poetry.lock).

## Jupyter

We install [JupyterLab](https://jupyter.org) on fly within the aforementioned
virtual environment. Executing

```bash
make jupyter
```

will install and start the Jupyter lab. The experiments are replicated in the experiments folder notebooks.

## Citation

If you want to reference our work in your research, please consider using the following BibTeX for the citation:
```
@misc{boyd2020retirement,
      title={A Tax-Efficient Model Predictive Control Policy for Retirement Funding},
      author={Kasper Johansson and Stephen Boyd},
      year={2025},
      url={https://web.stanford.edu/~boyd/papers/retirement.html},
      note={Preprint}
}
```

