# Project Template

This template combines three libraries to give you some basic training infrastructure:

- [seml](https://github.com/TUM-DAML/seml/) to load configuration files and run jobs


## Installation (Quick Guide)
We highly recommend using [`uv`](https://docs.astral.sh/uv/) for reproducible project management:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
To setup the right environment and activate it use
```sh
uv sync
source .venv/bin/activate
```
*Optionally*: Install pre-commit hooks via
```sh
pre-commit install
```
When executing commands with `seml` make sure to always first activate your virtual environment or use `uv run seml`. Do not use `uvx seml` as `uvx` will create a temporary virtual environments where your packages are not installed.

## Developement

**Project management**

For project management, we recommend [`uv`](https://docs.astral.sh/uv/). Please read the docs carefully. Here are the most important commands
* To add a package to your project use: `uv add <package>`, e.g., `uv add jax[cuda12]`.
* To update your environment: `uv sync`.
* To run a script without explicitly activating the environment, use `uv run main.py`.
* Activate your environment: `source .venv/bin/activate`

`uv` will create a lock file that exactly describes your current environment. Make sure to commit it. To recreate this environment, use `uv sync --locked`. This lock file enables the exact reproducibility of your current environment.

**IDE**

We recommend [VS Code](https://code.visualstudio.com) for development. Select the conda environment you created earlier as your default python interpreter. *Optionally*, use static typecheckers and linters like [ruff](https://github.com/astral-sh/ruff).

**Sacred**

`seml` is based on [Sacred](https://sacred.readthedocs.io/en/stable/index.html). Familiarize yourself with the rough concept behind this framework. Importantly, understand how [experiments](https://sacred.readthedocs.io/en/stable/experiment.html) work and how they can be [configured](https://sacred.readthedocs.io/en/stable/experiment.html#configuration) using config overrides and `named configs`.

**MongoDB**

`seml` will log your experiments on our local `MongoDB` server after you set it up according to the [installation guide]((https://github.com/TUM-DAML/seml/)). Familiarize yourself with the core functionality of `seml` experiments from the example configurations.


**Pytest**

During development you may want to test several functionalities. We recommend using [`pytest`](https://docs.pytest.org/en/8.0.x/) for this. To run your tests simply call
```sh
pytest
```


## Running experiments locally

To start a training locally, call `main.py` with the your settings, for example

```sh
./main.py with config/data/small.yaml config/model/big.yaml
```

You can use this for debugging, e.g. in an interactive slurm session or on your own machine.

## Running experiments on the cluster

Use `seml` to run experiments on the cluster. Pick a collection name, e.g. `example_experiment`. Each experiment should be referred to with an configuration file in `experiments/`. Use the `seml.description` field to keep track of your experiments. Add experiments using:

```bash
seml {your-collection-name} add config/seml/grid.yaml
```

Run them on the cluster using:

```bash
seml {your-collection-name} start
```

You can monitor the experiment using:

```bash
seml {your-collection-name} status
```

More advanced usage of seml can be found in the [documentation](https://github.com/TUM-DAML/seml/tree/master/examples).


## Analyzing results

You can analyze the results by inspecting output files your code generates or values you log in the MongoDB. For reference, see `notebooks/visualize_results.ipynb`.
