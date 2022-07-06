# rfinder

A starting point for spinning up a Python package.

## Installation

This project has been built with [poetry](https://python-poetry.org/) v1.2.0b1, so you can install your new package with a simple `poetry install`. I have not tested it on other platforms or older poetry versions.

I have included a environment file generator. It generates an environment file to contain variables for your particular use-case. The `poetry run generate-env` command will generate a .env file in the root of the project. You should then copy this with `cp .env .env.local` and make appropriate changes. This workflow prevents you from manually editing the .env file and later overwriting it with defaults by re-running the `generate-env` script. If the default .env changes, you will need to manually update your .env.local file.

The entire [environment](py_package/environment/) folder can of course be removed if not useful to you. If removed, you should also delete the `generate-env` script in the [pyproject.toml](pyproject.toml) file.

## Developing

To add packages dependencies with poetry you can use the `poetry add` command. This starter uses dependency groups to separate the dependencies into groups. The `dev` group is for development dependencies, and the `default` group is for deployment. This allows you to build your package with only deployment dependencies included in the bundle. To use these groups you can use the `poetry add my-chosen-dependency --group dev` or `poetry add my-chosen-dependency` for the default group.

This package is mypy and flake8 compliant. You can lint with `poetry run poe lint`.

The package is also formatted with isort and black. Format with `poetry run poe format`.

Although maybe contentious, I have committed the .vscode folder with my [settings.json](.vscode/settings.json) file. If you are using vscode this should make linting "on the go" easier.
