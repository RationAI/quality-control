# Quality Control

This repository contains a selection of image-processing routines, **forming a foundation
for QC methods** developed at RationAI. These functions are meant to be run on smaller WSI
regions and they **offer a straightforward and well-documented access** to the key parts
of the complete QC methods to allow for easier debugging and experimenting.

## Documentation

**[`RationAI Quality Control`](https://rationai.github.io/quality-control/)**

## Setting up a Development Environment

1. Create the development environment (including all optional and development dependencies):

    ```bash
    uv sync --all-groups
    ```

2. Install pre-commit hooks:

    ```bash
    uv run pre-commit install
    uv run pre-commit install --hook-type commit-msg
    ```
