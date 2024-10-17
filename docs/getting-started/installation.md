1.  ### Install **OpenSlide** for Handling WSIs

    [OpenSlide](https://openslide.org) is essential for handling whole-slide image (WSI) formats. To use `openslide-python`, you'll first need to install the OpenSlide library, followed by its Python bindings.

    #### Linux (Debian-based):

    ```bash
    sudo apt-get install openslide-tools
    ```

    #### macOS (via Homebrew):

    ```bash
    brew install openslide
    ```

    #### Windows:

    Download and install the [OpenSlide binaries](https://openslide.org/download/).

2.  ### Install the Core Quality Control Library

    You can install the core library using one of the following methods, depending on your package manager:

    === "pdm"

        ```bash
        pdm add git+https://gitlab.ics.muni.cz/rationai/digital-pathology/quality-control/quality-control.git
        ```

    === "pip"

        ```bash
        pip install git+https://gitlab.ics.muni.cz/rationai/digital-pathology/quality-control/quality-control.git
        ```

    #### Installing a Specific Version:

    To install a specific version of the **Quality Control** library, use the following command, replacing `v1.0.0` with your desired version.

    === "pdm"

        ```bash
        pdm add git+https://gitlab.ics.muni.cz/rationai/digital-pathology/quality-control/quality-control.git@v1.0.0
        ```

    === "pip"

        ```bash
        pip install git+https://gitlab.ics.muni.cz/rationai/digital-pathology/quality-control/quality-control.git@v1.0.0
        ```

    !!! info

        Replace `v1.0.0` with the specific version you want to install.

Once **OpenSlide**, and the core library are installed, you're ready to start using QC functions.
