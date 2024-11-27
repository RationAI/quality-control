1.  ### Install **pyvips** for Efficient Image Processing

    [Pyvips](https://pypi.org/project/pyvips/2.0.2/) is a powerful and fast image processing library, particularly suited for working with large images like WSIs. To install `pyvips`, ensure that the `libvips` system dependencies are installed first:

    #### Linux

    ```bash
    sudo apt-get install libvips-dev
    ```

    #### macOS (via Homebrew)

    ```bash
    brew install vips
    ```

    #### Windows

    Follow the instructions on the official [libvips installation page](https://www.libvips.org/install.html).


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

    #### Installing a Specific Version

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

Once **pyvips**, and the core library are installed, you're ready to start using QC functions.
