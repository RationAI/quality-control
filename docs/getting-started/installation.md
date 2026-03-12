1. ### Install **pyvips** for Efficient Image Processing

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

2. ### Install the Core Quality Control Library

    You can install the core library using one of the following methods, depending on your package manager:

    === "uv"
        (*Note that this command assumes an existing uv project.*)

        ```bash
        uv add git+https://github.com/RationAI/quality-control.git
        ```

    === "pip"

        ```bash
        pip install git+https://github.com/RationAI/quality-control.git
        ```

Once **pyvips**, and the core library are installed, you're ready to start using QC functions.
