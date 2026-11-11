1. ### Install **pyvips** for Efficient Image Processing

    [Pyvips](https://pypi.org/project/pyvips/) is a powerful and fast image processing library, particularly suited for working with large images like WSIs. To install `pyvips`, ensure that the `libvips` system dependencies are installed first:

    === "Linux"

        ```bash
        sudo apt-get install libvips-dev
        ```

    === "macOS (via Homebrew)"

        ```bash
        brew install vips
        ```

    === "Windows"

        For installation on Windows, follow the instructions on the official [libvips installation page](https://www.libvips.org/install.html).

2. ### Install **OpenSlide** for Reading WSIs

    The [rationai-masks](https://gitlab.ics.muni.cz/rationai/digital-pathology/libraries/masks) dependency
    (used for computing tissue masks in the examples) requires [OpenSlide](https://openslide.org/)
    through its `openslide-python` bindings. The OpenSlide library itself needs to be installed separately:

    === "openslide-bin (all platforms) via pip"

        ```bash
        pip install openslide-bin
        ```

    === "Linux"

        ```bash
        sudo apt-get install openslide-tools
        ```

    === "macOS (via Homebrew)"

        ```bash
        brew install openslide
        ```

        On Apple Silicon, Homebrew installs libraries into `/opt/homebrew/lib`, which Python
        does not search by default. If `import openslide` fails, use the `openslide-bin` option
        above, or set `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` before starting Python.

    For more details, see the [OpenSlide Python installation guide](https://openslide.org/api/python/#installing).

3. ### Install the Core Quality Control Library

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

Once **pyvips**, **OpenSlide**, and the core library are installed, you're ready to start using QC functions.
