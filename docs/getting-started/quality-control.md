# Using the Provided Quality Control Functions
This document presents several examples of using the provided QC functions. These examples
should provide you with good overview of the offered functionality and potential limitations.

## Suggested Workflow
This library is not suited for running the provided QC methods on complete WSIs.
Instead, the functions are expected to be run on smaller WSI regions and regular PNG/JPG images.
This should provide reasonable options for debugging potential problems, while avoiding
the technical overhead needed for processing the complete WSIs.

## Examples
The following examples should provide you with the neccessary information for running the provided QC functions.

### Detecting Residual Artifacts
In this example, we will work with the following image:

![The input image](data/residual_input.png)

#### Sample code
First, we need to import the necessary functionality and load the image into numpy array:

```python linenums="1"
import numpy as np
from PIL import Image

from rationai.staining import ColorConversion


img = np.asarray(Image.open("input.png").convert("RGB"))
```

Next, we will need to prepare the arguments for the [residual_artifacts_and_coverage](../api/residual_artifacts/residual-artifacts-and-coverage.md) function and call it on the input image:

```python linenums="9"

conversion = ColorConversion.RGB2HER
nucleus_area = 150
res_index = 2
threshold = 0.005

artifacts = residual_artifacts_and_coverage(
    img, conversion, nucleus_area, res_index, threshold
)
```

Since the tissue is stained with the **H&E protocol**, we used the `RGB2HER` color conversion with the corresponding index of the residual channel. The threshold value is based on the currently recommended value in the function's documentation. Finally, the nucleus area is manually approximated from the input image.

After obtaining the results, we can save the binary artifact mask and print the computed `coverage` number:

```python linenums="18"
mask = Image.fromarray(255 * artifacts["coverage_mask"].astype(np.uint8))
mask.save("residual_mask.png")

print("Coverage Number:", artifacts["coverage"])
```

The computed mask is returned as a binary image. Therefore, the computed values need to be scaled into the [0, 255] range before visualization.

#### Results
In our example, the `coverage` number came out to be `0.0766`, meaning that little more than 7% of the image's foreground area is covered by the artifact.
Finally, the computed artifact mask looks like this:

![The generated mask](data/residual_mask.png)