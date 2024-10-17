# dartplot

Plot diagnostics in observation and state space for DART with CAM or CESM.

This was designed to replicate the existing DART MATLAB diagnostics in python,
while adding some extended functionality and flexibility to add your own
additional functionality if you're more comfortable with python than MATLAB.

## Installation

You can install the package using `pip` directly from this repository:

```bash
pip install git+https://github.com/robin-clancy/dartplot.git
```

## Requirements

This package was originally designed to work with the NPL 2024b kernel on Cheyenne.<br>
You may need to install additional packages to get the required behaviour...

## Examples

The examples show the current range of functionality of the dartplot package.<br><br>
Obs space examples use example data included in the dartplot package.<br><br>
State space examples point to data that likely won't exist in the future and is too large to conveniently
include in the dartplot package. To run state space examples you may need to point to your own dataset.<br><br>

See:<br>
examples/<...>.ipynb