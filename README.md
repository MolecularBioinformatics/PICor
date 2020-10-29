# PICor: Statistical Isotope Correction

ICor is a python package for correcting mass spectrometry data for the effect of natural isotope abundance.


## Description

PICor takes pandas DataFrames of the measured integrated MS intensities as input, corrects them for natural isotope abundance and returns a DataFrame again.

PICor can also correct for overlapping isotopologues due to too low resoltion.For example the 13-C4 and 2-H4 isotopologues of the metabolite NAD can't be resolved at a resolution of 60,000 at 200 m/z.

## Installation

To install:
```bash
$ pip install picor
```

You need to have `pandas` and `scipy` installed.

## Usage

After importing PICor and loading your data (for example a csv file) with pandas you the correction works with:
```python
import pandas as pd
import picor

raw_data = pd.read_csv("data.csv", index="Time in h"))
corr_data = picor.calc_isotopologue_correction(
	raw_data,
	"NAD",
	)
print(corr_data)
```

You can activate a resolution depent correction by setting  `resolution_correction` to `True`. Specify the resolution and the reference m/z ratio with `resolution` and `mz_calibration`.


## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.

Jørn Dietze, UiT - The Arctic University of Tromsø, 2020
