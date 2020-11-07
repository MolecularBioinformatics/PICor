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

You need to have `docopt`, `pandas` and `scipy` installed.

## Usage

You can use PICor in two ways:

### Command Line

After the installation you can use PICor anywhere from the command line with `picor`.
```bash
picor tests/test_dataset.xlsx NAD -x "dummy column int" -x "dummy column str"
```
Files with `.csv` or `.xlsx` suffix can be used as input files.
You can choose the output file (in `.csv`format`) with the `-o` option.
`picor -h`shows all options.

### Python Module

After importing PICor and loading your data (for example a csv file) with pandas you the correction works with:
```python
import pandas as pd
import picor

raw_data = pd.DataFrame(
    {
        "No label": {0: 100, 1: 200, 2: 300, 3: 400, 4: 500, 5: 600},
        "1C13": {0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100},
        "4C13 6H02 3N15": {0: 30, 1: 40, 2: 50, 3: 60, 4: 70, 5: 80},
        "dummy column str": {0: "C", 1: "ER", 2: "C", 3: "ER", 4: "C", 5: "ER"},
    }
)
corr_data = picor.calc_isotopologue_correction(
    raw_data,
    "NAD",
    exclude_col=["dummy column str"],
)
print(corr_data)
```

In case the DataFrame contains columns (except the index colum) with other data than raw measurements, you can use either the `subset` with a list of all columns to be used or `exclude_col` with a list of the column to be skipped.

You can activate a resolution depent correction by setting  `resolution_correction` to `True`. Specify the resolution and the reference m/z ratio with `resolution` and `mz_calibration`.


Jørn Dietze, UiT - The Arctic University of Tromsø, 2020
