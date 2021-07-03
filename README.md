[![Build Status](https://travis-ci.com/MolecularBioinformatics/PICor.svg?branch=master)](https://travis-ci.com/MolecularBioinformatics/PICor)
[![codecov](https://codecov.io/gh/MolecularBioinformatics/PICor/branch/master/graph/badge.svg?token=DZIP4BMO3K)](https://codecov.io/gh/MolecularBioinformatics/PICor)
[![CodeFactor](https://www.codefactor.io/repository/github/molecularbioinformatics/picor/badge)](https://www.codefactor.io/repository/github/molecularbioinformatics/picor)

# PICor: Statistical Isotope Correction

PICor is a python package for correcting mass spectrometry data for the effect of natural isotope abundance.


## Description

PICor takes pandas DataFrames of the measured integrated MS intensities as input, corrects them for natural isotope abundance and returns a DataFrame again.

PICor can also correct for overlapping isotopologues due to too low resoltion. For example, the 13-C4 and 2-H4 isotopologues of the metabolite NAD can't be resolved at a resolution of 60,000 at 200 m/z.

## Installation

To install:

```bash
$ pip install picor
```

PICor depends on `docopt`, `pandas`, `openpyxl` and `scipy` and installs those with pip if not available.

## Usage

You can use PICor in two ways:

### Command Line

After the installation you can use PICor anywhere from the command line with `picor`.

```bash
picor tests/test_dataset.xlsx NAD -x "dummy column int" -x "dummy column str"
```

Files with `.csv` or `.xlsx` suffix can be used as input files.
You can choose the output file (in csv format) with the `-o` option.
If no output file is given, output will be printed to stdout.

`picor -h` shows all options.

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
    molecule_name="NAD",
    exclude_col=["dummy column str"],
)
print(corr_data)
```

In case the DataFrame contains columns (except the index colum) with other data than raw measurements, you can use either the `subset` with a list of all columns to be used or `exclude_col` with a list of the column to be skipped.

You can activate a resolution depent correction by setting  `resolution_correction` to `True`. Specify the resolution and the reference m/z ratio with `resolution` and `mz_calibration`.


## Molecule Specification
The molecule to be corrected can either be specified by name (`molecule_name`) or by formula and charge (`molecule_formula` and `molecule_charge`).
If a name is used it has to be specified in the file specified by `molecules_file` (file path). 

The molecule file has to be tab-separated with the columns `name`, `formula` and `charge`. The column labels have to match exactly. Look at the example file ìn `src/picor/metabolites.csv`.

## Input Data
Using the command line interface `picor` both excel (`.xlsx`) and comma-separated data (`.csv`) can be corrected.

Both data formats should be arranged with the different samples as rows and different labels/isotopologues as columns.
Additional columns with for example more information about the samples have to be added to the excluded columns. Either with `-x` (command line) or `exclude_col` parameter as a list (python interface).

### Example Data

| sample number | No label | 1C13 | 4C13 6H02 3N15 | sample condition |
|:------------- | -------- | ---- | -------------- | ---------------- |
| 0             | 100      | 100  | 30             | C                |
| 1             | 200      | 100  | 40             | ER               |
| 2             | 300      | 100  | 50             | C                |
| 3             | 400      | 100  | 60             | ER               |
| 4             | 500      | 100  | 70             | C                |
| 5             | 600      | 100  | 80             | ER               |


### Label Specification

The isotopologues or labels are specified as string in the table header (first line).
Labels can include either one or multiple isotopes or 'No label'.
The number of labeled atoms of each isotope has to be specified before the element, e.g. '3H02 2C13' for three deuterium and two 13-C atoms.
Spaces and underscores are allowed but not necessary in the label definition.

In case you want to add additional information you can a prefix separated by a colon (':'), e.g. "NAD:2C13". The prefix will be ignored. 


Jørn Dietze, UiT -The Arctic University of Norway, 2021
