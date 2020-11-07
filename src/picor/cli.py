"""Isotopologue correction using statistical distributions.

Read raw data as csv or excel file and correct for natural abundance.

Usage:
    picor FILE METABOLITE [-o OUTFILE]
                          [-s COL]...
                          [-x EXCOL]...
                          [--isotopes-file IFILE]
                          [--metabolites-file MFILE]
    picor FILE METABOLITE --res-correction
                          [--mz-calibration MZ]
                          [--isotopes-file IFILE]
                          [-o OUTFILE]
                          [-s COL]...
                          [-x EXCOL]...
                          [--resolution RES]
                          [--metabolites-file MFILE]
    picor (-h | --help)
    picor --version

Arguments:
    FILE        Path to excel or csv file (xlsx or csv)
    METABOLITE  Name as in metabolites-file

Options:
  -o --output OUTFILE       Output file path (csv format)
  -r --res-correction       Perform resolution correction
  --resolution RES          Resolution of measurement [default: 60000]
  --mz-calibration MZ       Mass-charge of calibration point [default: 200]
  -s --subset COL           Column for calculation; can be used multiple times
  -x --exclude-col EXCOL    Column to ignore; can be used multiple times
  --metabolites-file MFILE  Path to tab-separated metabolites file
                            Name, formula and charge as rows; e.g. Suc C4H4O3 -1
  --isotopes-file IFILE     Path to tab-separated isotope file
                            Element, mass, abundance and isotope as rows
                            E.g. H 1.008 0.99 H01
  -h --help                 Show this screen.
  --version                 Show version.

"""
import logging
from pathlib import Path
import pkg_resources
import sys

from docopt import docopt
import pandas as pd

import picor

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"

_logger = logging.getLogger(__name__)


def cli(arguments):
    """Run isotope correction with CLI interface."""
    infile = Path(arguments["FILE"])
    outfile = arguments["--output"]
    if infile.suffix == ".xlsx":
        raw_data = pd.read_excel(infile, index_col=0)
    elif infile.suffix == ".csv":
        raw_data = pd.read_csv(infile, index_col=0)
    else:
        raise ValueError("FILE can be either '.csv' or '.xlsx' file type")
    corr_data = picor.calc_isotopologue_correction(
        raw_data,
        arguments["METABOLITE"],
        subset=arguments["--subset"],
        exclude_col=arguments["--exclude-col"],
        resolution_correction=arguments["--res-correction"],
        mz_calibration=float(arguments["--mz-calibration"]),
        resolution=float(arguments["--resolution"]),
        isotopes_file=arguments["--isotopes-file"],
        metabolites_file=arguments["--metabolites-file"],
    )
    if outfile:
        corr_data.to_csv(outfile)
    print(corr_data)


def main():
    """Serve as entry point for CLI."""
    version = pkg_resources.get_distribution("picor").version
    arguments = docopt(__doc__, version=version)
    _logger.info(f"{arguments=}")
    sys.exit(cli(arguments))


if __name__ == "__main__":
    main()
