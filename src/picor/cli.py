"""Isotopologue correction using statistical distributions.

Read raw data as csv or excel file and correct for natural abundance.

Usage:
    picor FILE MOLECULE [-o OUTFILE]
                        [-s COL]...
                        [-x EXCOL]...
                        [--isotopes-file IFILE]
                        [--molecules-file MFILE]
                        [-v | -vv]
    picor FILE MOLECULE --res-correction
                        [--mz-calibration MZ]
                        [--isotopes-file IFILE]
                        [-o OUTFILE]
                        [-s COL]...
                        [-x EXCOL]...
                        [--resolution RES]
                        [--molecules-file MFILE]
                        [-v | -vv]
    picor (-h | --help)
    picor --version

Arguments:
    FILE       Path to excel or csv file (xlsx or csv)
    MOLECULE   Name as in molecules-file

Options:
  -o --output OUTFILE       Output file path (csv format)
  -r --res-correction       Perform resolution correction
  --resolution RES          Resolution of measurement [default: 60000]
  --mz-calibration MZ       Mass-charge of calibration point [default: 200]
  -s --subset COL           Column for calculation; can be used multiple times
  -x --exclude-col EXCOL    Column to ignore; can be used multiple times
  --molecules-file MFILE    Path to tab-separated molecules file
                            Name, formula and charge as rows; e.g. Suc C4H4O3 -1
  --isotopes-file IFILE     Path to tab-separated isotope file
                            Element, mass, abundance and isotope as rows
                            E.g. H 1.008 0.99 H01
  -v --verbose              Display more info like intermediate correction factors
                            (multiple increase verbosity, up to 2)
                            Prints to stderr
  -h --help                 Show this screen.
  --version                 Show version.

"""
import logging
from pathlib import Path
import sys
import pkg_resources

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
        raw_data = pd.read_excel(infile, index_col=0, engine="openpyxl")
    elif infile.suffix == ".csv":
        raw_data = pd.read_csv(infile, index_col=0)
    else:
        raise ValueError("FILE can be either '.csv' or '.xlsx' file type")
    if arguments["--verbose"] == 0:
        logging_level = "WARNING"
    elif arguments["--verbose"] == 1:
        logging_level = "INFO"
    else:
        logging_level = "DEBUG"
    corr_data = picor.calc_isotopologue_correction(
        raw_data,
        arguments["MOLECULE"],
        subset=arguments["--subset"],
        exclude_col=arguments["--exclude-col"],
        resolution_correction=arguments["--res-correction"],
        mz_calibration=float(arguments["--mz-calibration"]),
        resolution=float(arguments["--resolution"]),
        isotopes_file=arguments["--isotopes-file"],
        molecules_file=arguments["--molecules-file"],
        logging_level=logging_level,
    )
    if outfile:
        corr_data.to_csv(outfile)
    print(corr_data)


def main():
    """Serve as entry point for CLI."""
    version = pkg_resources.get_distribution("picor").version
    arguments = docopt(__doc__, version=version)
    _logger.info("arguments = {arguments}")
    sys.exit(cli(arguments))


if __name__ == "__main__":
    main()
