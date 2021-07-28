"""Isotopologue correction using statistical distributions.

Functions:
    calc_isotopologue_correction: Correct DataFrame with measurements.
"""
import logging
import os
from pathlib import Path

import picor.isotope_probabilities as ip
import picor.resolution_correction as rc

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"

_logger = logging.getLogger(__name__)


def calc_isotopologue_correction(
    raw_data,
    molecule_name=None,
    molecules_file=None,
    molecule_formula=None,
    molecule_charge=None,
    subset=False,
    exclude_col=False,
    resolution_correction=False,
    mz_calibration=200,
    resolution=60000,
    isotopes_file=None,
    logging_level="WARNING",
):
    """Calculate isotopologue correction factor for molecule.

    Takes pandas DataFrame and calculates isotopologue correction
    for molecule in molecules file, returns DataFrame with corrected values.
    Only C13 and N15 is supported as column labels right now e.g. 5C13

    Parameters
    ----------
    raw_data : pandas.DataFrame
        DataFrame of integrated lowest peaks per species vs time
    molecule_name : str
        Molecule name as in molecules_file.
    molecules_file : str or Path
        tab-separated file with name, formula and charge as rows
        e.g. Suc C4H4O3 -1
    molecule_formula : str
        Chemical formula as string.
        No spaces or underscores allowed.
        E.g. "C3H7O1"
    molecule_charge : int
        Charge as signed integer
    subset : list of str or False, optional
        List of column names to use for calculation
    exclude_col : list of str, optional
        Columns to ignore in calculation
    resolution_correction : bool, optional
        Run additonal correction for isotopologues overlaping
        due to low resolution. For example H02 and C13
    mz_calibration : float, optional
        mass-charge ratio of calibration point
    resolution : float, optional
        Resolution at calibration mz
    isotopes_file : Path or str, optional
        tab-separated file with element, mass, abundance and isotope as rows
        e.g. H 1.008 0.99 H01
    logging_level : str
        Logging is output to stderr
        Possible levels: "DEBUG", "INFO", "WARNING", "CRITICAL"
        Default level: "WARNING"

    Returns
    -------
    pandas.DataFrame
        Corrected data
    """
    logging.basicConfig(level=os.environ.get("LOGLEVEL", logging_level))
    dir_path = Path(__file__).parent
    if not isotopes_file:
        isotopes_file = dir_path / "isotopes.csv"
    if not molecules_file:
        molecules_file = dir_path / "metabolites.csv"
    if not subset:
        subset = raw_data.columns
        if exclude_col:
            subset = list(set(subset) - set(exclude_col))

    molecule_info = ip.MoleculeInfo.get_molecule_info(
        molecule_name, molecules_file, molecule_formula, molecule_charge, isotopes_file
    )
    subset = ip.LabelTuple(subset, molecule_info)
    res_corr_info = rc.ResolutionCorrectionInfo(
        resolution_correction, resolution, mz_calibration, molecule_info
    )
    if res_corr_info.do_correction:
        rc.warn_direct_overlap(subset, res_corr_info)
    data = correct_data(raw_data, subset, res_corr_info)
    return data


def correct_data(uncorrected_data, subset, res_corr_info):
    """Correct data based on labels in subset."""
    data = uncorrected_data.copy()
    for label1 in subset:
        corr = ip.calc_correction_factor(subset.molecule_info, label1,)
        assert corr >= 1, "Correction factor should be greater or equal 1"
        data[label1.as_string] = corr * data[label1.as_string]
        _logger.info(f"Correction factor {label1.as_string}: {corr}")
        for label2 in subset:
            if label1 >= label2:
                continue
            trans_prob = calc_transition_prob(label1, label2, res_corr_info)
            data[label2.as_string] = (
                data[label2.as_string] - trans_prob * data[label1.as_string]
            )
            data[label2.as_string].clip(lower=0, inplace=True)
            _logger.info(
                f"Transition prob {label1.as_string} -> {label2.as_string}: {trans_prob}"
            )
    return data


def calc_transition_prob(label1, label2, res_corr_info):
    """Calculate the probablity between two (un-)labelled isotopologues.

    Parameters
    ----------
    label1 : Label
        Type of isotopic label, e.g. Label("1N15")
    label2 : Label
        Type of isotopic label, e.g. Label("10C1301N15")
    res_corr_info : ResolutionCorrectionInfo
        Instance with resolution and mz calibration info.

    Returns
    -------
    float
        Transition probability
    """
    if label1 >= label2:
        return 0
    difference_labels = label2.subtract(label1)
    if res_corr_info.do_correction:
        trans_prob = rc.calc_indirect_overlap_prob(label1, label2, res_corr_info)
    else:  # Without resolution correction
        trans_prob = ip.calc_label_diff_prob(label1, difference_labels)
    return trans_prob
