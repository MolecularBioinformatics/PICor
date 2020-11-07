"""Isotopologue correction using statistical distributions.

Functions:
    calc_isotopologue_correction: Correct DataFrame with measurements.
"""
import os
import logging

import picor.isotope_probabilities as ip
import picor.resolution_correction as rc

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"

_logger = logging.getLogger(__name__)


def calc_isotopologue_correction(
    raw_data,
    metabolite,
    subset=False,
    exclude_col=False,
    resolution_correction=False,
    mz_calibration=200,
    resolution=60000,
    isotopes_file=None,
    metabolites_file=None,
):
    """Calculate isotopologue correction factor for metabolite.

    Takes pandas DataFrame and calculates isotopologue correction
    for metabolite in metabolites file, returns DataFrame with corrected values.
    Only C13 and N15 is supported as column labels right now e.g. 5C13
    :param  raw_data: pandas DataFrame
        DataFrame of integrated lowest peaks per species vs time
    :param metabolite: str
        metabolite name as in metabolites_file
    :param subset: list of str or False
        List of column names to use for calculation
    :param exclude_col: list of str
        Columns to ignore in calculation
    :param resolution_correction: bool (default: False)
        Run additonal correction for isotopologues overlaping
        due to low resolution. For example H02 and C13
    :param mz_calibration: float (default: 200)
        mass-charge ratio of calibration point
    :param resolution: float (default: 60_000)
        Resolution at calibration mz
    :param isotopes_file: Path to isotope file
        tab-separated file with element, mass, abundance and isotope as rows
        e.g. H 1.008 0.99 H01
    :param metabolites_file: Path to metabolites file
        tab-separated file with name, formula and charge as rows
        e.g. Suc C4H4O3 -1
    :return: pandas DataFrame
        Corrected data
    """
    if not isotopes_file:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        isotopes_file = os.path.join(dir_path, "isotopes.csv")
    if not metabolites_file:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        metabolites_file = os.path.join(dir_path, "metabolites.csv")
    data = raw_data.copy(deep=True)
    if not subset:
        subset = data.columns
        if exclude_col:
            subset = list(set(subset) - set(exclude_col))
    subset = ip.sort_labels(subset)

    if resolution_correction:
        mass = rc.calc_isotopologue_mass(
            metabolite,
            "No label",
            ip.get_isotope_mass_series(isotopes_file),
            metabolites_file,
            isotopes_file,
        )
        charge = rc.get_metabolite_charge(metabolite, metabolites_file)
        min_mass_diff = rc.calc_min_mass_diff(mass, charge, mz_calibration, resolution)
        rc.warn_direct_overlap(
            subset, metabolite, min_mass_diff, metabolites_file, isotopes_file
        )

    for label1 in subset:
        corr = ip.calc_correction_factor(
            metabolite, label1, isotopes_file, metabolites_file
        )
        assert corr >= 1, "Correction factor should be greater or equal 1"
        data[label1] = corr * data[label1]
        _logger.info(f"Correction factor {label1}: {corr}")
        for label2 in subset:
            if ip.label_shift_smaller(label1, label2):
                if resolution_correction:
                    indirect_overlap_prob = rc.calc_indirect_overlap_prob(
                        label1,
                        label2,
                        metabolite,
                        min_mass_diff,
                        metabolites_file,
                        isotopes_file,
                    )
                    data[label2] = data[label2] - indirect_overlap_prob * data[label1]
                    _logger.info(
                        f"Overlapping prob {label1} -> {label2}: {indirect_overlap_prob}"
                    )
                trans_prob = ip.calc_transition_prob(
                    label1, label2, metabolite, metabolites_file, isotopes_file
                )
                data[label2] = data[label2] - trans_prob * data[label1]
                data[label2].clip(lower=0, inplace=True)
                _logger.info(f"Transition prob {label1} -> {label2}: {trans_prob}")
    return data
