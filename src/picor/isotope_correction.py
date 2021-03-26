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
    molecule_name,
    subset=False,
    exclude_col=False,
    resolution_correction=False,
    mz_calibration=200,
    resolution=60000,
    isotopes_file=None,
    molecules_file=None,
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
        molecule name as in molecules_file
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
    molecules_file : Path or str, optional
        tab-separated file with name, formula and charge as rows
        e.g. Suc C4H4O3 -1

    Returns
    -------
    pandas.DataFrame
        Corrected data
    """
    if not isotopes_file:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        isotopes_file = os.path.join(dir_path, "isotopes.csv")
    if not molecules_file:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        molecules_file = os.path.join(dir_path, "metabolites.csv")
    data = raw_data.copy(deep=True)
    if not subset:
        subset = data.columns
        if exclude_col:
            subset = list(set(subset) - set(exclude_col))
    subset = ip.sort_labels(subset)

    molecule_info = ip.MoleculeInfo(molecule_name, molecules_file, isotopes_file)
    if resolution_correction:
        mass = molecule_info.calc_isotopologue_mass("No label",)
        charge = molecule_info.get_charge()
        min_mass_diff = rc.calc_min_mass_diff(mass, charge, mz_calibration, resolution)
        rc.warn_direct_overlap(
            subset, molecule_info, min_mass_diff,
        )

    for label1 in subset:
        corr = ip.calc_correction_factor(molecule_info, label1,)
        assert corr >= 1, "Correction factor should be greater or equal 1"
        data[label1] = corr * data[label1]
        _logger.info(f"Correction factor {label1}: {corr}")
        for label2 in subset:
            if ip.label_shift_smaller(label1, label2):
                if resolution_correction:
                    indirect_overlap_prob = rc.calc_indirect_overlap_prob(
                        label1, label2, molecule_info, min_mass_diff,
                    )
                    data[label2] = data[label2] - indirect_overlap_prob * data[label1]
                    _logger.info(
                        f"Overlapping prob {label1} -> {label2}: {indirect_overlap_prob}"
                    )
                trans_prob = ip.calc_transition_prob(label1, label2, molecule_info,)
                data[label2] = data[label2] - trans_prob * data[label1]
                data[label2].clip(lower=0, inplace=True)
                _logger.info(f"Transition prob {label1} -> {label2}: {trans_prob}")
    return data
