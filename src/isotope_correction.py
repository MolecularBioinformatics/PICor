"""Isotopologue correction using statistical distributions.

Functions:
    calc_isotopologue_correction: Correct DataFrame with measurements.
"""
import os

import src.isotope_probabilities as ip
import src.resolution_correction as rc


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
    verbose=False,
):
    """Calculate isotopologue correction factor for metabolite.

    Calculates isotopologue correction factor for metabolite in metabolites file
    Only C13 and N15 is supported as column labels right now e.g. 5C13
    :param  raw_data: pandas DataFrame
        DataFrame of integrated lowest peaks per species vs time
    :param metabolite: str
        metabolite name
    :param subset: list of str or False
        List of column names to use for calculation
    :param exclude_col: list of str
        Columns to ignore in calculation
    :param isotopes_file: Path to isotope file
        default location: scripts/isotope_correction/isotopes.csv
    :param metabolites_file: Path to metabolites file
        default location: scripts/isotope_correction/metabolites.csv
    :param verbose: bool (default: False)
        print correction and transition factors
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
        if verbose:
            print(f"Correction factor {label1}: {corr}")
        for label2 in subset:
            if ip.label_shift_smaller(label1, label2):
                prob = ip.calc_transition_prob(
                    label1, label2, metabolite, metabolites_file, isotopes_file
                )
                data[label2] = data[label2] - prob * data[label1]
                data[label2].clip(lower=0, inplace=True)
                if verbose:
                    print(f"Transition prob {label1} -> {label2}: {prob}")
    return data
