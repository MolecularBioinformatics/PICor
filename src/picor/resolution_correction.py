"""Isotopologue correction using statistical distributions.

Functions:
    calc_indirect_overlap_prob: probability of random H02 or C13
    warn_direct_overlap: warn if direct overlap due to resolution
    warn_indirect_overlap: warn if overlap due to H02 or C13 incorporation
"""
from functools import reduce
import itertools
from operator import mul
import re
import warnings

import pandas as pd
from scipy.special import binom

from picor.isotope_probabilities import (
    assign_light_isotopes,
    get_isotope_mass_series,
    get_isotope_abundance,
    get_metabolite_formula,
    parse_label,
    subtract_label,
)

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"

ABUNDANCE = None


def fwhm(mz_cal, mz, resolution):
    """Calculate Full width half maximum (FWHM)."""
    if mz_cal < 0 or mz <= 0 or resolution <= 0:
        raise ValueError("Arguments must be positive")
    return mz ** (3 / 2) / (resolution * mz_cal ** 0.5)


def calc_min_mass_diff(mass, charge, mz_cal, resolution):
    """Calculate minimal resolvable mass difference.

    For m and z return minimal mass difference that ca be resolved properly.
    :param mass: float
        Mass of molecule
    :param charge: int
        Charge of molecule
    :param mz_cal: float
        Mass/charge for which resolution was determined
    :param resolution: float
        Resolution
    :returns: float
    """
    if mass < 0:
        raise ValueError("'mass' must be positive.")
    mz = abs(mass / charge)
    return 1.66 * abs(charge) * fwhm(mz_cal, mz, resolution)


def calc_isotopologue_mass(
    metabolite_name, label, isotope_mass_series, metabolites_file, isotopes_file
):
    """Calculate mass of isotopologue.

    Given the metabolite name and label composition, return mass in atomic units.
    :param metabolite_name: str
        Name as in metabolite_file
    :param label: str or dict
        "No label" or formula, can contain whitespaces
    :param isotope_mass_series: pandas Series
        Isotope name (e.g. 'H02') as index and mass as values
        Output of 'get_isotope_mass_series'
    :returns: float
    """
    if isinstance(label, str):
        label_dict = parse_label(label)
    elif isinstance(label, dict):
        label_dict = label
    else:
        raise ValueError("label must be str or dict")
    label = pd.Series(label_dict, dtype="int64")
    metab = pd.Series(
        get_metabolite_formula(metabolite_name, metabolites_file, isotopes_file),
        dtype="int64",
    )
    metab = assign_light_isotopes(metab)
    light_isotopes = subtract_label(metab, label)
    formula_isotopes = pd.concat([light_isotopes, label])
    mass = isotope_mass_series.multiply(formula_isotopes).dropna().sum()
    return mass


def get_metabolite_charge(metabolite_name, metabolites_file):
    """Get charge of metabolite."""
    charges = pd.read_csv(
        metabolites_file,
        sep="\t",
        usecols=["name", "charge"],
        index_col="name",
        squeeze=True,
    )
    return charges[metabolite_name]


def calc_coarse_mass_difference(label1, label2):
    """Calculate difference in nucleons (e.g. 2 between H20 and D20).

    nucleons(label2) - nucleons(label1)
    """
    label1 = parse_label(label1)
    label2 = parse_label(label2)
    return sum(label2.values()) - sum(label1.values())


def is_isotologue_overlap(
    label1,
    label2,
    metabolite_name,
    min_mass_diff,
    isotope_mass_series,
    metabolites_file,
    isotopes_file,
):
    """Return True if label1 and label2 are too close to detection limit.

    Checks whether two isotopologues defined by label and metabolite name
    are below miniumum resolved mass difference.
    """
    mass1 = calc_isotopologue_mass(
        metabolite_name, label1, isotope_mass_series, metabolites_file, isotopes_file
    )
    mass2 = calc_isotopologue_mass(
        metabolite_name, label2, isotope_mass_series, metabolites_file, isotopes_file
    )
    return abs(mass1 - mass2) < min_mass_diff


def warn_indirect_overlap(
    label_list, metabolite_name, min_mass_diff, metabolites_file, isotopes_file
):
    """Warn if any of labels can have indirect overlap."""
    for label1, label2 in itertools.permutations(label_list, 2):
        prob = calc_indirect_overlap_prob(
            label1,
            label2,
            metabolite_name,
            min_mass_diff,
            metabolites_file,
            isotopes_file,
        )
        if prob:
            warnings.warn(
                f"{label1} indirect overlap with {label2} with prob {prob:.4f}"
            )


def calc_indirect_overlap_prob(
    label1, label2, metabolite_name, min_mass_diff, metabolites_file, isotopes_file
):
    """Calculate probability for overlap caused by random H02 and C13 incoporation.

    Only C13 and H02 attributions are considered so far.
    :param label1: str
        "No label" or formula e.g. "1C13 2H02"
    :param label2: str
        "No label" or formula e.g. "1C13 2H02"
    :param metabolite_name: str
        Name as in metabolites_file
    :param min_mass_diff: float
        Minimal resolvable mass difference by MS measurement
    :param metabolites_file: Path to metabolites file
        default location: ~/isocordb/Metabolites.dat
    :param isotopes_file: Path to isotope file
        default location: ~/isocordb/Isotopes.dat
    :return: float
        Transition probability of overlap
    """
    # Label overlap possible with additional atoms
    n_atoms = get_metabolite_formula(metabolite_name, metabolites_file, isotopes_file)
    isotope_mass_series = get_isotope_mass_series(isotopes_file)
    coarse_mass_difference = calc_coarse_mass_difference(label1, label2)
    if coarse_mass_difference <= 0:
        return 0
    label1 = parse_label(label1)
    label2 = parse_label(label2)
    label1_series = pd.Series(label1, dtype="float64")
    label2_series = pd.Series(label2, dtype="float64")

    # Check if standard transition is possible
    label_diff = label2_series.sub(label1_series, fill_value=0)
    if label_diff.ge(0).all():
        return 0

    probs = {}
    for n_c in range(0, coarse_mass_difference + 1):
        # breakpoint()
        n_h = coarse_mass_difference - n_c
        label_trans = {"C13": n_c, "H02": n_h}
        label_trans_series = pd.Series(label_trans, dtype="float64")
        label1_mod = dict(label1_series.add(label_trans_series, fill_value=0))
        if is_isotologue_overlap(
            label1_mod,
            label2,
            metabolite_name,
            min_mass_diff,
            isotope_mass_series,
            metabolites_file,
            isotopes_file,
        ):
            probs[n_c] = calc_label_diff_prob(
                label1, label_trans, n_atoms, isotopes_file
            )
    prob_total = sum(probs.values())
    return prob_total


def warn_direct_overlap(
    label_list, metabolite_name, min_mass_diff, metabolites_file, isotopes_file
):
    """Warn if any of labels can have overlap."""
    isotope_mass_series = get_isotope_mass_series(isotopes_file)
    for label1, label2 in itertools.permutations(label_list, 2):
        # Direct label overlap
        if is_isotologue_overlap(
            label1,
            label2,
            metabolite_name,
            min_mass_diff,
            isotope_mass_series,
            metabolites_file,
            isotopes_file,
        ):
            warnings.warn(f"Direct overlap of {label1} and {label2}")


def calc_label_diff_prob(label1, difference_labels, n_atoms, isotopes_file):
    """Calculate the transition probablity of difference in labelled atoms.

    :param label1: dict
        Isotope symbol (e.g. N15) as key and number of atoms as value
    :param difference_labels: dict
        Isotope symbol (e.g. N15) as key and number of atoms as value
    :param n_atoms: dict
        Element symbol as key and number of atoms as value
    :param isotopes_file: Path to isotope file
    :return: float
        Transition probability
    """
    global ABUNDANCE
    if not ABUNDANCE:
        ABUNDANCE = get_isotope_abundance(isotopes_file)

    prob = []
    for isotope, n_label in difference_labels.items():
        try:
            n_elem_1 = label1[isotope]
        except KeyError:
            n_elem_1 = 0
        elem = re.search(r"[A-Z][a-z]?", isotope).group(0)

        n_unlab = n_atoms[elem] - n_elem_1 - n_label
        abun_unlab = ABUNDANCE[elem][0]
        abun_lab = ABUNDANCE[elem][1]
        if n_label == 0:
            continue

        trans_pr = binom((n_atoms[elem] - n_elem_1), n_label)
        trans_pr *= abun_lab ** n_label
        trans_pr *= abun_unlab ** n_unlab
        prob.append(trans_pr)

    # Prob is product of single probabilities
    prob_total = reduce(mul, prob)
    assert prob_total <= 1, "Transition probability greater than 1"
    return prob_total
