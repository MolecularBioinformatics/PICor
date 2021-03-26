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

from picor.isotope_probabilities import parse_label

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


def fwhm(mz_cal, mz, resolution):
    """Calculate Full width half maximum (FWHM)."""
    if mz_cal < 0 or mz <= 0 or resolution <= 0:
        raise ValueError("Arguments must be positive")
    return mz ** (3 / 2) / (resolution * mz_cal ** 0.5)


def calc_min_mass_diff(mass, charge, mz_cal, resolution):
    """Calculate minimal resolvable mass difference.

    For m and z return minimal mass difference that ca be resolved properly.

    Parameters
    ----------
    mass : float
        Mass of molecule
    charge : int
        Charge of molecule
    mz_cal : float
        Mass/charge for which resolution was determined
    resolution : float
        Resolution

    Returns
    -------
    float
        Minimal resolvable mass differenceo

    Raises
    ------
    ValueError
        If molecule mass is negative.
    """
    if mass < 0:
        raise ValueError("'mass' must be positive.")
    mz = abs(mass / charge)
    return 1.66 * abs(charge) * fwhm(mz_cal, mz, resolution)


def calc_coarse_mass_difference(label1, label2):
    """Calculate difference in nucleons (e.g. 2 between H20 and D20).

    nucleons(label2) - nucleons(label1)
    """
    label1 = parse_label(label1)
    label2 = parse_label(label2)
    return sum(label2.values()) - sum(label1.values())


def is_isotologue_overlap(
    label1, label2, molecule_info, min_mass_diff,
):
    """Return True if label1 and label2 are too close to detection limit.

    Checks whether two isotopologues defined by label and metabolite name
    are below miniumum resolved mass difference.
    """
    mass1 = molecule_info.calc_isotopologue_mass(label1)
    mass2 = molecule_info.calc_isotopologue_mass(label2)
    return abs(mass1 - mass2) < min_mass_diff


def warn_indirect_overlap(
    label_list, molecule_info, min_mass_diff,
):
    """Warn if any of labels can have indirect overlap."""
    for label1, label2 in itertools.permutations(label_list, 2):
        prob = calc_indirect_overlap_prob(label1, label2, molecule_info, min_mass_diff,)
        if prob:
            warnings.warn(
                f"{label1} indirect overlap with {label2} with prob {prob:.4f}"
            )


def calc_indirect_overlap_prob(
    label1, label2, molecule_info, min_mass_diff,
):
    """Calculate probability for overlap caused by random H02 and C13 incoporation.

    Only C13 and H02 attributions are considered so far.

    Parameters
    ----------
    label1 : str
        "No label" or formula e.g. "1C13 2H02"
    label2 : str
        "No label" or formula e.g. "1C13 2H02"
    molecule_info : MoleculeInfo
        Instance with molecule and isotope information.
    min_mass_diff : float
        Minimal resolvable mass difference by MS measurement

    Returns
    -------
    float
        Transition probability of overlap
    """
    # Label overlap possible with additional atoms
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
        if is_isotologue_overlap(label1_mod, label2, molecule_info, min_mass_diff,):
            probs[n_c] = calc_label_diff_prob(label1, label_trans, molecule_info)
    prob_total = sum(probs.values())
    return prob_total


def warn_direct_overlap(
    label_list, molecule_info, min_mass_diff,
):
    """Warn if any of labels can have overlap."""
    for label1, label2 in itertools.permutations(label_list, 2):
        # Direct label overlap
        if is_isotologue_overlap(label1, label2, molecule_info, min_mass_diff,):
            warnings.warn(f"Direct overlap of {label1} and {label2}")


def calc_label_diff_prob(label1, difference_labels, molecule_info):
    """Calculate the transition probablity of difference in labelled atoms.

    Parameters
    ----------
    label1 : dict
        Isotope symbol (e.g. N15) as key and number of atoms as value
    difference_labels : dict
        Isotope symbol (e.g. N15) as key and number of atoms as value
    molecule_info : MoleculeInfo
        Instance with molecule and isotope information.
    Returns
    -------
    float
        Transition probability
    """
    n_atoms = molecule_info.formula
    abundance = molecule_info.isotopes.abundance

    prob = []
    for isotope, n_label in difference_labels.items():
        # get number of atoms of isotope, default to 0
        n_elem_1 = label1.get(isotope, 0)
        elem = re.search(r"[A-Z][a-z]?", isotope).group(0)

        n_unlab = n_atoms[elem] - n_elem_1 - n_label
        abun_unlab = abundance[elem][0]
        abun_lab = abundance[elem][1]
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
