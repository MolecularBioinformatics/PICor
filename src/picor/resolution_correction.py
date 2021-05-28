"""Isotopologue correction using statistical distributions.

Functions:
    calc_indirect_overlap_prob: probability of random H02 or C13
    warn_direct_overlap: warn if direct overlap due to resolution
    warn_indirect_overlap: warn if overlap due to H02 or C13 incorporation
"""
import itertools
import warnings

import pandas as pd

from picor.isotope_probabilities import calc_label_diff_prob, Label

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

    Parameters
    ----------
    label1 : Label
        Label of isotopologue 1
    label2 : Label
        Label of isotopologue 2

    Returns
    -------
    float
        Coarse difference

    Raises
    ------
    TypeError
        If label1 or label2 is not Label instance
    """
    if not isinstance(label1, Label) or not isinstance(label2, Label):
        raise TypeError("label1 and label2 have to be Label instances")
    return label2.mass - label1.mass


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
    """Calculate probability for overlap caused by random H02, C13 and O18 incoporation.

    Only O18, C13 and H02 attributions are considered so far.

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

    # Check if standard transition is possible
    label_diff = label1.get_diff_label_series(label2)
    if label_diff.ge(0).all():
        return 0

    probs = []
    contains_o = "O" in molecule_info.formula
    for (n_h, n_c, n_o) in generate_labels(coarse_mass_difference, contains_o):
        label_trans = {"H02": n_h, "C13": n_c}
        if contains_o:
            label_trans["O18"] = n_o
        label_trans_series = pd.Series(label_trans, dtype="float64")
        # TODO: label1_mod has to be label class
        label1_mod = dict(label1.as_series.add(label_trans_series, fill_value=0))
        if is_isotologue_overlap(label1_mod, label2, molecule_info, min_mass_diff,):
            print(label_trans)
            probs.append(calc_label_diff_prob(label1, label_trans, molecule_info))
    prob_total = sum(probs)
    return prob_total


def generate_labels(mass_diff, molecule_info):
    """Return combinations of H02, C13 and O18 for given mass diff."""
    elements = molecule_info.get_elements()
    for comb in itertools.product(range(0, mass_diff + 1), repeat=len(elements)):
        label = ip.Label(pd.Series(comb, index=elements))
        if label.get_coarse_mass_shift() == mass_diff:
            yield label


def warn_direct_overlap(
    label_list, molecule_info, min_mass_diff,
):
    """Warn if any of labels can have overlap."""
    for label1, label2 in itertools.permutations(label_list, 2):
        # Direct label overlap
        if is_isotologue_overlap(label1, label2, molecule_info, min_mass_diff,):
            warnings.warn(f"Direct overlap of {label1} and {label2}")
