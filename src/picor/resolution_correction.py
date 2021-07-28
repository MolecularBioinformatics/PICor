"""Isotopologue correction using statistical distributions.

Functions:
    calc_indirect_overlap_prob: probability of random H02 or C13
    warn_direct_overlap: warn if direct overlap due to resolution
    warn_indirect_overlap: warn if overlap due to H02 or C13 incorporation
"""
import itertools
import logging
import warnings

import pandas as pd

from picor.isotope_probabilities import calc_label_diff_prob, Label

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"

_logger = logging.getLogger(__name__)


class ResolutionCorrectionInfo:
    """Class to store all parameters for resolution correction."""

    def __init__(self, do_correction, resolution, mz_calibration, molecule_info):
        self.do_correction = bool(do_correction)
        self.resolution = float(resolution)
        self.mz_calibration = float(mz_calibration)
        self.molecule_info = molecule_info
        self.molecule_mass = Label("No label", molecule_info).calc_isotopologue_mass()
        self.min_mass_diff = self.calc_min_mass_diff(
            self.molecule_mass, molecule_info.charge, mz_calibration, resolution,
        )
        # TODO Replace hardcoded  indirect_overlap_cutoff with flexible approach
        self.indirect_overlap_cutoff = 5

    def __repr__(self):
        return (
            "ResolutionCorrectionInfo("
            f"do_correction={self.do_correction}, resolution={self.resolution}, "
            f"mz_calibration={self.mz_calibration}, molecule_info={self.molecule_info})"
        )

    @staticmethod
    def fwhm(mz_cal, mz, resolution):
        """Calculate Full width half maximum (FWHM)."""
        if mz_cal < 0 or mz <= 0 or resolution <= 0:
            raise ValueError("Arguments must be positive")
        return mz ** (3 / 2) / (resolution * mz_cal ** 0.5)

    @staticmethod
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
        return (
            1.66 * abs(charge) * ResolutionCorrectionInfo.fwhm(mz_cal, mz, resolution)
        )


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
    label1, label2, res_corr_info,
):
    """Return True if label1 and label2 are too close to detection limit.

    Checks whether two isotopologues defined by label and metabolite name
    are below miniumum resolved mass difference.
    """
    mass1 = label1.calc_isotopologue_mass()
    mass2 = label2.calc_isotopologue_mass()
    return abs(mass1 - mass2) < res_corr_info.min_mass_diff


def warn_indirect_overlap(label_list, res_corr_info):
    """Warn if any of labels can have indirect overlap."""
    for label1, label2 in itertools.permutations(label_list, 2):
        prob = calc_indirect_overlap_prob(label1, label2, res_corr_info)
        if prob:
            warning = f"{label1} indirect overlap with {label2} with prob {prob:.4f}"
            warnings.warn(warning)
            _logger.warning(warning)


def calc_indirect_overlap_prob(label1, label2, res_corr_info):
    """Calculate probability for overlap caused by random H02, C13 and O18 incoporation.

    Only O18, C13 and H02 attributions are considered so far.

    Parameters
    ----------
    label1 : str
        "No label" or formula e.g. "1C13 2H02"
    label2 : str
        "No label" or formula e.g. "1C13 2H02"
    res_corr_info : ResolutionCorrectionInfo
        Instance with resolution and mz calibration info.

    Returns
    -------
    float
        Transition probability of overlap
    """
    _logger.info(f"Indirect overlap prob for {label1} -> {label2}")
    # Label overlap possible with additional atoms
    coarse_mass_difference = calc_coarse_mass_difference(label1, label2)
    if (
        coarse_mass_difference <= 0
        or coarse_mass_difference >= res_corr_info.indirect_overlap_cutoff
    ):
        return 0

    probs = []
    for label_trans in generate_labels(coarse_mass_difference, res_corr_info):
        label1_mod = label1.add(label_trans)
        if is_isotologue_overlap(label1_mod, label2, res_corr_info):
            prob = calc_label_diff_prob(label1, label_trans)
            _logger.debug(f"For trans label {label_trans}: {prob}")
            probs.append(prob)
    prob_total = sum(probs)
    _logger.info(f"total indirect prob: {prob_total}\n")
    return prob_total


def generate_labels(mass_diff, res_corr_info):
    """Return combinations of possible isotopes e.g. H02 and C13 for given mass diff."""
    molecule_info = res_corr_info.molecule_info
    shift = molecule_info.isotopes.isotope_shift[molecule_info.get_isotopes()]
    isotopes = shift[shift > 0]  # only use isotope that cause mass shift
    for comb in itertools.product(range(0, mass_diff + 1), repeat=len(isotopes.index)):
        mass_shift = sum(isotopes * comb)
        if mass_shift != mass_diff:
            continue
        try:
            label = Label(pd.Series(comb, index=isotopes.index), molecule_info)
        except ValueError:
            continue
        yield label


def warn_direct_overlap(label_list, res_corr_info):
    """Warn if any of labels can have overlap."""
    for label1, label2 in itertools.permutations(label_list, 2):
        # Direct label overlap
        if is_isotologue_overlap(label1, label2, res_corr_info):
            warning = f"Direct overlap of {label1} and {label2}"
            warnings.warn(warning)
            _logger.warning(warning)
