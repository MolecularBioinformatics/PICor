"""Unit tests for resolution dependent isotope correction."""

import itertools
from pathlib import Path
import pytest
import unittest

from picor.isotope_probabilities import MoleculeInfo
import picor.resolution_correction as rc


__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


class TestMassCalculations(unittest.TestCase):
    """Molecule mass and minimum mass difference."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = MoleculeInfo(
        "Test1", molecules_file=molecules_file, isotopes_file=isotopes_file,
    )

    def test_calc_min_mass_diff_result(self):
        """Minimal mass difference."""
        res = rc.calc_min_mass_diff(680, 2, 200, 50000)
        self.assertAlmostEqual(res, 0.029436, places=6)

    def test_calc_min_mass_diff_negative_mass(self):
        """Minimal mass difference."""
        with self.assertRaises(ValueError):
            rc.calc_min_mass_diff(-680, 2, 200, 50000)

    def test_is_overlap_true(self):
        """Overlapping isotopologues"""
        res = rc.is_isotologue_overlap("5C13", "4C13 1H02", self.molecule_info, 0.04,)
        self.assertTrue(res)

    def test_is_overlap_false(self):
        """Non overlapping isotopologues"""
        res = rc.is_isotologue_overlap("14C13", "14H02", self.molecule_info, 0.04,)
        self.assertFalse(res)

    def test_coarse_mass_difference(self):
        """Difference in nucleons."""
        res = rc.calc_coarse_mass_difference("No label", "5C13 3N15 2H02")
        self.assertEqual(res, 10)

    def test_fwhm_result(self):
        """Result with valid input."""
        mz_cal, mz, resolution = 200, 500, 50_000
        res = rc.fwhm(mz_cal, mz, resolution)
        self.assertAlmostEqual(res, 0.01581139)

    def test_fwhm_bad_input(self):
        """Result with valid input."""
        for mz_cal, mz, resolution in itertools.product([200, -200], repeat=3):
            if all(ele > 0 for ele in [mz_cal, mz, resolution]):
                # skip case in which all are positive
                continue
            with self.subTest():
                with self.assertRaises(ValueError):
                    rc.fwhm(mz_cal, mz, resolution)


class TestOverlapWarnings(unittest.TestCase):
    """Overlap warnings."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = MoleculeInfo(
        "Test1", molecules_file=molecules_file, isotopes_file=isotopes_file,
    )

    def test_direct_overlap_warn(self):
        """Warning with overlapping labels."""
        with self.assertWarns(UserWarning):
            rc.warn_direct_overlap(
                ["4H02", "4C13"], self.molecule_info, 0.05,
            )

    def test_indirect_overlap_warn(self):
        """Warning with indirectly overlapping labels."""
        label_list = [["3H02", "4C13"], ["1H02 1C13", "5C13"]]
        for labels in label_list:
            with self.assertWarns(UserWarning):
                rc.warn_indirect_overlap(
                    labels, self.molecule_info, 0.05,
                )
