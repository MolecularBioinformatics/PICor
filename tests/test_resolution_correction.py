"""Unit tests for resolution dependent isotope correction."""

import itertools
from numbers import Number
from pathlib import Path
import pytest
import unittest

from picor.isotope_probabilities import MoleculeInfo, Label, LabelTuple
import picor.resolution_correction as rc


__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


class TestResolutionCorrectionInfo(unittest.TestCase):
    """ResolutionCorrectionInfo class tests."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = MoleculeInfo.get_molecule_info(
        molecule_name="Test1",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )

    def test_init_return_types(self):
        """Return correct types for attributes."""
        inst = rc.ResolutionCorrectionInfo(True, 100000, 600, self.molecule_info)
        self.assertIsInstance(inst.do_correction, bool)
        self.assertIsInstance(inst.resolution, Number)
        self.assertIsInstance(inst.mz_calibration, Number)
        self.assertIsInstance(inst.molecule_info, MoleculeInfo)
        self.assertIsInstance(inst.molecule_mass, Number)
        self.assertIsInstance(inst.min_mass_diff, Number)


class TestMassCalculations(unittest.TestCase):
    """Molecule mass and minimum mass difference."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = MoleculeInfo.get_molecule_info(
        molecule_name="Test1",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )
    res_corr_info = rc.ResolutionCorrectionInfo(False, 60000, 200, molecule_info)

    def test_calc_min_mass_diff_result(self):
        """Minimal mass difference."""
        res = rc.ResolutionCorrectionInfo.calc_min_mass_diff(680, 2, 200, 50000)
        self.assertAlmostEqual(res, 0.029436, places=6)

    def test_calc_min_mass_diff_negative_mass(self):
        """Minimal mass difference."""
        with self.assertRaises(ValueError):
            rc.ResolutionCorrectionInfo.calc_min_mass_diff(-680, 2, 200, 50000)

    def test_is_overlap_true(self):
        """Overlapping isotopologues"""
        res = rc.is_isotologue_overlap(
            Label("5C13", self.molecule_info),
            Label("4C13 1H02", self.molecule_info),
            self.res_corr_info,
        )
        self.assertTrue(res)

    def test_is_overlap_false(self):
        """Non overlapping isotopologues"""
        res = rc.is_isotologue_overlap(
            Label("14C13", self.molecule_info),
            Label("14H02", self.molecule_info),
            self.res_corr_info,
        )
        self.assertFalse(res)

    def test_coarse_mass_difference_result(self):
        """Difference in nucleons."""
        res = rc.calc_coarse_mass_difference(
            Label("No label", self.molecule_info),
            Label("5C13 3N15 2H02", self.molecule_info),
        )
        self.assertEqual(res, 10)

    def test_coarse_mass_difference_bad_type(self):
        """Non Label as input."""
        label = Label("No label", self.molecule_info)
        non_label = {"C13": 12}
        label_list = [(label, non_label), (non_label, label), (non_label, non_label)]
        for la1, la2 in label_list:
            with self.subTest():
                with self.assertRaises(TypeError):
                    rc.calc_coarse_mass_difference(la1, la2)

    def test_fwhm_result(self):
        """Result with valid input."""
        mz_cal, mz, resolution = 200, 500, 50_000
        res = rc.ResolutionCorrectionInfo.fwhm(mz_cal, mz, resolution)
        self.assertAlmostEqual(res, 0.01581139)

    def test_fwhm_bad_input(self):
        """Result with valid input."""
        for mz_cal, mz, resolution in itertools.product([200, -200], repeat=3):
            if all(ele > 0 for ele in [mz_cal, mz, resolution]):
                # skip case in which all are positive
                continue
            with self.subTest():
                with self.assertRaises(ValueError):
                    rc.ResolutionCorrectionInfo.fwhm(mz_cal, mz, resolution)


class TestGenerateLabels(unittest.TestCase):
    """generate_labels."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = MoleculeInfo.get_molecule_info(
        molecule_name="Test1",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )
    res_corr_info = rc.ResolutionCorrectionInfo(False, 60000, 200, molecule_info)

    def test_generate_labels_res(self):
        """Return correct result fo rsmall example."""
        res = list(rc.generate_labels(2, self.res_corr_info))
        res_corr = [
            Label(la, self.molecule_info)
            for la in [
                "1O18",
                "2O17",
                "1N151O17",
                "2N15",
                "1H021O17",
                "1H021N15",
                "2H02",
                "1C131O17",
                "1C131N15",
                "1C131H02",
                "2C13",
            ]
        ]
        self.assertListEqual(res, res_corr)

    def test_generate_labels_error(self):
        """Catch ValueError for generated labels too large for molecule."""
        mol = MoleculeInfo.get_molecule_info(
            molecule_formula="C2H8O1",
            molecule_charge=1,
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        res_corr_info = rc.ResolutionCorrectionInfo(False, 60000, 200, mol)
        res = len(list(rc.generate_labels(3, res_corr_info)))
        self.assertEqual(res, 9)


class TestOverlapWarnings(unittest.TestCase):
    """Overlap warnings."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = MoleculeInfo.get_molecule_info(
        molecule_name="Test1",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )
    res_corr_info = rc.ResolutionCorrectionInfo(False, 60000, 200, molecule_info)

    def test_direct_overlap_warn(self):
        """Warning with overlapping labels."""
        with self.assertWarns(UserWarning):
            rc.warn_direct_overlap(
                LabelTuple(["4H02", "4C13"], self.molecule_info), self.res_corr_info
            )

    def test_direct_overlap_not_warn(self):
        """No warning with non-overlapping labels."""
        with pytest.warns(None) as warnings:
            rc.warn_direct_overlap(
                LabelTuple(["3H02", "4C13"], self.molecule_info), self.res_corr_info
            )
            assert not warnings

    def test_indirect_overlap_warn(self):
        """Warning with indirectly overlapping labels."""
        res_corr_info_high = rc.ResolutionCorrectionInfo(
            False, 200000, 200, self.molecule_info
        )
        label_list = [
            LabelTuple(["3H02", "4C13"], self.molecule_info),
            LabelTuple(["1H02 1C13", "5C13"], self.molecule_info),
            LabelTuple(["2N15 1H02", "2N15 3C13"], self.molecule_info),
        ]
        for labels in label_list:
            with self.subTest():
                with self.assertWarns(UserWarning):
                    rc.warn_indirect_overlap(labels, res_corr_info_high)
