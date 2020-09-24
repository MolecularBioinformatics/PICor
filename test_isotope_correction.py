"""Unit tests for isotope correction."""

from pathlib import Path
import unittest

import pandas as pd

import src.isotope_correction as ic


class TestLabels(unittest.TestCase):
    """Label and formula parsing."""

    metabolites_file = Path("test/test_metabolites.csv")
    isotopes_file = Path("test/test_isotopes.csv")

    def test_formula_string(self):
        """Parse_formula parses string correctly."""
        data = "C30Si12H2NO1P2"
        res_corr = {"C": 30, "Si": 12, "H": 2, "N": 1, "O": 1, "P": 2}
        res = ic.parse_formula(data)
        self.assertEqual(res, res_corr)

    def test_formula_bad_type(self):
        """Parse_formula raises TypeError with list as input."""
        data = ["C12", "H12"]
        with self.assertRaises(TypeError):
            ic.parse_formula(data)

    def test_label_string(self):
        """Parse_label parses string correctly."""
        data = "2N153C13H02"
        res_corr = {"N15": 2, "C13": 3, "H02": 1}
        res = ic.parse_label(data)
        self.assertEqual(res, res_corr)

    def test_label_no_label(self):
        """Parse_label parses 'No label' string correctly."""
        data = "No label"
        res_corr = {}
        res = ic.parse_label(data)
        self.assertEqual(res, res_corr)

    def test_label_bad_type(self):
        """Parse_label raises TypeError with list as input."""
        data = ["1C13", "H02"]
        with self.assertRaises(TypeError):
            ic.parse_label(data)

    def test_label_bad_isotope(self):
        """Parse_label raises ValueError with unrecognized string as input."""
        bad_isotope_list = ["C14", "B11", "Be9", "H02C14"]
        for data in bad_isotope_list:
            with self.assertRaises(ValueError):
                ic.parse_label(data)

    def test_label_empty_string(self):
        """Parse_label raises ValueError with empty input string."""
        data = ""
        with self.assertRaises(ValueError):
            ic.parse_label(data)

    def test_get_metabolite_formula_result(self):
        """Return valid formula."""
        res = ic.get_metabolite_formula(
            "Test1", self.metabolites_file, self.isotopes_file
        )
        self.assertDictEqual(res, {"C": 21, "H": 28, "N": 7, "O": 14, "P": 2})

    def test_get_metabolite_formula_invalid_element(self):
        """Raise ValueError with invalid element in metabolite formula."""
        with self.assertRaises(ValueError):
            res = ic.get_metabolite_formula(
                "Test3", self.metabolites_file, self.isotopes_file
            )

    def test_sort_list(self):
        """Sort_labels gives correct order with list of strings."""
        data = ["4C13", "3C13", "N154C13", "No label"]
        res_corr = ["No label", "3C13", "4C13", "N154C13"]
        res = ic.sort_labels(data)
        self.assertEqual(res, res_corr)

    def test_sort_bad_type(self):
        """TypeError with string as input."""
        data = "N15"
        with self.assertRaises(TypeError):
            ic.sort_labels(data)

    def test_label_smaller_true(self):
        """Label_shift_smaller returns True for label1 being smaller."""
        label1 = "C132N15"  # Mass shift of 3
        label2 = "5N15"  # Mass shift of 5
        res = ic.label_shift_smaller(label1, label2)
        self.assertTrue(res)

    def test_label_smaller_false(self):
        """Label_shift_smaller returns False for label1 being larger."""
        label1 = "C1310N15"  # Mass shift of 11
        label2 = "5N15"  # Mass shift of 5
        res = ic.label_shift_smaller(label1, label2)
        self.assertFalse(res)

    def test_label_smaller_equal(self):
        """Label_shift_smaller returns False for labels with equal mass."""
        label1 = "C1310N15"  # Mass shift of 11
        label2 = "11N15"  # Mass shift of 5
        res = ic.label_shift_smaller(label1, label2)
        self.assertFalse(res)


class TestCorrectionFactor(unittest.TestCase):
    """Calculation of correction factor."""

    metabolites_file = Path("test/test_metabolites.csv")
    isotopes_file = Path("test/test_isotopes.csv")

    def test_result_no_label(self):
        """Result with 'No label'."""
        res = ic.calc_correction_factor(
            "Test1",
            label="No label",
            metabolites_file=self.metabolites_file,
            isotopes_file=self.isotopes_file,
        )
        # corr_factor = 1 / res.total[0]
        self.assertAlmostEqual(res, 1.33471643)

    def test_result_with_label(self):
        """Result with complex label."""
        res = ic.calc_correction_factor(
            "Test1",
            label="10C131N1512H02",
            metabolites_file=self.metabolites_file,
            isotopes_file=self.isotopes_file,
        )
        # corr_factor = 1 / res.total[0]
        self.assertAlmostEqual(res, 1.19257588)

    def test_result_wrong_label(self):
        """ValueError with impossible label"""
        with self.assertRaises(ValueError):
            ic.calc_correction_factor(
                "Test1",
                label="100C131N15",
                metabolites_file=self.metabolites_file,
                isotopes_file=self.isotopes_file,
            )


class TestTransitionProbability(unittest.TestCase):
    """Calculation of probability between to isotopologues"""

    metabolites_file = Path("test/test_metabolites.csv")
    isotopes_file = Path("test/test_isotopes.csv")

    def test_result_label1_smaller(self):
        """Result with label1 being smaller than label2"""
        label1 = "1N15"
        label2 = "2N152C13"
        metabolite = {"C": 30, "Si": 12, "H": 2, "N": 3}
        res = ic.calc_transition_prob(
            label1,
            label2,
            metabolite,
            metabolites_file=self.metabolites_file,
            isotopes_file=self.isotopes_file,
        )
        self.assertAlmostEqual(res, 0.00026729)

    def test_result_label1_equal(self):
        """Result with label1 being equal to label2"""
        label1 = "1N15"
        metabolite = {"C": 30, "Si": 12, "H": 2, "N": 3}
        res = ic.calc_transition_prob(
            label1,
            label1,
            metabolite,
            metabolites_file=self.metabolites_file,
            isotopes_file=self.isotopes_file,
        )
        self.assertEqual(res, 0)

    def test_result_label1_larger(self):
        """Result with label1 being larger than label2"""
        label1 = "2N152C13"
        label2 = "2N15"
        metabolite = {"C": 30, "Si": 12, "H": 2, "N": 3}
        res = ic.calc_transition_prob(
            label1,
            label2,
            metabolite,
            metabolites_file=self.metabolites_file,
            isotopes_file=self.isotopes_file,
        )
        self.assertEqual(res, 0)

    def test_result_metabolite_formula(self):
        """Result with metabolite formula"""
        label1 = "1N15"
        label2 = "2N152C13"
        metabolite = "Test1"
        res = ic.calc_transition_prob(
            label1,
            label2,
            metabolite,
            metabolites_file=self.metabolites_file,
            isotopes_file=self.isotopes_file,
        )
        self.assertAlmostEqual(res, 0.00042029)

    def test_wrong_type(self):
        """Type error with list as metabolite"""
        label1 = "1N15"
        label2 = "2N152C13"
        metabolite = ["C30", "Si12", "H2", "N3"]
        with self.assertRaises(TypeError):
            ic.calc_transition_prob(
                label1,
                label2,
                metabolite,
                metabolites_file=self.metabolites_file,
                isotopes_file=self.isotopes_file,
            )


class TestIsotopologueCorrection(unittest.TestCase):
    """Total Correction Factor for metabolite and DataFrame"""

    metabolites_file = Path("test/test_metabolites.csv")
    isotopes_file = Path("test/test_isotopes.csv")

    def test_result(self):
        """Result with default values"""
        data = pd.read_csv(Path("test/test_dataset.csv"), index_col=0)
        data.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        metabolite = "Test1"
        res = ic.calc_isotopologue_correction(
            data,
            metabolite,
            metabolites_file=self.metabolites_file,
            isotopes_file=self.isotopes_file,
        )
        data_corrected = pd.read_csv(
            Path("test/test_dataset_corrected.csv"), index_col=0
        )
        pd.testing.assert_frame_equal(data_corrected, res)


class TestMassCalculations(unittest.TestCase):
    """Molecule mass and minimum mass difference."""

    metabolites_file = Path("test/test_metabolites.csv")
    isotopes_file = Path("test/test_isotopes.csv")

    def test_mass_nolabel(self):
        """Molecule mass calculation without label."""
        isotope_mass_series = ic.get_isotope_mass_series(self.isotopes_file)
        res = ic.calc_isotopologue_mass(
            "Test1", "No label", isotope_mass_series, self.isotopes_file
        )
        self.assertAlmostEqual(res, 664.116947, places=5)

    def test_mass_label(self):
        """Molecule mass calculation with correct label."""
        isotope_mass_series = ic.get_isotope_mass_series(self.isotopes_file)
        res = ic.calc_isotopologue_mass(
            "Test1", "15C13", isotope_mass_series, self.isotopes_file
        )
        self.assertAlmostEqual(res, 679.167270, places=5)

    def test_mass_bad_label(self):
        """ValueError for Molecule mass calculation with too large label."""
        isotope_mass_series = ic.get_isotope_mass_series(self.isotopes_file)
        with self.assertRaises(ValueError):
            ic.calc_isotopologue_mass(
                "Test1", "55C13", isotope_mass_series, self.isotopes_file
            )

    def test_calc_min_mass_diff_result(self):
        """Minimal mass difference."""
        res = ic.calc_min_mass_diff(680, 2, 200, 50000)
        self.assertAlmostEqual(res, 0.029436, places=6)

    def test_calc_min_mass_diff_negative_mass(self):
        """Minimal mass difference."""
        with self.assertRaises(ValueError):
            ic.calc_min_mass_diff(-680, 2, 200, 50000)

    def test_is_overlap_true(self):
        """Overlapping isotopologues"""
        isotope_mass_series = ic.get_isotope_mass_series(self.isotopes_file)
        res = ic.is_isotologue_overlap(
            "5C13", "4C13 1H02", "Test1", 0.04, isotope_mass_series, self.isotopes_file,
        )
        self.assertTrue(res)

    def test_is_overlap_false(self):
        """Non overlapping isotopologues"""
        isotope_mass_series = ic.get_isotope_mass_series(self.isotopes_file)
        res = ic.is_isotologue_overlap(
            "14C13", "14H02", "Test1", 0.04, isotope_mass_series, self.isotopes_file,
        )
        self.assertFalse(res)

    def test_coarse_mass_difference(self):
        """Difference in nucleons."""
        isotope_mass_series = ic.get_isotope_mass_series(self.isotopes_file)
        res = ic.calc_coarse_mass_difference(
            "No label", "5C13 3N15 2H02", isotope_mass_series
        )
        self.assertEqual(res, 10)
