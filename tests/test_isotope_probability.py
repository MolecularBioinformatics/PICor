"""Unit tests for isotope correction."""

from pathlib import Path
import unittest

import picor.isotope_probabilities as ip


__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


class TestLabels(unittest.TestCase):
    """Label and formula parsing."""

    metabolites_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")

    def test_formula_string(self):
        """Parse_formula parses string correctly."""
        data = "C30Si12H2NO1P2"
        res_corr = {"C": 30, "Si": 12, "H": 2, "N": 1, "O": 1, "P": 2}
        res = ip.parse_formula(data)
        self.assertEqual(res, res_corr)

    def test_formula_bad_type(self):
        """Parse_formula raises TypeError with list as input."""
        data = ["C12", "H12"]
        with self.assertRaises(TypeError):
            ip.parse_formula(data)

    def test_label_string(self):
        """Parse_label parses string correctly."""
        data = "2N153C13H02"
        res_corr = {"N15": 2, "C13": 3, "H02": 1}
        res = ip.parse_label(data)
        self.assertEqual(res, res_corr)

    def test_label_no_label(self):
        """Parse_label parses 'No label' string correctly."""
        data = "No label"
        res_corr = {}
        res = ip.parse_label(data)
        self.assertEqual(res, res_corr)

    def test_label_bad_type(self):
        """Parse_label raises TypeError with list as input."""
        data = ["1C13", "H02"]
        with self.assertRaises(TypeError):
            ip.parse_label(data)

    def test_label_bad_isotope(self):
        """Parse_label raises ValueError with unrecognized string as input."""
        bad_isotope_list = ["C14", "B11", "Be9", "H02C14"]
        for data in bad_isotope_list:
            with self.assertRaises(ValueError):
                ip.parse_label(data)

    def test_label_empty_string(self):
        """Parse_label raises ValueError with empty input string."""
        data = ""
        with self.assertRaises(ValueError):
            ip.parse_label(data)

    def test_get_metabolite_formula_result(self):
        """Return valid formula."""
        res = ip.get_metabolite_formula(
            "Test1", self.metabolites_file, self.isotopes_file
        )
        self.assertDictEqual(res, {"C": 21, "H": 28, "N": 7, "O": 14, "P": 2})

    def test_get_metabolite_formula_invalid_element(self):
        """Raise ValueError with invalid element in metabolite formula."""
        with self.assertRaises(ValueError):
            ip.get_metabolite_formula(
                "Test3", self.metabolites_file, self.isotopes_file
            )

    def test_sort_list(self):
        """Sort_labels gives correct order with list of strings."""
        data = ["4C13", "3C13", "N154C13", "No label"]
        res_corr = ["No label", "3C13", "4C13", "N154C13"]
        res = ip.sort_labels(data)
        self.assertEqual(res, res_corr)

    def test_sort_bad_type(self):
        """TypeError with string as input."""
        data = "N15"
        with self.assertRaises(TypeError):
            ip.sort_labels(data)

    def test_label_smaller_true(self):
        """Label_shift_smaller returns True for label1 being smaller."""
        label1 = "C132N15"  # Mass shift of 3
        label2 = "5N15"  # Mass shift of 5
        res = ip.label_shift_smaller(label1, label2)
        self.assertTrue(res)

    def test_label_smaller_false(self):
        """Label_shift_smaller returns False for label1 being larger."""
        label1 = "C1310N15"  # Mass shift of 11
        label2 = "5N15"  # Mass shift of 5
        res = ip.label_shift_smaller(label1, label2)
        self.assertFalse(res)

    def test_label_smaller_equal(self):
        """Label_shift_smaller returns False for labels with equal mass."""
        label1 = "C1310N15"  # Mass shift of 11
        label2 = "11N15"  # Mass shift of 5
        res = ip.label_shift_smaller(label1, label2)
        self.assertFalse(res)


class TestCorrectionFactor(unittest.TestCase):
    """Calculation of correction factor."""

    metabolites_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")

    def test_result_no_label(self):
        """Result with 'No label'."""
        res = ip.calc_correction_factor(
            "Test1",
            label="No label",
            metabolites_file=self.metabolites_file,
            isotopes_file=self.isotopes_file,
        )
        # corr_factor = 1 / res.total[0]
        self.assertAlmostEqual(res, 1.33471643)

    def test_result_with_label(self):
        """Result with complex label."""
        res = ip.calc_correction_factor(
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
            ip.calc_correction_factor(
                "Test1",
                label="100C131N15",
                metabolites_file=self.metabolites_file,
                isotopes_file=self.isotopes_file,
            )


class TestTransitionProbability(unittest.TestCase):
    """Calculation of probability between to isotopologues"""

    metabolites_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")

    def test_result_label1_smaller(self):
        """Result with label1 being smaller than label2"""
        label1 = "1N15"
        label2 = "2N152C13"
        metabolite = {"C": 30, "Si": 12, "H": 2, "N": 3}
        res = ip.calc_transition_prob(
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
        res = ip.calc_transition_prob(
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
        res = ip.calc_transition_prob(
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
        res = ip.calc_transition_prob(
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
            ip.calc_transition_prob(
                label1,
                label2,
                metabolite,
                metabolites_file=self.metabolites_file,
                isotopes_file=self.isotopes_file,
            )
