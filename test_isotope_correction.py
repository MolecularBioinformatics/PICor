"""Unit tests for isotope correction."""
import unittest

import src.isotope_correction as ic


class TestLabels(unittest.TestCase):
    """Label and formula parsing."""

    def test_formula_string(self):
        """Test that parse_formula parses string correctly."""
        data = "C30Si12H2NO1P2"
        res_corr = {"C": 30, "Si": 12, "H": 2, "N": 1, "O": 1, "P": 2}
        res = ic.parse_formula(data)
        self.assertEqual(res, res_corr)

    def test_formula_bad_type(self):
        """Test that parse_formula raises TypeError with list as input."""
        data = ["C12", "H12"]
        with self.assertRaises(TypeError):
            ic.parse_formula(data)

    def test_label_string(self):
        """Test that parse_label parses string correctly."""
        data = "2N153C13H02"
        res_corr = {"N15": 2, "C13": 3, "H02": 1}
        res = ic.parse_label(data)
        self.assertEqual(res, res_corr)

    def test_label_no_label(self):
        """Test that parse_label parses 'No label' string correctly."""
        data = "No label"
        res_corr = {}
        res = ic.parse_label(data)
        self.assertEqual(res, res_corr)

    def test_label_bad_type(self):
        """Test that parse_label raises TypeError with list as input."""
        data = ["1C13", "H02"]
        with self.assertRaises(TypeError):
            ic.parse_label(data)

    def test_label_bad_isotope(self):
        """Test that parse_label raises ValueError with unrecognized string as input."""
        bad_isotope_list = ["C14", "B11", "Be9", "H02C14"]
        for data in bad_isotope_list:
            with self.assertRaises(ValueError):
                ic.parse_label(data)

    def test_label_empty_string(self):
        """Test that parse_label raises ValueError with empty input string."""
        data = ""
        with self.assertRaises(ValueError):
            ic.parse_label(data)

    def test_sort_list(self):
        """Test that sort_labels gives correct order with list of strings."""
        data = ["4C13", "3C13", "N154C13", "No label"]
        res_corr = ["No label", "3C13", "4C13", "N154C13"]
        res = ic.sort_labels(data)
        self.assertEqual(res, res_corr)

    def test_sort_bad_type(self):
        """Test that sort_labels raises TypeError with string as input."""
        data = "N15"
        with self.assertRaises(TypeError):
            ic.sort_labels(data)

    def test_label_smaller_true(self):
        """Test that label_shift_smaller returns True for label1 being smaller."""
        label1 = "C132N15"  # Mass shift of 3
        label2 = "5N15"  # Mass shift of 5
        res = ic.label_shift_smaller(label1, label2)
        self.assertTrue(res)

    def test_label_smaller_false(self):
        """Test that label_shift_smaller returns False for label1 being larger."""
        label1 = "C1310N15"  # Mass shift of 11
        label2 = "5N15"  # Mass shift of 5
        res = ic.label_shift_smaller(label1, label2)
        self.assertFalse(res)

    def test_label_smaller_equal(self):
        """Test that label_shift_smaller returns False for labels with equal mass."""
        label1 = "C1310N15"  # Mass shift of 11
        label2 = "11N15"  # Mass shift of 5
        res = ic.label_shift_smaller(label1, label2)
        self.assertFalse(res)


class TestCorrectionFactor(unittest.TestCase):
    """Calculation of correction factor."""

    def test_result_no_label(self):
        """Result with 'No label'."""
        res = ic.calc_correction_factor("NAD", label="No label")
        # corr_factor = 1 / res.total[0]
        self.assertAlmostEqual(res, 1.33471643)

    def test_result_with_label(self):
        """Result with complex label."""
        res = ic.calc_correction_factor("NAD", label="10C131N1512H02")
        # corr_factor = 1 / res.total[0]
        self.assertAlmostEqual(res, 1.19257588)

    def test_result_wrong_label(self):
        """ValueError with impossible label"""
        with self.assertRaises(ValueError):
            ic.calc_correction_factor("NAD", label="100C131N15")


class TestTransitionProbability(unittest.TestCase):
    """Calculation of probability between to isotopologues"""

    def test_result_label1_smaller(self):
        """Result with label1 being smaller than label2"""
        label1 = "1N15"
        label2 = "2N152C13"
        metabolite = {"C": 30, "Si": 12, "H": 2, "N": 3}
        res = ic.calc_transition_prob(
            label1,
            label2,
            metabolite,
            "~/isocordb/Metabolites.dat",
            "~/isocordb/Isotopes.dat",
        )
        self.assertAlmostEqual(res, 0.00040094)

    def test_result_label1_equal(self):
        """Result with label1 being equal to label2"""
        label1 = "1N15"
        metabolite = {"C": 30, "Si": 12, "H": 2, "N": 3}
        res = ic.calc_transition_prob(
            label1,
            label1,
            metabolite,
            "~/isocordb/Metabolites.dat",
            "~/isocordb/Isotopes.dat",
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
            "~/isocordb/Metabolites.dat",
            "~/isocordb/Isotopes.dat",
        )
        self.assertEqual(res, 0)

    def test_result_metabolite_formula(self):
        """Result with metabolite formula"""
        label1 = "1N15"
        label2 = "2N152C13"
        metabolite = "NAD"
        res = ic.calc_transition_prob(
            label1,
            label2,
            metabolite,
            "~/isocordb/Metabolites.dat",
            "~/isocordb/Isotopes.dat",
        )
        self.assertAlmostEqual(res, 0.00049034)

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
                "~/isocordb/Metabolites.dat",
                "~/isocordb/Isotopes.dat",
            )
