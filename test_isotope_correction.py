import unittest

import src.isotope_correction as ic


class TestLabels(unittest.TestCase):
    def test_formula_string(self):
        """Test that parse_formula parses string correctly
        """
        data = "C30Si12H2NO1P2"
        res_corr = {"C": 30, "Si": 12, "H": 2, "N": 1, "O": 1, "P": 2}
        res = ic.parse_formula(data)
        self.assertEqual(res, res_corr)

    def test_formula_bad_type(self):
        """Test that parse_formula raises TypeError with list as input
        """
        data = ["C12", "H12"]
        with self.assertRaises(TypeError):
            res = ic.parse_formula(data)

    def test_label_string(self):
        """Test that parse_label parses string correctly
        """
        data = "2N153C13H02"
        res_corr = {"N15": 2, "C13": 3, "H02": 1}
        res = ic.parse_label(data)
        self.assertEqual(res, res_corr)

    def test_label_no_label(self):
        """Test that parse_label parses 'No label' string correctly
        """
        data = "No label"
        res_corr = {}
        res = ic.parse_label(data)
        self.assertEqual(res, res_corr)

    def test_label_bad_type(self):
        """Test that parse_label raises TypeError with list as input
        """
        data = ["1C13", "H02"]
        with self.assertRaises(TypeError):
            res = ic.parse_label(data)

    def test_label_bad_isotope(self):
        """Test that parse_label raises ValueError with unrecognized string as input
        """
        bad_isotope_list = ["C14", "B11", "Be9", "H02C14"]
        for data in bad_isotope_list:
            with self.assertRaises(ValueError):
                res = ic.parse_label(data)

    def test_label_empty_string(self):
        """Test that parse_label raises ValueError with empty input string
        """
        data = ""
        with self.assertRaises(ValueError):
            res = ic.parse_label(data)

    def test_sort_list(self):
        """Test that sort_labels gives correct order with list of strings
        """
        data = ["4C13", "3C13", "N154C13", "No label"]
        res_corr = ["No label", "3C13", "4C13", "N154C13"]
        res = ic.sort_labels(data)
        self.assertEqual(res, res_corr)

    def test_sort_bad_type(self):
        """Test that sort_labels raises TypeError with string as input
        """
        data = "N15"
        with self.assertRaises(TypeError):
            res = ic.sort_labels(data)

    def test_label_smaller_true(self):
        """Test that label_shift_smaller returns True for label1 being smaller
        """
        label1 = "C132N15"  # Mass shift of 3
        label2 = "5N15"  # Mass shift of 5
        res = ic.label_shift_smaller(label1, label2)
        self.assertTrue(res)

    def test_label_smaller_false(self):
        """Test that label_shift_smaller returns False for label1 being larger
        """
        label1 = "C1310N15"  # Mass shift of 11
        label2 = "5N15"  # Mass shift of 5
        res = ic.label_shift_smaller(label1, label2)
        self.assertFalse(res)

    def test_label_smaller_equal(self):
        """Test that label_shift_smaller returns False for labels with equal mass
        """
        label1 = "C1310N15"  # Mass shift of 11
        label2 = "11N15"  # Mass shift of 5
        res = ic.label_shift_smaller(label1, label2)
        self.assertFalse(res)


# class Test
