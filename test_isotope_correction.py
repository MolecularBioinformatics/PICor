import unittest

import src.isotope_correction as ic


class TestParse(unittest.TestCase):
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

    def test_label_bad_type(self):
        """Test that parse_label raises TypeError with list as input
        """
        data = ["1C13", "H02"]
        with self.assertRaises(TypeError):
            res = ic.parse_label(data)
