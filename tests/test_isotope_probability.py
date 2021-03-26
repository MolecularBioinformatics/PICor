"""Unit tests for isotope correction."""

from pathlib import Path
import unittest

import pandas
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

    def test_label_split_label(self):
        """Parse_label parses composite label correctly."""
        data = "NAD:2N153C13H02"
        res_corr = {"N15": 2, "C13": 3, "H02": 1}
        res = ip.parse_label(data)
        self.assertEqual(res, res_corr)

    def test_label_bad_split_label(self):
        """Parse_label raises ValueError with bad composite label."""
        data = "NAD:NamPT:2N153C13H02"
        with self.assertRaises(ValueError):
            ip.parse_label(data)

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

    def test_isotope_to_element_result(self):
        """Return correct elements."""
        label = {"N15": 2, "C13": 3, "H02": 1}
        res_corr = {"N": 2, "C": 3, "H": 1}
        res = ip.isotope_to_element(label)
        self.assertEqual(res, res_corr)

    def test_isotope_to_element_bad_element(self):
        """Raise ValueError for undefined element in isotope label."""
        label = {"O18": 2, "C13": 3, "H02": 1}
        with self.assertRaises(ValueError):
            ip.isotope_to_element(label)

class TestIsotopeInfo(unittest.TestCase):
    """Isotope file parsing."""

    isotopes_file = Path("tests/test_isotopes.csv")

    def test_init_attributes_type(self):
        """Type of instance attributes."""
        instance = ip.IsotopeInfo(self.isotopes_file)
        self.assertIsInstance(instance.abundance, dict)
        self.assertIsInstance(instance.isotopes_file, Path)
        self.assertIsInstance(instance.isotope_mass_series, pandas.Series)

    def test_get_isotope_abundance_result(self):
        """Return correct data."""
        res = ip.IsotopeInfo.get_isotope_abundance(self.isotopes_file)
        self.assertListEqual(
            ['H', 'C', 'N', 'O', 'Si', 'P', 'S'],
            list(res.keys()),
        )
        self.assertListEqual(
            res["C"], [0.9893, 0.0107]
        )

    def test_get_isotope_mass_series_keys(self):
        """Return correct Series keys."""
        res = ip.IsotopeInfo.get_isotope_mass_series(self.isotopes_file)
        self.assertListEqual(
            ['H01', 'H02', 'C12'],
            list(res.keys())[:3],
        )

    def test_get_isotope_mass_series_fields(self):
        """Return correct data."""
        res = ip.IsotopeInfo.get_isotope_mass_series(self.isotopes_file)
        self.assertEqual(13.003354835, res["C13"])

    def test_get_isotope_mass_series_names(self):
        """Return correct index and data names."""
        res = ip.IsotopeInfo.get_isotope_mass_series(self.isotopes_file)
        self.assertEqual("isotope", res.keys().name)
        self.assertEqual("mass", res.name)


class TestMoleculeInfo(unittest.TestCase):
    """Label and formula parsing."""

    metabolites_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")

    def test_init_attributes_type(self):
        """Type of instance attributes."""
        ins = ip.MoleculeInfo(
            "Test1", self.metabolites_file, self.isotopes_file
        )
        self.assertIsInstance(ins.molecule_name, str)
        self.assertIsInstance(ins.isotopes, ip.IsotopeInfo)
        self.assertIsInstance(ins.molecule_list, pandas.DataFrame)
        self.assertIsInstance(ins.formula, dict)

    def test_init_formula_result(self):
        """Return right fields"""
        ins = ip.MoleculeInfo(
            "Test1", self.metabolites_file, self.isotopes_file
        )
        self.assertDictEqual(ins.formula, {"C": 21, "H": 28, "N": 7, "O": 14, "P": 2})

    def test_molecule_name_unknown(self):
        """Raise KeyError with undefined molecule name."""
        with self.assertRaises(KeyError):
            ip.MoleculeInfo(
                "Unknown", self.metabolites_file, self.isotopes_file
            )

    def test_get_metabolite_formula_invalid_element(self):
        """Raise ValueError with invalid element in molecule formula."""
        with self.assertRaises(ValueError):
            ip.MoleculeInfo(
                "Test3", self.metabolites_file, self.isotopes_file
            )


class TestCorrectionFactor(unittest.TestCase):
    """Calculation of correction factor."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = ip.MoleculeInfo(
        "Test1",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )

    def test_result_no_label(self):
        """Result without label."""
        res = ip.calc_correction_factor(self.molecule_info)
        self.assertAlmostEqual(res, 1.33471643)

    def test_result_with_label(self):
        """Result with complex label."""
        res = ip.calc_correction_factor(self.molecule_info, label="10C131N1512H02",)
        self.assertAlmostEqual(res, 1.19257588)

    def test_result_wrong_label(self):
        """ValueError with impossible label."""
        with self.assertRaises(ValueError):
            ip.calc_correction_factor(
                self.molecule_info,
                label="100C131N15",
            )


class TestTransitionProbability(unittest.TestCase):
    """Calculation of probability between to isotopologues."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = ip.MoleculeInfo(
        "Test4",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )

    def test_result_labels_dict(self):
        """Result with label1 being smaller than label2 and both dict."""
        label1 = {"N15": 1}
        label2 = {"N15": 2, "C13": 2}
        res = ip.calc_transition_prob(
            label1,
            label2,
            self.molecule_info,
        )
        self.assertAlmostEqual(res, 0.00026729)

    def test_result_label1_smaller(self):
        """Result with label1 being smaller than label2."""
        label1 = "2N15"
        label2 = "2N152C13"
        res = ip.calc_transition_prob(
            label1,
            label2,
            self.molecule_info,
        )
        self.assertAlmostEqual(res, 0.03685030)

    def test_result_label1_equal(self):
        """Result with label1 being equal to label2."""
        label1 = "1N15"
        res = ip.calc_transition_prob(
            label1,
            label1,
            self.molecule_info,
        )
        self.assertEqual(res, 0)

    def test_result_label1_larger(self):
        """Result with label1 being larger than label2."""
        label1 = "2N152C13"
        label2 = "2N15"
        res = ip.calc_transition_prob(
            label1,
            label2,
            self.molecule_info,
        )
        self.assertEqual(res, 0)

    def test_result_metabolite_formula(self):
        """Result with metabolite formula."""
        label1 = "1N15"
        label2 = "2N152C13"
        molecule_info = ip.MoleculeInfo(
            "Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        res = ip.calc_transition_prob(
            label1,
            label2,
            molecule_info,
        )
        self.assertAlmostEqual(res, 0.00042029)

    def test_wrong_type(self):
        """Type error with dict as metabolite."""
        label1 = "1N15"
        label2 = "2N152C13"
        molecule = {"C": 12, "N": 15}
        with self.assertRaises(TypeError):
            ip.calc_transition_prob(
                label1,
                label2,
                molecule,
            )
