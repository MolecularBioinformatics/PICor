"""Unit tests for isotope correction."""

from pathlib import Path
import unittest

import pandas
from pandas.testing import assert_series_equal

import picor.isotope_probabilities as ip


__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


class TestLabels(unittest.TestCase):
    """Label and formula parsing."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = ip.MoleculeInfo.get_molecule_info(
        molecule_name="Test1",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )

    def test_init_label_dict(self):
        """Init Label instance with dict as input."""
        data = {"C13": 2, "O18": 3, "H02": 2}
        res = ip.Label(data, self.molecule_info)
        self.assertEqual(res.as_dict, data)

    def test_init_label_series(self):
        """Init Label instance with pandas Series as input."""
        data = pandas.Series([2, 3, 2], index=["C13", "O18", "H02"])
        res_corr = {"C13": 2, "O18": 3, "H02": 2}
        res = ip.Label(data, self.molecule_info)
        self.assertEqual(res.as_dict, res_corr)

    def test_eq_true(self):
        """Test __eq__ for same Label Instance."""
        data = {"C13": 2, "O18": 3, "H02": 2}
        label1 = ip.Label(data, self.molecule_info)
        label2 = ip.Label(data, self.molecule_info)
        self.assertTrue(label1 == label2)

    def test_eq_false(self):
        """Test __eq__ for different Label Instance."""
        data1 = {"C13": 2, "O18": 3, "H02": 2}
        data2 = {"C13": 1, "O18": 3, "H02": 2}
        label1 = ip.Label(data1, self.molecule_info)
        label2 = ip.Label(data2, self.molecule_info)
        self.assertFalse(label1 == label2)

    def test_eq_false_type(self):
        """Test __eq__ for differnt types."""
        data = {"C13": 2, "O18": 3, "H02": 2}
        label = ip.Label(data, self.molecule_info)
        iso = ip.IsotopeInfo(self.isotopes_file)
        self.assertFalse(label == iso)

    def test_lt(self):
        """Test __lt__."""
        label1 = ip.Label({"C13": 2, "H02": 3}, self.molecule_info)
        label2 = ip.Label({"O18": 3, "H02": 1}, self.molecule_info)
        self.assertTrue(label1 < label2)

    def test_ge(self):
        """Test __ge__."""
        label1 = ip.Label({"C13": 2, "H02": 3}, self.molecule_info)
        label2 = ip.Label({"O18": 1, "H02": 1}, self.molecule_info)
        label3 = ip.Label({"O18": 1, "H02": 3}, self.molecule_info)
        self.assertTrue(label1 >= label2)
        self.assertTrue(label1 >= label3)

    def test_le(self):
        """Test __le__."""
        label1 = ip.Label({"C13": 2, "H02": 3}, self.molecule_info)
        label2 = ip.Label({"O18": 3, "H02": 1}, self.molecule_info)
        label3 = ip.Label({"O18": 2, "H02": 1}, self.molecule_info)
        self.assertTrue(label1 <= label2)
        self.assertTrue(label1 <= label3)

    def test_gt(self):
        """Test __gt__."""
        label1 = ip.Label({"C13": 2, "H02": 3}, self.molecule_info)
        label2 = ip.Label({"O18": 1, "H02": 1}, self.molecule_info)
        self.assertTrue(label1 > label2)

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
        res = ip.Label(data, self.molecule_info)
        self.assertEqual(res.as_dict, res_corr)

    def test_label_wrong_atoms(self):
        """ValueError for atom not in molecule."""
        with self.assertRaises(ValueError):
            ip.Label("S33", self.molecule_info)

    def test_label_too_many_atoms(self):
        """ValueError for too many atoms in label."""
        with self.assertRaises(ValueError):
            ip.Label("55C13", self.molecule_info)

    def test_label_no_label(self):
        """Parse_label parses 'No label' string correctly."""
        data = "No label"
        res_corr = {}
        res = ip.Label(data, self.molecule_info)
        self.assertEqual(res.as_dict, res_corr)

    def test_label_bad_type(self):
        """Parse_label raises TypeError with list as input."""
        data = ["1C13", "H02"]
        with self.assertRaises(TypeError):
            ip.Label(data, self.molecule_info)

    def test_parse_label_zero_value(self):
        """Parse_label removes zero value elements."""
        string = "3C13 0N15 1H02"
        res_corr = {"C13": 3, "H02": 1}
        res = ip.Label.parse_label(string)
        self.assertEqual(res, res_corr)

    def test_parse_label_bad_type(self):
        """Parse_label raises TypeError for wrong Type."""
        bad_type_list = [["C13"], {"C13": 2}, 2]
        for data in bad_type_list:
            with self.assertRaises(TypeError):
                ip.Label.parse_label(data)

    def test_label_empty_string(self):
        """Parse_label raises ValueError with empty input string."""
        data = ""
        with self.assertRaises(ValueError):
            ip.Label(data, self.molecule_info)

    def test_label_split_label(self):
        """Parse_label parses composite label correctly."""
        data = "NAD:2N153C13H02"
        res_corr = {"N15": 2, "C13": 3, "H02": 1}
        res = ip.Label(data, self.molecule_info)
        self.assertEqual(res.as_dict, res_corr)

    def test_label_bad_split_label(self):
        """Parse_label raises ValueError with bad composite label."""
        data = "NAD:NamPT:2N153C13H02"
        with self.assertRaises(ValueError):
            ip.Label(data, self.molecule_info)

    def test_str_result(self):
        """Return correct string representation."""
        data = "2N153C13H02"
        res = str(ip.Label(data, self.molecule_info))
        res_corr = "Label: 2N153C13H02"
        self.assertEqual(res, res_corr)

    def test_generate_label_string(self):
        """generate_label_string return correct string."""
        data = {"N15": 2, "C13": 3, "H02": 1}
        res_corr = "2N15 3C13 1H02"
        res = ip.Label.generate_label_string(data)
        self.assertEqual(res, res_corr)

    def test_sort_list(self):
        """Sort_labels gives correct order with list of strings."""
        data = [
            ip.Label(label, self.molecule_info)
            for label in ["4C13", "3C13", "N154C13", "No label"]
        ]
        res_corr = ["No label", "3C13", "4C13", "N154C13"]
        res = ip.sort_labels(data)
        for t, c in zip(res, res_corr):
            self.assertEqual(t.as_string, c)

    def test_isotope_to_element_result(self):
        """Return correct elements."""
        label = ip.Label("2N15 3C13 1H02", self.molecule_info)
        res_corr = {"N": 2, "C": 3, "H": 1}
        res = label.isotope_to_element(label)
        self.assertEqual(res, res_corr)

    def test_add_result(self):
        """Return correct elements."""
        label1 = ip.Label("1N15 1C13", self.molecule_info)
        label2 = ip.Label("2N15 3C13 1H02", self.molecule_info)
        res_corr = ip.Label(
            pandas.Series({"C13": 4, "H02": 1, "N15": 3}), self.molecule_info
        )
        res = label2.add(label1)
        assert_series_equal(res.as_series, res_corr.as_series)

    def test_add_different_molecule(self):
        """Raise ValueError if labels have different molecule_info instances."""
        molecule_info1 = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        molecule_info2 = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test4",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        label1 = ip.Label("1N15 1C13", molecule_info1)
        label2 = ip.Label("2N15 3C13 1H02", molecule_info2)
        with self.assertRaises(ValueError):
            label1.add(label2)

    def test_subtract_result(self):
        """Return correct elements."""
        label1 = ip.Label("1N15 1C13", self.molecule_info)
        label2 = ip.Label("2N15 3C13 1H02", self.molecule_info)
        res_corr = ip.Label(
            pandas.Series({"C13": 2, "H02": 1, "N15": 1}), self.molecule_info
        )
        res = label2.subtract(label1)
        assert_series_equal(res.as_series, res_corr.as_series)

    def test_subtract_bad_type(self):
        """Raise TypeError for dict as label."""
        label1 = ip.Label("2O18 3C13 1H02", self.molecule_info)
        label2 = {"N": 1, "C": 2, "H": 1}
        with self.assertRaises(TypeError):
            label1.subtract(label2)

    def test_subtract_different_molecule(self):
        """Raise ValueError if labels have different molecule_info instances."""
        molecule_info1 = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        molecule_info2 = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test4",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        label1 = ip.Label("1N15 1C13", molecule_info1)
        label2 = ip.Label("2N15 3C13 1H02", molecule_info2)
        with self.assertRaises(ValueError):
            label1.subtract(label2)

    def test_check_isotopes_error(self):
        """ValueError is raised for unknown isotope."""
        with self.assertRaises(ValueError):
            ip.Label("2O15 3C13 1H02", self.molecule_info)

    def test_mass_nolabel(self):
        """Molecule mass calculation without label."""
        label = ip.Label("No label", self.molecule_info)
        res = label.calc_isotopologue_mass()
        self.assertAlmostEqual(res, 664.116947, places=5)

    def test_mass_label(self):
        """Molecule mass calculation with correct label."""
        label = ip.Label("15C13", self.molecule_info)
        res = label.calc_isotopologue_mass()
        self.assertAlmostEqual(res, 679.167270, places=5)


class TestLabelTuple(unittest.TestCase):
    """LabelTuple creation."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = ip.MoleculeInfo.get_molecule_info(
        molecule_name="Test1",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )

    def test_init_result(self):
        """Correct results for list of labels."""
        data = ["C13", "2C13 2H02", "H02"]
        res = ip.LabelTuple(data, self.molecule_info)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[2], ip.Label("2C13 2H02", self.molecule_info))
        self.assertEqual(res.molecule_info, self.molecule_info)

    def test_getitem(self):
        """Get value."""
        data = ["C13", "2C13 2H02", "H02"]
        lt = ip.LabelTuple(data, self.molecule_info)
        self.assertEqual(lt[2], ip.Label("2C13 2H02", self.molecule_info))


class TestIsotopeInfo(unittest.TestCase):
    """Isotope file parsing."""

    isotopes_file = Path("tests/test_isotopes.csv")

    def test_init_attributes_type(self):
        """Type of instance attributes."""
        instance = ip.IsotopeInfo(self.isotopes_file)
        self.assertIsInstance(instance.abundance, pandas.Series)
        self.assertIsInstance(instance.isotopes_file, Path)
        self.assertIsInstance(instance.isotope_mass_series, pandas.Series)

    def test_eq_true(self):
        """Test __eq__ for same IsotopeInfo Instance."""
        iso1 = ip.IsotopeInfo(self.isotopes_file)
        iso2 = ip.IsotopeInfo(self.isotopes_file)
        self.assertTrue(iso1 == iso2)

    def test_eq_false(self):
        """Test __eq__ for different IsotopeInfo Instance."""
        isotopes_file2 = Path("tests/test_isotopes2.csv")
        iso1 = ip.IsotopeInfo(self.isotopes_file)
        iso2 = ip.IsotopeInfo(isotopes_file2)
        self.assertFalse(iso1 == iso2)

    def test_eq_false_type(self):
        """Test __eq__ for differnt types."""
        iso = ip.IsotopeInfo(self.isotopes_file)
        other = {}
        self.assertFalse(iso == other)

    def test_get_isotope_abundance_result(self):
        """Return correct data."""
        res = ip.IsotopeInfo.get_isotope_abundance(self.isotopes_file)
        res_corr = [
            "H01",
            "H02",
            "C12",
            "C13",
            "N14",
            "N15",
            "O16",
            "O17",
            "O18",
            "Si28",
            "Si29",
            "Si30",
            "P31",
            "S32",
            "S33",
            "S34",
            "S35",
            "S36",
        ]
        self.assertListEqual(
            res_corr, list(res.index),
        )
        self.assertEqual(res["C12"], 0.9893)

    def test_get_isotope_mass_series_keys(self):
        """Return correct Series keys."""
        res = ip.IsotopeInfo.get_isotope_mass_series(self.isotopes_file)
        self.assertListEqual(
            ["H01", "H02", "C12"], list(res.keys())[:3],
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

    def test_get_elements(self):
        """Return correct list of elements."""
        res = ip.IsotopeInfo.get_elements(self.isotopes_file)
        self.assertSetEqual({"H", "C", "O", "N", "Si", "S", "P"}, res)

    def test_get_isotopes_from_elements(self):
        """Correct list of isotopes."""
        iso = ip.IsotopeInfo(self.isotopes_file)
        ele = ["C", "O", "H"]
        res = iso.get_isotopes_from_elements(ele)
        res_corr = ["C12", "C13", "O16", "O17", "O18", "H01", "H02"]
        self.assertListEqual(res, res_corr)


class TestMoleculeInfo(unittest.TestCase):
    """Label and formula parsing."""

    molecules_file = Path("tests/test_metabolites.csv")
    molecules_file2 = Path("tests/test_metabolites_missing_charge.csv")
    isotopes_file = Path("tests/test_isotopes.csv")

    def test_init_formula(self):
        """Return right fields"""
        isotopes = ip.IsotopeInfo(self.isotopes_file)
        formula = {"C": 21, "H": 28, "N": 7, "O": 14, "P": 2}
        charge = 1
        ins = ip.MoleculeInfo(formula, charge, isotopes)
        self.assertIsInstance(ins, ip.MoleculeInfo)

    def test_get_molecule_info_formula_type(self):
        """Type of instance attributes."""
        ins = ip.MoleculeInfo.get_molecule_info(
            molecule_formula="C21H28N7O14P2",
            molecule_charge=1,
            isotopes_file=self.isotopes_file,
        )
        self.assertIsInstance(ins.isotopes, ip.IsotopeInfo)
        self.assertIsInstance(ins.charge, int)
        self.assertIsInstance(ins.formula, dict)

    def test_get_molecule_info_formula_result(self):
        """Return right fields"""
        ins = ip.MoleculeInfo.get_molecule_info(
            molecule_formula="C21H28N7O14P2",
            molecule_charge=1,
            isotopes_file=self.isotopes_file,
        )
        self.assertEqual(ins.charge, +1)
        self.assertDictEqual(ins.formula, {"C": 21, "H": 28, "N": 7, "O": 14, "P": 2})

    def test_get_molecule_info_name_type(self):
        """Type of instance attributes."""
        ins = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        self.assertIsInstance(ins.isotopes, ip.IsotopeInfo)
        self.assertIsInstance(ins.charge, int)
        self.assertIsInstance(ins.formula, dict)

    def test_get_molecule_info_name_result(self):
        """Return right fields"""
        ins = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        self.assertEqual(ins.charge, +1)
        self.assertDictEqual(ins.formula, {"C": 21, "H": 28, "N": 7, "O": 14, "P": 2})

    def test_get_molecule_info_wrong_arguments(self):
        """Raise ValueError when too many arguments."""
        with self.assertRaises(ValueError):
            ip.MoleculeInfo.get_molecule_info(
                molecule_name="Test1",
                molecule_formula="C21H28N7O14P2",
                molecules_file=self.molecules_file,
                isotopes_file=self.isotopes_file,
            )

    def test_eq_true(self):
        """Test __eq__ for same MoleculeInfo Instances."""
        molecule1 = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        molecule2 = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        self.assertTrue(molecule1 == molecule2)

    def test_eq_false(self):
        """Test __eq__ for different MoleculeInfo Instance."""
        molecule1 = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        molecule2 = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test4",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        self.assertFalse(molecule1 == molecule2)

    def test_eq_false_type(self):
        """Test __eq__ for different types."""
        molecule = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        iso = ip.IsotopeInfo(self.isotopes_file)
        self.assertFalse(molecule == iso)

    def test_molecule_name_unknown(self):
        """Raise KeyError with undefined molecule name."""
        with self.assertRaises(KeyError):
            ip.MoleculeInfo.get_molecule_info(
                molecule_name="Unknown",
                molecules_file=self.molecules_file,
                isotopes_file=self.isotopes_file,
            )

    def test_get_formula_invalid_element(self):
        """Raise ValueError with invalid element in molecule formula."""
        with self.assertRaises(ValueError):
            ip.MoleculeInfo.get_molecule_info(
                molecule_name="Test3",
                molecules_file=self.molecules_file,
                isotopes_file=self.isotopes_file,
            )

    def test_get_charge_result(self):
        """Correct molecule charge."""
        res = ip.MoleculeInfo.get_charge("Test1", self.molecules_file)
        self.assertEqual(res, 1)

    def test_get_charge_column_missing(self):
        """ValueError if value in 'charge' column is missing."""
        with self.assertRaises(ValueError):
            ip.MoleculeInfo.get_charge("Test4", self.molecules_file2)

    def test_get_elements_result(self):
        """Return correct list of elements."""
        molecule = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        res = molecule.get_elements()
        res_corr = ["C", "H", "N", "O", "P"]
        self.assertListEqual(res, res_corr)

    def test_get_isotopes_result(self):
        """Return correct list of isotopes."""
        molecule = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        res = molecule.get_isotopes()
        res_corr = [
            "C12",
            "C13",
            "H01",
            "H02",
            "N14",
            "N15",
            "O16",
            "O17",
            "O18",
            "P31",
        ]
        self.assertListEqual(res, res_corr)


class TestCorrectionFactor(unittest.TestCase):
    """Calculation of correction factor."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = ip.MoleculeInfo.get_molecule_info(
        molecule_name="Test1",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )

    def test_result_no_label(self):
        """Result without label."""
        res = ip.calc_correction_factor(self.molecule_info)
        self.assertAlmostEqual(res, 1.33471643)

    def test_result_with_label(self):
        """Result with complex label."""
        res = ip.calc_correction_factor(
            self.molecule_info, label=ip.Label("10C13 1N15 12H02", self.molecule_info),
        )
        self.assertAlmostEqual(res, 1.19257588)

    def test_result_wrong_label(self):
        """ValueError with impossible label."""
        with self.assertRaises(ValueError):
            ip.calc_correction_factor(
                self.molecule_info, label=ip.Label("100C131N15", self.molecule_info),
            )


class TestDiffProb(unittest.TestCase):
    """Calculation of difference probability."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = ip.MoleculeInfo.get_molecule_info(
        molecule_name="Test1",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )

    def test_result_empty_diff_label(self):
        """Result with empty diff label."""
        label = ip.Label("1C13", self.molecule_info)
        diff_label = ip.Label({}, self.molecule_info)
        res = ip.calc_label_diff_prob(label, diff_label)
        self.assertEqual(res, 0)

    def test_result_negative_diff_label(self):
        """Result with negative diff label."""
        label = ip.Label("1C13", self.molecule_info)
        diff_label = ip.Label({"C13": -1, "N15": 1}, self.molecule_info)
        res = ip.calc_label_diff_prob(label, diff_label)
        self.assertEqual(res, 0)

    def test_result_simple_diff_label(self):
        """Result with simple one atom diff label."""
        label = ip.Label("1C13", self.molecule_info)
        diff_label = ip.Label("1C13", self.molecule_info)
        res = ip.calc_label_diff_prob(label, diff_label)
        self.assertAlmostEqual(res, 0.1744399)

    def test_result_complex_diff_label(self):
        """Result with diff label containing multiple atoms."""
        label = ip.Label("1C13", self.molecule_info)
        diff_label = ip.Label("1C13 1N15", self.molecule_info)
        res = ip.calc_label_diff_prob(label, diff_label)
        self.assertAlmostEqual(res, 0.0043485)
