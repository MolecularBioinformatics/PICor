"""Unit tests for isotope correction."""

from pathlib import Path
import unittest

import pandas as pd

import picor.isotope_correction as ic
import picor.isotope_probabilities as ip
import picor.resolution_correction as rc

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


class TestIsotopologueCorrection(unittest.TestCase):
    """Total Correction Factor for molecule and DataFrame."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")

    def test_result(self):
        """Result with default values."""
        data = pd.read_csv(Path("tests/test_dataset.csv"), index_col=0)
        data.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        molecule_name = "NAD"
        res = ic.calc_isotopologue_correction(data, molecule_name,)
        data_corrected = pd.read_csv(
            Path("tests/test_dataset_corrected.csv"), index_col=0
        )
        pd.testing.assert_frame_equal(data_corrected, res)

    def test_result_files(self):
        """Result with given molecule and isotopes file paths."""
        data = pd.read_csv(Path("tests/test_dataset.csv"), index_col=0)
        data.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        molecule_name = "Test1"
        res = ic.calc_isotopologue_correction(
            data,
            molecule_name,
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        data_corrected = pd.read_csv(
            Path("tests/test_dataset_corrected.csv"), index_col=0
        )
        pd.testing.assert_frame_equal(data_corrected, res)

    def test_result_subset(self):
        """Result with explicit subset."""
        data = pd.read_csv(Path("tests/test_dataset.csv"), index_col=0)
        subset = ["No label", "1C13", "4C13 6H02 3N15"]
        molecule_name = "Test1"
        res = ic.calc_isotopologue_correction(
            data,
            molecule_name,
            subset=subset,
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        assert res.shape == data.shape
        res.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        data_corrected = pd.read_csv(
            Path("tests/test_dataset_corrected.csv"), index_col=0
        )
        pd.testing.assert_frame_equal(data_corrected, res)

    def test_result_exclude_col(self):
        """Result with exclude_col parameter."""
        data = pd.read_csv(Path("tests/test_dataset.csv"), index_col=0)
        exclude_col = ["dummy column int", "dummy column str"]
        molecule_name = "Test1"
        res = ic.calc_isotopologue_correction(
            data,
            molecule_name,
            exclude_col=exclude_col,
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        assert res.shape == data.shape
        res.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        data_corrected = pd.read_csv(
            Path("tests/test_dataset_corrected.csv"), index_col=0
        )
        pd.testing.assert_frame_equal(data_corrected, res)

    def test_resolution_result(self):
        """Result with default values"""
        data = pd.read_csv(Path("tests/test_dataset_resolution.csv"), index_col=0)
        data.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        molecule_name = "Test1"
        res = ic.calc_isotopologue_correction(
            data,
            molecule_name,
            resolution_correction=True,
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        data_corrected = pd.read_csv(
            Path("tests/test_dataset_resolution_corrected.csv"), index_col=0
        )
        pd.testing.assert_frame_equal(data_corrected, res)

    def test_resolution_ultra_high_result(self):
        """Result with ultra high resolution."""
        data = pd.read_csv(Path("tests/test_dataset.csv"), index_col=0)
        data.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        data.rename(columns={"4C13 6H02 3N15": "2H02"}, inplace=True)
        molecule_name = "Test1"
        res_w_cor = ic.calc_isotopologue_correction(
            data,
            molecule_name,
            resolution_correction=True,
            resolution=1e8,
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        res_wo_cor = ic.calc_isotopologue_correction(
            data,
            molecule_name,
            resolution_correction=False,
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        pd.testing.assert_frame_equal(res_wo_cor, res_w_cor, rtol=1e-4)

    def test_resolution_warning(self):
        """Warn when overlapping isotopologues."""
        data = pd.read_csv(Path("tests/test_dataset.csv"), index_col=0)
        data.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        data.rename(columns={"4C13 6H02 3N15": "1H02"}, inplace=True)
        molecule_name = "Test1"
        with self.assertWarns(UserWarning):
            ic.calc_isotopologue_correction(
                data,
                molecule_name,
                resolution_correction=True,
                molecules_file=self.molecules_file,
                isotopes_file=self.isotopes_file,
            )


class TestTransitionProbability(unittest.TestCase):
    """Calculation of probability between to isotopologues."""

    molecules_file = Path("tests/test_metabolites.csv")
    isotopes_file = Path("tests/test_isotopes.csv")
    molecule_info = ip.MoleculeInfo.get_molecule_info(
        molecule_name="Test4",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )
    molecule_info2 = ip.MoleculeInfo.get_molecule_info(
        molecule_name="Test1",
        molecules_file=molecules_file,
        isotopes_file=isotopes_file,
    )
    res_corr_info = rc.ResolutionCorrectionInfo(False, 60000, 200, molecule_info)

    def test_result_label1_smaller(self):
        """Result with label1 being smaller than label2."""
        label1 = ip.Label("2N15", self.molecule_info)
        label2 = ip.Label("2N152C13", self.molecule_info)
        res = ic.calc_transition_prob(label1, label2, self.res_corr_info)
        self.assertAlmostEqual(res, 0.03685030)

    def test_result_label1_equal(self):
        """Result with label1 being equal to label2."""
        label1 = ip.Label("1N15", self.molecule_info)
        res = ic.calc_transition_prob(label1, label1, self.res_corr_info)
        self.assertEqual(res, 0)

    def test_result_label_missmatch_without_res_corr(self):
        """Zero result with no possible transition without res_corr."""
        label1 = ip.Label("2N15 4C13", self.molecule_info2)
        label2 = ip.Label("1N15 6C13", self.molecule_info2)
        res_corr_info = rc.ResolutionCorrectionInfo(
            False, 60000, 200, self.molecule_info2
        )
        res = ic.calc_transition_prob(label1, label2, res_corr_info)
        self.assertEqual(res, 0)

    def test_result_label_missmatch_with_res_corr(self):
        """Possible result for missmatch with resolution correction."""
        label1 = ip.Label("2N15 4C13", self.molecule_info2)
        label2 = ip.Label("1N15 6C13", self.molecule_info2)
        res_corr_info = rc.ResolutionCorrectionInfo(
            True, 60000, 200, self.molecule_info2
        )
        res = ic.calc_transition_prob(label1, label2, res_corr_info)
        self.assertAlmostEqual(res, 0.1794381)

    def test_result_label1_larger(self):
        """Result with label1 being larger than label2."""
        label1 = ip.Label("2N152C13", self.molecule_info)
        label2 = ip.Label("2N15", self.molecule_info)
        res = ic.calc_transition_prob(label1, label2, self.res_corr_info)
        self.assertEqual(res, 0)

    def test_result_molecule_formula(self):
        """Result with molecule formula."""
        molecule_info = ip.MoleculeInfo.get_molecule_info(
            molecule_name="Test1",
            molecules_file=self.molecules_file,
            isotopes_file=self.isotopes_file,
        )
        label1 = ip.Label("1N15", molecule_info)
        label2 = ip.Label("2N152C13", molecule_info)
        res = ic.calc_transition_prob(label1, label2, self.res_corr_info)
        self.assertAlmostEqual(res, 0.00042029)

    def test_wrong_type(self):
        """Type error with dict as molecule."""
        label1 = ip.Label("1N15", self.molecule_info)
        label2 = ip.Label("2N152C13", self.molecule_info)
        molecule = {"C": 12, "N": 15}
        with self.assertRaises(TypeError):
            ic.calc_transition_prob(label1, label2, molecule, self.res_corr_info)
