"""Unit tests for isotope correction."""

from pathlib import Path
import unittest

import pandas as pd

import picor.isotope_correction as ic

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

    def test_resolution_result(self):
        """Result with default values"""
        data = pd.read_csv(Path("tests/test_dataset.csv"), index_col=0)
        data.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        data.rename(columns={"4C13 6H02 3N15": "2H02"}, inplace=True)
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
            resolution=1e6,
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
        pd.testing.assert_frame_equal(res_wo_cor, res_w_cor)

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
