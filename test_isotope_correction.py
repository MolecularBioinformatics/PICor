"""Unit tests for isotope correction."""

from pathlib import Path
import unittest

import pandas as pd

import src.isotope_correction as ic


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

    def test_resolution_result(self):
        """Result with default values"""
        data = pd.read_csv(Path("test/test_dataset.csv"), index_col=0)
        data.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        data.rename(columns={"4C13 6H02 3N15": "2H02"}, inplace=True)
        metabolite = "Test1"
        res = ic.calc_isotopologue_correction(
            data,
            metabolite,
            resolution_correction=True,
            metabolites_file=self.metabolites_file,
            isotopes_file=self.isotopes_file,
            verbose=True,
        )
        data_corrected = pd.read_csv(
            Path("test/test_dataset_resolution_corrected.csv"), index_col=0
        )
        pd.testing.assert_frame_equal(data_corrected, res)

    def test_resolution_warning(self):
        """Warn when overlapping isotopologues."""
        data = pd.read_csv(Path("test/test_dataset.csv"), index_col=0)
        data.drop(columns=["dummy column int", "dummy column str"], inplace=True)
        data.rename(columns={"4C13 6H02 3N15": "1H02"}, inplace=True)
        metabolite = "Test1"
        with self.assertWarns(UserWarning):
            ic.calc_isotopologue_correction(
                data,
                metabolite,
                resolution_correction=True,
                metabolites_file=self.metabolites_file,
                isotopes_file=self.isotopes_file,
            )
