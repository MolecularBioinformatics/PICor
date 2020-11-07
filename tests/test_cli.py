"""Unit tests for isotope correction."""

import subprocess


import pandas as pd

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


def test_result(tmp_path):
    """Output file result."""
    outfile = tmp_path / "test_out.csv"
    res_correct = "./tests/test_dataset_corrected.csv"
    cmd = f"picor tests/test_dataset.csv NAD -x 'dummy column int' -x 'dummy column str' -o {outfile}"
    process = subprocess.run(cmd, check=True, shell=True)
    df_res = pd.read_csv(outfile, index_col=0)
    df_res.drop(columns=["dummy column str", "dummy column int"], inplace=True)
    df_corr = pd.read_csv(res_correct, index_col=0)
    assert process.returncode == 0
    pd.testing.assert_frame_equal(df_res, df_corr)
