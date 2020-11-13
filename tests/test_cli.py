"""Unit tests for isotope correction."""

import subprocess


import pandas as pd

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


def test_result_csv(tmp_path):
    """Output file result with csv input."""
    outfile = tmp_path / "test_out.csv"
    res_correct = "./tests/test_dataset_corrected.csv"
    cmd = f"picor tests/test_dataset.csv NAD -x 'dummy column int' -x 'dummy column str' -o {outfile}"
    process = subprocess.run(cmd, shell=True)
    df_res = pd.read_csv(outfile, index_col=0)
    df_res.drop(columns=["dummy column str", "dummy column int"], inplace=True)
    df_corr = pd.read_csv(res_correct, index_col=0)
    assert process.returncode == 0
    pd.testing.assert_frame_equal(df_res, df_corr)


def test_result_xlsx(tmp_path):
    """Output file result with xlsx input."""
    outfile = tmp_path / "test_out.csv"
    res_correct = "./tests/test_dataset_corrected.csv"
    cmd = f"picor tests/test_dataset.xlsx NAD -x 'dummy column int' -x 'dummy column str' -o {outfile}"
    process = subprocess.run(cmd, shell=True)
    df_res = pd.read_csv(outfile, index_col=0)
    df_res.drop(columns=["dummy column str", "dummy column int"], inplace=True)
    df_corr = pd.read_csv(res_correct, index_col=0)
    assert process.returncode == 0
    pd.testing.assert_frame_equal(df_res, df_corr)


def test_no_output(tmp_path):
    """Output on stdout."""
    cmd = (
        f"picor tests/test_dataset.csv NAD -x 'dummy column int' -x 'dummy column str'"
    )
    process = subprocess.run(cmd, shell=True, capture_output=True)
    assert process.returncode == 0
    assert not process.stderr, "No output to stderr"
    assert len(process.stdout) > 0, "Results printed to stdout"


def test_wrong_filetype(tmp_path):
    """ValueError with unknown filetype as input."""
    cmd = (
        f"picor tests/test_dataset.tsv NAD -x 'dummy column int' -x 'dummy column str'"
    )
    process = subprocess.run(cmd, capture_output=True, shell=True)
    assert process.returncode == 1
    assert "ValueError" in str(process.stderr)
