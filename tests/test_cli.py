"""Unit tests for isotope correction."""

import subprocess
from subprocess import PIPE


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


def test_result_wrong_fileformat(tmp_path):
    """Check error with unknown input."""
    cmd = "picor tests/test_dataset.tsv NAD -x 'dummy column int' -x 'dummy column str'"
    process = subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    assert process.returncode == 1
    assert not process.stdout, "No output to stdout"
    assert len(process.stderr) > 0, "Error message printed to StdErr"


def test_result_resolution_correction(tmp_path):
    """Output file result with xlsx input."""
    outfile = tmp_path / "test_out.csv"
    res_correct = "./tests/test_dataset_resolution_corrected.csv"
    cmd = f"picor tests/test_dataset_resolution.csv NAD --res-correction -x 'dummy column int' -x 'dummy column str' -o {outfile}"
    subprocess.run(cmd, shell=True, check=True)
    df_res = pd.read_csv(outfile, index_col=0)
    df_res.drop(columns=["dummy column str", "dummy column int"], inplace=True)
    df_corr = pd.read_csv(res_correct, index_col=0)
    pd.testing.assert_frame_equal(df_res, df_corr)


def test_no_output(tmp_path):
    """Output on stdout."""
    cmd = "picor tests/test_dataset.csv NAD -x 'dummy column int' -x 'dummy column str'"
    process = subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    assert process.returncode == 0
    assert not process.stderr, "No output to stderr"
    assert len(process.stdout) > 0, "Results not printed to stdout"


def test_logging_level_info(tmp_path):
    """Logging to stderr with INFO level."""
    cmd = "picor tests/test_dataset.csv NAD -x 'dummy column int' -x 'dummy column str' -v"
    process = subprocess.run(cmd, shell=True, check=True, stdout=PIPE, stderr=PIPE)
    log = process.stderr.decode("utf-8")
    assert len(process.stderr) > 0, "Logging not printed to stderr"
    assert len(process.stdout) > 0, "Results not printed to stdout"
    assert "INFO:picor.isotope_correction:" in log, "INFO not in stderr"


def test_logging_level_debug(tmp_path):
    """Logging to stderr with DEBUG level."""
    cmd = "picor tests/test_dataset.csv NAD -x 'dummy column int' -x 'dummy column str' -r -v -v"
    process = subprocess.run(cmd, shell=True, check=True, stdout=PIPE, stderr=PIPE)
    log = process.stderr.decode("utf-8")
    assert len(process.stderr) > 0, "Logging not printed to stderr"
    assert len(process.stdout) > 0, "Results not printed to stdout"
    assert "DEBUG:picor.resolution_correction:" in log, "DEBUG not in stderr"
