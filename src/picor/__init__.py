# -*- coding: utf-8 -*-
"""Isotopologue correction for MS data sets.

Modules:
    isotope_correction.py: main function
    isotope_probabilities.py: subroutines and helper functions
    resolution_correction.py: subroutines for resolution depend correction
"""
from picor.isotope_correction import calc_isotopologue_correction
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "PICor"
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
