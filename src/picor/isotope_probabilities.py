"""Isotopologue correction functions.

Functions:
    calc_correction_factor: Get correction factor for metabolite and label.
    calc_transition_prob: Get transition probablity for two isotopologues.
"""
from functools import reduce
from operator import mul
import os
import re

import pandas as pd
from scipy.special import binom

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


ABUNDANCE = None
METABOLITES = None


def parse_formula(string):
    """Parse chemical formula and return dict.

    Parse chemical formula of type C3O3H6 and return dictionary
    of elements and number of atoms.
    Be careful, no input check is happening!

    :param string: Str of chemical formula
    :return : Dict of str and int
    """
    elements = re.findall(r"([A-Z][a-z]*)(\d*)", string)
    return {elem: int(num) if num != "" else 1 for elem, num in elements}


def parse_label(string):
    """Parse label e.g. "3C13 2H02"  and return dict.

    Parse label e.g. 5C13N15 and return dictionary of elements and number of atoms
    Only support H02 (deuterium), C13, N15 and 'No label'
    Prefix separated by colon can be used and will be ignored, e.g. "NA:2C13"
    Underscores can be used to separate different elements, e.g. "3C13_6H02"
    :param string: Str of chemical formula
    :return : Dict of str and int
    """
    if not isinstance(string, str):
        raise TypeError("label must be string")
    if string.count(":") > 1:
        raise ValueError("only one colon allowed in label to separate prefix")
    string = string.split(":")[-1]

    if string.lower() == "no label":
        return {}
    allowed_isotopes = ["H02", "C13", "N15"]
    label = re.findall(r"(\d*)([A-Z][a-z]*\d\d)", string)
    label_dict = {elem: int(num) if num != "" else 1 for num, elem in label}
    if not set(label_dict).issubset(allowed_isotopes) or not label_dict:
        # Check for empty list and only allowed isotopes
        raise ValueError("Label should be H02, C13, N15 or 'No label'")
    return label_dict


def isotope_to_element(label):
    """Change dict key from isotope to element (e.g. H02 -> H)."""
    atom_label = {}
    for elem in label:
        if elem == "C13":
            atom_label["C"] = label[elem]
        elif elem == "N15":
            atom_label["N"] = label[elem]
        elif elem == "H02":
            atom_label["H"] = label[elem]
        else:
            raise ValueError("Only H02, C13 and N15 are allowed as isotopic label")
    return atom_label


def sort_labels(labels):
    """Sort list of metabolite labels by coarse mass.

    Sort list of metabolite labels by coarse mass
    (number of neutrons) from lower to higher mass
    :param labels: list of str
        labels (e.g. ["No label","N15","5C13"])
    :return: list of str
        Sorted list
    """
    if isinstance(labels, str):
        raise TypeError("labels must be list-like but not string")
    masses = {}
    for label in labels:
        mass = parse_label(label)
        masses[label] = sum(mass.values())
    sorted_masses = sorted(masses.items(), key=lambda kv: kv[1])
    sorted_labels = [lab[0] for lab in sorted_masses]
    return sorted_labels


def label_shift_smaller(label1, label2):
    """Check whether the mass shift of label1 is smaller than label2.

    Calculates if mass shift (e.g. 5 for 5C13 label) is smaller for label1
    than for label2 and returns True if that is the case.
    Only support H02 (deuterium),  C13 and N15 as labels.
    label can also be "No label" or something similar.
    :param label1: str or dict
        Label of isotopologue 1
    :param label2: str or dict
        Label of isotopologue 2
    :return: Boolean
        True if label 1 is smaller than label 2, otherwise False
    """
    if isinstance(label1, str):
        label1 = parse_label(label1)
    if isinstance(label2, str):
        label2 = parse_label(label2)
    shift_label1 = sum(label1.values())
    shift_label2 = sum(label2.values())

    return shift_label1 < shift_label2


def get_isotope_abundance(isotopes_file):
    """Get abundace of different isotopes.

    Parse file with abundance of different isotopes
    :param isotopes_file: str
        File path to isotopes file
        comma separated csv with element and abundance columns
    :return: dict
    """
    isotopes = pd.read_csv(isotopes_file, sep="\t")
    isotopes.set_index("element", drop=True, inplace=True)

    abundance = {}
    for elem in isotopes.itertuples():
        if elem.Index not in abundance:
            abundance[elem.Index] = []
        abundance[elem.Index].append(elem.abundance)
    return abundance


def get_metabolite_formula(metabolite, metabolites_file, isotopes_file):
    """Get molecular formula from file.

    Parse and look up molecular formula of metabolite and
    return number of atoms per element
    :param metabolite: str
        Name of metabolite used in metabolites_file
    :param metabolites_file: str
        File path to isotopes file
        tab-separated csv with name and formula columns
    :param isotopes_file: Path to isotope file
        File path to isotopes file, tab separated
    :return: dict
        Elements and number
    """
    global ABUNDANCE
    if not ABUNDANCE:
        ABUNDANCE = get_isotope_abundance(isotopes_file)
    global METABOLITES
    if not isinstance(METABOLITES, pd.DataFrame):
        METABOLITES = pd.read_csv(metabolites_file, sep="\t", na_filter=False)
        METABOLITES["formula"] = METABOLITES["formula"].apply(parse_formula)
        METABOLITES.set_index("name", drop=True, inplace=True)

    try:
        n_atoms = METABOLITES.loc[metabolite].formula
    except KeyError as error:
        raise KeyError(
            f"Metabolite {error} couldn't be found in metabolites file"
        ) from error
    if not all(element in ABUNDANCE for element in n_atoms):
        raise ValueError("Unknown element in metabolite")
    return n_atoms


def assign_light_isotopes(metabolite_series):
    """Replace all element names with light isotopes ("C" -> "C13")."""
    result = metabolite_series.rename(
        {
            "H": "H01",
            "C": "C12",
            "N": "N14",
            "O": "O16",
            "Si": "Si28",
            "P": "P31",
            "S": "S32",
        }
    )
    return result


def subtract_label(metabolite_series, label_series):
    """Subtract label atoms from metabolite formula."""
    formula_difference = metabolite_series.copy()
    iso_dict = {"H02": "H01", "C13": "C12", "N15": "N14"}
    for heavy in label_series.keys():
        light = iso_dict[heavy]
        formula_difference[light] = metabolite_series[light] - label_series[heavy]
    if any(formula_difference < 0):
        raise ValueError("Too many labelled atoms")
    return formula_difference


def get_isotope_mass_series(isotopes_file):
    """Get series of isotope masses."""
    return pd.read_csv(
        isotopes_file,
        sep="\t",
        usecols=["mass", "isotope"],
        index_col="isotope",
        squeeze=True,
    )


def calc_correction_factor(
    metabolite, label=False, isotopes_file=None, metabolites_file=None
):
    """Calculate correction factor with metabolite composition defined by label.

    Label supports only H02 (deuterium), C13, N15 and 'No label'.
    :param metabolite: str of metabolite name
        Name must be found in metabolites_file
    :param label: False or str of type and number of atoms
        E.g. '5C13N15' or 'No label'. False equals to no label
    :param isotopes_file: Path to isotope file
        default location: scripts/isotope_correction/isotopes.csv
    :param metabolites_file: Path to metabolites file
        default location: scripts/isotope_correction/metabolites.csv
    :return: float
        Correction factor
    """
    if not isotopes_file:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        isotopes_file = os.path.join(dir_path, "isotopes.csv")
    if not metabolites_file:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        metabolites_file = os.path.join(dir_path, "metabolites.csv")
    global ABUNDANCE
    if not ABUNDANCE:
        ABUNDANCE = get_isotope_abundance(isotopes_file)

    # Parse label and store information in atom_label dict
    atom_label = {}
    if label:
        label = parse_label(label)
        atom_label = isotope_to_element(label)

    n_atoms = get_metabolite_formula(metabolite, metabolites_file, isotopes_file)
    prob = {}
    for elem in n_atoms:
        n_atom = (
            n_atoms[elem] - atom_label[elem] if elem in atom_label else n_atoms[elem]
        )
        if n_atom < 0:
            raise ValueError("Too many labelled atoms")
        prob[elem] = ABUNDANCE[elem][0] ** n_atom
    probability = reduce(mul, prob.values())
    return 1 / probability


def calc_transition_prob(
    label1, label2, metabolite_formula, metabolites_file, isotopes_file
):
    """Calculate the probablity between two (un-)labelled isotopologues.

    :param label1: str or dict
        Type of isotopic label, e.g. 1N15
    :param label2: str or dict
        Type of isotopic label, e.g. 10C1301N15
    :param metabolite_formula: str or dict
        Molecular formula or dict of elements and number
    :param isotopes_file: Path to isotope file
        default location: ~/isocordb/Isotopes.dat
    :param metabolites_file: Path to metabolites file
        default location: ~/isocordb/Metabolites.dat
    :return: float
        Transition probability
    """
    if isinstance(label1, str):
        label1 = parse_label(label1)
    if isinstance(label2, str):
        label2 = parse_label(label2)

    if not label_shift_smaller(label1, label2):
        return 0

    if isinstance(metabolite_formula, str):
        n_atoms = get_metabolite_formula(
            metabolite_formula, metabolites_file, isotopes_file
        )
    elif isinstance(metabolite_formula, dict):
        n_atoms = metabolite_formula
    else:
        raise TypeError(
            "metabolite_formula must be str (molecular formula) or dict of elements"
        )
    label1 = pd.DataFrame.from_dict(label1, orient="index")
    label2 = pd.DataFrame.from_dict(label2, orient="index")
    difference_labels = label2.sub(label1, fill_value=0).iloc[:, 0]
    difference_labels.index = difference_labels.index.str[:-2]
    label1.index = label1.index.str[:-2]
    label2.index = label2.index.str[:-2]
    # Checks if transition from label1 to 2 is possible
    if difference_labels.lt(0).any():
        return 0

    global ABUNDANCE
    if not ABUNDANCE:
        ABUNDANCE = get_isotope_abundance(isotopes_file)

    prob = []
    for elem in difference_labels.index:
        try:
            n_elem_1 = label1.loc[elem, 0]
        except KeyError:
            n_elem_1 = 0
        n_elem_2 = label2.loc[elem, 0]
        n_unlab = n_atoms[elem] - n_elem_2
        n_label = difference_labels[elem]
        abun_unlab = ABUNDANCE[elem][0]
        abun_lab = ABUNDANCE[elem][1]
        if n_label == 0:
            continue

        trans_pr = binom((n_atoms[elem] - n_elem_1), n_label)
        trans_pr *= abun_lab ** n_label
        trans_pr *= abun_unlab ** n_unlab
        prob.append(trans_pr)

    # Prob is product of single probabilities
    prob_total = reduce(mul, prob)
    assert prob_total <= 1, "Transition probability greater than 1"
    return prob_total
