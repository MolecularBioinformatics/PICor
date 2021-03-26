"""Isotopologue correction functions.

Functions:
    calc_correction_factor: Get correction factor for molecule and label.
    calc_transition_prob: Get transition probablity for two isotopologues.
"""
from functools import reduce
from operator import mul
import re

import pandas as pd
from scipy.special import binom

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


class IsotopeInfo:
    def __init__(self, isotopes_file):
        """
            :param isotopes_file: str
                File path to isotopes file
                comma separated csv with element and abundance columns
        """
        self.isotopes_file = isotopes_file
        self.abundance = self.get_isotope_abundance(isotopes_file)
        self.isotope_mass_series = self.get_isotope_mass_series(isotopes_file)

    @staticmethod
    def get_isotope_abundance(isotopes_file):
        """Get abundace of different isotopes.

        Parse file with abundance of different isotopes
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

    @staticmethod
    def get_isotope_mass_series(isotopes_file):
        """Get series of isotope masses."""
        return pd.read_csv(
            isotopes_file,
            sep="\t",
            usecols=["mass", "isotope"],
            index_col="isotope",
            squeeze=True,
        )


class MoleculeInfo:
    def __init__(self, molecule_name, molecules_file, isotopes_file):
        """
            :param isotopes_file: str
                File path to isotopes file
                comma separated csv with element and abundance columns
        """
        self.molecule_name = molecule_name
        self.isotopes = IsotopeInfo(isotopes_file)
        self.molecules_file = molecules_file
        self.molecule_list = self.get_molecule_list(molecules_file)
        self.formula = self.get_formula()

    @staticmethod
    def get_molecule_list(molecules_file):
        molecule_list = pd.read_csv(molecules_file, sep="\t", na_filter=False)
        molecule_list["formula"] = molecule_list["formula"].apply(parse_formula)
        molecule_list.set_index("name", drop=True, inplace=True)
        return molecule_list

    def get_formula(self):
        """Get molecular formula from file.

        Parse and look up molecular formula of molecule and
        return number of atoms per element
        :param molecule: str
            Name of molecule used in molecules_file
        :param molecules_file: str
            File path to isotopes file
            tab-separated csv with name and formula columns
        :param isotopes_file: Path to isotope file
            File path to isotopes file, tab separated
        :return: dict
            Elements and number
        """
        try:
            n_atoms = self.molecule_list.loc[self.molecule_name].formula
        except KeyError as error:
            raise KeyError(
                f"Molecule {error} couldn't be found in molecules file"
            ) from error
        if not all(element in self.isotopes.abundance for element in n_atoms):
            raise ValueError("Unknown element in molecule")
        return n_atoms
    
    def get_charge(self):
        """Get charge of molecule."""
        charges = pd.read_csv(
            self.molecules_file,
            sep="\t",
            usecols=["name", "charge"],
            index_col="name",
            squeeze=True,
        )
        try:
            charges = charges.astype(int)
        except ValueError:
            raise ValueError("Charge of at least one molecule missing in molecules_file")
        return charges[self.molecule_name]

    def get_molecule_light_isotopes(self):
        """Replace all element names with light isotopes ("C" -> "C13")."""
        molecule_series = pd.Series(self.get_formula(), dtype="int64",)
        result = molecule_series.rename(
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

    @staticmethod
    def subtract_label(molecule_series, label_series):
        """Subtract label atoms from molecule formula."""
        formula_difference = molecule_series.copy()
        iso_dict = {"H02": "H01", "C13": "C12", "N15": "N14"}
        for heavy in label_series.keys():
            light = iso_dict[heavy]
            formula_difference[light] = molecule_series[light] - label_series[heavy]
            if any(formula_difference < 0):
                raise ValueError("Too many labelled atoms")
        return formula_difference

    def calc_isotopologue_mass(
        self, label,
    ):
        """Calculate mass of isotopologue.

        Given the molecule name and label composition, return mass in atomic units.
        :param molecule_name: str
            Name as in molecule_file
        :param label: str or dict
            "No label" or formula, can contain whitespaces
        :returns: float
        """
        if isinstance(label, str):
            label_dict = parse_label(label)
        elif isinstance(label, dict):
            label_dict = label
        else:
            raise ValueError("label must be str or dict")
        label = pd.Series(label_dict, dtype="int64")
        molecule_series = self.get_molecule_light_isotopes()
        light_isotopes = self.subtract_label(molecule_series, label)
        formula_isotopes = pd.concat([light_isotopes, label])
        mass = (
            self.isotopes.isotope_mass_series.multiply(formula_isotopes).dropna().sum()
        )
        return mass


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
    """Sort list of molecule labels by coarse mass.

    Sort list of molecule labels by coarse mass
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


def calc_correction_factor(
    molecule_info, label=False,
):
    """Calculate correction factor with molecule composition defined by label.

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
    # if not isotopes_file:
    #     path = os.path.abspath(__file__)
    #     dir_path = os.path.dirname(path)
    #     isotopes_file = os.path.join(dir_path, "isotopes.csv")
    # if not metabolites_file:
    #     path = os.path.abspath(__file__)
    #     dir_path = os.path.dirname(path)
    #     metabolites_file = os.path.join(dir_path, "metabolites.csv")
    # global ABUNDANCE
    # if not ABUNDANCE:
    #     ABUNDANCE = get_isotope_abundance(isotopes_file)

    # Parse label and store information in atom_label dict
    atom_label = {}
    if label:
        label = parse_label(label)
        atom_label = isotope_to_element(label)

    n_atoms = molecule_info.formula
    abundance = molecule_info.isotopes.abundance
    prob = {}
    for elem in n_atoms:
        n_atom = (
            n_atoms[elem] - atom_label[elem] if elem in atom_label else n_atoms[elem]
        )
        if n_atom < 0:
            raise ValueError("Too many labelled atoms")
        prob[elem] = abundance[elem][0] ** n_atom
    probability = reduce(mul, prob.values())
    return 1 / probability


def calc_transition_prob(label1, label2, molecule_info):
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

    if not isinstance(molecule_info, MoleculeInfo):
        raise TypeError("molecule_info must be instance of MoleculeInfo class")
    n_atoms = molecule_info.formula
    label1 = pd.Series(label1)
    label2 = pd.Series(label2)
    difference_labels = label2.sub(label1, fill_value=0)
    difference_labels.index = difference_labels.index.str[:-2]
    label1.index = label1.index.str[:-2]
    label2.index = label2.index.str[:-2]
    # Checks if transition from label1 to 2 is possible
    if difference_labels.lt(0).any():
        return 0

    abundance = molecule_info.isotopes.abundance

    prob = []
    for elem in difference_labels.index:
        n_elem_1 = label1.get(elem, 0)
        n_elem_2 = label2.get(elem, 0)
        n_unlab = n_atoms[elem] - n_elem_2
        n_label = difference_labels[elem]
        abun_unlab = abundance[elem][0]
        abun_lab = abundance[elem][1]
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
