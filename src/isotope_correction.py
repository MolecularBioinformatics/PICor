"""Isotopologue correction using statistical distributions.

Functions:
    calc_isotopologue_correction: Correct DataFrame with measurements.
    calc_correction_factor: Get correction factor for metabolite and label.
    calc_transition_prob: Get transition probablity for two isotopologues.
"""
from math import prod
import os
import re
import warnings

import pandas as pd
from scipy.special import binom


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
    """Parse label e.g.  and return dict.

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
    sorted_labels = [l[0] for l in sorted_masses]
    return sorted_labels


def label_shift_smaller(label1, label2):
    """Check whether the mass shift of label1 is smaller than label2.

    Calculates if mass shift (e.g. 5 for 5C13 label) is smaller for label1
    than for label2 and returns True if that is the case.
    Only support H02 (deuterium),  C13 and N15 as labels.
    label can also be "No label" or something similar.
    :param label1: str
        Label of isotopologue 1
    :param label2: str
        Label of isotopologue 2
    :return: Boolean
        True if label 1 is smaller than label 2, otherwise False
    """
    label1 = parse_label(label1)
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
    for elem in isotopes.loc[["C", "N", "O", "H"]].itertuples():
        if elem.Index not in abundance:
            abundance[elem.Index] = []
        abundance[elem.Index].append(elem.abundance)
    return abundance


def get_metabolite_formula(metabolite, metabolites_file):
    """Get molecular formula from file.

    Parse and look up molecular formula of metabolite and
    return number of atoms per element
    :param metabolite: str
        Name of metabolite used in metabolites_file
    :param metabolites_file: str
        File path to isotopes file
        tab-separated csv with name and formula columns
    :return: dict
        Elements and number
    """
    global METABOLITES
    if not isinstance(METABOLITES, pd.DataFrame):
        METABOLITES = pd.read_csv(metabolites_file, sep="\t", na_filter=False)
        METABOLITES["formula"] = METABOLITES["formula"].apply(parse_formula)
        METABOLITES.set_index("name", drop=True, inplace=True)

    try:
        n_atoms = {}
        n_atoms["C"] = METABOLITES.loc[metabolite].formula["C"]
        n_atoms["N"] = METABOLITES.loc[metabolite].formula["N"]
        n_atoms["O"] = METABOLITES.loc[metabolite].formula["O"]
        n_atoms["H"] = METABOLITES.loc[metabolite].formula["H"]
    except KeyError as error:
        raise KeyError(
            f"Metabolite {error} couldn't be found in metabolites file"
        ) from error
    return n_atoms


def calc_probability_table(
    metabolite, label=False, isotopes_file=None, metabolites_file=None
):
    """Calculate table of isotopologue probabilities.

    Label supports only H02 (deuterium), C13, N15 and 'No label'.
    :param metabolite: str of metabolite name
        Name must be found in metabolites_file
    :param label: False or str of type and number of atoms
        E.g. '5C13N15' or 'No label'. False equals to no label
    :param isotopes_file: Path to isotope file
        default location: scripts/isotope_correction/isotopes.csv
    :param metabolites_file: Path to metabolites file
        default location: scripts/isotope_correction/metabolites.csv
    :return : Pandas DataFrame
        Probabilities for different isotopologues for each element and total probability
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

    n_atoms = get_metabolite_formula(metabolite, metabolites_file)
    prob = {}
    for elem in n_atoms:
        start_count = 0
        # Lowers number of atoms in case a label is present
        if elem not in atom_label:
            n_atom = n_atoms[elem]
        else:
            n_atom = n_atoms[elem] - atom_label[elem]
            if n_atom < 0:
                raise ValueError("Too many labelled atoms")
        abund = ABUNDANCE[elem]
        p = {}
        # Calculation for elements with 2 Isotopes
        if len(abund) == 2:
            for i in range(start_count, n_atom + 1):
                n_comb = binom(n_atom, i)
                p[i] = n_comb
                p[i] *= abund[0] ** (n_atom - i)
                p[i] *= abund[1] ** (i)
        # Calculation for elements with 3 Isotopes
        elif len(abund) == 3:
            if len(abund) > 3:
                warnings.warn(
                    "Only first three isotopes per element are used for calculation"
                )
            for i in range(start_count, n_atom + 1):
                for j in range(0, i + 1):
                    shift = (i - j) + 2 * j
                    pr = binom(n_atom, i - j)
                    pr *= binom(n_atom, j)
                    pr *= abund[0] ** (n_atom - i)
                    pr *= abund[1] ** (i - j)
                    pr *= abund[2] ** (j)
                    if shift in p:
                        p[shift] += pr
                    else:
                        p[shift] = pr

        prob[elem] = p

    isotopologue_prob = {}
    for i in prob["C"]:
        for j in prob["N"]:
            for k in prob["O"]:
                for l in prob["H"]:
                    if not i + j + k + l in isotopologue_prob:
                        isotopologue_prob[i + j + k + l] = 0
                    isotopologue_prob[i + j + k + l] += (
                        prob["C"][i] * prob["N"][j] * prob["O"][k] * prob["H"][l]
                    )

    p = pd.DataFrame(prob)
    p["total"] = pd.Series(isotopologue_prob)
    p.fillna(value=0.0, inplace=True)
    return p


def fwhm(m_cal, m_z, resolution):
    """Calculate Full width half maximum (FWHM)."""
    return m_z ** (3 / 2) / (resolution * m_cal ** 0.5)


def calc_min_mass_diff(mass, charge, m_cal, resolution):
    """Calculate minimal resolvable mass difference.

    For m and z return minimal mass difference that ca be resolved properly.
    :param mass: float
        Mass of molecule
    :param charge: int
        Charge of molecule
    :param m_cal: float
        Mass for which resolition has been determined
    :param resolution: float
        Resolution
    :returns: float
    """
    m_z = mass / charge
    return 1.66 * charge * fwhm(m_cal, m_z, resolution)


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
        },
    )
    return result


def subtract_label(metabolite_series, label_series):
    """Subtract label atoms from metabolite formula."""
    formula_difference = metabolite_series.copy()
    iso_dict = {
        "H02": "H01",
        "C13": "C12",
        "N15": "N14",
    }
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


def calc_isotopologue_mass(metabolite_name, label, isotope_mass_series):
    """Calculate mass of isotopologue.

    Given the metabolite name and label composition, return mass in atomic units.
    :param metabolite_name: str
        Name as in metabolite_file
    :param label: str
        "No label" or formula, can contain whitespaces
    :param isotope_mass_series: pandas Series
        Isotope name (e.g. 'H02') as index and mass as values
        Output of 'get_isotope_mass_series'
    :returns: float
    """
    label = pd.Series(parse_label(label), dtype="int64")
    metab = pd.Series(
        get_metabolite_formula(metabolite_name, "src/metabolites.csv"), dtype="int64"
    )
    metab = assign_light_isotopes(metab)
    light_isotopes = subtract_label(metab, label)
    formula_isotopes = pd.concat([light_isotopes, label])
    mass = isotope_mass_series.multiply(formula_isotopes).dropna().sum()
    return mass


def is_isotologue_overlapp(
    label1, label2, metabolite_name, min_mass_diff, isotope_mass_series
):
    """Return True if label1 and label2 are too close to detection limit.

    Checks whether two isotopologues defined by label and metabolite name
    are below miniumum resolved mass difference.
    """
    mass1 = calc_isotopologue_mass(metabolite_name, label1, isotope_mass_series)
    mass2 = calc_isotopologue_mass(metabolite_name, label2, isotope_mass_series)
    return abs(mass1 - mass2) < min_mass_diff


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

    n_atoms = get_metabolite_formula(metabolite, metabolites_file)
    prob = {}
    for elem in n_atoms:
        n_atom = (
            n_atoms[elem] - atom_label[elem] if elem in atom_label else n_atoms[elem]
        )
        if n_atom < 0:
            raise ValueError("Too many labelled atoms")
        prob[elem] = ABUNDANCE[elem][0] ** n_atom
    probability = prod(prob.values())
    return 1 / probability


def calc_transition_prob(
    label1, label2, metabolite_formula, metabolites_file, isotopes_file
):
    """Calculate the probablity between two (un-)labelled isotopologues.

    :param label1: str
        Type of isotopic label, e.g. 1N15
    :param label1: str
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
    if not label_shift_smaller(label1, label2):
        return 0
    label1 = parse_label(label1)
    label2 = parse_label(label2)

    if isinstance(metabolite_formula, str):
        n_atoms = get_metabolite_formula(metabolite_formula, metabolites_file)
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
    prob_total = prod(prob)
    assert prob_total <= 1, "Transition probability greater than 1"
    return prob_total


def calc_isotopologue_correction(
    raw_data,
    metabolite,
    subset=False,
    exclude_col=False,
    isotopes_file=None,
    metabolites_file=None,
    verbose=False,
):
    """Calculate isotopologue correction factor for metabolite.

    Calculates isotopologue correction factor for metabolite in metabolites file
    Only C13 and N15 is supported as column labels right now e.g. 5C13
    :param  raw_data: pandas DataFrame
        DataFrame of integrated lowest peaks per species vs time
    :param metabolite: str
        metabolite name
    :param subset: list of str or False
        List of column names to use for calculation
    :param exclude_col: list of str
        Columns to ignore in calculation
    :param isotopes_file: Path to isotope file
        default location: scripts/isotope_correction/isotopes.csv
    :param metabolites_file: Path to metabolites file
        default location: scripts/isotope_correction/metabolites.csv
    :param verbose: bool (default: False)
        print correction and transition factors
    :return: pandas DataFrame
        Corrected data
    """
    if not isotopes_file:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        isotopes_file = os.path.join(dir_path, "isotopes.csv")
    if not metabolites_file:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        metabolites_file = os.path.join(dir_path, "metabolites.csv")
    data = raw_data.copy(deep=True)
    if not subset:
        subset = data.columns
        if exclude_col:
            subset = list(set(subset) - set(exclude_col))
    subset = sort_labels(subset)

    for label1 in subset:
        corr = calc_correction_factor(
            metabolite, label1, isotopes_file, metabolites_file
        )
        assert corr >= 1, "Correction factor should be greater or equal 1"
        data[label1] = corr * data[label1]
        if verbose:
            print(f"Correction factor {label1}: {corr}")
        for label2 in subset:
            if label_shift_smaller(label1, label2):
                prob = calc_transition_prob(
                    label1, label2, metabolite, metabolites_file, isotopes_file
                )
                data[label2] = data[label2] - prob * data[label1]
                data[label2].clip(lower=0, inplace=True)
                if verbose:
                    print(f"Transition prob {label1} -> {label2}: {prob}")
    return data
