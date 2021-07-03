"""Isotopologue correction functions.

Functions:
    calc_correction_factor: Get correction factor for molecule and label.
    calc_transition_prob: Get transition probablity for two isotopologues.
"""
from collections.abc import Sequence
from functools import reduce
from operator import mul
import re

import pandas as pd
from scipy.special import binom

__author__ = "Jørn Dietze"
__copyright__ = "Jørn Dietze"
__license__ = "gpl3"


class IsotopeInfo:
    """Class containing isotope data."""

    def __init__(self, isotopes_file):
        """Data class containing isotope information.

        Parameter
        ---------
        isotopes_file : str or Path
            File path to isotopes file
            comma separated csv with element and abundance columns
        """
        self.isotopes_file = isotopes_file
        self.abundance = self.get_isotope_abundance(isotopes_file)
        self.isotope_mass_series = self.get_isotope_mass_series(isotopes_file)
        self.isotope_shift = self.get_shift(isotopes_file)
        self.elements = self.get_elements(isotopes_file)

    def __repr__(self):
        return f"IsotopeInfo('{self.isotopes_file}')"

    def __eq__(self, other):
        if isinstance(other, IsotopeInfo):
            return self.isotopes_file == other.isotopes_file
        return False

    @staticmethod
    def get_isotope_abundance(isotopes_file):
        """Get abundace of different isotopes.

        Parse file with abundance of different isotopes

        Parameters
        ----------
        isotopes_file : str or Path
            File path to isotopes file
            comma separated csv with element and abundance columns

        Returns
        -------
        dict
            Abundance of isotope.
        """
        abundance = pd.read_csv(
            isotopes_file,
            sep="\t",
            usecols=["abundance", "isotope"],
            index_col="isotope",
            squeeze=True,
        )

        return abundance

    @staticmethod
    def get_isotope_mass_series(isotopes_file):
        """Get series of isotope masses.

        Parameters
        ----------
        isotopes_file : str or Path
            File path to isotopes file
            comma separated csv with element and abundance columns

        Returns
        -------
        pandas.Series
            Isotope names and masses
        """
        return pd.read_csv(
            isotopes_file,
            sep="\t",
            usecols=["mass", "isotope"],
            index_col="isotope",
            squeeze=True,
        )

    @staticmethod
    def get_shift(isotopes_file):
        """Get series of mass shift between isotopes."""
        iso_data = pd.read_csv(isotopes_file, sep="\t", index_col="isotope",)
        iso_data = (
            iso_data.groupby("element")
            .apply(
                lambda gdf: gdf.assign(
                    shift=lambda df: round(df["mass"] - min(df["mass"]))
                )
            )
            .droplevel(0)
        )
        return iso_data["shift"].astype("int64")

    @staticmethod
    def get_elements(isotopes_file):
        """Get set of elements in isotopes_file."""
        elements = pd.read_csv(
            isotopes_file, sep="\t", usecols=["element"], squeeze=True,
        )
        return set(elements)

    def get_isotopes_from_elements(self, element_list):
        """Get list of isotopes based on elements."""
        iso_data = pd.read_csv(
            self.isotopes_file,
            sep="\t",
            index_col="element",
            usecols=["element", "isotope"],
            squeeze=True,
        )
        return list(iso_data[element_list])

    def get_isotopes(self):
        """Get list of all isotopes."""
        return list(self.isotope_mass_series.keys())

    @staticmethod
    def get_element_from_isotope(isotope):
        """Return element based on isotope."""
        return re.match(r"[A-Z][a-z]?", isotope).group()

    def get_lightest_isotope_from_element(self, element):
        """Return lightest isotpe based on element, e.g. 'C12' from 'C'."""
        iso_data = pd.read_csv(
            self.isotopes_file,
            sep="\t",
            index_col="element",
            usecols=["element", "isotope"],
            squeeze=True,
        )
        iso_data = iso_data.groupby("element").min()
        return iso_data[element]

    def get_lightest_isotope_from_isotope(self, isotope):
        """Return lightest isotope of element for given isotope."""
        return self.get_lightest_isotope_from_element(
            self.get_element_from_isotope(isotope)
        )


class MoleculeInfo:
    """Class containing molecule data."""

    def __init__(self, formula, charge, isotopes):
        """Class contains molecule formula, charge and isotope data.

        Parameters
        ----------
        formula : dict
            Elements as keys
            Number of atoms as values
            E.g. {"C":3, "H":8}
        charge : int
            Charge as signed integer
        isotopes : IsotopeInfo
            Contains isotope data.
        """
        self.isotopes = isotopes
        self.formula = formula
        self.charge = int(charge)

    def __repr__(self):
        return f"MoleculeInfo({self.formula}, {self.charge}, {self.isotopes})"

    def __eq__(self, other):
        if isinstance(other, MoleculeInfo):
            return (
                self.formula == other.formula
                and self.charge == other.charge
                and self.isotopes == other.isotopes
            )
        return False

    @classmethod
    def get_molecule_info(
        cls,
        molecule_name=None,
        molecules_file=None,
        molecule_formula=None,
        molecule_charge=None,
        isotopes_file=None,
    ):
        """Get MoleculeInfo instance from either formula or name.

        Either name and molecules file or formula and charge are reqired.

        Parameters
        ----------
        molecule_name : str
            Molecule name as in molecules_file.
        molecules_file : str or Path
            tab-separated file with name, formula and charge as rows
            e.g. Suc C4H4O3 -1
        molecule_formula : str
            Chemical formula as string.
            No spaces or underscores allowed.
            E.g. "C3H7O1"
        molecule_charge : int
            Charge as signed integer
        isotopes_file : str or Path
            File path to isotopes file
            comma separated csv with element and abundance columns

        Raises
        ------
        ValueError
            If both molecule_name and molecule_formula are specified.

        Returns
        -------
        MoleculeInfo
            Instance of class with formula, charge and isotope data.
        """
        if molecule_name and molecules_file and not molecule_formula:
            molecule = cls.create_from_name(
                molecule_name, molecules_file, isotopes_file
            )
        elif molecule_formula and molecule_charge and not molecule_name:
            molecule = cls.create_from_formula(
                molecule_formula, molecule_charge, isotopes_file
            )
        else:
            raise ValueError(
                "Either molecule name and file or molecule formula and charge have to be specified"
            )
        return molecule

    @classmethod
    def create_from_name(cls, molecule_name, molecules_file, isotopes_file):
        """Get MoleculeInfo instance from molecule name and file."""
        isotopes = IsotopeInfo(isotopes_file)
        molecule_list = cls.get_molecule_list(molecules_file)
        molecule_formula = cls.get_formula(molecule_name, molecule_list, isotopes)
        molecule_charge = cls.get_charge(molecule_name, molecules_file)
        return cls(molecule_formula, molecule_charge, isotopes)

    @classmethod
    def create_from_formula(cls, molecule_formula, molecule_charge, isotopes_file):
        """Get MoleculeInfo instance from formula and charge."""
        isotopes = IsotopeInfo(isotopes_file)
        formula = parse_formula(molecule_formula)
        return cls(formula, molecule_charge, isotopes)

    @staticmethod
    def get_molecule_list(molecules_file):
        """Read and return molecule_list."""
        molecule_list = pd.read_csv(molecules_file, sep="\t", na_filter=False)
        molecule_list["formula"] = molecule_list["formula"].apply(parse_formula)
        molecule_list.set_index("name", drop=True, inplace=True)
        return molecule_list

    @staticmethod
    def get_formula(molecule_name, molecule_list, isotopes):
        """Return molecular formula.

        Parse and look up molecular formula of molecule and
        return number of atoms per element

        Returns
        -------
        dict
            Elements as keys
            Number of atoms as values

        Raises
        ------
        KeyError
            If molecule is not contained in molecules_file.
        ValueError
            If unknown element in molecule formula.
        """
        try:
            n_atoms = molecule_list.loc[molecule_name].formula
        except KeyError as error:
            raise KeyError(
                f"Molecule {error} couldn't be found in molecules file"
            ) from error
        if not all(element in isotopes.elements for element in n_atoms):
            raise ValueError("Unknown element in molecule")
        return n_atoms

    @staticmethod
    def get_charge(molecule_name, molecules_file):
        """Get charge of molecule."""
        charges = pd.read_csv(
            molecules_file,
            sep="\t",
            usecols=["name", "charge"],
            index_col="name",
            squeeze=True,
        )
        try:
            charges = charges.astype(int)
        except ValueError as exc:
            raise ValueError(
                "Charge of at least one molecule missing in molecules_file or missformed file"
            ) from exc
        return charges[molecule_name]

    def get_elements(self):
        """Return elements of molecules."""
        return list(self.formula.keys())

    def get_isotopes(self):
        """Return all possible isotopes in molecule."""
        elements = self.get_elements()
        return self.isotopes.get_isotopes_from_elements(elements)

    def get_molecule_light_isotopes(self):
        """Replace all element names with light isotopes ("C" -> "C13")."""
        molecule_series = pd.Series(self.formula, dtype="int64",)
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
    def subtract_label(molecule_series, label):
        """Subtract label atoms from molecule formula.

        Parameters
        ----------
        molecule_series : Series
            Formula as Series with isotopes as index
        label : Label
            Type of isotopic label, e.g. Label("1N15")

        Returns
        -------
        Series
            Formula with isotopes as index

        Raises
        ------
        ValueError
            If too many atoms are subtracted and negative number of atoms is reached
        """
        formula_difference = molecule_series.copy()
        for heavy in label.as_series.keys():
            light = label.molecule_info.isotopes.get_lightest_isotope_from_isotope(
                heavy
            )
            formula_difference[light] = molecule_series[light] - label.as_series[heavy]
        return formula_difference


class Label:
    """Class for storing label information."""

    def __init__(self, label, molecule_info):
        """Class contains label in different repressentations (dict, series, str).

        Parameters
        ----------
        label : str, dict, Series
            Either string: "No label" or label formula, e.g. "2C13 3H02"
            or dict: {} or e.g. {"C13": 2, "H02": 3}
            or pandas Series: Series() or e.g. Series([2, 3], index=["C13", "H02"])
        molecule_info : MoleculeInfo
            Instance with molecule and isotope information.

        Raises
        ------
        TypeError
            If label not str, dict or pandas Series
        """
        # TODO Use OrderedDict to have unique solution
        if isinstance(label, str):
            self.as_string = label
            self.as_dict = self.parse_label(label)
            self.as_series = pd.Series(self.as_dict, dtype="int64")
        elif isinstance(label, dict):
            self.as_dict = {ele: n for ele, n in label.items() if n}
            self.as_series = pd.Series(self.as_dict, dtype="int64")
            self.as_string = self.generate_label_string(self.as_dict)
        elif isinstance(label, pd.Series):
            self.as_series = label[label != 0].astype("int64")
            self.as_dict = dict(self.as_series)
            self.as_string = self.generate_label_string(self.as_dict)
        else:
            raise TypeError("label has to be str, dict or pandas Series")
        self.molecule_info = molecule_info
        self.mass = self.get_coarse_mass_shift()  # Coarse mass of label alone
        self.check_isotopes()
        self.check_number_atoms()

    def __repr__(self):
        return f"Label({self.as_dict}, {self.molecule_info})"

    def __str__(self):
        return f"Label: {self.as_string}"

    def __bool__(self):
        return bool(self.as_dict)

    def __eq__(self, other):
        if isinstance(other, Label):
            return (
                self.as_dict == other.as_dict
                and self.molecule_info == other.molecule_info
            )
        return False

    def __gt__(self, other):
        return self.mass > other.mass

    def __lt__(self, other):
        return self.mass < other.mass

    def __ge__(self, other):
        return self.mass >= other.mass

    def __le__(self, other):
        return self.mass <= other.mass

    def check_isotopes(self):
        """Raise ValueError for label isotopes not in isotope list."""
        for iso in self.as_dict.keys():
            if iso not in self.molecule_info.isotopes.get_isotopes():
                raise ValueError(f"Label atom {iso} not in isotopes file")

    def check_number_atoms(self):
        """Raise ValueError if number of atoms in label is greater than molecule."""
        for iso, num in self.as_dict.items():
            elem = self.molecule_info.isotopes.get_element_from_isotope(iso)
            try:
                if num > self.molecule_info.formula[elem]:
                    raise ValueError("Too many atoms in label.")
            except KeyError as err:
                raise ValueError("Label contains atoms not in molecule.") from err

    def subtract(self, other):
        """Return difference between two labels as new Label instance.
        self - other
        """
        if not isinstance(other, Label):
            raise TypeError("Both labels has to be Label instance")
        if self.molecule_info != other.molecule_info:
            raise ValueError("Labels have different molecule_info.")
        return Label(
            self.as_series.sub(other.as_series, fill_value=0).astype("int64"),
            self.molecule_info,
        )

    def add(self, other):
        """Add other label and return new Label instance.

        Parameters
        ----------
        other : Label

        Returns
        -------
        Label
            New summed label

        Raises
        ------
        ValueError
            Both labels have different molecule information.
        """
        if self.molecule_info != other.molecule_info:
            raise ValueError("Labels have different molecule_info.")
        return Label(
            self.as_series.add(other.as_series, fill_value=0), self.molecule_info
        )

    def get_coarse_mass_shift(self):
        """Return mass of label compared to unlabelled molecule.
        E.g. 5 for "2C13 2H02"
        """
        iso_shift = self.molecule_info.isotopes.isotope_shift
        return int(self.as_series.multiply(iso_shift).sum())

    def calc_isotopologue_mass(self):
        """Calculate mass of isotopologue.

        Given the molecule name and label composition, return mass in atomic units.

        Returns
        -------
        float
            Molecule mass
        """
        molecule_series = self.molecule_info.get_molecule_light_isotopes()
        light_isotopes = self.molecule_info.subtract_label(molecule_series, self)
        formula_isotopes = pd.concat([light_isotopes, self.as_series])
        mass = (
            self.molecule_info.isotopes.isotope_mass_series.multiply(formula_isotopes)
            .dropna()
            .sum()
        )
        return float(mass)

    @staticmethod
    def generate_label_string(label_as_dict, sep=" "):
        """Generate string representation of label instance."""
        return sep.join([f"{n_atom}{iso}" for iso, n_atom in label_as_dict.items()])

    @staticmethod
    def parse_label(string):
        """Parse label e.g. "3C13 2H02"  and return dict.

        Parse label e.g. 5C13N15 and return dictionary of elements and number of atoms
        Prefix separated by colon can be used and will be ignored, e.g. "NA:2C13"
        Underscores can be used to separate different elements, e.g. "3C13_6H02"

        Parameters
        ----------
        string : str
            Chemical formula

        Returns
        -------
        dict
            Elements as keys
            Number as values

        Raises
        ------
        TypeError
            If input is not string.
        ValueError
           If label contains more than one colon.
        ValueError
            If label string is empty.
        """
        if not isinstance(string, str):
            raise TypeError("label must be string")
        if string.count(":") > 1:
            raise ValueError("only one colon allowed in label to separate prefix")
        string = string.split(":")[-1]
        if string == "":
            raise ValueError(
                "Empty string as label not allowed.\n"
                "Use 'No label' instead or specify label."
            )

        if string.lower() == "no label":
            return {}
        label = re.findall(r"(\d*)([A-Z][a-z]?\d\d)", string)
        label_dict = {}
        for num, elem in label:
            if num == "":
                label_dict[elem] = 1
            elif int(num) > 0:
                label_dict[elem] = int(num)
        return label_dict

    @staticmethod
    def isotope_to_element(label):
        """Change dict key from isotope to element (e.g. H02 -> H)."""
        return {
            re.match(r"[A-Z][a-z]?", elem).group(): n
            for elem, n in label.as_dict.items()
        }


class LabelTuple(Sequence):
    """Class to manage multiple labels."""

    __slots__ = ("_items", "molecule_info")

    def __init__(self, label_list, molecule_info):
        """Class contains tuple of labels from list of strings.

        Parameters
        ----------
        label_list : list
            List of label strings
        molecule_info : MoleculeInfo
            Instance with molecule and isotope information.
        """
        labels = [Label(label, molecule_info) for label in label_list]
        self._items = sort_labels(labels)
        self.molecule_info = molecule_info

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return f"LabelTuple({[label.as_string for label in self._items]}, {self.molecule_info})"


def parse_formula(string):
    """Parse chemical formula and return dict.

    Parse chemical formula of type C3O3H6 and return dictionary
    of elements and number of atoms.
    Be careful, no input check is happening!

    Parameters
    ----------
    string : str
        chemical formula as string

    Returns
    -------
    dict
        Elements as keys
        Number as values
    """
    elements = re.findall(r"([A-Z][a-z]?)(\d*)", string)
    return {elem: int(num) if num != "" else 1 for elem, num in elements}


def sort_labels(labels):
    """Sort list of molecule labels by coarse mass.

    Sort list of molecule labels by coarse mass
    (number of neutrons) from lower to higher mass

    Parameters
    ----------
    labels : list of Labels
        All elements of list must be of instance Label

    Returns
    -------
    list of Labels
        Sorted list
    """
    masses = [(label, label.mass) for label in labels]
    sorted_masses = sorted(masses, key=lambda tup: tup[1])
    sorted_labels = [lab[0] for lab in sorted_masses]
    return sorted_labels


def calc_correction_factor(
    molecule_info, label=False,
):
    """Calculate correction factor with molecule composition defined by label.

    Label supports only H02 (deuterium), C13, N15 and 'No label'.

    Parameters
    ----------
    molecule_info : MoleculeInfo
        Instance with molecule and isotope information.
    label : False or Label
        Instance of Label
        Type and number of atoms.
        E.g. Label('5C13N15') or Label('No label'). False equals to no label

    Returns
    -------
    float
        Correction factor

    Raises
    ------
    ValueError
        If there's more labelled atoms than atoms.
    """
    atom_label = Label.isotope_to_element(label) if label else {}

    n_atoms = molecule_info.formula
    abundance = molecule_info.isotopes.abundance
    prob = {}
    for elem in n_atoms:
        n_atom = (
            n_atoms[elem] - atom_label[elem] if elem in atom_label else n_atoms[elem]
        )
        isotope = molecule_info.isotopes.get_lightest_isotope_from_element(elem)
        prob[elem] = abundance[isotope] ** n_atom
    probability = reduce(mul, prob.values())
    return 1 / probability


def calc_label_diff_prob(label, difference_label):
    """Calculate the transition probablity of difference in labelled atoms.

    Parameters
    ----------
    label : Label
        Type of isotopic label, e.g. Label("1N15")
    difference_label : Label
        Type of isotopic label, e.g. Label("1N15")

    Returns
    -------
    float
        Transition probability
    """
    if not difference_label:
        return 0

    # No classical transition possible with negative values
    if difference_label.as_series.lt(0).any():
        return 0
    n_atoms = label.molecule_info.formula
    abundance = label.molecule_info.isotopes.abundance

    prob = []
    for isotope_lab, n_diff_label in difference_label.as_series.items():
        # breakpoint()
        # get number of atoms of isotope, default to 0
        elem = label.molecule_info.isotopes.get_element_from_isotope(isotope_lab)
        isotope_unlab = label.molecule_info.isotopes.get_lightest_isotope_from_isotope(
            isotope_lab
        )

        n_label = label.as_dict.get(isotope_lab, 0)
        n_unlab = n_atoms[elem] - n_label - n_diff_label
        abun_lab = abundance[isotope_lab]
        abun_unlab = abundance[isotope_unlab]

        trans_pr = binom((n_atoms[elem] - n_label), n_diff_label)
        trans_pr *= abun_lab ** n_diff_label
        trans_pr *= abun_unlab ** n_unlab
        prob.append(trans_pr)

    # Prob is product of single probabilities
    prob_total = reduce(mul, prob)
    assert prob_total <= 1, "Transition probability greater than 1"
    return prob_total
