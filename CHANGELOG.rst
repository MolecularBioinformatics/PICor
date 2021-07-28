=========
Changelog
=========

Version 0.6.2
=============

- Add logging level parameter to calc_isotopologue_correction
- Add --verbose CLI parameter to get correction factors
- Increase resolution correction performance


Version 0.6.1
=============

- fixes a bug with the resolution correction for certain cases:
  In some configurations the mass difference between different labeled species
  resulted in a combinatoric explosion causing calculations to run almost for ever.
- cutoff value for the calculation was introduced and is currently set to 5.

Version 0.6.0
=============

- Change calc_isotopologue_correction interface. Either molecule name or
  molecule formula or charge is now acceptable
- Fix resolution correction
- Resolution correction now corrects for all isotopes in molecule
- All isotopes specified in isotopes_file are now allowed as label

Version 0.5.0
=============

- Refactor code: Introducing MoleculeInfo class

Version 0.4.1
=============

- Add more test to reach 100% test coverage

Version 0.4.0
=============

- Expand README
- Add CLI interface callable by `picor -h`

Version 0.3.3
=============

- Fix compatibility with python 3.6 and 3.7

Version 0.3
===========

- Change name to PICor
- Change package structure based on PyScaffold

Version 0.2
===========

- Add resolution correction  

Version 0.1
===========

- FIX: Nasty bug in transition probability calculations
