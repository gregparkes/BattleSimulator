# ChangeLog

All notable changes to this project will be documented in this file.

## [0.3.5] - 2019-10-18
### Added
- `Terrain`. This is a major expansion giving 3D pseudodepth to animated battles. Depth now influences:
	- Decreased movement speed on hills
	- Increased range of units on hills
	- Increased damage output of units targeting downhill enemies
- Boundary checking for units
- `pip` support
- `CHANGELOG.md` file
- First release on GitHub 0.3.5
- `jitcode` for streamlined `numba` use
- Global `target` functions for initialization
- Armor calculations for units
- `example4.ipynb` to reflect changes

### Changed
- `__init__` file dependencies
- Removed file requirement for `Battle()`
- `Distribution` constructor now accepts copy-constructor of another Distribution
- Made `Battle` attributes properties to protect them

### Removed
- Alternative constructor for `Distribution`


## [0.3.4] - 2019-10-11
### Added
- Integrated terrain background
- Terrain weighting in unit movement calculations

### Changed
- Streamlined `Terrain`, `Distribution` into `Battle` object

### Removed
- `test_move` test suite


## [0.3.2] - 2019-10-07
### Added
- Teaching material
- General *decision-based AI* with `aggressive` and `hit_and_run` options
- `Terrain` definition
- LICENSE.

### Changed
- Size-dependency on quiver-plots to units.
- Streamlined legend
- Memory usage from `M_` object

### Removed
- `set_position` in `Battle`

***

Ensure that any use of this material is appropriately referenced and in compliance with the license.