# Third-party data and licences

## NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set

The remaining-useful-life data used here is the NASA C-MAPSS "Turbofan Engine
Degradation Simulation Data Set" (sub-sets FD001-FD004), produced by the NASA
Prognostics Center of Excellence (PCoE).

- **Not redistributed here.** The data is downloaded and verified by
  `scripts/fetch_data.py` into `data/raw/` (git-ignored). It is pinned by
  SHA-256; see that script for the hashes.
- **Source / mirror.** NASA PCoE repository, mirrored by the PHM Society at
  <https://data.phmsociety.org/nasa/> (item 6, "Turbofan Engine Degradation
  Simulation Data Set"). The authoritative NASA listing is
  <https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data>
  (download was unavailable there at time of writing).
- **Terms.** The PCoE/PHM repository asks that users employ the data at their
  own risk, acknowledge the repository and the data donors, and cite the data
  set. No restrictive licence is attached (U.S. Government-produced data).
- **Citation.** A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage
  Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," in
  *Proc. 1st Int. Conf. on Prognostics and Health Management (PHM08)*, Denver,
  CO, 2008. (Included in the archive as "Damage Propagation Modeling.pdf".)

## Python dependencies

See `requirements.txt`. NumPy (BSD-3), pandas (BSD-3), PyArrow (Apache-2.0),
scikit-learn (BSD-3); dev: pytest (MIT), ruff (MIT).
