# Marine Dynamics Reference Models

This file records evidence used by the underwater dynamics implementation. A
citation is not treated as a parameter source unless the paper identifies the
vehicle, units, and whether a value was measured or fitted.

## Governing structure

- Fossen, T. I. (2011), *Handbook of Marine Craft Hydrodynamics and Motion
  Control*, Wiley, DOI `10.1002/9781119994138`. This is the source for the
  marine-craft notation `M nu_dot + C(nu) nu + D(nu_r) nu_r + g(eta) = tau`.
- The project implements the structure as a wrench layer. PhysX retains the
  rigid-body integration and gravity responsibilities; see
  `docs/HYDRODYNAMICS_VALIDATION.md`.

## Direct BlueROV2 Heavy evidence

Espinal, A., Cerrada, C., Chaos, D., and Moreno-Salinas, D. (2026),
"Nonlinear identification of a closed-loop semi-physical model for the
vertical heave velocity of the BlueRov2 Heavy," *Jornadas de Automatica*, 47,
DOI `10.17979/ja-cea.2026.47.13848`.

- Vehicle: BlueROV2 Heavy with 8 thrusters (4 horizontal, 4 vertical).
- Evidence: experimental PRBS and step tests in Stabilize mode; DVL heave
  velocity was filtered and split into estimation and validation data.
- Identified semi-physical heave model: `w_dot = -alpha * w * abs(w) + tau`.
- Fitted values reported in Table 5: `alpha = 2.81 1/m`, `Kp = 0.55 1/s`,
  `Ki = 0.40 1/s^2`; the ascending/descending steady-state polynomial
  coefficients are `(3.5786, -0.1851, -0.1038)` and
  `(-5.0866, 0.6991, 0.0018)` respectively, with cutoff `c = 0.15`.
- Interpretation: this is an experimentally validated single-axis,
  closed-loop reference. The fitted alpha is not directly interchangeable
  with a free-swimming 6-DoF quadratic damping coefficient, and is therefore
  not inserted as 6-DoF ground truth.
- PDF source used for inspection:
  `https://revistas.udc.es/index.php/JA_CEA/article/download/13848/9778`.

## Related experimental hydrodynamics

- Avila, J. P. J. and Adamowski, J. C. (2011), "Experimental evaluation of
  the hydrodynamic coefficients of a ROV through Morison's equation," *Ocean
  Engineering* 38(17), 2162-2170, DOI `10.1016/j.oceaneng.2011.09.032`.
  Experimental ROV coefficient methodology; not a BlueROV2 parameter table.
- Lack, S., Rentzow, E., Jeinsch, T. (2019), "Experimental parameter
  identification for an open-frame ROV," *IFAC-PapersOnLine* 52(21), 271-276,
  DOI `10.1016/j.ifacol.2019.12.319`. Experimental identification methodology;
  vehicle and coefficients differ from this BlueROV2 configuration.
- Ridao, P., Tiano, A., El-Fakdi, A., Carreras, M., Zirilli, A. (2004), "On
  the identification of non-linear models of unmanned underwater vehicles,"
  *Control Engineering Practice* 12(12), 1483-1499, DOI
  `10.1016/j.conengprac.2004.01.004`.

These papers support the calibration workflow and the distinction between
published structure and vehicle-specific fitted coefficients. They do not
justify treating the current project nominal drag, inertia, or added mass as
measured BlueROV2 Heavy truth.

## Simulator reference

- Visual asset repository: `https://github.com/bvibhav/stonefish_bluerov2`.
- Imported commit: `6448383af6b7ef6083b0eac2c08102660591e318`.
- License: Apache-2.0 (attribution in `assets/bluerov2/THIRD_PARTY.md`).
- This repository provides visual/Stonefish assets, not a validated Isaac
  parameter set. Stonefish is an independent simulator for future replay.

## Current parameter status

`configs/robots/bluerov2_hydro.yaml` is the authoritative provenance contract.
Mass, inertia, displaced volume, CoM/CoB, damping, added mass, and thrust
curves are currently `estimated`, `geometry-derived`, or project-contract
values with low confidence unless marked otherwise. The only directly
BlueROV2 Heavy experimental coefficients presently recorded are the heave
reference values above, and they remain single-axis reference evidence.
