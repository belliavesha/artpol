# artpol
Use examples of the artful technique for polarization modeling using planar space-time approximation 

Scripts that compute X-ray polarization with the explicit, analytic transport of ARTPOL
(Loktev et al. 2022, 2024): light bending in the approximation of Poutanen (2020), photon
trajectories in a plane, so that the polarization angle relative to the normal of that plane is
conserved, and the Lorentz rotation of the polarization basis between the comoving frame of the
emitter and the static frame. No geodesic is integrated; every quantity is an algebraic function
of the position of the emitting element. A polarized spectrum of a thin disk takes a few tens of
milliseconds on one core.

The scripts accompany the chapter on analytic models of X-ray polarization and produce its ARTPOL figures and tables.

## Requirements

Python 3 with `numpy` and `matplotlib`, and `scipy` for `funnel_reflection_views.py`. Labels are
typeset with LaTeX when `latex` is on the `PATH`, and with matplotlib's own mathtext otherwise.
Each script is run on its own, e.g. `python kerr_disk_artpol.py`, and writes its output files
into the current directory.

`kerr_disk_artpol.jl` needs Julia with the packages `PyPlot` and `DelimitedFiles`; run it as
`julia -t auto kerr_disk_artpol.jl` to spread the rings over threads.

## Scripts

| Script | What it computes | Output | Run time |
|---|---|---|---|
| `kerr_disk_artpol.py` | Thin Kerr disk with a scattering atmosphere: polarized spectrum and image, a = 0.8, i = 70 deg. The reference implementation of ARTPOL for disks. | `disk_spectrum_a0.8_i70.{txt,pdf}`, `disk_image_a0.8_i70.pdf` | ~1.5 s, of which 25 ms for the spectrum |
| `kerr_disk_artpol.jl` | The same calculation in Julia, with the same model and grids. At the end it prints PD and PA at 2, 4 and 8 keV next to the values of the Python script. | `disk_spectrum_a0.8_i70_jl.{txt,pdf}`, `disk_image_a0.8_i70_jl.pdf` | |
| `artpol_table.py` | PD and PA of the thin disk at 2, 4 and 8 keV for a = 0, 0.5, 0.8 and i = 30, 80 deg (Table 2 of the chapter), and the same for a higher accretion rate. | `artpol_table.{txt,tex}` | < 1 s |
| `kerr_disk_synchrotron.py` | Optically thin synchrotron emission of a geometrically thick disk (eleven layers, vertical field) in the configuration of Fig. 15 of Veledina & Pélissier (2025), and the (F_Q, F_U) tracks of an orbiting hot spot (their Fig. 9). | `synchrotron_thick_disk.{pdf,txt}`, `synchrotron_hotspot_tracks.pdf` | ~20 s |
| `ns_spreading_layer.py` | Spreading layer on a neutron star whose spin axis is tilted from the orbital axis, with an opaque disk in the orbital plane, after Bobrikova et al. (arXiv:2610.31300): two configurations, sky images at 5 keV and PD and PA spectra. | `ns_spreading_layer.{pdf,txt}` | ~1.5 s |
| `funnel_reflection_views.py` | Light from a central source reflected once by the outflowing walls of a super-Eddington funnel (after Ahlberg et al., arXiv:2610.23089), seen at 15, 33 and 45 deg from the funnel axis, with the conical grid drawn on the visible wall. | `funnel_reflection_views.{pdf,txt}` | ~3 s |

## Physics and approximations

**Thin disk** (`kerr_disk_artpol.py`, `kerr_disk_artpol.jl`, `artpol_table.py`).
Novikov-Thorne disk with the Page-Thorne dissipation profile, zero torque at the ISCO, a diluted
blackbody with f_col = 1.7, and the Chandrasekhar-Sobolev atmosphere in the fits of Viironen &
Poutanen (2004): angular distribution a_es = (1 + 2.3 mu - 0.3 mu^2)/2.383 and PD
p_es = 0.1171 (1 - mu)/(1 + 3.582 mu), polarized in the plane of the disk. The Kerr metric enters
through the ISCO, the dissipation profile, the Keplerian velocity relative to the locally
non-rotating frame, beta = A (Omega_K - omega)/(r^2 sqrt(Delta)), and the redshift
g = delta (X + Y cos xi) with X = r sqrt(Delta/A) and Y = 2a/sqrt(A). Light bending, the lensing
factor and the rotation of the polarization angle are those of a Schwarzschild metric with the
same mass. The disk runs from the ISCO to 6000 r_g (3000 R_S) on 80 logarithmic rings, taken at
their centres, and 140 azimuths, with phi = 0 counted once. Only the direct image is computed: no
returning radiation and no higher-order images. Loktev et al. (2024) give the accuracy of this
hybrid against exact Kerr ray tracing; the errors grow with spin and inclination and are largest
for photons passing behind the hole.

Units: radii in the code are in r_g = GM/c^2, and the images are in R_S = 2 r_g. The accretion rate
`M_dot` is in the code's own units: the dissipation scales as m * M_dot, so the physical rate is
Mdot = 1.39e17 m^3 M_dot g/s, and `M_dot` has to be changed together with the mass `m`. For
m = 10 and M_dot = 0.0033 (the values of Loktev et al. 2024) this is Mdot = 4.6e17 g/s and
kT_* = 1.29 keV, with T_* as defined in that paper. In the dimensionless energy x = E/kT_* the
polarization spectra do not depend on m or Mdot; in keV they shift as Mdot^(1/4).

The colour of the disk images is g^4 T_eff^4, without the limb darkening a_es; it shows where the
flux comes from and does not enter any of the computed Stokes parameters.

**Synchrotron thick disk** (`kerr_disk_synchrotron.py`). Power-law electrons with
alpha_E = 1 (PD 75%), electric vector perpendicular to the projected comoving field, emissivity
proportional to (sin theta')^(1 + alpha_E), and the flux element dS L g^(3 + alpha_E)/delta of
Veledina & Pélissier (2025), Eq. 35. Matter above the mid-plane moves with the Keplerian velocity
of its equatorial radius. The outer radius (12 R_S) and the constant emissivity are choices of the
script. The default spin is 0, the case of the paper; for a > 0 the equatorial Kerr functions are
evaluated at the spherical radius of elevated matter, an approximation that has not been tested.

**Spreading layer** (`ns_spreading_layer.py`). Spherical star with R_S/R = 0.34, bending in the
approximation of Beloborodov (2002) with lensing factor 1, a layer rotating at
beta = 0.35 sin(theta), the same atmosphere as for the disk (a_es = 0.421 + 0.868 mu), and the
polarization angle of Poutanen (2020), rotated by the angle between the projected spin and orbital
axes. An infinitely thin, opaque disk in the orbital plane hides the lower hemisphere.

**Funnel** (`funnel_reflection_views.py`). Conical funnel with grazing angle 10 deg and wall length
10^3 base radii, walls flowing outward along themselves at 0.2c, an isotropic unpolarized point
source at the centre of the base, and one reflection off a semi-infinite, conservative
electron-scattering atmosphere. The reflection law (Chandrasekhar 1960) is computed exactly: the
H-functions and the N-matrix are solved on a 64-point Gauss-Legendre grid. Light that would hit the
opposite wall after the reflection is dropped. No gravity. The source spectrum, a power law with
photon index 2 and a 30 keV cutoff, is a choice of the script; Thomson reflection is grey, so the
energy dependence enters only through the Doppler factors of the wall, and for a pure power law
the PD and PA do not depend on energy.

## Conventions

The polarization angle is counted counterclockwise on the sky (from north through east) from the
projected symmetry axis: the disk axis for the disk scripts, the orbital axis for
`ns_spreading_layer.py`, and the funnel axis for `funnel_reflection_views.py`. In the images east
is on the left. The thin-disk spectra plot chi = PA - 90 deg, the turn away from the low-energy
direction, which is perpendicular to the projected axis; `artpol_table.py` gives PA = 90 deg + chi.
In the polarization sticks of the images, the length is proportional to the PD.

## Checks

- The polarization angle of `kerr_disk_artpol.py` equals the closed-form chi_GR + chi_SR of
  Loktev et al. (2022) to 1e-11 deg (a = 0, i = 10-80 deg, random points), and a ring at
  r = 10^6 r_g returns exactly p_es(i) with U/Q < 1e-8 for a = 0 and 0.9.
- The dissipation profile agrees with a direct numerical integration of the Page-Thorne formula
  to 1e-6, for a from -0.5 to 0.94. The Lorentz factor equals 1/sqrt(1 - beta^2) to 1e-12.
- The model, grids and transfer function of `kerr_disk_artpol.jl` reproduce the PD and PA spectra
  of `kerr_disk_artpol.py` to 1e-11 (checked on a line-by-line transcription of the Julia code),
  and the intensity spectrum up to a constant factor; the script prints its own PD and PA at
  2, 4 and 8 keV next to the Python values.
- The vector transport of `kerr_disk_synchrotron.py` reproduces the polarization angle and
  redshift of `kerr_disk_artpol.py` to 1e-13 deg and 1e-15 for an equatorial emitter with
  the same local polarization (a vertical field polarizes like the scattering atmosphere).
- The polarization angle of `ns_spreading_layer.py` agrees to 1e-12 deg with an independent
  vector construction (aberration of the polarization basis, then planar transport).
- The reflection law of `funnel_reflection_views.py` conserves flux to 1e-6 for any incidence
  angle, and agrees with an independent polarized Monte Carlo of Thomson scattering in a
  semi-infinite atmosphere (3 x 10^5 photons each at mu_0 = 0.3 and 0.8, 24 bins in mu and
  azimuth): the share of the reflected flux to 4% and q = Q/I and u = U/I to 0.02, within the
  Monte Carlo noise. The polarization sticks lie nearly perpendicular to the projected direction
  to the source, as they should for reflection.

## References

- Ahlberg V., Veledina A., Churazov E., Khabibullin I., arXiv:2610.23089
- Beloborodov A. M. 2002, ApJ 566, L85
- Bobrikova A., Veledina A., Poutanen J., Loktev V., arXiv:2610.31300
- Chandrasekhar S. 1960, Radiative Transfer (Dover)
- Loktev V., Veledina A., Poutanen J. 2022, A&A 660, A25
- Loktev V., Veledina A., Poutanen J., Nättilä J., Suleimanov V. F. 2024, A&A 685, A84
- Page D. N., Thorne K. S. 1974, ApJ 191, 499
- Poutanen J. 2020, A&A 640, A24
- Veledina A., Pélissier M. 2025, A&A 693, A273 (arXiv:2412.08359)
- Viironen K., Poutanen J. 2004, A&A 426, 985
