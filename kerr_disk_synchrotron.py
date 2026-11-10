"""Vectorized ARTPOL image of an optically thin, synchrotron-emitting thick disk around a black hole,
and polarized-flux tracks of an orbiting hot spot.

The disk image recomputes the configuration of Fig. 15 of Veledina & Pelissier (2024): Schwarzschild
(a = 0), i = 30 deg, vertical field B = B_z, eta = 30 deg at the ISCO (h = 1.73 R_S), eleven layers.
Intensity and polarization are shown in one panel (VP24 show them in two). Only the vertical-field
case is drawn; the toroidal case is kept in DISK_FIELDS as a comment.

Synchrotron emission, field geometry and disk geometry follow Veledina & Pelissier (2024,
arXiv:2412.08359; VP24), Sects. 2 and 4; the Kerr part follows ARTPOL (Loktev et al. 2024), as in
kerr_disk_artpol.py:
  * Kerr metric, spin A_SPIN (set to 0 for the chapter figure); light bending, lensing factor and the transport of polarization
    as in ARTPOL: Schwarzschild-type bending (Poutanen 2020), trajectories in a plane, so the angle
    of the polarization vector to the normal of that plane is conserved (Loktev et al. 2022);
  * Keplerian rotation (counterclockwise): ZAMO-frame speed beta = A (Omega_K - omega)/(r^2 sqrt(Delta))
    and redshift g = Gamma [X + Y beta + (beta X + Y) cos(xi')] = delta (X + Y cos xi), with X = r sqrt(Delta/A)
    and Y = 2a/sqrt(A) (xi, xi': angle between photon and velocity in the ZAMO and comoving frames),
    as in kerr_disk_artpol.py; matter above the mid-plane moves with the velocity of its equatorial
    radius r_e (VP24, Sect. 2.1);
  * synchrotron emission of power-law electrons, I' ~ E'^-alpha_E (sin theta')^(1+alpha_E), alpha_E = 1,
    PD P_s = (p+1)/(p+7/3) = 75%; the electric vector is perpendicular to the projection of the comoving
    field B' (VP24, Eq. 29); B' = b_eq [cos(phi') u + sin(phi') (n x u)] + b_z n (VP24, Eq. 24):
    toroidal phi' = 0, radial phi' = 90 deg, vertical (poloidal) b_z = 1;
  * disk: Stokes flux of an element ~ dS L g^(3+alpha_E)/delta (sin theta')^(1+alpha_E) (VP24, Eq. 35),
    dS = r_e dr_e dphi/sqrt(1 - 2/r + a^2/r^2) as in ARTPOL; thick disk of VP24 Fig. 15: constant
    height with eta = 30 deg at the ISCO, eleven layers;
  * hot spot (VP24, Sect. 4.1 and Fig. 9): equatorial spot of constant comoving size, flux
    ~ L g^(3+alpha_E) (sin theta')^(1+alpha_E); radii 3, 4.5, 7 R_S and i = 150 deg as in Fig. 9.
Choices not fixed by VP24: the disk extends to R_OUT = 12 R_S with constant emissivity; no light-travel
delays for the spot. For A_SPIN > 0 (not used for the chapter figure) the equatorial Kerr functions
are evaluated at the spherical radius for elevated matter, an approximation that is not validated.
Synchrotron spectra are power laws at every point, so PD and PA do not depend on energy.

Output: synchrotron_thick_disk.pdf (log intensity with polarization sticks, stick length = PD),
synchrotron_hotspot_tracks.pdf (F_Q-F_U tracks), synchrotron_thick_disk.txt (total PD and PA).
"""
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

t_start = time.perf_counter()

# ---------------------------------------------------------------- parameters
A_SPIN = 0.0                # Schwarzschild, as in VP24
INC = 30.0                  # deg, image (as in VP24, Fig. 15)
ALPHA_E = 1.0
P_S = (2*ALPHA_E + 2)/(2*ALPHA_E + 1 + 7/3)
ETA_ISCO = 30.0             # deg, disk half-thickness angle at the ISCO (VP24 Fig. 15)
N_LAYERS = 11
R_OUT = 24.0                # r_g (12 R_S); not given in VP24
DISK_FIELDS = {# 'toroidal': dict(b_eq=1.0, phip=0.0, b_z=0.0),
               'vertical': dict(b_eq=0.0, phip=0.0, b_z=1.0)}      # VP24 Fig. 15
SPOT_FIELDS = {r"radial, $\phi'=90^\circ$": dict(b_eq=1.0, phip=90.0, b_z=0.0),
               r"toroidal, $\phi'=0^\circ$": dict(b_eq=1.0, phip=0.0, b_z=0.0),
               r'vertical, $B=B_z$': dict(b_eq=0.0, phip=0.0, b_z=1.0)}
SPOT_INC = 150.0            # deg (VP24 Fig. 9 middle column)
SPOT_RADII = [3.0, 4.5, 7.0]   # R_S
NR, NPHI = 400, 1920        # disk grid per layer (fine enough for 500 pixels)
XSPAN = 8.0                 # R_S, half-width of the image
NPIX = 500


def dot(a, b):
    return np.sum(a*b, axis=-1)

def unit(a):
    return a/np.linalg.norm(a, axis=-1, keepdims=True)

def basis(k, l):
    """Polarization basis B(k, l): e1 = projection of l on the sky of k, e2 = k x l."""
    return unit(l - dot(k, l)[..., None]*k), unit(np.cross(k, l))

def r_isco_f(a):
    z1 = 1 + (1 - a*a)**(1/3)*((1 + a)**(1/3) + (1 - a)**(1/3))
    z2 = np.sqrt(3*a*a + z1*z1)
    return 3 + z2 - np.sqrt((3 - z1)*(3 + z1 + 2*z2))

R_ISCO = r_isco_f(A_SPIN)                       # r_g
H_DISK = R_ISCO*np.tan(np.radians(ETA_ISCO))   # r_g


def elements(re, h, ph, field, inc, a=A_SPIN):
    """Everything for matter at equatorial radius re, height h, azimuth ph (arrays, r_g units).
    Returns sky position (R_S), sky PA chi, and the factors entering the disk and spot fluxes."""
    re, h, ph = np.broadcast_arrays(re, h, ph)
    i = np.radians(inc)
    o = np.array([np.sin(i), 0.0, np.cos(i)])
    nrm = np.array([0.0, 0.0, 1.0])
    r = np.hypot(re, h)
    rhat = np.stack([np.cos(ph)*re/r, np.sin(ph)*re/r, h/r], -1)
    cpsi = dot(rhat, o)
    spsi = np.sqrt(np.clip(1 - cpsi**2, 1e-30, None))
    u, y = 2/r, 1 - cpsi
    with np.errstate(divide='ignore', invalid='ignore'):
        x = (1 - u)*y*(1 + u*u*y*y/112 - np.e/1e2*u*y*(np.log(1 - y/2) + y/2))          # ARTPOL trajectory()
        L = 1 + 3*u*u*y*y/112 - np.e/1e2*u*y*(2*np.log(1 - y/2) + y*(4 - 3*y)/(4 - 2*y))  # trajectoryD()
    calp = 1 - x
    ok = np.isfinite(calp) & (np.abs(calp) < 1) & (L > 0)
    calp = np.where(ok, calp, 0.0)
    salp = np.sqrt(1 - calp**2)
    k0 = (salp[..., None]*o + (spsi*calp - cpsi*salp)[..., None]*rhat)/spsi[..., None]

    # Kerr Keplerian motion of the equatorial radius (ZAMO frame) and redshift
    Del_e = re*re - 2*re + a*a
    Alf_e = re**4 + a*a*re*(re + 2)
    beta = Alf_e/np.sqrt(Del_e)/re**2*(1/(re**1.5 + a) - 2*a*re/Alf_e)
    gam = 1/np.sqrt(1 - beta**2)
    Del, Alf = r*r - 2*r + a*a, r**4 + a*a*r*(r + 2)
    Chi = r*np.sqrt(Del/Alf)
    ochar = 2*a/np.sqrt(Alf)
    uvel = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], -1)
    cxi = dot(uvel, k0)
    delta = 1/(gam*(1 - beta*cxi))
    k0p = delta[..., None]*(k0 - (gam*beta)[..., None]*uvel + ((gam - 1)*cxi)[..., None]*uvel)
    cxip = dot(uvel, k0p)
    g = gam*(Chi + ochar*beta + cxip*(beta*Chi + ochar))

    # comoving field and the synchrotron electric vector
    nxu = np.cross(nrm, uvel)
    B = field['b_eq']*(np.cos(np.radians(field['phip']))*uvel + np.sin(np.radians(field['phip']))*nxu) \
        + field['b_z']*np.broadcast_to(nrm, uvel.shape)
    B = unit(B)
    sth = np.sqrt(np.clip(1 - dot(B, k0p)**2, 0, None))
    fp = unit(np.cross(k0p, B))
    # aberration in the basis B(k, u), then planar transport
    a1, a2 = basis(k0p, uvel)
    b1, b2 = basis(k0, uvel)
    f = dot(fp, a1)[..., None]*b1 + dot(fp, a2)[..., None]*b2
    w = unit(np.cross(rhat, o))
    fobs = dot(f, w)[..., None]*w + dot(f, np.cross(k0, w))[..., None]*np.cross(o, w)
    e1 = np.array([-np.cos(i), 0.0, np.sin(i)])          # projected disk axis
    e2 = np.cross(o, nrm); e2 /= np.linalg.norm(e2)
    chi = np.arctan2(dot(fobs, e2), dot(fobs, e1))       # PA, counterclockwise from the projected axis

    bimp = r*salp/np.sqrt(1 - u)/2                       # impact parameter in R_S
    X = bimp*rhat[..., 1]/spsi
    Y = bimp*dot(rhat, e1)/spsi
    spot = np.where(ok, L*g**(3 + ALPHA_E)*sth**(1 + ALPHA_E), 0.0)
    disk = np.where(ok, L*g**(3 + ALPHA_E)/delta*sth**(1 + ALPHA_E)/np.sqrt(1 - 2/r + a*a/r/r), 0.0)
    return dict(X=X, Y=Y, chi=chi, spot=spot, disk=disk, ok=ok, g=g)


def disk_layers(field, inc=INC, nr=NR, nphi=NPHI):
    re_edges = np.geomspace(R_ISCO, R_OUT, nr + 1)
    re = np.sqrt(re_edges[1:]*re_edges[:-1]); dre = np.diff(re_edges)
    ph = (np.arange(nphi) + 0.5)*2*np.pi/nphi
    RE, PH = np.meshgrid(re, ph, indexing='ij')
    dA = (re*dre)[:, None]*(2*np.pi/nphi)
    out = []
    for h in H_DISK*np.linspace(-1, 1, N_LAYERS):
        el = elements(RE, h, PH, field, inc)
        m = el['ok']
        out.append(dict(X=el['X'][m], Y=el['Y'][m], chi=el['chi'][m], W=(el['disk']*dA)[m]))
    return {k: np.concatenate([d[k] for d in out]) for k in out[0]}


def cic_image(el, npix=NPIX, span=XSPAN):
    """Stokes images; each element is shared between the four nearest pixels (cloud-in-cell)."""
    edges = np.linspace(-span, span, npix + 1)
    dx = edges[1] - edges[0]
    fx = (el['X'] + span)/dx - 0.5; fy = (el['Y'] + span)/dx - 0.5
    ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int)
    tx, ty = fx - ix, fy - iy
    out = []
    for wgt in (el['W'], el['W']*P_S*np.cos(2*el['chi']), el['W']*P_S*np.sin(2*el['chi'])):
        img = np.zeros(npix*npix)
        for ddx, ddy, ww in ((0, 0, (1 - tx)*(1 - ty)), (1, 0, tx*(1 - ty)), (0, 1, (1 - tx)*ty), (1, 1, tx*ty)):
            jx, jy = ix + ddx, iy + ddy
            m = (jx >= 0) & (jx < npix) & (jy >= 0) & (jy < npix)
            img += np.bincount(jy[m]*npix + jx[m], weights=(wgt*ww)[m], minlength=npix*npix)
        out.append(img.reshape(npix, npix))
    return (edges, *out)


# ---------------------------------------------------------------- disk images
t0 = time.perf_counter()
res = {}
for name, fld in DISK_FIELDS.items():
    el = disk_layers(fld)
    I = el['W'].sum(); q = (el['W']*P_S*np.cos(2*el['chi'])).sum()/I; u = (el['W']*P_S*np.sin(2*el['chi'])).sum()/I
    res[name] = dict(img=cic_image(el), pd=100*np.hypot(q, u), pa=np.degrees(0.5*np.arctan2(u, q)))
t_disk = time.perf_counter() - t0

# ---------------------------------------------------------------- hot-spot tracks
t0 = time.perf_counter()
phs = np.linspace(0, 2*np.pi, 721)
tracks = {}
for name, fld in SPOT_FIELDS.items():
    tracks[name] = []
    for rS in SPOT_RADII:
        el = elements(np.full_like(phs, 2*rS), 0.0, phs, fld, SPOT_INC)
        F = el['spot']
        Fn = F/F.mean()                                  # normalized to the orbit-averaged flux
        tracks[name].append((rS, Fn*P_S*np.cos(2*el['chi']), Fn*P_S*np.sin(2*el['chi'])))
t_spot = time.perf_counter() - t0


# ---------------------------------------------------------------- plot 1: disk images (Fig. 4 style)
t0 = time.perf_counter()
bwr = matplotlib.colormaps['plasma_r'].resampled(256)
newcolors = bwr(np.linspace(0, 1, 256))[::-1]
# no black lowest bin here: the zero of intensity at the critical point would look like a second hole
newcmp = matplotlib.colors.ListedColormap(newcolors)
fontsize = 9
stickcolor = (0.0, 0.65, 0.6)   # teal: readable on the bright, nearly uniform disk
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix'})
fig, axs = plt.subplots(1, len(DISK_FIELDS), figsize=(4.0*len(DISK_FIELDS), 3.3), sharey=True,
                        gridspec_kw=dict(wspace=0.05), squeeze=False)
axs = axs[0]
nb = 20                                                   # stick cells per side
LABEL_PHI = 300                                           # disk azimuth of the ring labels, deg
for ax, (name, r) in zip(axs, res.items()):
    edges, I, Q, U = r['img']
    with np.errstate(divide='ignore'):
        logI = np.where(I > 0, np.log10(I/np.percentile(I[I > 0], 99.5)), np.nan)   # robust to single-pixel spikes; empty pixels transparent
    phg = np.linspace(0, 2*np.pi, 721)
    el_in = elements(np.full_like(phg, R_ISCO), 0.0, phg, DISK_FIELDS[name], INC)
    ax.fill(el_in['X'], el_in['Y'], color='k', zorder=0)      # black inside the image of the ISCO
    im = ax.pcolormesh(edges, edges, logI, cmap=newcmp, vmin=-2.0, vmax=0, shading='flat', rasterized=True, zorder=1)
    # mid-plane grid: rings at 5, 7.5, 10 R_S and rays every 30 deg (thin black lines)
    for rS in (5, 7.5, 10):
        el = elements(np.full_like(phg, 2*rS), 0.0, phg, DISK_FIELDS[name], INC)
        ax.plot(el['X'], el['Y'], 'k-', lw=0.5, zorder=2)
        j = np.argmin(abs(phg - np.radians(LABEL_PHI)))
        ax.text(el['X'][j], el['Y'][j], '%g' % rS, fontsize=fontsize - 2, ha='center', va='center', zorder=4,
                clip_on=True)
    reg = np.geomspace(R_ISCO, R_OUT, 200)
    for phl in np.radians(np.arange(0, 360, 30)):
        el = elements(reg, 0.0, np.full_like(reg, phl), DISK_FIELDS[name], INC)
        ax.plot(el['X'], el['Y'], 'k-', lw=0.5, zorder=2)
    # sticks: Stokes summed over cells, length proportional to PD
    k = NPIX//nb
    Ic = I[:nb*k, :nb*k].reshape(nb, k, nb, k).sum((1, 3))
    Qc = Q[:nb*k, :nb*k].reshape(nb, k, nb, k).sum((1, 3))
    Uc = U[:nb*k, :nb*k].reshape(nb, k, nb, k).sum((1, 3))
    cc = 0.5*(edges[:nb*k:k] + edges[k:nb*k + 1:k])
    XC, YC = np.meshgrid(cc, cc)
    m = Ic > 0.02*Ic.max()
    p = np.hypot(Qc, Uc)[m]/Ic[m]; a = 0.5*np.arctan2(Uc, Qc)[m]
    half = 0.5*0.9*(cc[1] - cc[0])*p/P_S                  # P_s = full cell
    segs = np.stack([np.stack([XC[m] + half*np.sin(a), YC[m] - half*np.cos(a)], -1),
                     np.stack([XC[m] - half*np.sin(a), YC[m] + half*np.cos(a)], -1)], 1)
    ax.add_collection(LineCollection(segs, colors=[stickcolor], linewidths=1.0, zorder=3))
    hb = 0.5*0.9*(cc[1] - cc[0])*0.5/P_S                  # 50% bar
    ax.plot([0.7*XSPAN - hb, 0.7*XSPAN + hb], [5.9]*2, '-', color=stickcolor, lw=1.5, zorder=6)
    ax.text(0.7*XSPAN, 6.1, r'$50\%$', color=stickcolor, ha='center', va='bottom', fontsize=fontsize - 2, zorder=6)
    ax.add_patch(matplotlib.patches.Rectangle((0.7*XSPAN - 1.1, 5.55), 2.2, 1.4, color='white', alpha=0.85, zorder=4))
    ax.text(0.03, 0.97, 'PD = %.0f%%, PA = %.0f$^\\circ$' % (r['pd'], r['pa']), color='black',
            transform=ax.transAxes, va='top', ha='left', fontsize=fontsize - 1, zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
    ax.set_xlim(-XSPAN, XSPAN); ax.set_ylim(-5.5, 7.0); ax.set_aspect('equal')
    ax.set_xlabel(r'$X\ [R_{\rm S}]$', fontsize=fontsize)
    ax.tick_params(direction='in', top=True, right=True, labelsize=fontsize - 2)
axs[0].set_ylabel(r'$Y\ [R_{\rm S}]$', fontsize=fontsize)
cax = axs[-1].inset_axes([1.03, 0.0, 0.045, 1.0])          # colour bar as tall as the image
cb = fig.colorbar(im, cax=cax)
cb.set_label(r'$\log_{10} I/I_{\max}$', fontsize=fontsize)
cb.set_ticks([-2, -1.5, -1, -0.5, 0])
cb.ax.tick_params(labelsize=fontsize - 2)
# parameters go to the figure caption (no suptitle)
fig.savefig('synchrotron_thick_disk.pdf', bbox_inches='tight')

# ---------------------------------------------------------------- plot 2: hot-spot tracks (VP24 Fig. 9 style)
cmaps = ['Greys', 'Reds', 'Greens']
fig2, axs2 = plt.subplots(1, len(SPOT_FIELDS), figsize=(5.2*len(SPOT_FIELDS), 5.0), gridspec_kw=dict(wspace=0.3))
for ax, (name, trs) in zip(axs2, tracks.items()):
    lim = 1.1*max(np.abs(np.r_[fq, fu]).max() for _, fq, fu in trs)
    for (rS, fq, fu), cm in zip(trs, cmaps):
        pts = np.stack([fq, fu], -1)
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], 1), cmap=cm, norm=plt.Normalize(-0.5, 1), linewidths=2)
        lc.set_array(phs[:-1]/(2*np.pi))                # saturation grows along the orbit
        ax.add_collection(lc)
        ax.plot([], [], color=plt.get_cmap(cm)(0.8), lw=2, label=r'$r_{\rm e}=%.1f\,R_{\rm S}$' % rS)
    ax.axhline(0, color='0.7', lw=0.5); ax.axvline(0, color='0.7', lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel(r'$F_Q/\langle F_I\rangle$'); ax.set_ylabel(r'$F_U/\langle F_I\rangle$')
    ax.set_title(name)
    ax.tick_params(direction='in', top=True, right=True)
axs2[0].legend(frameon=False, fontsize=9)
fig2.suptitle(r'Equatorial hot spot, Kerr $a=%.1f$, $i=%g^\circ$ (colour saturates along the orbit, from $\varphi=0$)'
              % (A_SPIN, SPOT_INC), fontsize=12)
fig2.savefig('synchrotron_hotspot_tracks.pdf', bbox_inches='tight')
t_plot = time.perf_counter() - t0

np.savetxt('synchrotron_thick_disk.txt', np.array([[n, r['pd'], r['pa']] for n, r in enumerate(res.values())]),
           header='field (%s) PD[%%] PA[deg]' % ', '.join(DISK_FIELDS) + '  energy independent (power law); '
                  'PA counterclockwise from the projected disk axis; a=%.1f i=%g' % (A_SPIN, INC))
for name, r in res.items():
    print('%-20s PD = %5.2f%%  PA = %6.2f deg' % (name, r['pd'], r['pa']))
print('disk: %.2f s, hot spots: %.2f s, plots: %.2f s, total: %.2f s'
      % (t_disk, t_spot, t_plot, time.perf_counter() - t_start))
