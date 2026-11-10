"""Polarized emission of a spreading layer (SL) on a neutron star whose spin axis is tilted
from the orbital axis, following Bobrikova, Veledina, Poutanen & Loktev (arXiv:2610.31300),
Sect. 2 and Appendix A, with the same assumptions:

  * spherical NS, compactness u = R_S/R = 0.34; light bending in the Beloborodov (2002)
    approximation, lensing factor D = 1;
  * SL between co-latitudes theta_min and 180 - theta_min of the NS (spin) frame, rotating
    counterclockwise with beta(theta) = beta_eq sin(theta), beta_eq = 0.35;
  * local emission of a semi-infinite electron-scattering atmosphere: diluted blackbody with
    a_es = 0.421 + 0.868 cos(alpha'), f_c T_eff = 2 keV, and PD
    P_es = -0.1171 (1 - cos alpha')/(1 + 3.582 cos alpha') (Viironen & Poutanen 2004);
  * PA from Poutanen (2020), Eqs. (A.14)-(A.15), plus the rotation chi_m of the projected
    spin axis from the projected orbital axis, Eqs. (13)-(14);
  * an infinitely thin, opaque accretion disk in the orbital plane through the NS centre
    hides everything with n.z < 0, Eq. (16).

Two configurations, both with orbital inclination i = 75 deg (as in the paper's Figs. 3-4) and
the SL edge tied to the obliquity, theta_min = 90 - eta, as in the paper's Figs. 5-6 (the SL
reaches the disk):
  A:  Phi = 45 deg,  eta = 30 deg, theta_min = 60 deg
  B:  Phi = 135 deg, eta = 15 deg, theta_min = 75 deg

Output: ns_spreading_layer.pdf (sky images at 5 keV with polarization sticks, and the
dimensionless luminosity x l_x, PD and PA against energy) and ns_spreading_layer.txt.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

t0 = time.perf_counter()

U_NS = 0.34          # compactness R_S/R
BETA_EQ = 0.35       # equatorial SL velocity in units of c
FCT = 2.0            # f_c k T_eff, keV
E = np.logspace(np.log10(0.6), np.log10(40), 70)   # keV  (x = E/2 keV from 0.3 to 20)
E_IMG = 5.0          # keV, energy of the images (and of the numbers printed)
CONFIGS = [dict(name='A', i=75, eta=30, Phi=45, thmin=60),
           dict(name='B', i=75, eta=15, Phi=135, thmin=75)]
NTH, NPH = 300, 720  # grid in cos(theta) and phi


def a_es(c):
    return 0.421 + 0.868*c

def p_es(c):
    return -0.1171*(1 - c)/(1 + 3.582*c)


def layer(cfg, E=E, nth=NTH, nph=NPH):
    i, eta, Phi = np.radians([cfg['i'], cfg['eta'], cfg['Phi']])
    thmin = np.radians(cfg['thmin'])
    u = U_NS

    # NS inclination and the angles that relate the NS and orbital frames, Eqs. (3), (13), (17)
    cis = np.cos(i)*np.cos(eta) + np.sin(i)*np.sin(eta)*np.cos(Phi)
    sis = np.sqrt(1 - cis**2)
    chim = np.arctan2(-np.sin(eta)*np.sin(Phi)/sis, (np.cos(eta)*np.sin(i) - np.sin(eta)*np.cos(i)*np.cos(Phi))/sis)
    Phis = np.arctan2(-np.sin(i)*np.sin(Phi)/sis, (np.sin(eta)*np.cos(i) - np.cos(eta)*np.sin(i)*np.cos(Phi))/sis)

    # SL grid: uniform in cos(theta) and phi (dS = R^2 dcos(theta) dphi)
    ce = np.linspace(np.cos(thmin), -np.cos(thmin), nth + 1)
    pe = np.linspace(0, 2*np.pi, nph + 1)
    cth = 0.5*(ce[1:] + ce[:-1]); ph = 0.5*(pe[1:] + pe[:-1])
    dcos, dph = abs(ce[1] - ce[0]), pe[1] - pe[0]
    CT, PH = np.meshgrid(cth, ph, indexing='ij')
    ST = np.sqrt(1 - CT**2)
    cph, sph = np.cos(PH), np.sin(PH)

    # bending, Eqs. (A.1)-(A.2), and the two visibility conditions
    cpsi = cis*CT + sis*ST*cph
    spsi = np.sqrt(np.clip(1 - cpsi**2, 1e-300, None))
    calp = 1 - (1 - u)*(1 - cpsi)
    salp = np.sqrt(np.clip(1 - calp**2, 0, None))
    vis = (calp > 0) & (np.cos(eta)*CT + np.sin(eta)*ST*np.cos(Phis - PH) > 0)

    # Doppler effect of the SL rotation, Eqs. (A.4)-(A.9)
    beta = BETA_EQ*ST
    gam = 1/np.sqrt(1 - beta**2)
    cxi = -salp/spsi*sis*sph
    delta = 1/(gam*(1 - beta*cxi))
    calp_c = np.clip(delta*calp, 0, 1)                     # cos(alpha') in the comoving frame

    # PA, Eqs. (A.14)-(A.15), in the NS-frame basis; then rotated by chi_m
    B = sis*ST + cis*CT*cph
    fA = spsi/salp*B + (calp - cpsi)/(salp*spsi)*(cph - B*cpsi)
    fC = spsi/salp*CT + (calp - cpsi)/(salp*spsi)*(cis - CT*cpsi)
    chi = np.arctan2(ST*sph + beta*fA, -sis*CT + cis*ST*cph - beta*sph*fC)
    chit = chi + chim
    P = p_es(calp_c)

    # Stokes x l_x, Eq. (22): (1/Delta)(15/pi^5) Int cos(alpha) (delta sqrt(1-u))^4 a_es x'^4/(e^x' - 1) [1, P cos 2chi, P sin 2chi]
    Delta = np.cos(thmin)
    g = delta*np.sqrt(1 - u)
    x = E/FCT
    w = np.where(vis, calp*g**4*a_es(calp_c), 0.0)*dcos*dph   # (nth, nph)
    xp = x[None, None, :]/g[..., None]
    with np.errstate(over='ignore'):
        spec = xp**4/np.expm1(xp)
    WI = (w[..., None]*spec)
    I = 15/np.pi**5/Delta*WI.sum((0, 1))
    Q = 15/np.pi**5/Delta*(WI*(P*np.cos(2*chit))[..., None]).sum((0, 1))
    Uq = 15/np.pi**5/Delta*(WI*(P*np.sin(2*chit))[..., None]).sum((0, 1))

    # sky positions: impact parameter b = R sin(alpha)/sqrt(1-u) along the projected normal,
    # at position angle (from the projected spin axis towards k x omega) atan2(-n.e*_y, ...) + chi_m
    def sky(cth_, ph_):
        st_ = np.sqrt(1 - cth_**2)
        cps = cis*cth_ + sis*st_*np.cos(ph_)
        ca = 1 - (1 - u)*(1 - cps)
        b = np.sqrt(np.clip(1 - ca**2, 0, None))/np.sqrt(1 - u)
        nx = -cis*st_*np.cos(ph_) + sis*cth_          # n . e*_x,  e*_x = (-cos i*, 0, sin i*)
        ny = -st_*np.sin(ph_)                          # n . e*_y,  e*_y = (0, -1, 0)
        ang = np.arctan2(ny, nx) + chim                # position angle from the projected orbital axis
        return b*np.sin(ang), b*np.cos(ang), ca         # (towards e^z_y, towards e^z_x)

    CE, PE = np.meshgrid(ce, pe, indexing='ij')
    Xg, Yg, _ = sky(CE, PE)
    img = np.where(vis, (g**3*a_es(calp_c)*(E_IMG/FCT/g)**3/np.expm1(E_IMG/FCT/g)), np.nan)   # specific intensity I_E (no cos(alpha): it is an image)

    # the rest of the star (no emission), for the drawing
    ce_s = np.linspace(1, -1, 181); pe_s = np.linspace(0, 2*np.pi, 361)
    CS, PS = np.meshgrid(ce_s, pe_s, indexing='ij')
    Xs, Ys, cas = sky(CS, PS)
    cmid = 0.25*(CS[1:, 1:] + CS[:-1, 1:] + CS[1:, :-1] + CS[:-1, :-1])
    pmid = 0.25*(PS[1:, 1:] + PS[:-1, 1:] + PS[1:, :-1] + PS[:-1, :-1])
    smid = np.sqrt(1 - cmid**2)
    vstar = ((1 - (1 - u)*(1 - (cis*cmid + sis*smid*np.cos(pmid))) > 0)
             & (np.cos(eta)*cmid + np.sin(eta)*smid*np.cos(Phis - pmid) > 0))

    # polarization sticks: Stokes summed over SL segments at E_IMG
    ji = np.argmin(abs(E - E_IMG))
    Fi = WI[..., ji]
    qi, ui = P*np.cos(2*chit), P*np.sin(2*chit)
    Xc, Yc, _ = sky(CT, PH)
    sticks = []
    for a in np.array_split(np.arange(nth), 3):
        for c in np.array_split(np.arange(nph), 18):
            w_ = Fi[np.ix_(a, c)]
            if w_.sum() <= 5e-3*Fi.sum():
                continue
            q_ = (w_*qi[np.ix_(a, c)]).sum()/w_.sum(); u_ = (w_*ui[np.ix_(a, c)]).sum()/w_.sum()
            sticks.append(((w_*Xc[np.ix_(a, c)]).sum()/w_.sum(), (w_*Yc[np.ix_(a, c)]).sum()/w_.sum(),
                           np.hypot(q_, u_), 0.5*np.arctan2(u_, q_)))
    return dict(I=I, Q=Q, U=Uq, Xg=Xg, Yg=Yg, img=img, Xs=Xs, Ys=Ys, vstar=vstar, sticks=sticks,
                i_ns=np.degrees(np.arccos(cis)), chim=np.degrees(chim), i=i)


res = [layer(c) for c in CONFIGS]
t_calc = time.perf_counter() - t0


# ------------------------------------------------------------------ plots
cols = ['#e6194b', '#4363d8']
vmax = max(np.nanmax(r['img']) for r in res)
cmap = plt.get_cmap('plasma').with_extremes(bad=(0, 0, 0, 0))
bmax = 1/np.sqrt(1 - U_NS)                               # apparent radius of the star, in R
L = 0.5*0.15*bmax/0.02                                   # a 2% stick is 0.15 R_app long
XLIM = 1.5                                               # image half-width, in R
j5 = np.argmin(abs(E - E_IMG))

fig = plt.figure(figsize=(14, 10.5))
gs = fig.add_gridspec(2, 6, height_ratios=[1.3, 1], hspace=0.25, wspace=1.0)
for j, (cfg, r) in enumerate(zip(CONFIGS, res)):
    ax = fig.add_subplot(gs[0, 3*j:3*j + 3])
    ax.set_facecolor('white')
    ax.add_patch(Ellipse((0, 0), 2*2.6*bmax, 2*2.6*bmax*abs(np.cos(r['i'])), color='0.85', zorder=0))  # disk, behind the star
    ax.pcolormesh(r['Xs'], r['Ys'], np.where(r['vstar'], 1.0, np.nan), cmap='Set2', vmin=0, vmax=3,
                  shading='flat', zorder=1, rasterized=True)
    im = ax.pcolormesh(r['Xg'], r['Yg'], np.log10(r['img']/vmax), cmap=cmap, vmin=-1.5, vmax=0,
                       shading='flat', zorder=2, rasterized=True)
    segs = [((x - L*p*np.sin(a), y - L*p*np.cos(a)), (x + L*p*np.sin(a), y + L*p*np.cos(a)))
            for x, y, p, a in r['sticks']]                # PA counterclockwise from the projected orbital axis
    ax.add_collection(LineCollection(segs, colors='#06c2ac', linewidths=2, zorder=3))
    ca, sa = np.cos(np.radians(r['chim'])), np.sin(np.radians(r['chim']))
    ax.annotate('', xy=(1.45*bmax*sa, 1.45*bmax*ca), xytext=(1.05*bmax*sa, 1.05*bmax*ca),
                arrowprops=dict(arrowstyle='->', color='k', lw=1.5), zorder=4)   # projected NS spin axis
    ax.annotate('', xy=(0, 1.45*bmax), xytext=(0, 1.05*bmax),
                arrowprops=dict(arrowstyle='->', color='0.5', lw=1.5, ls='--'), zorder=4)  # projected orbital axis
    ax.plot([-1.2 - 0.01*L, -1.2 + 0.01*L], [1.62]*2, 'k-', lw=2)
    ax.text(-1.2, 1.67, '1%', ha='center', va='bottom', fontsize=10)
    ax.set_xlim(XLIM, -XLIM)                               # counterclockwise = towards the left, as on the sky
    ax.set_ylim(-0.7, 1.9)
    ax.set_aspect('equal')
    ax.set_title(r'%s: $i=%d^\circ$, $\eta=%d^\circ$, $\Phi=%d^\circ$, $\theta_{\min}=%d^\circ$ ($i_*=%.0f^\circ$)'
                 % (cfg['name'], cfg['i'], cfg['eta'], cfg['Phi'], cfg['thmin'], r['i_ns']), fontsize=11)
    ax.set_xlabel(r'$X/R$'); ax.set_ylabel(r'$Y/R$')
    ax.tick_params(direction='in', top=True, right=True)
cb = fig.colorbar(im, ax=fig.axes[:2], fraction=0.02, pad=0.02)
cb.set_label(r'$\log_{10} I_{%g\,{\rm keV}}/I_{\max}$' % E_IMG)

axI = fig.add_subplot(gs[1, 0:2]); axP = fig.add_subplot(gs[1, 2:4]); axA = fig.add_subplot(gs[1, 4:6])
rows = []
for n, (cfg, r, c) in enumerate(zip(CONFIGS, res, cols)):
    pd = 100*np.hypot(r['Q'], r['U'])/r['I']
    pa = np.degrees(0.5*np.arctan2(r['U'], r['Q']))
    axI.loglog(E, r['I'], color=c, lw=2, label=cfg['name'])
    axP.semilogx(E, pd, color=c, lw=2)
    axA.semilogx(E, pa, color=c, lw=2)
    rows.append(np.column_stack([np.full_like(E, n + 1), np.full_like(E, cfg['Phi']), np.full_like(E, cfg['eta']),
                                 np.full_like(E, cfg['thmin']), E, E/FCT, r['I'], r['Q'], r['U'], pd, pa]))
axI.set_ylim(1e-3, 3*max(r['I'].max() for r in res))
axI.set_ylabel(r'$x\,l_x$'); axP.set_ylabel('PD (%)'); axA.set_ylabel(r'PA ($^\circ$)')
for ax in (axI, axP, axA):
    ax.set_xlabel(r'$E$ (keV), $f_{\rm c}kT_{\rm eff}=2$ keV'); ax.axvspan(2, 8, color='0.93', zorder=0)
    ax.tick_params(direction='in', which='both', top=True, right=True)
axI.legend(frameon=False)
# fig.suptitle(r'Spreading layer on a tilted NS: $u=%.2f$, $\beta_{\rm eq}=%.2f$; images at %g keV, '
#              r'solid arrow: NS spin axis, dashed: orbital axis; IXPE band shaded' % (U_NS, BETA_EQ, E_IMG), fontsize=12)
fig.savefig('ns_spreading_layer.pdf', bbox_inches='tight')
np.savetxt('ns_spreading_layer.txt', np.vstack(rows),
           header='config(1=A, 2=B) Phi[deg] eta[deg] theta_min[deg] E[keV] x I Q U PD[%] PA[deg]  '
                  '(x l_x Stokes; PA counterclockwise from the projected orbital axis)')

for cfg, r in zip(CONFIGS, res):
    print('%s (Phi=%d, eta=%d, theta_min=%d): i* = %.1f deg, chi_m = %.1f deg, PD(%.1f keV) = %.2f%%, PA = %.1f deg'
          % (cfg['name'], cfg['Phi'], cfg['eta'], cfg['thmin'], r['i_ns'], r['chim'], E[j5],
             100*np.hypot(r['Q'][j5], r['U'][j5])/r['I'][j5], np.degrees(0.5*np.arctan2(r['U'][j5], r['Q'][j5]))))
print('computing: %.2f s' % t_calc)
