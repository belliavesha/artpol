"""ARTPOL thin Kerr disk: polarized spectrum and image (a = 0.8, i = 70 deg).

Novikov-Thorne disk with the Page-Thorne dissipation profile, diluted blackbody with f_col = 1.7,
Chandrasekhar-Sobolev atmosphere (a_es, p_es of Viironen & Poutanen 2004), and the ARTPOL transport
of Loktev et al. (2022, 2024). m = 10, M_dot = 0.0033 (code units, see README), r_out = 6000 r_g;
80 logarithmic rings (at their centres) x 140 azimuths, 80 energies from 10^-1.5 to 10^1.5 keV.
kerr_disk_artpol.jl is the same calculation in Julia.

Output: disk_spectrum_a0.8_i70.{txt,pdf}, disk_image_a0.8_i70.pdf.
"""
import time
import numpy as np
import matplotlib
import shutil
if shutil.which('latex'):           # typeset labels with LaTeX if it is installed
    matplotlib.rcParams['text.usetex'] = True
else:                               # otherwise matplotlib's own mathtext, Computer Modern look
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 24
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

t_start = time.perf_counter()

bwr = matplotlib.colormaps['plasma_r'].resampled(256)
newcolors = bwr(np.linspace(0, 1, 256))
newcolors[:, :] = newcolors[::-1, :]
newcolors[:1, :] = np.array([0, 0, 0, 1])
newcmp = matplotlib.colors.ListedColormap(newcolors)
newcmp.set_bad((1, 1, 1, 1))


#  local model
def wrapol(a):
    return (a + np.pi/2) % np.pi - np.pi/2

def BB(T, E):
    # cgs; E and T in K
    C = 1.3339153359099091e-16  # 2*6.6261e-27/2.9979245e10**2 * 20836617636.1327**3
    with np.errstate(over='ignore'):
        return C*E**3/np.expm1(E/T)

def BBC(T, E, f):
    return BB(T*f, E)/f**4

def beaming(cs):
    return (1.0 + 2.3*cs - 0.3*cs*cs)/(1 + 4.6/3 - 0.6/4)

def poladeg(cs):
    return -(1.0 - cs)/(1.0 + 3.582*cs)*.1171


#  disk (r in r_g)
def r_isco_f(chi):
    Z_1 = 1 + (1 - chi*chi)**(1/3)*((1 + chi)**(1/3) + (1 - chi)**(1/3))
    Z_2 = np.sqrt(3*chi*chi + Z_1*Z_1)
    sgn = -1 if chi > 0 else 1
    return 3 + Z_2 + sgn*np.sqrt((3 - Z_1)*(3 + Z_1 + 2*Z_2))

def rel_cor(r, a):
    x0 = np.sqrt(r_isco_f(a))
    x = np.sqrt(r)
    x1 = 2*np.cos((np.arccos(a) - np.pi)/3)
    x2 = 2*np.cos((np.arccos(a) + np.pi)/3)
    x3 = -2*np.cos(np.arccos(a)/3)
    A = 3*(x1 - a)**2/x1/(x1 - x2)/(x1 - x3)*np.log((x - x1)/(x0 - x1))
    B = 3*(x2 - a)**2/x2/(x2 - x1)/(x2 - x3)*np.log((x - x2)/(x0 - x2))
    C = 3*(x3 - a)**2/x3/(x3 - x2)/(x3 - x1)*np.log((x - x3)/(x0 - x3))
    return (x - x0 - 1.5*a*np.log(x/x0) - A - B - C)*r/(x**3 - 3*x + 2*a)

def Tcol(r, chi, M_dot, m):
    omega2 = m*r**(-3)
    Q_0 = np.maximum(3/8/np.pi*omega2*M_dot*rel_cor(r, chi), 0.0)
    return (Q_0/9.891097125498387e-33)**0.25


#  transfer
def trajectory(u, y):
    return (1 - u)*y*(1 + u*u*y*y/112 - np.e/1e2*u*y*(np.log(1 - y/2) + y/2))

def trajectoryD(u, y):
    return 1 + 3*u*u*y*y/112 - np.e/1e2*u*y*(2*np.log(1 - y/2) + y*(4 - 3*y)/(4 - 2*y))

def trans(r, phi, i, a=0):
    """r and phi are arrays of the same shape.

    Returns PA (rad), cos of the comoving emission angle, redshift g,
    solid-angle factor dO, impact parameter bbar (r_g) and sky position angle chii.
    Vectors are written out in components (x, y, z) with N = z,
    K = (sin i, 0, cos i), R = (cos phi, sin phi, 0), V = (-sin phi, cos phi, 0).
    """
    r, phi = np.broadcast_arrays(np.asarray(r, float), np.asarray(phi, float))
    si, ci = np.sin(i), np.cos(i)
    cp, sp = np.cos(phi), np.sin(phi)

    u = 2/r
    B = 1 + a/r**1.5
    D = 1 - 2/r + a*a/r/r
    A = 1 + a*a/r/r + 2*a*a/r**3
    b = 1 - 3/r + 2*a/r**1.5
    Del = r*r - 2*r + a*a
    Alf = r**4 + a*a*r*(r + 2)
    Chi = r*np.sqrt(Del/Alf)
    Omg = 2*a*r/Alf
    Gamma = B*np.sqrt(D/A/b)
    beta = Alf/np.sqrt(Del)/r**2*(1/(r**1.5 + a) - Omg)

    cospsi = si*cp                      # K.R
    sinpsi = np.sqrt(1 - cospsi**2)
    y = 1 - cospsi
    x = trajectory(u, y)
    D_beauty = trajectoryD(u, y)
    cosalpha = 1 - x
    sinalpha = np.sqrt(1 - cosalpha**2)

    # S = (sinalpha K + (sinpsi cosalpha - sinalpha cospsi) R)/sinpsi
    c1 = sinalpha/sinpsi
    c2 = (sinpsi*cosalpha - sinalpha*cospsi)/sinpsi
    Sx = c1*si + c2*cp
    Sy = c2*sp
    Sz = c1*ci

    v1 = -sp*si                         # V.K
    r1 = cospsi                         # R.K
    n1 = ci                             # N.K
    chi1 = np.arctan2(-v1, n1*r1)

    kr = Sx*cp + Sy*sp                  # S.R
    kv = -Sx*sp + Sy*cp                 # S.V  (= coszet)
    kn = Sz                             # S.N
    coszet = kv
    delta = 1/(1 - beta*coszet)/Gamma
    # P = (S - Gamma beta V + (Gamma - 1) coszet V) delta ; V has no z component
    cV = -Gamma*beta + (Gamma - 1)*coszet
    Px = (Sx - cV*sp)*delta
    Py = (Sy + cV*cp)*delta
    Pz = Sz*delta
    PR = Px*cp + Py*sp
    PV = -Px*sp + Py*cp
    PN = Pz
    chi2 = np.arctan2(-kn, kv*kr)
    chi3 = np.arctan2(-PR, PV*PN)

    ochar = 2*a/np.sqrt(Alf)
    gin = Gamma*(Chi + ochar*beta + PV*(beta*Chi + ochar))
    dO = kn/np.sqrt(D)*D_beauty
    bbar = r*sinalpha/np.sqrt(1 - u)

    # chii = base_rot(N, R, K): angle between the sky projections of R and N
    E1R = np.stack([cp - cospsi*si, sp, -cospsi*ci])     # R - (K.R) K
    E1N = np.array([-ci*si, 0.0, 1 - ci*ci])              # N - (K.N) K
    E2N = np.array([0.0, -si, 0.0])                       # K x N
    E1R = E1R/np.linalg.norm(E1R, axis=0)
    E1N = E1N/np.linalg.norm(E1N)
    E2N = E2N/np.linalg.norm(E2N)
    coschi = np.tensordot(E1N, E1R, axes=(0, 0))
    sinchi = np.tensordot(E2N, E1R, axes=(0, 0))
    chii = np.arctan2(sinchi, coschi)

    return chi1 + chi2 + chi3, PN, gin, dO, bbar, chii


# parameters
chi = 0.8
inc = 70
kev_to_K = 11604525.0061657
M_dot = 0.0033
fcor = 1.7
m = 10
fN = 80
kevs = np.logspace(-1.5, 1.5, fN)

r_isco = r_isco_f(chi)
r_in = r_isco + 1e-7
r_out = 6000
phiN = 140
dphi = 2*np.pi/phiN
rN = 80
dR = np.log(r_out/r_in)/rN
rs = r_in*np.exp((np.arange(rN) + 0.5)*dR)                 # ring centres
phis = np.linspace(0, 2*np.pi, phiN, endpoint=False)   # phi = 0 once, 2 pi excluded


#  spectrum
t0 = time.perf_counter()
rr, pp = np.meshgrid(rs, phis, indexing='ij')          # (rN, phiN)
PA, cs, g, dO, bbar, phii = trans(rr, pp, np.radians(inc), a=chi)
T = Tcol(rr, chi, M_dot, m)
seg = rr*rr*dR*dphi
ok = np.isfinite(PA)

pd = poladeg(cs)
w = (g**3*seg*dO*beaming(cs))[ok]                      # weight of each cell
Bmat = BBC(T[ok][:, None], kevs[None, :]*kev_to_K/g[ok][:, None], fcor)   # (cells, energies)
q = (np.cos(2*PA)*pd)[ok]
u = (np.sin(2*PA)*pd)[ok]
sssp = w @ Bmat
ssqsp = (w*q) @ Bmat
ssusp = (w*u) @ Bmat
t_spec = time.perf_counter() - t0

pdsp = np.sqrt(ssusp**2 + ssqsp**2)/sssp*100
pasp = np.degrees(wrapol(np.arctan2(ssusp, ssqsp)/2 + np.pi/2))
np.savetxt("disk_spectrum_a0.8_i70.txt", np.column_stack((kevs, sssp, ssqsp, ssusp, pdsp, pasp)),
           header="E[keV] I Q U (arbitrary units) PD[%] PA[deg]")


#  image
t0 = time.perf_counter()
xspan = 25/2    # R_S
yspan = xspan
cutout = 40     # r_g
NP = 13
ls0 = 11
Nphisecs = 12
phisecs = np.linspace(0, 2*np.pi, Nphisecs + 1)
philvls = phisecs + phisecs[1]/2

def sky(r, phi):
    """Sky position (X, Y) in R_S of disk points (r, phi), Eq. (50), plus PA and cos."""
    pa, cs, g, dO, bbar, phii = trans(r, phi, np.radians(inc), a=chi)
    bbar = bbar/2
    return -bbar*np.sin(phii), bbar*np.cos(phii), pa, cs, g

rimg = np.logspace(np.log10(r_isco + 1e-7), np.log10(cutout), 200)
phimg = np.linspace(0, 2*np.pi, 361)
ri, pi_ = np.meshgrid(rimg, phimg, indexing='ij')
Xim, Yim, _, _, gim = sky(ri, pi_)
img = np.log10(np.maximum(gim**4*Tcol(ri, chi, M_dot, m)**4, np.finfo(float).tiny))
img = img - img.max()

rings = [sky(np.full_like(phimg, rl), phimg)[:2] for rl in (10, 20, 30)]
labels = [sky(np.array([rl]), np.array([np.radians(200)]))[:2] for rl in (10, 20, 30)]
rays = [sky(rimg, np.full_like(rimg, phl))[:2] for phl in phisecs[:-1]]

radsecs = np.array([r_isco, 10, 20, 30, cutout])
radsecs = np.sqrt(radsecs[:-1]*radsecs[1:])
rc, pc = np.meshgrid(radsecs, philvls[:-1], indexing='ij')
xcc, ycc, pa, cs, _ = sky(rc.ravel(), pc.ravel())
half = 2*xspan*ls0*poladeg(cs)/NP                       # half-length of each stick (5% ~ 2 R_S)
sticks = np.stack([np.stack([xcc - np.cos(pa)*half, ycc - np.sin(pa)*half], -1),
                   np.stack([xcc + np.cos(pa)*half, ycc + np.sin(pa)*half], -1)], 1)
t_img = time.perf_counter() - t0


# plots
t0 = time.perf_counter()
linewidth = 2
fontsize = matplotlib.rcParams['font.size']
fig, (plotBBSP, plotPDSP, plotPASP) = plt.subplots(3, figsize=(6, 12), sharex='col',
                                                   gridspec_kw={'height_ratios': [4, 3, 3]})
fig.subplots_adjust(hspace=0, wspace=0)
peak = (sssp*kevs).max()
plotBBSP.plot(kevs, sssp*kevs, '-', color='#e6194b', linewidth=linewidth)
plotBBSP.set_xscale("log")
plotBBSP.set_yscale("log")
plotBBSP.set_ylim(peak*1e-4, peak*3)
plotPDSP.plot(kevs, pdsp, '-', color='#e6194b', linewidth=linewidth)
plotPASP.plot(kevs, pasp, '-', color='#e6194b', linewidth=linewidth)
for axv in [plotPDSP, plotPASP]:
    axv.yaxis.set_major_locator(ticker.MaxNLocator(6))
    axv.yaxis.set_minor_locator(ticker.AutoMinorLocator())
for axv in [plotPDSP, plotPASP, plotBBSP]:
    axv.tick_params(axis="both", which="both", pad=10, top=True, right=True, direction='in')
    axv.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
plotBBSP.text(0.95, 0.95, r'$i= %i^\circ $' % inc, transform=plotBBSP.transAxes,
              verticalalignment="top", horizontalalignment='right')
plotPDSP.yaxis.set_major_formatter(ticker.StrMethodFormatter(r"${x:.2g}$"))
plotPASP.yaxis.set_major_formatter(ticker.StrMethodFormatter(r"${x:.2g}^\circ$"))
plotPASP.set_xlabel(r"$E$ (keV)", fontsize=fontsize)
plotPASP.set_xlim(kevs[0], kevs[-1])
plotPASP.set_ylabel(r"$\chi$", fontsize=fontsize)
plotPDSP.set_ylabel(r"$p\,(\%)$", fontsize=fontsize)
plotBBSP.set_ylabel(r"$xl_x$", fontsize=fontsize)
fig.savefig("disk_spectrum_a0.8_i70.pdf", bbox_inches='tight')

stickcolor = (0.0, 0.4, 0.70)
figIMG, ax = plt.subplots(1, 1, figsize=(8, 7))
ax.set_facecolor('white')
ax.fill(Xim[0], Yim[0], color='k', zorder=0)
im = ax.pcolormesh(Xim, Yim, img, cmap=newcmp, vmin=-3.1, vmax=0, shading='gouraud', zorder=1)
for (xr, yr) in rings + rays:
    ax.plot(xr, yr, '-', color='black', linewidth=0.5, zorder=2)
for rl, (xl, yl) in zip((10, 20, 30), labels):
    ax.text(xl[0], yl[0], "%1.0f" % (rl/2), fontsize=16, ha='center', va='center', zorder=4)
ax.add_collection(matplotlib.collections.LineCollection(sticks, colors=[stickcolor], linewidths=1.5, zorder=3))
h5 = 2*xspan*ls0*0.05/NP
ax.plot([xspan*0.7 - h5, xspan*0.7 + h5], [yspan*0.82]*2, '-', linewidth=1.5, color=stickcolor, zorder=3)
ax.text(xspan*0.7, yspan*0.88, r"$5\%$", fontsize=16, ha='center', va='bottom', color=stickcolor)
cbar = figIMG.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=fontsize, axis="both", which="both", direction='out')
cbar.ax.set_ylabel(r"$\log_{10} I/I_{\rm{max}}$", fontsize=fontsize, labelpad=10)
ticks = [-xspan*4/5, -2*xspan/5, 0, 2*xspan/5, xspan*4/5]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="both", pad=10, top=True, right=True, direction='in')
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlim(-xspan, xspan)
ax.set_ylim(-7, yspan)
ax.set_aspect('equal')
ax.set_xlabel(r"$X\, [R_{\rm{S}}]$", fontsize=fontsize)
ax.set_ylabel(r"$Y\, [R_{\rm{S}}]$", fontsize=fontsize)
ax.text(0.05, 0.95, r'$a=%.1f,\ i= %i^\circ$' % (chi, inc), transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='left')
figIMG.savefig("disk_image_a0.8_i70.pdf", bbox_inches='tight')
t_plot = time.perf_counter() - t0

print(f"spectrum ({rN}x{phiN} cells, {fN} energies): {t_spec*1e3:7.1f} ms")
print(f"image grid, rings, rays, sticks:        {t_img*1e3:7.1f} ms")
print(f"plotting and saving two PDFs:           {t_plot*1e3:7.1f} ms")
print(f"total, including imports:               {time.perf_counter() - t_start:7.2f} s")
