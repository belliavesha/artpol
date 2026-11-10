"""Table 2 of the chapter (Example A) computed with ARTPOL: PD and PA of the direct emission of a
thermal Kerr disk at 2, 4 and 8 keV for a = 0, 0.5, 0.8 and i = 30, 80 deg.

Functions copied unchanged from kerr_disk_artpol.py; disk parameters as
there and as in Fig. 5 of Loktev et al. (2024): m = 10, M_dot = 0.0033 (script units), f_col = 1.7,
r_out = 6000 r_g = 3000 R_S, 80 log-spaced annuli x 140 azimuths.
PA convention: chi is the angle plotted in the PA panels of Fig. 5 of Loktev et al. (2024), zero for
polarization perpendicular to the projected disk axis; the table gives PA = 90 deg + chi, measured
from the projected axis, so 90 deg means perpendicular to it.
Also prints the same table for M_dot = 0.1 to show that the accretion rate shifts the curves in energy.

Output: artpol_table.txt and artpol_table.tex (rows for the LaTeX table).
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



kev_to_K = 11604525.0061657
fcor, m = 1.7, 10
kevs = np.array([2.0, 4.0, 8.0])
SPINS, INCS = [0.0, 0.5, 0.8], [30, 80]


def disk_pol(chi, inc, M_dot, E=kevs, r_out=6000, rN=80, phiN=140):
    """PD [%] and chi [deg] (as in kerr_disk_artpol.py) of the disk at energies E [keV]."""
    r_in = r_isco_f(chi)
    dR = np.log(r_out/r_in)/rN
    dphi = 2*np.pi/phiN
    rs = np.logspace(np.log10(r_in + 1e-7), np.log10(r_out), rN + 1)[:-1]*np.exp(dR/2)
    phis = np.linspace(0, 2*np.pi, phiN, endpoint=False)
    rr, pp = np.meshgrid(rs, phis, indexing='ij')
    PA, cs, g, dO, bbar, phii = trans(rr, pp, np.radians(inc), a=chi)
    T = Tcol(rr, chi, M_dot, m)
    seg = rr*rr*dR*dphi
    ok = np.isfinite(PA)
    pd = poladeg(cs)
    w = (g**3*seg*dO*beaming(cs))[ok]
    B = BBC(T[ok][:, None], E[None, :]*kev_to_K/g[ok][:, None], fcor)
    I = w @ B; Q = (w*(np.cos(2*PA)*pd)[ok]) @ B; U = (w*(np.sin(2*PA)*pd)[ok]) @ B
    return np.hypot(Q, U)/I*100, np.degrees(wrapol(np.arctan2(U, Q)/2 + np.pi/2))


def p_newton(inc):
    return -100*poladeg(np.cos(np.radians(inc)))


rows, tex = [], []
for inc in INCS:
    for k, a in enumerate(SPINS):
        pd, ch = disk_pol(a, inc, 0.0033)
        rows.append([inc, a] + [v for e in range(3) for v in (pd[e], 90 + ch[e])])
        cells = ' & '.join(r'$%.2f\%%$, $%.1f^\circ$' % (pd[e], 90 + ch[e]) for e in range(3))
        lead = r'$%d^\circ$' % inc if k == 0 else ''
        newt = r'$%.1f\%%$' % p_newton(inc) if k == 0 else ''
        tex.append(r'%-11s & %-4g & %s & %s \\' % (lead, a, cells, newt))
np.savetxt('artpol_table.txt', np.array(rows), fmt='%.4g',
           header='i[deg] a PD2[%] PA2[deg] PD4[%] PA4[deg] PD8[%] PA8[deg]; PA = 90 + chi from the projected axis; '
                  'm=10, M_dot=0.0033, f_col=1.7, r_out=3000 R_S')
open('artpol_table.tex', 'w').write('\n'.join(tex) + '\n')
print('\n'.join(tex))
print('Newtonian p_es: i=30 %.2f%%, i=80 %.2f%%' % (p_newton(30), p_newton(80)))

print('\nsame with M_dot = 0.1 (energy scale x (0.1/0.0033)^(1/4) = %.2f):' % (0.1/0.0033)**0.25)
for inc in INCS:
    for a in SPINS:
        pd, ch = disk_pol(a, inc, 0.1)
        print('i=%d a=%.1f  ' % (inc, a) + '  '.join('%.2f%% %.1f' % (pd[e], 90 + ch[e]) for e in range(3)))
