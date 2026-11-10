"""Sky images of light reflected once by the walls of a super-Eddington funnel, for three viewing angles.

Conical funnel with base radius 1, grazing angle ALPHA, wall length RLEN; walls flowing outward along themselves at BETA;
isotropic unpolarized point source at the centre; one reflection off a semi-infinite conservative
electron-scattering atmosphere (Chandrasekhar 1960), with the polarization carried from the comoving
frame of the wall to the observer by the Lorentz rotation of the bases of Loktev et al. (2022).
After Ahlberg, Veledina, Churazov & Khabibullin (arXiv:2610.23089), single reflection only.

The funnel axis points up on the page and the observer sits at angle i from it, for the three
angles of VIEWS (15, 33.2 and 45 deg). Only light that leaves through the opening is kept, so every image lies inside the image of
the rim. Coordinates are centred on the centre of the rim, in units of the base radius rho_in; X grows
to the right, so east (e_E) is on the left. The PA is counted counterclockwise on the page (from
north through east) from the projected funnel axis (up).
Overlaid: the rim (opening ellipse) and the conical grid of the wall (circles of constant height and
generatrices every 30 deg), drawn only where the wall is visible.

Output: funnel_reflection_views.pdf, funnel_reflection_views.txt (PD and PA of the total at E_IMG).
"""
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import root

t0 = time.perf_counter()

# ------------------------------------------------------------------ parameters
ALPHA = np.radians(10.0)   # funnel grazing angle (paper: 5-25 deg; Fig. 2c uses 10 deg)
RLEN = 1e3                 # distance from the funnel centre to its edge, in base radii (paper: 10^3)
BETA = 0.2                 # outflow speed along the walls (paper Fig. 2c: 0-0.4)
VIEWS = [15.0, 33.2, 45.0]   # deg, angle between the funnel axis and the line of sight
GAMMA, ECUT = 2.0, 30.0    # incident cutoff power law: E F_E ~ E^(2-GAMMA) exp(-E/ECUT)
E = np.logspace(0, 2, 61)  # keV
E_IMG = 6.0                # energy of the sky images, keV
NZ, NPHI = 600, 360        # wall grid (z log-spaced in 1+z, azimuth uniform)

H = RLEN*np.cos(ALPHA)
ZETA = np.arctan((RLEN*np.sin(ALPHA) - 1)/H)
RTOP = 1 + H*np.tan(ZETA)


# ------------------------------------------------------------------ reflection law
def gl01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5*(x + 1), 0.5*w

class Reflection:
    """Reflection of unpolarized light by a semi-infinite Thomson-scattering atmosphere,
    in the form of Madhusudhan & Burrows (2012, Eqs. 23-28), after Abhyankar & Fymat (1970)
    and Chandrasekhar (1960). stokes() returns I_l, I_r, U per unit F for an incident flux pi F,
    in the basis (e1 = projected normal, e2 = k x n). Checked: flux conservation for omega = 1
    to 1e-8, and the single-scattering limit against the Thomson phase matrix to 1e-7."""
    def __init__(self, omega=1.0, n=64, iters=500):
        self.om = omega
        self.mu, self.w = mu, w = gl01(n)
        self.H1n = self._H(lambda m: 3/8*omega*(1 - m**2)*(1 + 2*m**2), iters)
        self.H2n = self._H(lambda m: 3/16*omega*(1 + m**2)**2, iters)
        M = self.M(mu)
        N = M.copy()
        for _ in range(iters):     # plain iteration, then a root finder (the conservative case converges slowly)
            J = np.einsum('k,kji,kjl,mk->mil', w, N, M, 1/(mu[:, None] + mu[None, :]))
            A = np.eye(2)[None] - 0.5*omega*mu[:, None, None]*J
            N = np.linalg.solve(A.transpose(0, 2, 1), M.transpose(0, 2, 1)).transpose(0, 2, 1)
        def resid(x):
            Nx = x.reshape(N.shape)
            J = np.einsum('k,kji,kjl,mk->mil', w, Nx, M, 1/(mu[:, None] + mu[None, :]))
            A = np.eye(2)[None] - 0.5*omega*mu[:, None, None]*J
            return (np.einsum('mij,mjk->mik', Nx, A) - M).ravel()
        self.Nn = root(resid, N.ravel(), method='hybr', options={'xtol': 1e-14}).x.reshape(N.shape)

    @staticmethod
    def M(m):
        m = np.asarray(m, float)
        out = np.zeros(m.shape + (2, 2))
        out[..., 0, 0] = m**2; out[..., 0, 1] = np.sqrt(2)*(1 - m**2); out[..., 1, 0] = 1.0
        return np.sqrt(3)/2*out

    def _H(self, psi, iters):
        mu, w = self.mu, self.w
        P = psi(mu); s0 = np.sqrt(1 - 2*np.sum(w*P)); Hn = np.ones_like(mu)
        K = mu[None, :]*P[None, :]/(mu[:, None] + mu[None, :])*w[None, :]
        for _ in range(iters):
            Hn = 1/(s0 + K @ Hn)
        return P, s0, Hn

    def H(self, which, m):
        P, s0, Hn = self.H1n if which == 1 else self.H2n
        m = np.asarray(m, float)[..., None]
        return 1/(s0 + np.sum(self.w*self.mu*P*Hn/(m + self.mu), axis=-1))

    def N(self, m):
        m = np.asarray(m, float)
        J = np.einsum('k,kji,kjl,...k->...il', self.w, self.Nn, self.M(self.mu), 1/(m[..., None] + self.mu))
        A = np.eye(2) - 0.5*self.om*m[..., None, None]*J
        return np.einsum('...ij,...jk->...ik', self.M(m), np.linalg.inv(A))

    def stokes(self, mu0, mu, dphi):
        """dphi = azimuth(k_out) - azimuth(k_in) around the normal, both for propagation directions."""
        N0 = self.N(mu0); N = self.N(mu)
        s1 = N0[..., 0, 0] + N0[..., 1, 0]; s2 = N0[..., 0, 1] + N0[..., 1, 1]
        Al = N[..., 0, 0]*s1 + N[..., 0, 1]*s2
        Ar = N[..., 1, 0]*s1 + N[..., 1, 1]*s2
        H1 = self.H(1, mu)*self.H(1, mu0); H2 = self.H(2, mu)*self.H(2, mu0)
        BU = 3*mu0*np.sqrt((1 - mu**2)*(1 - mu0**2))*H1
        Cr = 0.75*(1 - mu0**2)*H2
        D = self.om/(8*(mu + mu0))*mu0
        Il = D*(Al - mu*BU*np.cos(dphi) - mu**2*Cr*np.cos(2*dphi))
        Ir = D*(Ar + Cr*np.cos(2*dphi))
        U = D*(BU*np.sin(dphi) + 2*mu*Cr*np.sin(2*dphi))
        return Il, Ir, U


# ------------------------------------------------------------------ vector helpers (last axis = 3)
def dot(a, b):
    return np.sum(a*b, axis=-1)

def unit(a):
    return a/np.linalg.norm(a, axis=-1, keepdims=True)

def basis(k, l):
    """Polarization basis B(k, l) of Loktev et al. (2022): e1 = projection of l, e2 = k x l."""
    e1 = unit(l - dot(k, l)[..., None]*k)
    e2 = unit(np.cross(k, l))
    return e1, e2

def aberrate(k, b, beta):
    """Direction in the frame moving with velocity beta*b, and Doppler factor delta = E/E'."""
    g = 1/np.sqrt(1 - beta**2)
    kb = dot(k, b)[..., None]
    delta = 1/(g*(1 - beta*kb))
    kp = delta*(k - g*beta*b + (g - 1)*kb*b)
    return kp, delta[..., 0]


# ------------------------------------------------------------------ funnel wall grid (funnel frame, z = axis)
u_edges = np.linspace(0, np.log1p(H), NZ + 1)
z_edges = np.expm1(u_edges)
p_edges = np.linspace(0, 2*np.pi, NPHI + 1)
zc = 0.5*(z_edges[1:] + z_edges[:-1])
pc = 0.5*(p_edges[1:] + p_edges[:-1])
Z, PH = np.meshgrid(zc, pc, indexing='ij')
RHO = 1 + Z*np.tan(ZETA)
cph, sph = np.cos(PH), np.sin(PH)
P = np.stack([RHO*cph, RHO*sph, Z], -1)                                          # wall points
NRM = np.stack([-cph*np.cos(ZETA), -sph*np.cos(ZETA), np.full_like(Z, np.sin(ZETA))], -1)  # normal into the cavity
BDIR = np.stack([cph*np.sin(ZETA), sph*np.sin(ZETA), np.full_like(Z, np.cos(ZETA))], -1)   # flow direction
T1 = np.stack([-sph, cph, np.zeros_like(Z)], -1)                                 # tangent basis, T1 x T2 = NRM
T2 = np.cross(NRM, T1)
dA = (RHO*np.diff(p_edges)[None, :]*np.diff(z_edges)[:, None])/np.cos(ZETA)
DIST = np.linalg.norm(P, axis=-1)
K1 = P/DIST[..., None]                                                           # from the source to the wall

REFL = Reflection(1.0)


def azimuth(v):
    return np.arctan2(dot(v, T2), dot(v, T1))



def visible(Pw, Nw, k):
    """Wall points Pw (normals Nw) that face the observer k and are seen through the opening."""
    t_top = (H - Pw[..., 2])/k[2]
    q = Pw + t_top[..., None]*k
    return (dot(Nw, k) > 0) & (np.hypot(q[..., 0], q[..., 1]) <= RTOP)


def funnel_view(inc):
    """Sky image, sticks and total Stokes at E_IMG for the funnel axis at angle inc to the line of sight."""
    k = np.array([np.sin(inc), 0.0, np.cos(inc)])     # funnel frame, observer in the x-z plane
    eN, eE = basis(k, np.array([0, 0, 1.0]))          # up = projected funnel axis; eE = k x axis
    centre = np.array([0, 0, H])                      # centre of the rim
    sky = lambda p: (-dot(p - centre, eE), dot(p - centre, eN))   # X to the right, Y up, about the rim centre

    vis = visible(P, NRM, k)
    eta = dot(NRM, k)
    kv = np.broadcast_to(k, P.shape)[vis]
    nv, bv, pv, distv = NRM[vis], BDIR[vis], P[vis], DIST[vis]
    k1p, d1 = aberrate(K1[vis], bv, BETA)
    k2p, d2 = aberrate(kv, bv, BETA)
    mu0 = -dot(nv, k1p)
    mu = dot(nv, k2p)
    ok = (mu > 0) & (mu0 > 0)
    t1, t2 = T1[vis], T2[vis]
    az = lambda v: np.arctan2(dot(v, t2), dot(v, t1))
    Il, Ir, U = REFL.stokes(np.clip(mu0, 1e-9, 1), np.clip(mu, 1e-9, 1), az(k2p) - az(k1p))
    r = np.where(ok, (Il + Ir)/np.pi, 0.0)
    pd = np.hypot(Il - Ir, U)/(Il + Ir)
    psi = 0.5*np.arctan2(U, Il - Ir)
    e1n, e2n = basis(k2p, nv)                         # comoving basis of the Chandrasekhar solution
    f = np.cos(psi)[:, None]*e1n + np.sin(psi)[:, None]*e2n
    b1p, b2p = basis(k2p, bv)                         # B(k', beta) -> B(k, beta): no rotation
    chi = np.arctan2(dot(f, b2p), dot(f, b1p))
    b1, b2 = basis(kv, bv)
    f = np.cos(chi)[:, None]*b1 + np.sin(chi)[:, None]*b2
    ang = np.arctan2(dot(f, eE), dot(f, eN))          # PA from up towards e_E (east, on the left of the page): counterclockwise
    qs, us = pd*np.cos(2*ang), pd*np.sin(2*ang)
    pagex, pagey = -dot(f, eE), dot(f, eN)            # f on the page (X right = -e_E, Y up)
    assert np.allclose(np.abs(-np.sin(ang)*pagex + np.cos(ang)*pagey), np.hypot(pagex, pagey)), 'stick orientation'

    Iel = d2**3*r/d1
    finc = (np.array([E_IMG])*(d1/d2)[:, None])**(1 - GAMMA)*np.exp(-E_IMG*(d1/d2)[:, None]/ECUT)/distv[:, None]**2
    img = np.full(Z.shape, np.nan)
    img[vis] = Iel*finc[:, 0]
    Fimg = img[vis]*eta[vis]*dA[vis]                  # flux of each element at E_IMG
    Itot, Qtot, Utot = Fimg.sum(), (Fimg*qs).sum(), (Fimg*us).sum()

    zg, pg = np.meshgrid(z_edges, p_edges, indexing='ij')
    rg = 1 + zg*np.tan(ZETA)
    Xg, Yg = sky(np.stack([rg*np.cos(pg), rg*np.sin(pg), zg], -1))

    # polarization sticks binned on the page (flux-weighted q, u)
    Xv, Yv = sky(pv)
    nb = 7
    xe = np.linspace(-RTOP, RTOP, nb + 1); ye = np.linspace(-RTOP*np.cos(inc), RTOP*np.cos(inc), nb + 1)
    ix = np.clip(np.digitize(Xv, xe) - 1, 0, nb - 1); iy = np.clip(np.digitize(Yv, ye) - 1, 0, nb - 1)
    sticks = []
    for a in range(nb):
        for b in range(nb):
            m = (ix == a) & (iy == b)
            w = Fimg[m]
            if m.sum() < 20 or w.sum() <= 0.002*Fimg.sum():
                continue
            xq = (w*qs[m]).sum()/w.sum(); xu = (w*us[m]).sum()/w.sum()
            sticks.append(((w*Xv[m]).sum()/w.sum(), (w*Yv[m]).sum()/w.sum(), np.hypot(xq, xu), 0.5*np.arctan2(xu, xq)))

    # conical grid on the visible wall: circles of constant height and generatrices, masked where hidden
    ph = np.linspace(0, 2*np.pi, 721)
    zz = np.linspace(0, H, 2001)
    lines = []
    def wall(z, p):
        rho = 1 + z*np.tan(ZETA)
        Pw = np.stack([rho*np.cos(p), rho*np.sin(p), z], -1)
        Nw = np.stack([-np.cos(p)*np.cos(ZETA), -np.sin(p)*np.cos(ZETA), np.full_like(z, np.sin(ZETA))], -1)
        X, Y = sky(Pw)
        v = visible(Pw, Nw, k)
        return np.where(v, X, np.nan), np.where(v, Y, np.nan)
    for frac in (0.2, 0.4, 0.6, 0.8):
        lines.append(wall(np.full_like(ph, frac*H), ph))
    for p0 in np.radians(np.arange(0, 360, 30)):
        lines.append(wall(zz, np.full_like(zz, p0)))
    rim = sky(np.stack([RTOP*np.cos(ph), RTOP*np.sin(ph), np.full_like(ph, H)], -1))
    return dict(Xg=Xg, Yg=Yg, img=img, sticks=sticks, cell=min(np.diff(xe)[0], np.diff(ye)[0]),
                lines=lines, rim=rim, pd=100*np.hypot(Qtot, Utot)/Itot,
                pa=np.degrees(0.5*np.arctan2(Utot, Qtot)) % 180, vis_frac=vis.mean())


res = [funnel_view(np.radians(i)) for i in VIEWS]
t_calc = time.perf_counter() - t0

# ------------------------------------------------------------------ plot
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                     'font.size': 9})
fig, axs = plt.subplots(1, len(VIEWS), figsize=(7.0, 2.75), sharey=True, gridspec_kw=dict(wspace=0.06))
vmax = max(np.nanmax(r['img']) for r in res)
cmap = plt.get_cmap('plasma').with_extremes(bad='white')
smax = max(s[2] for r in res for s in r['sticks'])
for ax, inc, r in zip(axs, VIEWS, res):
    im = ax.pcolormesh(r['Xg'], r['Yg'], np.log10(r['img']/vmax), cmap=cmap, vmin=-3, vmax=0,
                       shading='flat', rasterized=True)
    for x, y in r['lines']:
        ax.plot(x, y, '-', color='white', lw=0.4, alpha=0.7)
    ax.plot(*r['rim'], 'k-', lw=0.8)
    L = 0.45*r['cell']/smax                            # longest stick = 0.9 of a bin
    segs = [((x - L*p*np.sin(a), y + L*p*np.cos(a)), (x + L*p*np.sin(a), y - L*p*np.cos(a)))
            for x, y, p, a in r['sticks']]               # PA counterclockwise from up: direction (-sin a, cos a)
    ax.add_collection(LineCollection(segs, colors='blue', linewidths=1.2))
    ax.set_xlim(-1.05*RTOP, 1.05*RTOP); ax.set_ylim(-1.05*RTOP, 1.05*RTOP)
    ax.set_aspect('equal')
    ax.set_title(r'$i=%.0f^\circ$: PD $=%.0f\%%$, PA $=%.0f^\circ$' % (inc, r['pd'], r['pa']), fontsize=8.5)
    ax.set_xlabel(r'$X\ [\rho_{\rm in}]$')
    ax.tick_params(direction='in', top=True, right=True)
axs[0].set_ylabel(r'$Y\ [\rho_{\rm in}]$')
cax = axs[-1].inset_axes([1.04, 0.0, 0.05, 1.0])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r'$\log_{10} I_{%g\,{\rm keV}}/I_{\max}$' % E_IMG)
fig.savefig('funnel_reflection_views.pdf', bbox_inches='tight')
np.savetxt('funnel_reflection_views.txt', np.array([[i, r['pd'], r['pa']] for i, r in zip(VIEWS, res)]),
           header='i[deg] PD[%%] PA[deg] at %g keV, PA counterclockwise from the projected funnel axis; '
                  'alpha=%g deg, R=%g, beta=%g' % (E_IMG, np.degrees(ALPHA), RLEN, BETA))
for i, r in zip(VIEWS, res):
    print('i = %4.1f deg: PD(%g keV) = %5.2f%%, PA = %5.1f deg, visible wall fraction %.3f'
          % (i, E_IMG, r['pd'], r['pa'], r['vis_frac']))
print('computing: %.2f s' % t_calc)
