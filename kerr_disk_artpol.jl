# ARTPOL thin Kerr disk in Julia: the same model and grids as kerr_disk_artpol.py, and the same
# polarized spectrum and image (a = 0.8, i = 70 deg).
#
# Model: Novikov-Thorne disk with the Page-Thorne dissipation profile, diluted blackbody with
# f_col = 1.7, Chandrasekhar-Sobolev atmosphere (a_es, p_es of Viironen & Poutanen 2004),
# ARTPOL transport (Loktev et al. 2022, 2024). m = 10, M_dot = 0.0033 in the units of the Python
# script, r_out = 6000 r_g, 80 rings (logarithmic, at the ring centres) x 140 azimuths (phi = 0
# counted once), 80 energies from 10^-1.5 to 10^1.5 keV.
#
# Implementation:
#  1. trans() works on scalars only: the vectors R, N, K, V, S, P are written out in components,
#     so a call allocates nothing.
#  2. The rings are distributed over threads (start Julia with `julia -t auto`), each ring
#     filling its own column, and the columns are summed afterwards in a fixed order,
#     so the result does not depend on the number of threads.
#  3. g^3 cancels between the transfer factor and the blackbody B(f_col T, E/g), so the inner
#     energy loop is one expm1 and a few multiplications per energy.
#
# Run: julia -t auto kerr_disk_artpol.jl   (packages: PyPlot, DelimitedFiles)

using DelimitedFiles
# The script only writes PDFs: use a non-interactive matplotlib backend, so that no Tk/Tcl
# window system is needed (set MPLBACKEND yourself to override). Must come before PyPlot.
haskey(ENV, "MPLBACKEND") || (ENV["MPLBACKEND"] = "Agg")
using PyPlot

const BBCONST = 5.040366120264132e+22
const KEV_TO_K = 11604525.0061657


wrapold(a) = rad2deg.(mod.((a.+(π/2)) , π) .-(π/2))     # angle in [-90, 90) deg

beaming(cs) = (1.0 + 2.3*cs -0.3*cs*cs)/(1+4.6/3-0.6/4)
poladeg(cs) = -(1.0-cs)/(1.0+3.582*cs)*.1171

function r_isco_f(chi)
    Z_1 = 1 + (1-chi*chi)^(1.0/3.0)*((1+chi)^(1.0/3.0)+(1-chi)^(1.0/3.0))
    Z_2 = √(chi*chi*3 + Z_1*Z_1)
    sgn = chi > 0 ? -1 : 1                                  # prograde: minus, retrograde: plus
    return 3 + Z_2 + sgn*√((3 - Z_1)*(3 + Z_1 +2*Z_2))
end

"""Page-Thorne factor f(r, a), normalized so that f -> 1 - sqrt(r_isco/r) far out (r in r_g)."""
function rel_cor(r, a)
    x0 = √(r_isco_f(a))
    x = √(r)
    x1 = 2*cos((acos(a) - π)/3)
    x2 = 2*cos((acos(a) + π)/3)
    x3 = -2*cos(acos(a)/3)
    A = 3*(x1 - a)^2/x1/(x1 - x2)/(x1 - x3)*log((x - x1)/(x0 - x1))
    B = 3*(x2 - a)^2/x2/(x2 - x1)/(x2 - x3)*log((x - x2)/(x0 - x2))
    C = 3*(x3 - a)^2/x3/(x3 - x2)/(x3 - x1)*log((x - x3)/(x0 - x3))
    return (x - x0 - 1.5*a*log(x/x0) - A - B - C)*r/(x^3 - 3*x + 2*a)
end

"""Effective temperature (K) at r (r_g), as Tcol() of kerr_disk_artpol.py."""
function Tcol(r,chi,M_dot,m)
    omega2 = m*r^(-3)
    Q_0 = max(3.0/8.0/pi*omega2*M_dot*rel_cor(r, chi), 0.0)
    return √(√(Q_0/9.891097125498387e-33))
end

trajectory(u,y)  = ( 1 - u )*y*( 1 + u*u*y*y/112 - ℯ/1e2*u*y*( log( 1 - y/2 ) + y/2 ) )
trajectoryD(u,y) = 1 + 3*u*u*y*y/112 - ℯ/1e2*u*y*( 2*log( 1 - y/2 ) + y*(4-3*y)/(4-2*y) )


"""
    trans(r, phi, si, ci, a)

Allocation-free version of trans() of kerr_disk_artpol.py; si, ci = sin i, cos i.
N = (0,0,1), K = (si,0,ci), R = (cos phi, sin phi, 0), V = (-sin phi, cos phi, 0).
Returns PA, cos of the comoving emission angle, redshift g, dO, impact parameter bbar,
and the sky position angle chii.
"""
@inline function trans(r, phi, si, ci, a)
    u = 2/r
    B = 1 + a/r^(1.5)
    D = 1 - 2/r + a*a/r/r
    A = 1 + a*a/r/r + 2*a*a/r/r/r
    b = 1 - 3/r + 2*a/r^(1.5)
    Del = r*r - 2*r + a*a
    Alf = r^4 + a*a*r*(r+2)
    Chi = r*√(Del/Alf)
    Omg = 2*a*r/Alf
    Gamma = B*√(D/A/b)
    beta = √(1 - 1/Gamma^2)                         # = A (Omega_K - omega)/(r^2 sqrt(Delta))

    sp, cp = sincos(phi)
    cospsi = si*cp                                  # K.R
    sinpsi = √(1 - cospsi^2)
    y = 1 - cospsi
    x = trajectory(u,y)
    D_cal = trajectoryD(u,y)
    cosalpha = 1 - x
    sinalpha = √(1 - cosalpha^2)

    c1 = sinalpha/sinpsi                            # S = c1 K + c2 R
    c2 = (sinpsi*cosalpha - sinalpha*cospsi)/sinpsi
    Sx = c1*si + c2*cp
    Sy = c2*sp
    Sz = c1*ci

    chi1 = atan(sp*si, ci*cospsi)                   # atan(-V.K, (N.K)(R.K))

    kr = Sx*cp + Sy*sp                              # S.R
    kv = -Sx*sp + Sy*cp                             # S.V = coszet
    kn = Sz                                         # S.N
    delta = 1/(1 - beta*kv)/Gamma
    cV = -Gamma*beta + (Gamma - 1)*kv               # P = (S + cV V) delta
    Px = (Sx - cV*sp)*delta
    Py = (Sy + cV*cp)*delta
    Pz = Sz*delta
    PR = Px*cp + Py*sp
    PV = -Px*sp + Py*cp
    chi2 = atan(-kn, kv*kr)
    chi3 = atan(-PR, PV*Pz)

    ochar = Omg*√(Alf)/r
    gin = Gamma*(Chi + ochar*beta + PV*(beta*Chi + ochar))
    dO = kn/√(D)*D_cal
    bbar = r*sinalpha/√(1 - u)
    chii = chi1 + π                                 # base_ort(N,R,K) + π
    return chi1+chi2+chi3, Pz, gin, dO, bbar, chii
end


"""Stokes spectra E*(I, Q, U) on the grids of kerr_disk_artpol.py; Q, U with the local PD p_es < 0."""
function spectrum(inc, chi; kevs = 10 .^ range(-1.5, 1.5, length = 80), r_out = 6000, phiN = 140, rN = 80,
                  M_dot = 0.0033, m = 10.0, fcor = 1.7)
    fN = length(kevs)
    Ek = kevs .* KEV_TO_K
    r_in = r_isco_f(chi) + 1e-7
    dphi = 2π/phiN
    dR = log(r_out/r_in)/rN
    rs = [r_in*exp((k - 0.5)*dR) for k in 1:rN]          # ring centres
    phis = [(k - 1)*dphi for k in 1:phiN]                # phi = 0 once, 2 pi excluded
    si, ci = sincos(deg2rad(inc))

    Ir = zeros(fN, rN); Qr = zeros(fN, rN); Ur = zeros(fN, rN)
    Threads.@threads :static for ir in 1:rN
        r = rs[ir]
        fT = fcor*Tcol(r, chi, M_dot, m)
        seg = r*r*dR*dphi
        @inbounds for phi in phis
            PA, cs, g, dO, bbar, chii = trans(r, phi, si, ci, chi)
            isfinite(PA) || continue
            pd = poladeg(cs)                        # negative: polarization in the disk plane
            w = BBCONST*seg*dO*beaming(cs)/fcor^4   # g^3 cancels against B(f T, E/g); 1/f^4 dilution
            s2, c2 = sincos(2PA)
            q = c2*pd
            uu = s2*pd
            gT = g*fT
            for i in 1:fN
                E = Ek[i]
                F = w*E^3/expm1(E/gT)
                Ir[i,ir] += F
                Qr[i,ir] += F*q
                Ur[i,ir] += F*uu
            end
        end
    end
    sssp  = vec(sum(Ir, dims = 2)) .* kevs
    ssqsp = vec(sum(Qr, dims = 2)) .* kevs
    ssusp = vec(sum(Ur, dims = 2)) .* kevs
    return kevs, sssp, ssqsp, ssusp
end

"""PD (%) and PA chi (deg, as plotted; 0 = perpendicular to the projected disk axis)."""
pol(I, Q, U) = (sqrt.(Q .^ 2 .+ U .^ 2) ./ I .* 100, wrapold(atan.(U, Q) ./ 2 .+ π/2))


sky(r, phi, si, ci, chi) = begin
    PA, cs, g, dO, bbar, chii = trans(r, phi, si, ci, chi)
    sc, cc = sincos(chii)
    (-bbar/2*sc, bbar/2*cc, PA, cs, g)
end

"""Image-plane coordinates (R_S) and log bolometric intensity on a polar disk grid."""
function image_data(inc, chi; cutout = 40, nr = 200, nphi = 361, M_dot = 0.0033, m = 10.0)
    r_isco = r_isco_f(chi)
    si, ci = sincos(deg2rad(inc))
    rimg = exp.(range(log(r_isco + 1e-7), log(cutout), length = nr))
    phimg = range(0, 2π, length = nphi)
    Xim = zeros(nr, nphi); Yim = zeros(nr, nphi); img = zeros(nr, nphi)
    Threads.@threads :static for j in 1:nr
        r = rimg[j]
        T4 = Tcol(r, chi, M_dot, m)^4
        @inbounds for (k, phi) in enumerate(phimg)
            X, Y, PA, cs, g = sky(r, phi, si, ci, chi)
            Xim[j,k] = X; Yim[j,k] = Y
            img[j,k] = log10(max(g^4*T4, floatmin(Float64)))
        end
    end
    img .-= maximum(img)
    return rimg, phimg, Xim, Yim, img
end


function plot_spectrum(kevs, sssp, ssqsp, ssusp)
    pdsp, pasp = pol(sssp, ssqsp, ssusp)
    figl, axls = plt.subplots(3, sharex = "col", sharey = "row", figsize = (4,10))
    axstv, axmuv, axpsiv = axls[1], axls[2], axls[3]
    axstv.plot(kevs, abs.(sssp), "-")
    axstv.set_yscale("log")
    axstv.set_xscale("log")
    axstv.set_ylim(maximum(sssp)*1e-4, maximum(sssp)*3)
    axpsiv.plot(kevs, pasp, "-")
    axpsiv.set_xscale("log")
    axmuv.plot(kevs, pdsp, "-")
    axmuv.set_xscale("log")
    axpsiv.set_xlabel(raw"$E,\,\mathrm{keV}$")
    for ax in (axmuv, axstv, axpsiv)
        ax.tick_params(axis = "both", which = "both", direction = "in", top = true, right = true)
    end
    axpsiv.set_ylabel(raw"$\chi,\,[^\circ]$")
    axstv.set_ylabel(raw"$\nu I_\nu$")
    axmuv.set_ylabel(raw"$p,\,\%$")
    figl.tight_layout(h_pad = 0., w_pad = 0.)
    figl.savefig("disk_spectrum_a0.8_i70_jl.pdf")
    open("disk_spectrum_a0.8_i70_jl.txt", "w") do io
        println(io, "# E[keV] E*I E*Q E*U (arbitrary units) PD[%] PA[deg]")
        writedlm(io, hcat(kevs, sssp, ssqsp, ssusp, pdsp, pasp))
    end
end


function plot_image(inc, chi, rimg, phimg, Xim, Yim, img)
    r_isco = r_isco_f(chi)
    si, ci = sincos(deg2rad(inc))
    xspan = 25/2; yspan = xspan; cutout = 40
    NP = 13; ls0 = 11
    phisecs = range(0, 2π, length = 13)
    philvls = phisecs .+ phisecs[2]/2
    fontsize = 16
    stickcolor = (0.0, 0.4, 0.70)

    fig, ax = plt.subplots(1, 1, figsize = (8,7))
    ax.set_facecolor("white")
    ax.fill(Xim[1,:], Yim[1,:], color = "k", zorder = 0)
    cmap = PyPlot.matplotlib.cm.plasma.with_extremes(under = "black")
    im = ax.pcolormesh(Xim, Yim, img, cmap = cmap, vmin = -3.1, vmax = 0, shading = "gouraud", zorder = 1)
    for rl in (10, 20, 30)
        xy = [sky(rl, phi, si, ci, chi) for phi in phimg]
        ax.plot(first.(xy), getindex.(xy, 2), "-", color = "black", linewidth = 0.5, zorder = 2)
        xl, yl = sky(rl, deg2rad(200), si, ci, chi)
        ax.text(xl, yl, string(rl ÷ 2), fontsize = fontsize, ha = "center", va = "center", zorder = 4)
    end
    for phl in phisecs[1:end-1]
        xy = [sky(rl, phl, si, ci, chi) for rl in rimg]
        ax.plot(first.(xy), getindex.(xy, 2), "-", color = "black", linewidth = 0.5, zorder = 2)
    end
    radsecs = [r_isco, 10, 20, 30, cutout]
    radsecs = sqrt.(radsecs[1:end-1] .* radsecs[2:end])
    segs = [begin
                xcc, ycc, pa, cs, g = sky(roi, phn, si, ci, chi)
                h = 2*xspan*ls0*poladeg(cs)/NP
                ((xcc - cos(pa)*h, ycc - sin(pa)*h), (xcc + cos(pa)*h, ycc + sin(pa)*h))
            end for roi in radsecs for phn in philvls[1:end-1]]
    ax.add_collection(PyPlot.matplotlib.collections.LineCollection(segs, colors = [stickcolor], linewidths = 1.5, zorder = 3))
    h5 = 2*xspan*ls0*0.05/NP
    ax.plot([xspan*0.7 - h5, xspan*0.7 + h5], [yspan*0.82, yspan*0.82], "-", linewidth = 1.5, color = stickcolor, zorder = 3)
    ax.text(xspan*0.7, yspan*0.88, raw"$5\%$", fontsize = fontsize, ha = "center", va = "bottom", color = stickcolor)
    cbar = fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
    cbar.ax.set_ylabel(raw"$\log_{10} I/I_{\rm max}$", fontsize = fontsize)
    ax.tick_params(axis = "both", which = "both", direction = "in", top = true, right = true, labelsize = fontsize)
    ax.set_xlim(-xspan, xspan)
    ax.set_ylim(-7, yspan)
    ax.set_aspect("equal")
    ax.set_xlabel(raw"$X\, [R_{\rm S}]$", fontsize = fontsize)
    ax.set_ylabel(raw"$Y\, [R_{\rm S}]$", fontsize = fontsize)
    ax.text(0.05, 0.95, "a = $chi, i = $(inc)°", transform = ax.transAxes, va = "top", ha = "left", fontsize = fontsize)
    fig.savefig("disk_image_a0.8_i70_jl.pdf", bbox_inches = "tight")
end


inc, chi = 70, 0.8
spectrum(inc, chi); image_data(inc, chi)              # first calls compile the code
t_spec = @elapsed (spec = spectrum(inc, chi))
t_img  = @elapsed (imd = image_data(inc, chi))
t_plot = @elapsed begin
    plot_spectrum(spec...)
    plot_image(inc, chi, imd...)
end
println("threads: ", Threads.nthreads())
println("spectrum (80x140 cells, 80 energies):  ", round(1e3*t_spec, digits = 1), " ms")
println("image grid (200x361):                  ", round(1e3*t_img, digits = 1), " ms")
println("plotting and saving:                   ", round(1e3*t_plot, digits = 1), " ms")

# check against the Python version: PD and PA at 2, 4, 8 keV
k248, I248, Q248, U248 = spectrum(inc, chi; kevs = [2.0, 4.0, 8.0])
pd248, pa248 = pol(I248, Q248, U248)
println("PD at 2, 4, 8 keV: ", round.(pd248, digits = 3), " %   (kerr_disk_artpol.py: [1.825, 1.551, 1.467])")
println("PA at 2, 4, 8 keV: ", round.(pa248, digits = 2), " deg (kerr_disk_artpol.py: [-4.94, -10.34, -14.39])")
