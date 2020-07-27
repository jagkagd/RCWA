from RCWA import RCWA
from Layer import PVG
import numpy as np
from numpy import sqrt, cos, array, pi
import matplotlib.pyplot as plt

ng = 1.58
dn = 0.15
no = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. - dn/2
ne = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. + dn/2
# ng = sqrt((2*no**2+ne**2)/3)
alpha = 60/2
k0 = 2*pi/532e-9
pb = 2 * 2*pi/(2*k0*ng*cos(np.deg2rad(alpha)))

wls = np.linspace(400, 700, 301)*1e-9
# wl = 532e-9
thetas = np.linspace(0, 60, 61)
phis = np.linspace(0, 359, 360)
# alphas = [0]
DERl = []
DERr = []
DETl = []
DETr = []
DERl2 = []
DERr2 = []
DETl2 = []
DETr2 = []
k1iv = []
k3iv = []
DE = []
nn = 1
ds = np.linspace(0, 5, 101)*1e-6
# for d in ds:
lay1 = PVG(5e-6, pb, alpha, 1, no, ne)
lay2 = PVG(5e-6, pb, alpha, -1, no, ne) 
lay3 = PVG(0e-6, pb, alpha, -1, no, ne)
for wl in wls:
    # for theta in thetas:
    # DERl0 = []
    # DERr0 = []
    # DETl0 = []
    # DETr0 = []
    # for phi in phis:
    # for alpha in alphas/2:
    #lay = AnisHomo(5e-6, 1.5, 1, 2, 2.5, 3., 3.5)
    # lay = SinGrating(d)
    pb = 2 * 2*pi/(2*k0*ng*cos(np.deg2rad(alpha)))
    # Kx = 2*k0*ng*cos(np.deg2rad(alpha))*sin(np.deg2rad(alpha))
    # Kz = 2*k0*ng*cos(np.deg2rad(alpha))*cos(np.deg2rad(alpha))
    # lay = PVG2(5e-6, Kx, Kz, 1, no, ne)
    # lay = VHG(5e-6, ng, 0.5, alpha, pb/2)
    rcwa = RCWA(ng, ng, [lay1, lay2], nn, [1, 1j], [1j, 1])
    derl, derr, detl, detr = rcwa.solve(0, 0, wl, 1, 1j)
    DERl.append(derl)
    DERr.append(derr)
    DETl.append(detl)
    DETr.append(detr)
    k1iv.append(rcwa.k1iv)
    k3iv.append(rcwa.k3iv)
    # # DERl2.append(derl2)
    # # DERr2.append(derr2)
    # # DETl2.append(detl2)
    # # DETr2.append(detr2)
    # DERl.append(array(DERl0))
    # DERr.append(array(DERr0))
    # DETl.append(array(DETl0))
    # DETr.append(array(DETr0))

DERl = array(DERl).T
DERr = array(DERr).T
DETl = array(DETl).T
DETr = array(DETr).T
k1iv = array(k1iv).T
k3iv = array(k3iv).T

xxs = wls
plt.subplot(121)
plt.plot(xxs, DERl[nn], 'r')
plt.plot(xxs, DERr[nn], 'r--')
if nn > 0:
    plt.plot(xxs, DERl[nn-1], 'g')
    plt.plot(xxs, DERl[nn+1], 'b')
    plt.plot(xxs, DERr[nn-1], 'g--')
    plt.plot(xxs, DERr[nn+1], 'b--')
    plt.plot(xxs, DERl[nn+1]/(DERl[nn+1]+DETl[nn]), 'k')
plt.ylim(0, 1.5)
# plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
plt.subplot(122)
plt.plot(xxs,  DETl[nn], 'r')
plt.plot(xxs,  DETr[nn], 'r--')
if nn > 0:
    plt.plot(xxs,  DETl[nn-1], 'g')
    plt.plot(xxs,  DETl[nn+1], 'b')
    plt.plot(xxs,  DETr[nn-1], 'g--')
    plt.plot(xxs,  DETr[nn+1], 'b--')
plt.ylim(0, 1.5)
plt.show()

# plt.subplot(211)
# thetam, phim = np.meshgrid(thetas, phis)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T)
# # plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
# plt.subplot(212)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T/(DERl[:, :, nn+1].T+DETl[:, :, nn].T))
# plt.show()