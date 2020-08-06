from RCWA import RCWA
from Layer import PVG, PVG_dp, PVG3, genGradient
import numpy as np
from numpy import sqrt, cos, array, pi, rad2deg, deg2rad, arcsin as asin, sin
import matplotlib.pyplot as plt
from scipy.optimize import 

def foo(d):
    ng = 1.56
    dn = 0.2
    no = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. - dn/2
    ne = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. + dn/2
    alpha = rad2deg(asin(1/ng*sin(deg2rad(20))))/2;
    wl0 = 940e-9
    k0 = 2*pi/wl0
    Kx = -2*k0*ng*sin(deg2rad(alpha))*cos(deg2rad(alpha))
    Kz = -2*k0*ng*sin(deg2rad(alpha))*sin(deg2rad(alpha))
    px = 2*pi/Kx
    pz = 2*pi/Kz

    thetas = np.linspace(-40, 40, 41)
    DERl = []
    DERr = []
    DETl = []
    DETr = []
    nn = 1
    lays = genGradient(PVG2, 10, 15e-6 , px, pz*2., pz*0.5, -1, no, ne)
    for theta in thetas:
        thetai = rad2deg(asin(sin(deg2rad(theta))/ng))
        rcwa = RCWA(ng, ng, lays, nn, [1, 1j], [1j, 1])
        derl, derr, detl, detr = rcwa.solve(0, thetai, wl0, 1, 1j)
        DETr.append(detr)

    DETr = array(DETr).T
    loss = np.sum(np.abs(DETr[nn-1]-1))
    return loss

xxs = thetas
plt.subplot(211)
plt.plot(xxs, DERl[nn], 'r')
plt.plot(xxs, DERr[nn], 'r--')
if nn > 0:
    plt.plot(xxs, DERl[nn-1], 'g')
    plt.plot(xxs, DERl[nn+1], 'b')
    plt.plot(xxs, DERr[nn-1], 'g--')
    plt.plot(xxs, DERr[nn+1], 'b--')
    plt.plot(xxs, DERl[nn+1]/(DERl[nn+1]+DETl[nn]), 'k')
#plt.ylim(0, 1.2)
# plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
plt.subplot(212)
plt.plot(xxs,  DETl[nn], 'r')
plt.plot(xxs,  DETr[nn], 'r--')
if nn > 0:
    plt.plot(xxs,  DETl[nn-1], 'g')
    plt.plot(xxs,  DETl[nn+1], 'b')
    plt.plot(xxs,  DETr[nn-1], 'g--')
    plt.plot(xxs,  DETr[nn+1], 'b--')
plt.ylim(0, 1.2)
plt.show()

# plt.subplot(211)
# thetam, phim = np.meshgrid(thetas, phis)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T)
# # plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
# plt.subplot(212)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T/(DERl[:, :, nn+1].T+DETl[:, :, nn].T))
# plt.show()