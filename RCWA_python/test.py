from RCWA import RCWA
from Layer import PVG, PVG_dp, PVG2, genGradient
import numpy as np
from numpy import sqrt, cos, array, pi, rad2deg, deg2rad, arcsin as asin, sin
import matplotlib.pyplot as plt

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
polR = []
polT = []
nn = 1
lays = genGradient(PVG2, 1, wl0*0.5/dn , px, pz, pz, -1, no, ne)
for theta in thetas:
    thetai = rad2deg(asin(sin(deg2rad(theta))/ng))
    rcwa = RCWA(ng, ng, lays, nn, [1, 1j], [1j, 1])
    derl, derr, detl, detr = rcwa.solve(0, thetai, wl0, 1, 1j)
    DERl.append(derl)
    DERr.append(derr)
    DETl.append(detl)
    DETr.append(detr)
    polR.append(rcwa.polR)
    polT.append(rcwa.polT)

DERl = array(DERl).T
DERr = array(DERr).T
DETl = array(DETl).T
DETr = array(DETr).T
polR = np.swapaxes(polR, 0, 1)
polT = np.swapaxes(polT, 0, 1)

xxs = thetas
plt.subplot(221)
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
plt.subplot(222)
plt.plot(xxs,  DETl[nn], 'r')
plt.plot(xxs,  DETr[nn], 'r--')
if nn > 0:
    plt.plot(xxs,  DETl[nn-1], 'g')
    plt.plot(xxs,  DETl[nn+1], 'b')
    plt.plot(xxs,  DETr[nn-1], 'g--')
    plt.plot(xxs,  DETr[nn+1], 'b--')
plt.ylim(0, 1.2)

plt.subplot(223)
Eu = polT[nn, :, 0]
Ev = polT[nn, :, 1]
plt.plot(xxs, np.abs(Ev/Eu), 'C0')
plt.twinx()
plt.plot(xxs, np.angle(Ev/Eu)/np.pi, 'C1')
plt.show()

# plt.subplot(211)
# thetam, phim = np.meshgrid(thetas, phis)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T)
# # plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
# plt.subplot(212)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T/(DERl[:, :, nn+1].T+DETl[:, :, nn].T))
# plt.show()