from RCWA import RCWA
from Layer import PVG, PVG_dp, PVG2, genGradient
import numpy as np
from numpy import sqrt, cos, array, pi, rad2deg, deg2rad, arcsin as asin, sin
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, minimize, SR1
from scipy import optimize

ng = 1.56
dn = 0.2
no = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. - dn/2
ne = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. + dn/2
alpha = rad2deg(asin(1/ng*sin(deg2rad(40))))/2
wl0 = 940e-9
k0 = 2*pi/wl0
Kx = -2*k0*ng*sin(deg2rad(alpha))*cos(deg2rad(alpha))
Kz = -2*k0*ng*sin(deg2rad(alpha))*sin(deg2rad(alpha))
px = 2*pi/Kx
pz = 2*pi/Kz
thetas = np.linspace(-20, 20, 41)
# theta0 = 0
# wls = np.linspace(800, 1000, 201)*1e-9
nn = 1

def foo(ps):
    d1, Kz1, d2, Kz2 = ps
    DETr = []
    lays = [PVG2(wl0*d1/dn, Kx, Kz1*Kz, -1, no, ne), PVG2(wl0*d2/dn, Kx, Kz2*Kz, -1, no, ne)]
    for theta in thetas:
        thetai = rad2deg(asin(sin(deg2rad(theta))/ng))
        rcwa = RCWA(ng, ng, lays, nn, [1, 1j], [1j, 1])
        _, _, _, detr = rcwa.solve(0, thetai, wl0, 1, 1j)
        DETr.append(detr)

    DETr = array(DETr).T
    eff = DETr[nn-1]
    loss = np.sum(np.abs(eff-1))
    return loss

# x0 = [0.533632448, 0.053913255, 0.541489986, 2.120916468]
x0 = [0.5, 1, 0.25, 1]
bound = Bounds([0, -5, 0, -5], [1, 5, 1, 5])
# bounds = np.array([[0, -10, 0, -10], [1, 10, 1, 10]]).T
#results = {}
# res = minimize(foo, x0, bounds=bound, jac="2-point", options={'disp': True})
#res = optimize.shgo(foo, bounds, n=60, iters=3, sampling_method='sobol', options={'disp': True, 'fmin': 0})
class Data:
    pass

res = Data()
res.x = x0
print(res.x)

DERl = []
DERr = []
DETl = []
DETr = []
DETr1 = []
DETr2 = []
# DETr3 = []
d1, Kz1, d2, Kz2 = res.x
lays = [PVG2(wl0*d1/dn, Kx, Kz1*Kz, -1, no, ne, 6*pi/8), PVG2(wl0*d2/dn, Kx, Kz2*Kz, -1, no, ne, 0)]
for theta in thetas:
    thetai = rad2deg(asin(sin(deg2rad(theta))/ng))
    rcwa = RCWA(ng, ng, [lays[0]], nn, [1, 1j], [1j, 1])
    # rcwa1 = RCWA(ng, ng, [lays[0]], nn, [1, 1j], [1j, 1])
    # rcwa2 = RCWA(ng, ng, [lays[1]], nn, [1, 1j], [1j, 1])
    # rcwa3 = RCWA(ng, ng, [lays[2]], nn, [1, 1j], [1j, 1])
    derl, derr, detl, detr = rcwa.solve(0, thetai, wl0, 1, 1j)
    # _, _, _, detr1 = rcwa1.solve(0, thetai, wl0, 1, 1j)
    # _, _, _, detr2 = rcwa2.solve(0, thetai, wl0, 1, 1j)
    # _, _, _, detr3 = rcwa3.solve(0, thetai, wl0, 1, 1j)
    DERl.append(derl)
    DERr.append(derr)
    DETl.append(detl)
    DETr.append(detr)
    # DETr1.append(detr1)
    # DETr2.append(detr2)
    # DETr3.append(detr3)

DERl = array(DERl).T
DERr = array(DERr).T
DETl = array(DETl).T
DETr = array(DETr).T
# DETr1 = array(DETr1).T
# DETr2 = array(DETr2).T
# DETr3 = array(DETr3).T

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
    # plt.plot(xxs,  DETr1[nn-1], 'k')
    # plt.plot(xxs,  DETr2[nn-1], 'k--')
    # plt.plot(xxs,  DETr3[nn-1], 'k.')
plt.ylim(0, 1)
plt.show()

# plt.subplot(211)
# thetam, phim = np.meshgrid(thetas, phis)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T)
# # plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
# plt.subplot(212)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T/(DERl[:, :, nn+1].T+DETl[:, :, nn].T))
# plt.show()