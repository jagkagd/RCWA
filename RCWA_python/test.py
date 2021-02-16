from RCWA import RCWA
from Layer import PVG2
import numpy as np
from numpy import sqrt, cos, array, pi, rad2deg, deg2rad, arcsin as asin, sin
import matplotlib.pyplot as plt
from Beam import Gaussian

ng = 1.56
dn = 0.2
no = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. - dn/2
ne = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. + dn/2
alpha = rad2deg(asin(1/ng*sin(deg2rad(20))))/2
wl0 = 885e-9
k0 = 2*pi/wl0
Kx = -2*k0*ng*sin(deg2rad(alpha))*cos(deg2rad(alpha))
Kz = -2*k0*ng*sin(deg2rad(alpha))*sin(deg2rad(alpha))
px = 2*pi/Kx
pz = 2*pi/Kz
d = wl0*0.5/dn
print(Kx, Kz)

# ws = np.logspace(-3, 0, 10)*1e-3
thetas = np.linspace(-20, 20, 101)
DERl = []
DERr = []
DETl = []
DETr = []
nn = 2
for theta in thetas:
# for w in ws:
    # R, Theta = np.mgrid[0:(4/w):20j, 0:(2*np.pi):20j]
    # fxs, fys = R*np.cos(Theta), R*np.sin(Theta)
    # dfS = R*np.gradient(R, axis=0)*np.gradient(Theta, axis=1)
    # beam = Gaussian(wl0, 10, w/np.deg2rad(10))
    # print(w)
    lays = [PVG2(d, -Kx, -Kz, 1, no, ne, 0)]
    rcwa = RCWA(ng, ng, lays, nn, [1, 1j], [1j, 1])
    # derl, derr, detl, detr = rcwa.solve_beam(0, 0, 1j, 1, beam, fxs, fys, dfS)
    derl, derr, detl, detr = rcwa.solve(0, theta, wl0, 1, 1j)
    DERl.append(derl)
    DERr.append(derr)
    DETl.append(detl)
    DETr.append(detr)

DERl = array(DERl).T
DERr = array(DERr).T
DETl = array(DETl).T
DETr = array(DETr).T

xxs = thetas
# xxs = ws
plt.subplot(211)
plt.plot(xxs, DERl[nn], 'r')
plt.plot(xxs, DERr[nn], 'r--')
plt.plot(xxs, DERl[nn-1], 'g')
plt.plot(xxs, DERl[nn+1], 'b')
plt.plot(xxs, DERr[nn-1], 'g--')
plt.plot(xxs, DERr[nn+1], 'b--')
#plt.ylim(0, 1.2)
# plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
plt.subplot(212)
plt.plot(xxs,  DETl[nn], 'r')
plt.plot(xxs,  DETr[nn], 'r--')
plt.plot(xxs,  DETl[nn-1], 'g')
plt.plot(xxs,  DETl[nn+1], 'b')
plt.plot(xxs,  DETr[nn-1], 'g--')
plt.plot(xxs,  DETr[nn+1], 'b--')
# plt.ylim(0, 1)
# plt.subplot(221)
# plt.pcolormesh(fxs, fys, res[:, :, -1, nn-1], shading='gouraud', vmin=0.5, vmax=1)
# plt.colorbar()
# plt.subplot(222)
# plt.pcolormesh(fxs, fys, beam.amplitude(fxs, fys), shading='gouraud')
# plt.colorbar()
# plt.subplot(223)
# plt.pcolormesh(fxs, fys, res[:, :, -2, nn-1], shading='gouraud')
# plt.colorbar()
# plt.figure()
# plt.pcolormesh(fxs, fys, dfS, shading='gouraud')
# plt.colorbar()
plt.show()