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
alpha = rad2deg(asin(1/ng*sin(deg2rad(40))))/2
wl0 = 940e-9
k0 = 2*pi/wl0
Kx = -2*k0*ng*sin(deg2rad(alpha))*cos(deg2rad(alpha))
Kz = -2*k0*ng*sin(deg2rad(alpha))*sin(deg2rad(alpha))
px = 2*pi/Kx
pz = 2*pi/Kz
d = wl0*0.5/dn

ws = [0.3e-3]
DERl = []
DERr = []
DETl = []
DETr = []
nn = 2
w = ws[0]
R, Theta = np.mgrid[0:(4/w):20j, 0:(2*np.pi):20j]
fxs, fys = R*np.cos(Theta), R*np.sin(Theta)
dfS = (R + np.gradient(R, axis=0)/2)*np.gradient(R, axis=0)*np.gradient(Theta, axis=1)
print(w)
lays = [PVG2(d, Kx, Kz, -1, no, ne, 0)]
rcwa = RCWA(ng, ng, lays, nn, [1, 1j], [1j, 1])
beam = Gaussian(wl0, 10, w/np.deg2rad(10))
res = rcwa.solve_beam(0, 0, 1, 1j, beam, fxs, fys, dfS)

degx, degy = np.rad2deg(np.arctan2(fxs, k0)), np.rad2deg(np.arctan2(fys, k0))

plt.subplot(221)
plt.pcolormesh(degx, degy, res[:, :, -1, nn-1], shading='gouraud')
plt.colorbar()
plt.subplot(222)
plt.pcolormesh(degx, degy, beam.amplitude(fxs, fys), shading='gouraud')
plt.colorbar()
plt.subplot(223)
plt.pcolormesh(degx, degy, res[:, :, -2, nn-1], shading='gouraud')
plt.colorbar()
plt.subplot(224)
plt.pcolormesh(degx, degy, res[:, :, -1, nn-1]*beam.amplitude(fxs, fys) / np.sum(beam.amplitude(fxs, fys)*dfS) , shading='gouraud')
plt.colorbar()
plt.show()