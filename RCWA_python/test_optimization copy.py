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

res = []
for Kz1 in np.linspace(-5, 5, 21):
    print(Kz1)
    res1 = []
    for Kz2 in np.linspace(-5, 5, 21):
        x0 = [0.2, Kz1, 0.6, Kz2]
        res1.append(foo(x0))
    res.append(res1)
plt.imshow(res)
plt.show()
#res = optimize.shgo(foo, bounds, n=60, iters=3, sampling_method='sobol', options={'disp': True, 'fmin': 0})
