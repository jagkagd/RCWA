from RCWA import RCWA
from Layer import PVG2
import numpy as np
from numpy import sqrt, cos, array, pi, rad2deg, deg2rad, arcsin as asin, sin
import matplotlib.pyplot as plt
from Stack import EChiral, solve_stack, KPick, KRotator, Field
from pprint import pprint

ng = 1.56
dn = 0.2
no = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. - dn/2
ne = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. + dn/2
alpha = rad2deg(asin(1/ng*sin(deg2rad(40))))/2
wl0 = 800e-9
k0 = 2*pi/wl0
# Kx = -2*k0*ng*sin(deg2rad(alpha))*cos(deg2rad(alpha))
# Kz = -2*k0*ng*sin(deg2rad(alpha))*sin(deg2rad(alpha))
d = wl0*0.5/dn
pprint([(i, rad2deg(asin(1/ng*sin(deg2rad(i))))) for i in [0, 30, 50, 60, 70]])

def Kcalc(alpha, wl0, ng):
    alpha = rad2deg(asin(1/ng*sin(deg2rad(alpha))))/2
    k0 = 2*pi/wl0
    Kx = -2*k0*ng*sin(deg2rad(alpha))*cos(deg2rad(alpha))
    Kz = -2*k0*ng*sin(deg2rad(alpha))*sin(deg2rad(alpha))
    if Kz > 0:
        return Kx, Kz
    else:
        return -Kx, -Kz

nn = 2
stacklist = {
    -20: [
        PVG2(d, *Kcalc(40, wl0, ng), 1, no, ne, 0), 
        KPick(0),
        EChiral(),
        PVG2(d, *Kcalc(-40, wl0, ng), 1, no, ne, 0), 
        KPick(1),
        PVG2(d, *Kcalc(20, wl0, ng), 1, no, ne, 0),
        KPick(0),
    ],
    -10: [
        PVG2(d, *Kcalc(40, wl0, ng), 1, no, ne, 0), 
        KPick(0),
        EChiral(),
        PVG2(d, *Kcalc(-40, wl0, ng), 1, no, ne, 0), 
        KPick(1),
        PVG2(d, *Kcalc(20, wl0, ng), 1, no, ne, 0),
        KPick(0),
    ],
    0: [
        PVG2(d, *Kcalc(40, wl0, ng), 1, no, ne, 0), 
        KPick(0),
        PVG2(d, *Kcalc(-40, wl0, ng), 1, no, ne, 0), 
        KPick(0),
        PVG2(d, *Kcalc(20, wl0, ng), 1, no, ne, 0),
        KPick(0),
    ],
    30: [
        PVG2(d, *Kcalc(40, wl0, ng), 1, no, ne, 0), 
        KPick(0),
        PVG2(d, *Kcalc(-40, wl0, ng), 1, no, ne, 0), 
        KPick(0),
        EChiral(),
        PVG2(d, *Kcalc(20, wl0, ng), 1, no, ne, 0),
        KPick(1),
    ],
    50: [
        EChiral(),
        PVG2(d, *Kcalc(40, wl0, ng), 1, no, ne, 0), 
        KPick(1),
        PVG2(d, *Kcalc(-40, wl0, ng), 1, no, ne, 0), 
        KPick(0),
        PVG2(d, *Kcalc(20, wl0, ng), 1, no, ne, 0),
        KPick(0),
    ],
    70: [
        EChiral(),
        PVG2(d, *Kcalc(40, wl0, ng), 1, no, ne, 0), 
        KPick(1),
        PVG2(d, *Kcalc(-40, wl0, ng), 1, no, ne, 0), 
        KPick(0),
        EChiral(),
        PVG2(d, *Kcalc(20, wl0, ng), 1, no, ne, 0),
        KPick(1),
    ],
}

def defeff(deflx, defly):
    defl = {-20: -10, -10: -6, 0: 0, 30: -6, 50: -5, 70: 1}
    incx0, incy0 = defl[deflx], defl[defly] 
    incx, incy = np.deg2rad(incx0), np.deg2rad(incy0)
    k0v = [
        np.tan(incx)/np.sqrt(1+np.tan(incx)**2),
        np.tan(incy)/np.sqrt(1+np.tan(incy)**2), 
        np.sqrt(1-np.tan(incx)**2/(1+np.tan(incx)**2)-np.tan(incy)**2/(1+np.tan(incy)**2))
    ]
    # print(k0v)
    stacks = stacklist[deflx] + [KRotator(90)] + stacklist[defly]

    field = solve_stack(nn, ng, wl0, stacks, [1, 1j], [1j, 1], k0v, 1/sqrt(2), -1j/sqrt(2))[0]
    # print(field.ang())
    # print(np.abs(field.amp)**2 * np.real(field.k0uv[-1]/k0v[-1]))
    # return (field.ang(), np.abs(field.amp)**2 * np.real(field.k0uv[-1]/k0v[-1]))
    return np.abs(field.amp)**2 * np.real(field.k0uv[-1]/k0v[-1])

# print(defeff(-20, 0))
pprint([['{:.2f}'.format(defeff(i, j)) for j in [0, 30, 50, 70]] for i in [0, 30, 50, 70]])

# xxs = thetas
# plt.subplot(211)
# plt.plot(xxs, DERl[nn], 'r')
# plt.plot(xxs, DERr[nn], 'r--')
# plt.plot(xxs, DERl[nn-1], 'g')
# plt.plot(xxs, DERl[nn+1], 'b')
# plt.plot(xxs, DERr[nn-1], 'g--')
# plt.plot(xxs, DERr[nn+1], 'b--')
# #plt.ylim(0, 1.2)
# # plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
# plt.subplot(212)
# plt.plot(xxs,  DETl[nn], 'r')
# plt.plot(xxs,  DETr[nn], 'r--')
# plt.plot(xxs,  DETl[nn-1], 'g')
# plt.plot(xxs,  DETl[nn+1], 'b')
# plt.plot(xxs,  DETr[nn-1], 'g--')
# plt.plot(xxs,  DETr[nn+1], 'b--')
# # plt.ylim(0, 1)
# # plt.subplot(221)
# # plt.pcolormesh(fxs, fys, res[:, :, -1, nn-1], shading='gouraud', vmin=0.5, vmax=1)
# # plt.colorbar()
# # plt.subplot(222)
# # plt.pcolormesh(fxs, fys, beam.amplitude(fxs, fys), shading='gouraud')
# # plt.colorbar()
# # plt.subplot(223)
# # plt.pcolormesh(fxs, fys, res[:, :, -2, nn-1], shading='gouraud')
# # plt.colorbar()
# # plt.figure()
# # plt.pcolormesh(fxs, fys, dfS, shading='gouraud')
# # plt.colorbar()
# plt.show()