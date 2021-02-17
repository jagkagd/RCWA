from RCWA import RCWA
from Layer import PVG2
import numpy as np
from numpy import sqrt, cos, array, pi, rad2deg, deg2rad, arcsin as asin, sin
import matplotlib.pyplot as plt
from matplotlib import ticker
from Stack import EChiral, solve_stack, KPick, KRotator, Field, plot_fields
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
        PVG2(d, *Kcalc(-20, wl0, ng), -1, no, ne, 0),
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
        EChiral(),
        PVG2(d, *Kcalc(-20, wl0, ng), -1, no, ne, 0),
        KPick(0),
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

    fields = solve_stack(nn, ng, wl0, stacks, [1, 1j], [1j, 1], k0v, 1/sqrt(2), -1j/sqrt(2), mode='all')
    # field = solve_stack(nn, ng, wl0, stacks, [1, 1j], [1j, 1], k0v, 1/sqrt(2), -1j/sqrt(2))[0]
    # return np.abs(field.amp)**2 * np.real(field.k0uv[-1]/k0v[-1])
    return fields

# pprint([['{:.2f}'.format(defeff(i, j)) for j in [0, 30, 50, 70]] for i in [0, 30, 50, 70]])

fieldss = defeff(30, 30)
k0v = [0, 0, 1]
# fieldss = solve_stack(nn, ng, wl0, [
#     # EChiral(),
#     KRotator(90),
#     PVG2(d, *Kcalc(20, wl0, ng), 1, no, ne, 0),
#     KPick(1),
# ], [1, 1j], [1j, 1], k0v, 1/sqrt(2), -1j/sqrt(2), mode='all')
# print(fieldss)
# plt.figure(figsize=(8, 4.3))
titles = ['Inc.', 'x + 40 deg', 'x - 40 deg', 'x + 20 deg', 'y + 40 deg', 'y - 40 deg', 'y + 20 deg']
for (i, fields) in enumerate(fieldss):
    plt.subplot(331+i)
    ax = plt.gca()
    plot_fields(ax, fields, [0, 0, 1], ng)
    plt.xlim(-90, 90)
    plt.ylim(-90, 90)
    ax.set_aspect(1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.grid('on')
    ax.tick_params(which='both', direction='in')
    # ax.set_title(titles[i])
plt.subplots_adjust(top=0.95,
bottom=0.05,
left=0.05,
right=0.98,
hspace=0.2,
wspace=0.25)
plt.show()