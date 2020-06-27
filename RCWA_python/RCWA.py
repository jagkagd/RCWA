import scipy as S
from scipy.spatial.transform import Rotation as R
from scipy.linalg import eig, inv, solve as linsolve
from scipy.integrate import quad
from numpy import array, split, zeros, sin, cos, tan, pi, diag, sqrt, deg2rad, exp, real
from numpy import arctan2 as atan2, arccos as acos, arcsin as asin
import numpy as np
import pysnooper
import matplotlib.pyplot as plt
import matplotlib

def dot(x, y):
    return sum([xi*yi for (xi, yi) in zip(x, y)])

def norml(v):
    vv = array(v)
    return vv / sqrt(dot(vv, np.conj(vv)))

def cquad(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def eulerMatrix(seq, angs, i, j):
    if seq == 'zyz':
        eulerMatrixElem = lambda i, j: \
            (lambda a, b, c: [
                [cos(a)*cos(b)*cos(c)-sin(a)*sin(c), -cos(c)*sin(a)-cos(a)*cos(b)*sin(c), cos(a)*sin(b)],
                [cos(b)*cos(c)*sin(a)+cos(a)*sin(c),  cos(a)*cos(c)-cos(b)*sin(a)*sin(c), sin(a)*sin(b)],
                [-cos(c)*sin(b), sin(b)*sin(c), cos(b)]
            ][i][j])
    return eulerMatrixElem(i, j)(*angs)

def k2uv_xy(k):
    kx, ky, kz = k
    kk = sqrt(kx**2 + ky**2 + kz**2)
    ktan = sqrt(kx**2 + ky**2)
    if abs(ktan) != 0.:
        return array([
            [kz/kk * kx/ktan, -ky/ktan],
            [kz/kk * ky/ktan,  kx/ktan]
        ], dtype=np.complex)
    else:
        return array([
            [kz/kk, 0],
            [0, 1]
        ], dtype=np.complex)

class Layer():
    def __init__(self):
        self.Kx = 0
        self.Kz = 0
        self.Kv = array([0, 0, 0])

    def eps(self, p, q, n):
        raise NotImplementedError()

    def epsxx1(self, n):
        raise NotImplementedError()

    def epsm(self, p, q, mn):
        res = np.zeros((mn, mn), dtype=np.complex)
        for i in range(mn):
            for j in range(mn):
                res[i][j] = self.eps(p, q, i-j)
        return res.round(12)
    
    # @pysnooper.snoop(max_variable_length=None)
    def getA(self, m, k0, ki):
        '''
        K = [Kx, 0, Kz]
        k2m = [kix, kiy, 0] - m*K = [kix-m*Kx, kiy, -m*Kz]
        E = \sum (Sxm(z) + Sym(z) + Szm(z))*exp(-1j*k2m*r)
        H = sqrt(epsilon0/mu0) \sum (Uxm(z) + Uym(z) + Uzm(z))*exp(-1j*k2m*r)
        V = [Sx, Sy, Ux, Uy]
        dV/dt = 1j * k0 * A * V = W @ diag(q) @ inv(W) V
        '''
        kix, kiy, _ = ki
        ms = np.arange(-m, m+1)
        mn = 2*m+1
        Id = np.eye(mn)

        kxim = diag(kix-ms*self.Kx) / k0
        k1ym = kiy * Id / k0
        kzim = diag(-ms*self.Kz) / k0

        exxm = self.epsm(0, 0, mn)
        exym = self.epsm(0, 1, mn)
        exzm = self.epsm(0, 2, mn)
        eyym = self.epsm(1, 1, mn)
        eyzm = self.epsm(1, 2, mn)
        ezzm = self.epsm(2, 2, mn)
        ezzm_1 = inv(ezzm).round(12)
        #exxm_1 = self.epsxx1m(mn)
        # exxm_1_1 = inv(exxm_1).round(12)
        # print('exxm ', exxm, 'exxm_1 ', exxm_1, 'exxm_1_1 ', exxm_1_1)

        A11m = kzim + kxim @ ezzm_1 @ exzm
        A12m =  kxim @ ezzm_1 @ eyzm
        A13m = -kxim @ ezzm_1 @ k1ym
        A14m = -Id + kxim @ ezzm_1 @ kxim

        A21m = k1ym @ ezzm_1 @ exzm
        A22m = kzim + k1ym @ ezzm_1 @ eyzm
        A23m = Id - k1ym @ ezzm_1 @ k1ym
        A24m = k1ym @ ezzm_1 @ kxim

        A31m =  kxim @ k1ym + exym - eyzm @ ezzm_1 @ exzm
        A32m = -kxim @ kxim + eyym - eyzm @ ezzm_1 @ eyzm
        A33m = kzim + eyzm @ ezzm_1 @ k1ym
        A34m = -eyzm @ ezzm_1 @ kxim

        A41m =  k1ym @ k1ym - exxm + exzm @ ezzm_1 @ exzm
        A42m = -k1ym @ kxim - exym + exzm @ ezzm_1 @ eyzm
        A43m = -exzm @ ezzm_1 @ k1ym
        A44m = kzim + exzm @ ezzm_1 @ kxim

        A = np.asarray(np.bmat([
            [A11m, A12m, A13m, A14m],
            [A21m, A22m, A23m, A24m],
            [A31m, A32m, A33m, A34m],
            [A41m, A42m, A43m, A44m]
        ])).round(10)
        return A

class Homo(Layer):
    def __init__(self, d, n):
        super().__init__()
        self.d = d
        self.n = n

    def eps(self, p, q, n):
        if n == 0 and p == q:
            return self.n**2
        else:
            return 0

class VHG(Layer):
    def __init__(self, d, n, dn, delta, period):
        super().__init__()
        self.d = d
        self.n = n
        self.dn = dn
        self.period = period
        self.delta = deg2rad(delta)
        self.K = 2*pi/(self.period)
        self.Kx = self.K*sin(self.delta)
        self.Kz = self.K*cos(self.delta)
        self.K = array([self.Kx, 0, self.Kz])

    def eps(self, p, q, n):
        if p == q:
            if n == 0:
                return self.n**2
            elif n == 1 or n == -1:
                return 1/2*self.dn
            else:
                return 0
        else:
            return 0

class AnisHomo(Layer):
    def __init__(self, d, exx, exy, exz, eyy, eyz, ezz):
        super().__init__()
        self.d = d
        self.epsv = [[exx, exy, exz], [0, eyy, eyz], [0, 0, ezz]]
    
    def eps(self, p, q, n):
        if n == 0:
            return self.epsv[p][q]
        else:
            return 0

class SinGrating(Layer):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.pitch = 1.02350e-6
        self.delta = deg2rad(84.071) # angle (rad) between z axis
        self.K = 2*pi/(self.pitch)
        self.Kx = self.K*sin(self.delta)
        self.Kz = self.K*cos(self.delta)
        self.no = 2.3370
        self.ne = 2.2425
        self.eo = no**2
        self.ee = ne**2
        self.Kv = array([self.Kx, 0, self.Kz])
    
    def eps(self, p, q, n):
        eo, ee, delta = self.eo, self.ee, self.delta
        r13 = 10e-12
        r33 = 32.2e-12
        r22 = 6.7e-12
        r51 = 32e-12
        E0 = 0.494e6
        r11p_ = r33*self.ne**4*sin(delta)*E0
        r22p = r13*self.no**4*sin(delta)*E0
        r12p = r51*self.no**2*self.ne**2*cos(delta)*E0
        r23p = r22*self.no**4*cos(delta)*E0

        r11p = 4e-4

        exx = lambda Kr: ee - r11p*sin(Kr)
        exy = lambda Kr: 0
        exz = lambda Kr: r12p*sin(Kr)
        eyy = lambda Kr: eo-r22p*sin(Kr)
        eyz = lambda Kr: r23p*sin(Kr)
        ezz = eyy
        es = [[exx, exy, exz], [0, eyy, eyz], [0, 0, ezz]]

        ejkn = lambda Kr: 1/(2*pi) * exp(-1.j*n*Kr)

        return cquad(lambda Kr: es[p][q](Kr)*ejkn(Kr), -pi, pi)[0]

class PVG(Layer):
    def __init__(self, d, pitch, delta, chi, no, ne):
        super().__init__()
        self.d = d
        self.pitch = pitch # LC rotate 2pi (m)
        self.period = pitch / 2 # period of grating, LC rotate pi (m)
        self.delta = deg2rad(delta) # angle (rad) between z axis
        self.chi = chi # chirality
        self.K = 2*pi/(self.period)
        self.Kx = self.K*sin(self.delta)
        self.Kz = self.K*cos(self.delta)
        # self.px = 2*pi/self.Kx
        # self.pz = 2*pi/self.Kz
        self.no = no
        self.ne = ne
        self.eo = no**2
        self.ee = ne**2
        self.Kv = array([self.Kx, 0, self.Kz])

    def eps(self, p, q, n):
        eo, ee, delta, chi = self.eo, self.ee, self.delta, self.chi

        exx = lambda Kr: eo + ((ee - eo)*cos(delta)**2*cos((chi*Kr)/2.)**2*(1 - cos((chi*Kr)/2.)**2*sin(delta)**2))/(cos(delta)**2*cos((chi*Kr)/2.)**2 + sin((chi*Kr)/2.)**2)
        exy = lambda Kr: ((ee - eo)*cos(delta)*cos((chi*Kr)/2.)*(1 - cos((chi*Kr)/2.)**2*sin(delta)**2)*sin((chi*Kr)/2.))/(cos(delta)**2*cos((chi*Kr)/2.)**2 + sin((chi*Kr)/2.)**2)
        exz = lambda Kr: -((ee - eo)*cos(delta)*cos((chi*Kr)/2.)**2*sin(delta)*sqrt(1 - cos((chi*Kr)/2.)**2*sin(delta)**2))/sqrt(cos(delta)**2*cos((chi*Kr)/2.)**2 + sin((chi*Kr)/2.)**2)
        eyy = lambda Kr: eo + ((ee - eo)*(1 - cos((chi*Kr)/2.)**2*sin(delta)**2)*sin((chi*Kr)/2.)**2)/(cos(delta)**2*cos((chi*Kr)/2.)**2 + sin((chi*Kr)/2.)**2)
        eyz = lambda Kr: -((ee - eo)*cos((chi*Kr)/2.)*sin(delta)*sqrt(1 - cos((chi*Kr)/2.)**2*sin(delta)**2)*sin((chi*Kr)/2.))/sqrt(cos(delta)**2*cos((chi*Kr)/2.)**2 + sin((chi*Kr)/2.)**2)
        ezz = lambda Kr: eo + (ee - eo)*cos((chi*Kr)/2.)**2*sin(delta)**2
        es = [[exx, exy, exz], [exy, eyy, eyz], [exz, eyz, ezz]]

        ejkn = lambda Kr: 1/(2*pi) * exp(-1.j*n*Kr)

        return cquad(lambda Kr: es[p][q](Kr)*ejkn(Kr), -pi, pi)[0]

class PVG2(Layer):
    def __init__(self, d, Kx, Kz, chi, no, ne):
        super().__init__()
        self.d = d
        self.chi = chi # chirality
        self.no = no
        self.ne = ne
        self.eo = no**2
        self.ee = ne**2
        self.Kx = Kx
        self.Kz = Kz
        self.Kv = array([self.Kx, 0, self.Kz])

    def eps(self, p, q, n):
        eo, ee, chi = self.eo, self.ee, self.chi
        exx = lambda n: {1: (ee-eo)/4, 0: (ee+eo)/2}.get(abs(n), 0)
        exy = lambda n: {1: -chi*1j*(ee-eo)/4, -1: chi*1j*(ee-eo)/4}.get(n, 0)
        exz = lambda n: 0
        eyy = lambda n: {1: (eo-ee)/4, 0: (ee+eo)/2}.get(abs(n), 0)
        eyz = lambda n: 0
        ezz = lambda n: {0: eo}.get(n, 0)
        eps = [[exx, exy, exz], [exy, eyy, eyz], [exz, eyz, ezz]]
        return eps[p][q](n)

class RCWA():
    '''
    epsilon = \sum epsilon \exp(1j*m*K*r)
    '''
    def __init__(self, n1, n3, layers, m, base1=[1, 1j], base2=[1, -1j]):
        self.n1 = n1
        self.n3 = n3
        self.base1 = norml(base1)
        self.base2 = norml(base2)
        self.m = m
        self.ms = np.arange(-m, m+1)
        self.M = len(self.ms)
        self.layers = layers
        if isinstance(layers, list):
            pxs = [layer.px for layer in self.layers]
            if not np.all(pxs == pxs[0]):
                raise("All layers should have the same x period.")
    
    def getLayerQW(self, k0, kiv):
        '''
        A = W @ diag(q) @ inv(W)
        '''
        def normW(W):
            W2 = np.zeros_like(W, dtype=np.complex)
            for i, w in enumerate(W.T): 
                W2[:, i] = (w.T/np.sum(w*w.conj())).copy()
            return W2

        if isinstance(self.layers, list):
            Ams = [layer.getA(self.m, k0, kiv) for layer in self.layers]
            qWs = [eig(Am) for Am in Ams]
            return [[1.j * k0 * q.round(12), normW(W).round(12)] for (q, W) in qWs]
        else:
            Am = self.layers.getA(self.m, k0, kiv)
            q, W = eig(Am)
            return (1.j * k0 * q.round(12), W)

    # @pysnooper.snoop(max_variable_length=None)
    def solve(self, phi, theta, wl, Eu, Ev):
        '''
        phi: projected angle between x axis in x-y plane 
        theta: angle between z axis
        light to z+
        '''
        self.phi = deg2rad(phi)
        self.theta = deg2rad(theta)
        self.k1uv = array([
            eulerMatrix('zyz', [self.phi, self.theta, 0], 0, 2),
            eulerMatrix('zyz', [self.phi, self.theta, 0], 1, 2),
            eulerMatrix('zyz', [self.phi, self.theta, 0], 2, 2)
        ])

        self.DERl = np.zeros(self.M)
        self.DERr = np.zeros_like(self.DERl)
        self.DETl = np.zeros_like(self.DERl)
        self.DETr = np.zeros_like(self.DERl)
        n1, n3 = self.n1, self.n3
        Ei = norml(array([Eu, Ev]))
        k0k = 2*pi/wl
        k0v = self.k1uv * k0k
        kiv = k0v * n1
        k1k = k0k * n1
        k3k = k0k * n3
        Q, W = self.getLayerQW(k0k, kiv)

        k1iv = array([kiv-i*self.layers.Kv for i in self.ms], dtype=np.complex)
        II = array([[0, -1], [1, 0]])
        for i in range(len(k1iv)):
            kx, ky, _ = k1iv[i]
            if kx**2 + ky**2 <= k1k**2:
                k1iv[i][-1] = -sqrt(k1k**2 - kx**2 - ky**2)
            else:
                k1iv[i][-1] = 1j*sqrt(-k1k**2 + kx**2 + ky**2)
        UVk1ixym = np.vstack([k2uv_xy(k1v) for k1v in k1iv])
        UVk1ixm = UVk1ixym[::2]
        UVk1iym = UVk1ixym[1::2]
        SRlxm = diag(UVk1ixm @ self.base1)
        SRlym = diag(UVk1iym @ self.base1)
        SRrxm = diag(UVk1ixm @ self.base2)
        SRrym = diag(UVk1iym @ self.base2)
        URlxm = n1 * diag(UVk1ixm @ II @ self.base1)
        URlym = n1 * diag(UVk1iym @ II @ self.base1)
        URrxm = n1 * diag(UVk1ixm @ II @ self.base2)
        URrym = n1 * diag(UVk1iym @ II @ self.base2)

        k3iv = array([kiv-i*self.layers.Kv for i in self.ms], dtype=np.complex)
        for i in range(len(k3iv)):
            kx, ky, _ = k3iv[i]
            if kx**2 + ky**2 <= k3k**2:
                k3iv[i][-1] = sqrt(k3k**2 - kx**2 - ky**2)
            else:
                k3iv[i][-1] = -1j*sqrt(-k3k**2 + kx**2 + ky**2)
        UVk3ixym = np.vstack([k2uv_xy(k3v) for k3v in k3iv])
        UVk3ixm = UVk3ixym[::2]
        UVk3iym = UVk3ixym[1::2]
        STlxm = diag(UVk3ixm @ self.base1)
        STlym = diag(UVk3iym @ self.base1)
        STrxm = diag(UVk3ixm @ self.base2)
        STrym = diag(UVk3iym @ self.base2)
        UTlxm = n3 * diag(UVk3ixm @ II @ self.base1)
        UTlym = n3 * diag(UVk3iym @ II @ self.base1)
        UTrxm = n3 * diag(UVk3ixm @ II @ self.base2)
        UTrym = n3 * diag(UVk3iym @ II @ self.base2)

        d = self.layers.d
        kziv = array([-i*self.layers.Kv[-1] for i in self.ms])

        ZERO = np.zeros((self.M, self.M))
        pPredict = np.imag(Q) <= 0
        mPredict = np.imag(Q) > 0
        Qp, Qm = Q[pPredict], Q[mPredict]
        Wp, Wm = W[:, pPredict], W[:, mPredict]
        eQpzm = lambda z: diag(exp(Qp*z))
        eQmzm = lambda z: diag(exp(Qm*(z-d)))
        ekiz = lambda z: diag(exp(-1j*kziv*z))
        Wsxm0, Wsym0, Wuxm0, Wuym0 = split(Wp, 4)
        Wsxm1, Wsym1, Wuxm1, Wuym1 = split(Wm, 4)
        Wsxm, Wsym, Wuxm, Wuym = [Wsxm0, Wsxm1], [Wsym0, Wsym1], [Wuxm0, Wuxm1], [Wuym0, Wuym1]
        Wm = lambda Wm, z: ekiz(z) @ np.hstack([Wm[0] @ eQpzm(z), ekiz(-d) @ Wm[1] @ eQmzm(z)])

        P = np.asarray(np.bmat([
            [SRlxm, SRrxm, ZERO,  ZERO,  -Wm(Wsxm, 0)],
            [SRlym, SRrym, ZERO,  ZERO,  -Wm(Wsym, 0)],
            [ZERO,  ZERO,  STlxm, STrxm, -Wm(Wsxm, d)],
            [ZERO,  ZERO,  STlym, STrym, -Wm(Wsym, d)],
            [URlxm, URrxm, ZERO,  ZERO,  -Wm(Wuxm, 0)],
            [URlym, URrym, ZERO,  ZERO,  -Wm(Wuym, 0)],
            [ZERO,  ZERO,  UTlxm, UTrxm, -Wm(Wuxm, d)],
            [ZERO,  ZERO,  UTlym, UTrym, -Wm(Wuym, d)],
        ]))
        # print(P.shape)

        kuvxy = array([
            [eulerMatrix('zyz', [self.phi, self.theta, 0], 0, 0), 
             eulerMatrix('zyz', [self.phi, self.theta, 0], 0, 1)], 
            [eulerMatrix('zyz', [self.phi, self.theta, 0], 1, 0), 
             eulerMatrix('zyz', [self.phi, self.theta, 0], 1, 1)]
        ])
        delta = self.ms == 0
        UVkiixym = np.vstack([kuvxy]*self.M)
        UVkiixm = UVkiixym[::2]
        UVkiiym = UVkiixym[1::2]
        ZERO = zeros(self.M)
        p = array([
            -1 * delta * (UVkiixm @ Ei),
            -1 * delta * (UVkiiym @ Ei),
            ZERO,
            ZERO,
            -1 * delta * n1 * (UVkiixm @ II @ Ei),
            -1 * delta * n1 * (UVkiiym @ II @ Ei),
            ZERO,
            ZERO
        ]).flatten()
        res = linsolve(P, p)
        rlv, rrv, tlv, trv, csxv, csyv, cuxv, cuyv = split(res, 8)
        self.DERl = -abs(rlv)**2 * real(k1iv[:, -1])/kiv[-1]
        self.DERr = -abs(rrv)**2 * real(k1iv[:, -1])/kiv[-1]
        self.DETl = abs(tlv)**2 * real(k3iv[:, -1])/kiv[-1]
        self.DETr = abs(trv)**2 * real(k3iv[:, -1])/kiv[-1]
        #print(k1iv, k3iv)
        return self.DERl, self.DERr, self.DETl, self.DETr
    
ng = 1.56
dn = 0.15
no = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. - dn/2
ne = (dn + 2*sqrt(-2*dn**2 + 9*ng**2))/6. + dn/2
# ng = sqrt((2*no**2+ne**2)/3)
alpha = 45/2
k0 = 2*pi/532e-9
pb = 2 * 2*pi/(2*k0*ng*cos(np.deg2rad(alpha)))

wls = np.linspace(400, 700, 201)*1e-9
#wls = [532e-9]
thetas = np.linspace(0, 60, 61)
phis = np.linspace(0, 359, 360)
# alphas = [0]
DERl = []
DERr = []
DETl = []
DETr = []
DE = []
nn = 1
ds = np.linspace(0, 5, 101)*1e-6
# for d in ds:
for wl in wls:
    # for theta in thetas:
    # for phi in phis:
    # for alpha in alphas/2:
    #lay = AnisHomo(5e-6, 1.5, 1, 2, 2.5, 3., 3.5)
    # lay = SinGrating(d)
    pb = 2 * 2*pi/(2*k0*ng*cos(np.deg2rad(alpha)))
    # Kx = 2*k0*ng*cos(np.deg2rad(alpha))*sin(np.deg2rad(alpha))
    # Kz = 2*k0*ng*cos(np.deg2rad(alpha))*cos(np.deg2rad(alpha))
    # lay = PVG2(5e-6, Kx, Kz, 1, no, ne)
    lay = PVG(5e-6, pb, alpha, 1, no, ne)
    # lay = VHG(5e-6, ng, 0.5, alpha, pb/2)
    rcwa = RCWA(ng, ng, lay, nn, [1, 1j], [1, -1j])
    derl, derr, detl, detr = rcwa.solve(0, -10, wl, 1, 1j)
    DERl.append(derl)
    DERr.append(derr)
    DETl.append(detl)
    DETr.append(detr)
    # DERl2.append(derl2)
    # DERr2.append(derr2)
    # DETl2.append(detl2)
    # DETr2.append(detr2)
    # DERl.append(array(DERl0))
    # DERr.append(array(DERr0))
    # DETl.append(array(DETl0))
    # DETr.append(array(DETr0))

DERl = array(DERl).T
DERr = array(DERr).T
DETl = array(DETl).T
DETr = array(DETr).T

xxs = wls
plt.subplot(211)
plt.plot(xxs, DERl[nn], 'r')
plt.plot(xxs, DERl[nn-1], 'g')
plt.plot(xxs, DERl[nn+1], 'b')
plt.plot(xxs, DERr[nn], 'r--')
plt.plot(xxs, DERr[nn-1], 'g--')
plt.plot(xxs, DERr[nn+1], 'b--')
plt.plot(xxs, DERl[nn+1]/(DERl[nn+1]+DETl[nn]), 'k')
# plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
plt.subplot(212)
plt.plot(xxs,  DETl[nn], 'r')
plt.plot(xxs,  DETl[nn-1], 'g')
plt.plot(xxs,  DETl[nn+1], 'b')
plt.plot(xxs,  DETr[nn], 'r--')
plt.plot(xxs,  DETr[nn-1], 'g--')
plt.plot(xxs,  DETr[nn+1], 'b--')
plt.show()

# plt.subplot(211)
# thetam, phim = np.meshgrid(thetas, phis)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T)
# # plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
# plt.subplot(212)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T/(DERl[:, :, nn+1].T+DETl[:, :, nn].T))
# plt.show()