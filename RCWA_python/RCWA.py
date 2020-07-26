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

def normW(W):
    W2 = np.zeros_like(W, dtype=np.complex)
    for i, w in enumerate(W.T): 
        W2[:, i] = (w.T/np.sum(w*w.conj())).copy()
    return W2

def redheffer(L, M):
    tlp, rlm, rlp, tlm = L
    trp, rrm, rrp, trm = M
    II = np.eye(len(tlp))
    iv1 = inv(II - rlm @ rrp)
    iv2 = inv(II - rrp @ rlm)
    return (
        trp @ iv1 @ tlp, rrm + trp @ iv1 @ rlm @ trm, 
        rlp + tlm @ iv2 @ rrp @ tlp, tlm @ iv2 @ trm
        )

def makeTm(L, M):
    A, Wm, B, Vm = L
    Wp, E, Vp, D = M
    temp1 = Wm @ inv(Vm)
    temp2 = Wp @ inv(Vp)
    iv1 = inv(Wp - temp1 @ Vp)
    iv2 = inv(Wm - temp2 @ Vm)
    return [
        iv1 @ (A - temp1 @ B), 
        iv1 @ (temp1 @ D - E),
        iv2 @ (temp2 @ B - A), 
        iv2 @ (E - temp2 @ D)
    ]

class Layer():
    def __init__(self, d):
        self.Kx = 0
        self.Kz = 0
        self.d = d
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
    
    def getQW(self, m, k0, kiv):
        A = self.getA(m, k0, kiv)
        q, W = eig(A)
        W = normW(W)
        Q = 1.j * k0 * q.round(12)
        pPredict = np.imag(Q) <= 0
        mPredict = np.imag(Q) > 0
        Qp, Qm = Q[pPredict], Q[mPredict]
        WVp, WVm = W[:, pPredict], W[:, mPredict]
        M = len(WVp)
        Wp, Vp = WVp[:M//2], WVp[M//2:]
        Wm, Vm = WVm[:M//2], WVm[M//2:]
        return (Qp, Qm, Wp, Vp, Wm, Vm)
    
    def getTm(self, m, k0, kiv):
        d = self.d
        ms = np.arange(-m, m+1)
        Qp, Qm, Wp, Vp, Wm, Vm = self.getQW(m, k0, kiv)
        K = lambda z: diag(exp(-1j*np.hstack([-ms*self.Kz, -ms*self.Kz])*z))
        Xp = lambda z: diag(exp(Qp*z))
        Xm = lambda z: diag(exp(Qm*z))
        A = K(d) @ Wp @ Xp(d)
        B = K(d) @ Vp @ Xp(d)
        E = K(-d) @ Wm @ Xm(-d)
        D = K(-d) @ Vm @ Xm(-d)
        return [(Wp, E, Vp, D), (A, Wm, B, Vm)]
    
    
    def getSymTm(self, n1, m, k0, kiv):
        L = Homo(0, n1).getTm(m, k0, kiv)[1]
        M = self.getTm(m, k0, kiv)[0]
        temp1 = makeTm(L, M)
        return redheffer(temp1, temp1[::-1])

class Homo(Layer):
    def __init__(self, d, n):
        super().__init__(d)
        self.n = n
        self.Kx = 0

    def eps(self, p, q, n):
        if n == 0 and p == q:
            return self.n**2
        else:
            return 0

class VHG(Layer):
    def __init__(self, d, n, dn, delta, period):
        super().__init__(d)
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
        super().__init__(d)
        self.epsv = [[exx, exy, exz], [0, eyy, eyz], [0, 0, ezz]]
    
    def eps(self, p, q, n):
        if n == 0:
            return self.epsv[p][q]
        else:
            return 0

class PVG(Layer):
    def __init__(self, d, pitch, delta, chi, no, ne):
        super().__init__(d)
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
        super().__init__(d)
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

class LiquidCrylstalRCWA():
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
        Kxs = np.array([layer.Kx for layer in self.layers])
        if not np.all(Kxs == Kxs[0]):
            raise Exception("All layers should have the same x period.")
        self.lKx = Kxs[0]
    
    def getLayerT(self, n1, k0, kiv):
        return [layer.getTm(self.m, k0, kiv) for layer in self.layers]

    def getIn(self, k1iv, Ei):
        II = array([[0, -1], [1, 0]])
        ZERO = np.zeros((self.M, self.M))
        n1 = self.n1

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

        SI = np.asarray(np.bmat([
            [diag(delta * (UVkiixm @ Ei)), ZERO], 
            [ZERO, diag(delta * (UVkiiym @ Ei))]
        ]))
        SR = np.asarray(np.bmat([
            [SRlxm, SRrxm],
            [SRlym, SRrym]
        ]))
        UI = n1 * np.asarray(np.bmat([
            [diag(delta * (UVkiixm @ II @ Ei)), ZERO], 
            [ZERO, diag(delta * (UVkiiym @ II @ Ei))]
        ]))
        UR = np.asarray(np.bmat([
            [URlxm, URrxm],
            [URlym, URrym]
        ]))
        return [(0, 0, 0, 0), (SI, SR, UI, UR)]
    
    def getOut(self, k3iv):
        II = array([[0, -1], [1, 0]])
        ZERO = np.zeros((self.M, self.M))
        n3 = self.n3

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
        ST = np.asarray(np.bmat([
            [STlxm, STrxm],
            [STlym, STrym]
        ]))
        UT = np.asarray(np.bmat([
            [UTlxm, UTrxm],
            [UTlym, UTrym]
        ]))
        ZERO2 = np.zeros((2*self.M, 2*self.M))
        return [(ST, ZERO2, UT, ZERO2), (0, 0, 0, 0)]

    # @pysnooper.snoop(max_variable_length=None)
    def solve(self, phi, theta, wl, Eu, Ev):
        '''
        phi: projected angle between x axis in x-y plane 
        theta: angle between z axis
        light to z+
        '''
        self.phi = deg2rad(phi)
        self.theta = deg2rad(theta)
        self.k0uv = array([
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
        k0v = self.k0uv * k0k
        kiv = k0v * n1
        k1k = k0k * n1
        k3k = k0k * n3

        k1iv = array([kiv-i*np.r_[self.lKx, 0, 0] for i in self.ms], dtype=np.complex)
        for i in range(len(k1iv)):
            kx, ky, _ = k1iv[i]
            if kx**2 + ky**2 <= k1k**2:
                k1iv[i][-1] = -sqrt(k1k**2 - kx**2 - ky**2)
            else:
                k1iv[i][-1] = 1j*sqrt(-k1k**2 + kx**2 + ky**2)

        k3iv = array([kiv-i*np.r_[self.lKx, 0, 0] for i in self.ms], dtype=np.complex)
        for i in range(len(k3iv)):
            kx, ky, _ = k3iv[i]
            if kx**2 + ky**2 <= k3k**2:
                k3iv[i][-1] = sqrt(k3k**2 - kx**2 - ky**2)
            else:
                k3iv[i][-1] = -1j*sqrt(-k3k**2 + kx**2 + ky**2)

        Tl = sum(self.getLayerT(self.n1, k0k, kiv), [])
        T0 = self.getIn(k1iv, Ei)
        Te = self.getOut(k3iv)
        Ts = (T0 + Tl + Te)[1:-1]

        ZERO = np.zeros((2*self.M, 2*self.M), dtype=np.complex)

        res = [0]*len(Ts)
        for i in range(0, len(Ts), 2):
            A, Wm, B, Vm = Ts[i]
            Wp, E, Vp, D = Ts[i+1]
            ZEROS = [ZERO, ZERO]
            res[i]   = ZEROS*(i//2) + [A, Wm, -Wp, -E] + ZEROS*(len(Ts)//2-1-i//2)
            res[i+1] = ZEROS*(i//2) + [B, Vm, -Vp, -D] + ZEROS*(len(Ts)//2-1-i//2)
        P = np.asarray(np.bmat(res))[:, (self.M*2):(-2*self.M)]
        delta = self.ms == 0

        SI, _, UI, _ = T0[1]
        p = np.asarray(np.bmat(
            [-np.diag(SI), -np.diag(UI)] + 
            [np.diag(ZERO)] * 2 * len(self.layers)
        )).flatten()
        res = linsolve(P, p)
        rlv, rrv = split(res[:2*self.M], 2)
        tlv, trv = split(res[-2*self.M:], 2)

        self.DERl = -abs(rlv)**2 * real(k1iv[:, -1])/kiv[-1]
        self.DERr = -abs(rrv)**2 * real(k1iv[:, -1])/kiv[-1]
        self.DETl =  abs(tlv)**2 * real(k3iv[:, -1])/kiv[-1]
        self.DETr =  abs(trv)**2 * real(k3iv[:, -1])/kiv[-1]
        self.k1iv = k1iv/k0k
        self.k3iv = k3iv/k0k
        #print(k1iv, k3iv)
        return self.DERl, self.DERr, self.DETl, self.DETr
    
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
lay4 = Homo(0, ng)
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
    rcwa = LiquidCrylstalRCWA(ng, ng, [lay1, lay2], nn, [1, 1j], [1j, 1])
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
plt.subplot(221)
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
plt.subplot(222)
plt.plot(xxs,  DETl[nn], 'r')
plt.plot(xxs,  DETr[nn], 'r--')
if nn > 0:
    plt.plot(xxs,  DETl[nn-1], 'g')
    plt.plot(xxs,  DETl[nn+1], 'b')
    plt.plot(xxs,  DETr[nn-1], 'g--')
    plt.plot(xxs,  DETr[nn+1], 'b--')
plt.ylim(0, 1.5)
plt.subplot(223)
plt.plot(xxs, np.abs(k1iv[nn, 0]), 'r')
plt.plot(xxs, np.abs(k1iv[nn, 1]), 'r--')
plt.plot(xxs, np.abs(k1iv[nn, 2]), 'r.')
if nn > 0:
    plt.plot(xxs, np.abs(k1iv[nn-1, 0]), 'g')
    plt.plot(xxs, np.abs(k1iv[nn-1, 1]), 'g--')
    plt.plot(xxs, np.abs(k1iv[nn-1, 2]), 'g.')
    plt.plot(xxs, np.abs(k1iv[nn+1, 0]), 'b')
    plt.plot(xxs, np.abs(k1iv[nn+1, 1]), 'b--')
    plt.plot(xxs, np.abs(k1iv[nn+1, 2]), 'b.')
# plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
plt.subplot(224)
plt.plot(xxs, np.abs(k3iv[nn, 0]), 'r')
plt.plot(xxs, np.abs(k3iv[nn, 1]), 'r--')
plt.plot(xxs, np.abs(k3iv[nn, 2]), 'r.')
if nn > 0:
    plt.plot(xxs, np.abs(k3iv[nn-1, 0]), 'g')
    plt.plot(xxs, np.abs(k3iv[nn-1, 1]), 'g--')
    plt.plot(xxs, np.abs(k3iv[nn-1, 2]), 'g.')
    plt.plot(xxs, np.abs(k3iv[nn+1, 0]), 'b')
    plt.plot(xxs, np.abs(k3iv[nn+1, 1]), 'b--')
    plt.plot(xxs, np.abs(k3iv[nn+1, 2]), 'b.')
plt.show()

# plt.subplot(211)
# thetam, phim = np.meshgrid(thetas, phis)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T)
# # plt.plot(wls,  DERr[nn]+DERl[nn], 'y')
# plt.subplot(212)
# plt.pcolormesh(thetam/60*cos(np.deg2rad(phim)), thetam/60*sin(np.deg2rad(phim)), DERl[:, :, nn+1].T/(DERl[:, :, nn+1].T+DETl[:, :, nn].T))
# plt.show()