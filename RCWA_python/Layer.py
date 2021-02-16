from scipy.linalg import eig, inv
from scipy.integrate import quad
from numpy import array, split, zeros, sin, cos, tan, pi, diag, sqrt, deg2rad, exp, real, arctan
import numpy as np

def cquad(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

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
        # exxm_1 = self.epsxx1m(mn)
        # exxm_1_1 = inv(exxm_1).round(12)

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
        # W = normW(W)
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
    def __init__(self, d, Kx, Kz, chi, no, ne):
        super().__init__(d)
        self.chi = chi # chirality
        self.Kx = Kx
        self.Kz = Kz
        self.no = no
        self.ne = ne
        self.eo = no**2
        self.ee = ne**2
        self.Kv = array([self.Kx, 0, self.Kz])
        self.delta = arctan(self.Kx/self.Kz)

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

class PVG_dp(PVG):
    def __init__(self, d, pitch, delta, chi, no, ne):
        self.pitch = pitch # LC rotate 2pi (m)
        self.period = pitch / 2 # period of grating, LC rotate pi (m)
        self.delta = deg2rad(delta) # angle (rad) between z axis
        self.K = 2*pi/(self.period)
        Kx = self.K*sin(self.delta)
        Kz = self.K*cos(self.delta)
        super().__init__(d, Kx, Kz, chi, no, ne)

class PVG2(Layer):
    def __init__(self, d, Kx, Kz, chi, no, ne, phi0):
        super().__init__(d)
        self.chi = chi # chirality
        self.no = no
        self.ne = ne
        self.eo = no**2
        self.ee = ne**2
        self.Kx = Kx
        self.Kz = Kz
        self.Kv = array([self.Kx, 0, self.Kz])
        self.phi0 = phi0

    def eps(self, p, q, n):
        eo, ee, chi, phi0 = self.eo, self.ee, self.chi, self.phi0
        exx = lambda n: {1: (ee-eo)/4*exp(-2j*phi0), -1: (ee-eo)/4*exp(2j*phi0), 0: (ee+eo)/2}.get(n, 0)
        exy = lambda n: {1: -chi*1j*exp(-2j*phi0)*(ee-eo)/4, -1: chi*1j*exp(2j*phi0)*(ee-eo)/4}.get(n, 0)
        exz = lambda n: 0
        eyy = lambda n: {1: (eo-ee)/4*exp(-2j*phi0), -1: (eo-ee)/4*exp(2j*phi0), 0: (ee+eo)/2}.get(n, 0)
        eyz = lambda n: 0
        ezz = lambda n: {0: eo}.get(n, 0)
        eps = [[exx, exy, exz], [exy, eyy, eyz], [exz, eyz, ezz]]
        return eps[p][q](n)

def genGradient(layClass, n, d, px, pz0, pz1, chi, no, ne):
    d0 = d/n
    d0s = np.array(list(range(n)))*1./n*d
    pzf = lambda z: 1/(1/pz0 + (1/pz1-1/pz0)*(z/d))
    pzs = [pzf(di) for di in d0s]
    return [layClass(d0, 2*pi/px, 2*pi/pz, chi, no, ne) for pz in pzs]
