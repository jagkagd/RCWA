from typing import List
from scipy.linalg import eig, inv, solve as linsolve
from numpy import array, split, zeros, sin, cos, tan, pi, diag, sqrt, deg2rad, exp, real
import numpy as np
from Beam import Plane

def dot(x, y):
    return sum([xi*yi for (xi, yi) in zip(x, y)])

def norml(v):
    vv = array(v)
    return vv / sqrt(dot(vv, np.conj(vv)))

def eulerMatrix(seq, angs, i, j=None):
    if i == 'm':
        if seq == 'zyz':
            return (lambda a, b, c: [
                [cos(a)*cos(b)*cos(c)-sin(a)*sin(c), -cos(c)*sin(a)-cos(a)*cos(b)*sin(c), cos(a)*sin(b)],
                [cos(b)*cos(c)*sin(a)+cos(a)*sin(c),  cos(a)*cos(c)-cos(b)*sin(a)*sin(c), sin(a)*sin(b)],
                [-cos(c)*sin(b), sin(b)*sin(c), cos(b)]
            ])(*angs)
    else:
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
        Kxs = np.array([layer.Kx for layer in self.layers])
        if not np.all(Kxs == Kxs[0]):
            raise Exception("All layers should have the same x period.")
        self.lKx = Kxs[0]
    
    def getLayerT(self, n1, k0, kiv):
        return [layer.getTm(self.m, k0, kiv) for layer in self.layers]

    def getIn(self, k1iv, k0uv, Ei):
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

        # kuvxy = array([
        #     [eulerMatrix('zyz', [phi, theta, 0], 0, 0), 
        #      eulerMatrix('zyz', [phi, theta, 0], 0, 1)], 
        #     [eulerMatrix('zyz', [phi, theta, 0], 1, 0), 
        #      eulerMatrix('zyz', [phi, theta, 0], 1, 1)]
        # ])
        kuvxy = k2uv_xy(k0uv)
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

    def solve(self, phi, theta, wl, Eu, Ev):
        '''
        phi: projected angle between x axis in x-y plane 
        theta: angle between z axis
        light to z+
        '''
        phi = deg2rad(phi)
        theta = deg2rad(theta)
        k0uv = array([
            eulerMatrix('zyz', [phi, theta, 0], 0, 2),
            eulerMatrix('zyz', [phi, theta, 0], 1, 2),
            eulerMatrix('zyz', [phi, theta, 0], 2, 2)
        ])
        return self.solve_k0(k0uv, wl, Eu, Ev)
    
    def solve_k0(self, k0uv, wl: float, Eu, Ev, info='T'):
        self.DERl = np.zeros(self.M)
        self.DERr = np.zeros_like(self.DERl)
        self.DETl = np.zeros_like(self.DERl)
        self.DETr = np.zeros_like(self.DERl)
        n1, n3 = self.n1, self.n3
        Ei = norml(array([Eu, Ev]))
        k0k = 2*pi/wl
        k0v = k0uv * k0k
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
        T0 = self.getIn(k1iv, k0uv, Ei)
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

        SI, _, UI, _ = T0[1]
        p = np.asarray(np.bmat(
            [-np.diag(SI), -np.diag(UI)] + 
            [np.diag(ZERO)] * 2 * len(self.layers)
        )).flatten()
        res = linsolve(P, p)
        rlv, rrv = split(res[:2*self.M], 2)
        tlv, trv = split(res[-2*self.M:], 2)

        self.rlv = rlv
        self.rrv = rrv
        self.tlv = tlv
        self.trv = trv

        self.DERl = -abs(rlv)**2 * real(k1iv[:, -1])/kiv[-1]
        self.DERr = -abs(rrv)**2 * real(k1iv[:, -1])/kiv[-1]
        self.DETl =  abs(tlv)**2 * real(k3iv[:, -1])/kiv[-1]
        self.DETr =  abs(trv)**2 * real(k3iv[:, -1])/kiv[-1]
        self.k1iv = k1iv/k0k
        self.k3iv = k3iv/k0k
        self.polR = np.array([norml(Eo) for Eo in rlv[:, np.newaxis]*self.base1 + rrv[:, np.newaxis]*self.base2])
        self.polT = np.array([norml(Eo) for Eo in tlv[:, np.newaxis]*self.base1 + trv[:, np.newaxis]*self.base2])
        if info == 'T':
            return self.DERl, self.DERr, self.DETl, self.DETr
        elif info == 'detail':
            return (rlv, rrv, tlv, trv), (k1iv, k3iv)
        else:
            raise NotImplementedError()
    
    def solve_beam(self, phi, theta, Eu, Ev, beam, fxs, fys, dfS):
        # x * y * [Rl, Rr, Tl, Tr] * [-m, ..., m]
        wl = beam.wl
        k0 = 2*np.pi/wl*self.n1
        if isinstance(beam, Plane):
            k1 = eulerMatrix('zyz', [phi, theta, 0], 'm') @ np.array([0, 0, k0])
            T0 = np.array(self.solve_k0(k1, wl, Eu, Ev))
            return T0
        T0 = np.zeros((*fxs.shape, 4, self.M))
        T = np.zeros_like(T0)
        for i in range(fxs.shape[0]):
            for j in range(fxs.shape[1]):
                kx0, ky0 = fxs[i, j], fys[i, j]
                kz0 = np.sqrt(k0**2 - kx0**2 - ky0**2)
                k1 = eulerMatrix('zyz', [phi, theta, 0], 'm') @ np.array([kx0, ky0, kz0])
                T0[i, j] = np.array(self.solve_k0(k1/k0, wl, Eu, Ev))
                T[i, j] = T0[i, j] * beam.amplitude(kx0, ky0)
        res = np.einsum('ijkl, ij -> kl', T, dfS) / np.sum(beam.amplitude(fxs, fys)*dfS)
        # return [(T0[:, :, i, :], res[i]) for i in range(4)]
        # return T0
        return res
        
