from typing import List
from numpy import conj, angle, pi, array, sqrt, deg2rad, rad2deg, sin, cos, abs, exp, arctan, real, sum, arcsin
from RCWA import RCWA, eulerMatrix, norml
from Layer import Layer
from pprint import pprint
from matplotlib.patches import Ellipse

class Field:
    def __init__(self, k0uv, Euv, rot=0):
        self.k0uv = array(k0uv)
        Eu, Ev = Euv
        self.amp = sqrt(Eu*conj(Eu) + Ev*conj(Ev))
        self.Euv = norml(Euv)
        self.rot = rot
    def __repr__(self):
        # return f'k0uv: [{self.k0uv[0]:.3g} {self.k0uv[1]:.3g} {self.k0uv[2]:.3g}] Euv: [{self.Euv[0]:.3g} {self.Euv[1]:.3g}] Amp: {self.amp:.3g}'
        return f'{self.rot}'
    def ang(self):
        kx, ky, _ = self.k0uv
        rot = deg2rad(self.rot)
        kx, ky = cos(-rot)*kx + sin(-rot)*ky, -sin(-rot)*kx + cos(-rot)*ky
        return rad2deg(real(arctan(array([kx/sqrt(1-kx**2), ky/sqrt(1-ky**2)]))))
    def pol_ellipse(self):
        Eu, Ev = self.Euv
        Eua, Eup = abs(Eu), angle(Eu)
        Eva, Evp = abs(Ev), angle(Ev)
        Evup = Evp-Eup
        A = real(sqrt((1+0j+sqrt(1+0j-(2*Eua*Eva)**2*sin(Evup)**2))/2))
        B = real(sqrt((1+0j-sqrt(1+0j-(2*Eua*Eva)**2*sin(Evup)**2))/2))
        if Eua**2 - Eva**2 == 0:
            return (A, B, pi/4)
        else:
            return (A, B, real(arctan((2*Eua*Eva*cos(Evup))/(Eua**2-Eva**2))/2))
    def chiral(self):
        Eu, Ev = self.Euv
        return 1 if angle(Ev/Eu) < 0 else -1
    def __eq__(self, f2):
        return sum(abs(self.k0uv - f2.k0uv)) < 1e-6
    def __neq__(self, f2):
        return not (self.__eq__(f2))
    def __add__(self, f2):
        return Field(self.k0uv, self.amp*array(self.Euv) + f2.amp*array(f2.Euv), self.rot)
    def T(self, k0v):
        return abs(self.amp)**2 * real(self.k0uv[-1]/k0v[-1])

class Fields:
    def __init__(self, fields=None):
        if fields is None:
            fields = []
        self.fields = fields
    def __ior__(self, field: Field):
        if abs(field.amp) < 1e-10:
            return self
        for (i, f) in enumerate(self.fields):
            if f == field:
                self.fields[i] = f + field
                break
        else:
            self.fields.append(field)
        # self.fields.append(field)
        return self
    def __iter__(self):
        return iter(self.fields)
    def __len__(self):
        return len(self.fields)
    def __getitem__(self, slice):
        return self.fields[slice]
    def __repr__(self):
        return ' '.join(['['] + [' ' + repr(field) for field in self.fields] + [']'])

class KPick:
    def __init__(self, no: int):
        self.no = no
    def __call__(self, fields: Fields):
        return fields[(len(fields)-1)//2 + self.no]

class KRotator:
    # grating turn from x to y with ang
    def __init__(self, ang):
        self.ang = deg2rad(ang)
    def __call__(self, field: Field):
        kx, ky, kz = field.k0uv
        return Field([
             cos(self.ang)*kx + sin(self.ang)*ky,
            -sin(self.ang)*kx + cos(self.ang)*ky,
            kz
        ], field.amp*field.Euv, field.rot+rad2deg(self.ang))

class EChiral:
    def __init__(self):
        pass
    def __call__(self, field: Field):
        Eu, Ev = field.Euv
        # Eua, Eut = abs(Eu), angle(Eu)
        # Eva, Evt = abs(Ev), angle(Ev)
        # return Field(field.k0uv, field.amp*array([Eua, Eva*exp(1j*(-(Evt-Eut)))]))
        return Field(field.k0uv, field.amp*array([Eu, -Ev]), field.rot)

def solve_stack(nn, nbg, wl0, stacks, base1, base2, k0v, Eu, Ev, mode='1'):
    fieldss = []
    k0 = 2*pi/wl0
    base1 = norml(base1)
    base2 = norml(base2)
    # k1 = eulerMatrix('zyz', [phi, theta, 0], 'm') @ array([0, 0, k0])
    Euv = norml([Eu, Ev])
    fields = Fields([Field(k0v, Euv)])
    fieldss.append(fields)
    # pprint(fields)
    stacks = [[s] if isinstance(s, Layer) else s for s in stacks]
    for stack in stacks:
        if isinstance(stack, list) and isinstance(stack[0], Layer):
            temp = Fields()
            for field in fields:
                res = RCWA(nbg, nbg, stack, nn, base1, base2)
                (_, _, tlv, trv), (_, k3iv) = res.solve_k0(field.k0uv, wl0, *field.Euv, 'detail')
                for (tl, tr, k3i) in zip(tlv, trv, k3iv):
                    temp |= Field(k3i/k0/nbg, (tl*array(base1) + tr*array(base2))*field.amp, field.rot)
            fields = temp
            fieldss.append(fields)
        elif isinstance(stack, KPick):
            if mode == '1':
                fields = Fields([stack(fields)])
        elif isinstance(stack, KRotator) or isinstance(stack, EChiral):
            fields = Fields([stack(field) for field in fields])
        else:
            raise NotImplementedError()
        # pprint(fields)
    return fieldss

def plot_field(ax, field: Field, k0v, nbg):
    angx, angy = field.ang()
    ang2ang = lambda ang: rad2deg(arcsin(nbg*sin(deg2rad(ang))))
    A, B, theta = field.pol_ellipse()
    alpha = field.T(k0v)
    if alpha < 0: alpha = 0
    if alpha > 1: alpha = 1
    ec = (1, 0, 0, alpha) if field.chiral() == 1 else (0, 0, 1, alpha)
    ell = Ellipse((ang2ang(-angx), ang2ang(-angy)), A*10, B*10, angle=rad2deg(theta), linewidth=1., fill=False, edgecolor=ec)
    ax.add_patch(ell)

def plot_fields(ax, fields: Field, k0v, nbg):
    for field in fields:
        plot_field(ax, field, k0v, nbg)
