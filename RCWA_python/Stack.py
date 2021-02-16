from typing import List
from numpy import conj, angle, pi, array, sqrt, deg2rad, rad2deg, sin, cos, abs, exp, arctan, real, sum
from RCWA import RCWA, eulerMatrix, norml
from Layer import Layer
from pprint import pprint

class Field:
    def __init__(self, k0uv, Euv):
        self.k0uv = array(k0uv)
        Eu, Ev = Euv
        self.amp = sqrt(Eu*conj(Eu) + Ev*conj(Ev))
        self.Euv = norml(Euv)
    def __repr__(self):
        return f'k0uv: [{self.k0uv[0]:.3g} {self.k0uv[1]:.3g} {self.k0uv[2]:.3g}] Euv: [{self.Euv[0]:.3g} {self.Euv[1]:.3g}] Amp: {self.amp:.3g}'
    def ang(self):
        kx, ky, kz = self.k0uv
        return rad2deg(real(arctan(array([kx/sqrt(1-kx**2), ky/sqrt(1-ky**2)]))))
    def __eq__(self, f2):
        return sum(abs(self.k0uv - f2.k0uv)) < 1e-6
    def __neq__(self, f2):
        return not (self.__eq__(f2))
    def __add__(self, f2):
        return Field(self.k0uv, self.amp*array(self.Euv) + f2.amp*array(f2.Euv))

class Fields:
    def __init__(self, fields=None):
        if fields is None:
            fields = []
        self.fields = fields
    def __ior__(self, field: Field):
        if field.amp < 1e-6:
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
        return '\n'.join(['['] + [' ' + repr(field) for field in self.fields] + [']'])

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
        ], field.amp*field.Euv)

class EChiral:
    def __init__(self):
        pass
    def __call__(self, field: Field):
        Eu, Ev = field.Euv
        # Eua, Eut = abs(Eu), angle(Eu)
        # Eva, Evt = abs(Ev), angle(Ev)
        # return Field(field.k0uv, field.amp*array([Eua, Eva*exp(1j*(-(Evt-Eut)))]))
        return Field(field.k0uv, field.amp*array([Eu, -Ev]))

def solve_stack(nn, nbg, wl0, stacks, base1, base2, k0v, Eu, Ev, mode='1'):
    k0 = 2*pi/wl0
    base1 = norml(base1)
    base2 = norml(base2)
    # k1 = eulerMatrix('zyz', [phi, theta, 0], 'm') @ array([0, 0, k0])
    Euv = norml([Eu, Ev])
    fields = Fields([Field(k0v, Euv)])
    # pprint(fields)
    stacks = [[s] if isinstance(s, Layer) else s for s in stacks]
    for stack in stacks:
        if isinstance(stack, list) and isinstance(stack[0], Layer):
            temp = Fields()
            for field in fields:
                res = RCWA(nbg, nbg, stack, nn, base1, base2)
                (_, _, tlv, trv), (_, k3iv) = res.solve_k0(field.k0uv, wl0, *field.Euv, 'detail')
                for (tl, tr, k3i) in zip(tlv, trv, k3iv):
                    temp |= Field(k3i/k0/nbg, (tl*array(base1) + tr*array(base2))*field.amp)
            fields = temp
        elif isinstance(stack, KPick):
            if mode == '1':
                fields = Fields([stack(fields)])
        elif isinstance(stack, KRotator) or isinstance(stack, EChiral):
            fields = Fields([stack(field) for field in fields])
        else:
            raise NotImplementedError()
        # pprint(fields)
    return fields
