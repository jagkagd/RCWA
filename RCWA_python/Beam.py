import numpy as np


class Beam:
    def __init__(self, wl):
        self.wl = wl
    
    def amplitude(self, fx, fy):
        raise NotImplementedError()


class Plane(Beam):
    def __init__(self, wl):
        super().__init__(wl)

    def amplitude(self, fx, fy):
        return (fx, fy) == (0, 0)


class Gaussian(Beam):
    def __init__(self, wl, theta0, z):
        super().__init__(wl)
        self.theta0 = np.deg2rad(theta0)
        self.z = z
        self.W0 = self.wl/np.pi/self.theta0
        self.z0 = np.pi*self.W0**2/self.wl
        self.Wz = self.W0*np.sqrt(1+(self.z/self.z0)**2)
    
    def amplitude(self, fx, fy):
        return 1/2*np.exp(-1/4*(fx**2+fy**2)*self.Wz**2)*self.W0*self.Wz
