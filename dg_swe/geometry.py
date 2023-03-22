import numpy as np


class BaseFace:

    def __init__(self, name, radius):
        self.radius = radius
        valid_names = ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']
        if not name in valid_names:
            raise ValueError(f'name: expected one of: {valid_names}. Found {name}.')
        self.name = name
        self._connections = None

    def lat_long(self, x, y, z):

        lat = np.arcsin(z / self.radius)
        long = np.arctan2(x, -y)

        return lat, long

    def lat_long_vecs(self, x, y, z):

        lat, long = self.lat_long(x, y, z)
        long_vec_x = np.cos(long)
        long_vec_y = np.sin(long)
        long_vec_z = 0 * long

        lat_vec_x = -np.sin(lat) * np.sin(long)
        lat_vec_y = np.sin(lat) * np.cos(long)
        lat_vec_z = np.cos(lat)

        return lat_vec_x, lat_vec_y, lat_vec_z, long_vec_x, long_vec_y, long_vec_z

    def to_cartesian(self, x1, y1):
        raise NotImplementedError

    def covariant_basis(self, x1, y1):
        raise NotImplementedError

    def is_connected(self, other):

        x1s = np.array([0.5, 0, -0.5, 0.0])
        y1s = np.array([0, 0.5, 0.0, -0.5])

        x2, y2, z2 = other.to_cartesian(x1s, y1s)

        for i in range(4):
            x1, y1, z1 = self.to_cartesian(x1s[i], y1s[i])
            match = np.isclose(x1, x2) & np.isclose(y1, y2) & np.isclose(z1, z2)
            if match.any():
                assert match.sum() == 1
                return i, np.where(match)[0][0]

        return None

    @property
    def connections(self):
        if self._connections is None:
            names = ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']
            names.remove(self.name)
            self._connections = []
            for name in names:
                _face = self.__class__(name, radius=self.radius)
                con = self.is_connected(_face)
                if con is not None:
                    i1, i2 = con
                    x1s = np.array([0.5, 0, -0.5, 0.0])
                    y1s = np.array([0, 0.5, 0.0, -0.5])

                    if x1s[i1] == 0:
                        x1 = 0.5
                        y1 = y1s[i1]
                    else:
                        x1 = x1s[i1]
                        y1 = 0.5

                    if x1s[i2] == 0:
                        x2 = 0.5
                        y2 = y1s[i2]
                    else:
                        x2 = x1s[i2]
                        y2 = 0.5

                    if any(not np.isclose(a, b) for a, b in zip(self.to_cartesian(x1, y1), _face.to_cartesian(x2, y2))):
                        print(self.name, name)
                        print(x1, y1)
                        print(x2, y2)
                        print(self.to_cartesian(x1, y1))
                        print(_face.to_cartesian(x2, y2))
                        print('Found difference!!!!!!!')
                        print()
                    self._connections.append((name, con))

        return self._connections


class EquiangularFace(BaseFace):

    def __init__(self, name, radius):

        super().__init__(name, radius)

    def to_cartesian(self, x1, y1):

        x1 = 0.5 * np.pi * x1 # change range from [-0.5, 0.5] --> [-pi / 4, pi / 4]
        y1 = 0.5 * np.pi * y1

        hyp = np.sqrt(1 + np.tan(x1) ** 2 + np.tan(y1) ** 2)

        if self.name == 'zp':
            z = 1 / hyp
            x = z * np.tan(x1)
            y = z * np.tan(y1)
        elif self.name == 'zn':
            z = -1 / hyp
            y = -z * np.tan(x1)
            x = -z * np.tan(y1)
        elif self.name == 'xp':
            x = 1 / hyp
            y = x * np.tan(x1)
            z = x * np.tan(y1)
        elif self.name == 'xn':
            x = -1 / hyp
            z = -x * np.tan(x1)
            y = -x * np.tan(y1)
        elif self.name == 'yp':
            y = 1 / hyp
            z = y * np.tan(x1)
            x = y * np.tan(y1)
        elif self.name == 'yn':
            y = -1 / hyp
            x = -y * np.tan(x1)
            z = -y * np.tan(y1)
        else:
            raise RuntimeError

        return self.radius * x, self.radius * y, self.radius * z

    def covariant_basis(self, x1, y1):

        x, y, z = self.to_cartesian(x1, y1)

        x /= self.radius
        y /= self.radius
        z /= self.radius

        x1 = 0.5 * np.pi * x1 # change range from [-0.5, 0.5] --> [-pi / 4, pi / 4]
        y1 = 0.5 * np.pi * y1

        hyp = np.sqrt(1 + np.tan(x1) ** 2 + np.tan(y1) ** 2)

        secx1 = (1 / np.cos(x1))
        secy1 = (1 / np.cos(y1))

        dxdz1 = x
        dydz1 = y
        dzdz1 = z

        if self.name == 'zp':
            dzdx1 = -(1 / hyp ** 3) * np.tan(x1) * secx1 ** 2
            dzdy1 = -(1 / hyp ** 3) * np.tan(y1) * secy1 ** 2

            dxdx1 = dzdx1 * np.tan(x1) + z * secx1 ** 2
            dxdy1 = dzdy1 * np.tan(x1)

            dydx1 = dzdx1 * np.tan(y1)
            dydy1 = dzdy1 * np.tan(y1) + z * secy1 ** 2

        elif self.name == 'zn':
            dzdx1 = (1 / hyp ** 3) * np.tan(x1) * secx1 ** 2
            dzdy1 = (1 / hyp ** 3) * np.tan(y1) * secy1 ** 2

            dydx1 = -(dzdx1 * np.tan(x1) + z * secx1 ** 2)
            dydy1 = -dzdy1 * np.tan(x1)

            dxdx1 = -dzdx1 * np.tan(y1)
            dxdy1 = -(dzdy1 * np.tan(y1) + z * secy1 ** 2)
        elif self.name == 'xp':
            dxdx1 = - (1 / hyp ** 3) * np.tan(x1) * secx1 ** 2
            dxdy1 = - (1 / hyp ** 3) * np.tan(y1) * secy1 ** 2

            dydx1 = (dxdx1 * np.tan(x1) + x * secx1 ** 2)
            dydy1 = dxdy1 * np.tan(x1)

            dzdx1 = dxdx1 * np.tan(y1)
            dzdy1 = dxdy1 * np.tan(y1) + x * secy1 ** 2

        elif self.name == 'xn':
            dxdx1 = (1 / hyp ** 3) * np.tan(x1) * secx1 ** 2
            dxdy1 = (1 / hyp ** 3) * np.tan(y1) * secy1 ** 2

            dzdx1 = -(dxdx1 * np.tan(x1) + x * secx1 ** 2)
            dzdy1 = -dxdy1 * np.tan(x1)

            dydx1 = -dxdx1 * np.tan(y1)
            dydy1 = -(dxdy1 * np.tan(y1) + x * secy1 ** 2)
        elif self.name == 'yp':
            dydx1 = - (1 / hyp ** 3) * np.tan(x1) * secx1 ** 2
            dydy1 = - (1 / hyp ** 3) * np.tan(y1) * secy1 ** 2

            dzdx1 = (dydx1 * np.tan(x1) + y * secx1 ** 2)
            dzdy1 = dydy1 * np.tan(x1)

            dxdx1 = dydx1 * np.tan(y1)
            dxdy1 = dydy1 * np.tan(y1) + y * secy1 ** 2
        elif self.name == 'yn':
            dydx1 = (1 / hyp ** 3) * np.tan(x1) * secx1 ** 2
            dydy1 = (1 / hyp ** 3) * np.tan(y1) * secy1 ** 2

            dxdx1 = -(dydx1 * np.tan(x1) + y * secx1 ** 2)
            dxdy1 = -dydy1 * np.tan(x1)

            dzdx1 = -dydx1 * np.tan(y1)
            dzdy1 = -(dydy1 * np.tan(y1) + y * secy1 ** 2)
        else:
            raise RuntimeError

        dxdx1 *= 0.5 * np.pi * self.radius
        dxdy1 *= 0.5 * np.pi * self.radius
        dydx1 *= 0.5 * np.pi * self.radius
        dydy1 *= 0.5 * np.pi * self.radius
        dzdx1 *= 0.5 * np.pi * self.radius
        dzdy1 *= 0.5 * np.pi * self.radius

        out = [dxdx1, dxdy1, dxdz1, dydx1, dydy1, dydz1, dzdx1, dzdy1, dzdz1]

        return out


class SadournyFace(BaseFace):

    def __init__(self, name, radius):

        super().__init__(name, radius)

    def to_cartesian_normalized(self, x1, y1):

        hyp = np.sqrt(y1 ** 2 + x1 ** 2 + 0.5 ** 2)

        if self.name == 'zp':
            z = 0.5 / hyp
            x = x1 / hyp
            y = y1 / hyp
        elif self.name == 'zn':
            z = -0.5 / hyp
            y = x1 / hyp
            x = y1 / hyp
        elif self.name == 'xp':
            x = 0.5 / hyp
            y = x1 / hyp
            z = y1 / hyp
        elif self.name == 'xn':
            x = -0.5 / hyp
            z = x1 / hyp
            y = y1 / hyp
        elif self.name == 'yp':
            y = 0.5 / hyp
            z = x1 / hyp
            x = y1 / hyp
        elif self.name == 'yn':
            y = -0.5 / hyp
            x = x1 / hyp
            z = y1 / hyp
        else:
            raise RuntimeError

        return x * self.radius, y * self.radius, z * self.radius

    def covariant_basis(self, x1, y1):

        a = np.sqrt(y1 ** 2 + x1 ** 2)
        hyp = np.sqrt(a ** 2 + 0.5 ** 2)
        x, y, z = self.to_cartesian(x1, y1)
        x /= self.radius
        y /= self.radius
        z /= self.radius

        dxdz1 = x
        dydz1 = y
        dzdz1 = z

        if self.name == 'zp':
            dxdx1 = (1 / hyp) * (1 - (x1 / hyp) ** 2)
            dxdy1 = -x1 * y1 / (hyp ** 3)

            dydx1 = -x1 * y1 / (hyp ** 3)
            dydy1 = (1 / hyp) * (1 - (y1 / hyp) ** 2)

            dzdx1 = -0.5 * x1 / hyp ** 3
            dzdy1 = -0.5 * y1 / hyp ** 3
        elif self.name == 'zn':
            dydx1 = (1 / hyp) * (1 - (x1 / hyp) ** 2)
            dydy1 = -x1 * y1 / (hyp ** 3)

            dxdx1 = -x1 * y1 / (hyp ** 3)
            dxdy1 = (1 / hyp) * (1 - (y1 / hyp) ** 2)

            dzdx1 = 0.5 * x1 / hyp ** 3
            dzdy1 = 0.5 * y1 / hyp ** 3
        elif self.name == 'xp':
            dydx1 = (1 / hyp) * (1 - (x1 / hyp) ** 2)
            dydy1 = -x1 * y1 / (hyp ** 3)

            dzdx1 = -x1 * y1 / (hyp ** 3)
            dzdy1 = (1 / hyp) * (1 - (y1 / hyp) ** 2)

            dxdx1 = -0.5 * x1 / hyp ** 3
            dxdy1 = -0.5 * y1 / hyp ** 3
        elif self.name == 'xn':
            dzdx1 = (1 / hyp) * (1 - (x1 / hyp) ** 2)
            dzdy1 = -x1 * y1 / (hyp ** 3)

            dydx1 = -x1 * y1 / (hyp ** 3)
            dydy1 = (1 / hyp) * (1 - (y1 / hyp) ** 2)

            dxdx1 = 0.5 * x1 / hyp ** 3
            dxdy1 = 0.5 * y1 / hyp ** 3
        elif self.name == 'yp':
            dzdx1 = (1 / hyp) * (1 - (x1 / hyp) ** 2)
            dzdy1 = -x1 * y1 / (hyp ** 3)

            dxdx1 = -x1 * y1 / (hyp ** 3)
            dxdy1 = (1 / hyp) * (1 - (y1 / hyp) ** 2)

            dydx1 = -0.5 * x1 / hyp ** 3
            dydy1 = -0.5 * y1 / hyp ** 3
        elif self.name == 'yn':
            dxdx1 = (1 / hyp) * (1 - (x1 / hyp) ** 2)
            dxdy1 = -x1 * y1 / (hyp ** 3)

            dzdx1 = -x1 * y1 / (hyp ** 3)
            dzdy1 = (1 / hyp) * (1 - (y1 / hyp) ** 2)

            dydx1 = 0.5 * x1 / hyp ** 3
            dydy1 = 0.5 * y1 / hyp ** 3
        else:
            raise RuntimeError

        # multiply all actual derivatives by radius
        # keep d/dz1 unit length
        for arr in (dxdx1, dxdy1, dydx1, dydy1, dzdx1, dzdy1):
            arr *= self.radius

        out = [dxdx1, dxdy1, dxdz1, dydx1, dydy1, dydz1, dzdx1, dzdy1, dzdz1]
        return out


if __name__ == '__main__':
    # face centre, edge centres, and corners
    x1s = np.array([0, 0, 0.5, 0, -0.5, 0.5, 0.5, -0.5, -0.5])
    y1s = np.array([0, 0.5, 0, -0.5, 0, 0.5, -0.5, 0.5, -0.5])
    vals = np.linspace(-0.5, 0.5, 1000)
    diff = np.mean(np.diff(vals))

    for name in ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']:
        face1 = EquiangularFace(name)
        face2 = SadournyFace(name)

        print(face1.name, face2.name)
        print('Points equal:', np.allclose(face1.to_cartesian(x1s, y1s), face2.to_cartesian(x1s, y1s)))
        print()

    for name in ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']:
        face1 = EquiangularFace(name)
        print(face1.name)
        out1 = face1.covariant_basis(0.5, vals)
        x, y, z = face1.to_cartesian(0.5, vals)
        errorx = abs((np.diff(x) / diff) - out1[1][1:])
        errory = abs((np.diff(y) / diff) - out1[4][1:])
        errorz = abs((np.diff(z) / diff) - out1[7][1:])
        print((abs(errorx).max(), abs(errory).max(), abs(errorz).max()))

        out1 = face1.covariant_basis(-0.5, vals)
        x, y, z = face1.to_cartesian(-0.5, vals)
        errorx = abs((np.diff(x) / diff) - out1[1][1:])
        errory = abs((np.diff(y) / diff) - out1[4][1:])
        errorz = abs((np.diff(z) / diff) - out1[7][1:])
        print((abs(errorx).max(), abs(errory).max(), abs(errorz).max()))

        out1 = face1.covariant_basis(vals, 0.5)
        x, y, z = face1.to_cartesian(vals, 0.5)
        errorx = abs((np.diff(x) / diff) - out1[0][1:])
        errory = abs((np.diff(y) / diff) - out1[3][1:])
        errorz = abs((np.diff(z) / diff) - out1[6][1:])
        print((abs(errorx).max(), abs(errory).max(), abs(errorz).max()))

        out1 = face1.covariant_basis(vals, -0.5)
        x, y, z = face1.to_cartesian(vals, -0.5)
        errorx = abs((np.diff(x) / diff) - out1[0][1:])
        errory = abs((np.diff(y) / diff) - out1[3][1:])
        errorz = abs((np.diff(z) / diff) - out1[6][1:])
        print((abs(errorx).max(), abs(errory).max(), abs(errorz).max()))
        print()



    dxdxi, dxdeta, dxdzeta, dydxi, dydeta, dydzeta, dzdxi, dzdeta, dzdzeta = face1.covariant_basis(0, 0)
    J = dxdxi * (dydeta * dzdzeta - dydzeta * dzdeta)
    J += dydxi * (dzdeta * dxdzeta - dzdzeta * dxdeta)
    J += dzdxi * (dxdeta * dydzeta - dxdzeta * dydeta)
    print('Centre J:', J)

    dxdxi, dxdeta, dxdzeta, dydxi, dydeta, dydzeta, dzdxi, dzdeta, dzdzeta = face1.covariant_basis(0, 0.5)
    J = dxdxi * (dydeta * dzdzeta - dydzeta * dzdeta)
    J += dydxi * (dzdeta * dxdzeta - dzdzeta * dxdeta)
    J += dzdxi * (dxdeta * dydzeta - dxdzeta * dydeta)
    print('Edge J:', J)

    dxdxi, dxdeta, dxdzeta, dydxi, dydeta, dydzeta, dzdxi, dzdeta, dzdzeta = face1.covariant_basis(0.5, 0.5)
    J = dxdxi * (dydeta * dzdzeta - dydzeta * dzdeta)
    J += dydxi * (dzdeta * dxdzeta - dzdzeta * dxdeta)
    J += dzdxi * (dxdeta * dydzeta - dxdzeta * dydeta)
    print('Corner J:', J)
    print(np.sqrt(dxdzeta ** 2 + dydzeta ** 2 + dzdzeta ** 2))
