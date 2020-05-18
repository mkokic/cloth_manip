import numpy as np
from particle import Particle
from constraint import Constraint


class Cloth(object):
    def __init__(self, width, height, pW, pH, mass, damp):
        self.width = width
        self.height = height
        self.pW = pW
        self.pH = pH
        self.mass = mass
        self.damp = damp

        self.pSepX = width / pW
        self.pSepY = height / pH

        self.pN = pW * pH
        self.pM = self.pN / float(mass)
        self.pD = 1.0 - damp

        self._create_particles()
        self._create_constraints()

        self.particles[int(self.pN / 2 + self.pW / 2)].force[2] -= 1

    def _create_particles(self):
        self.particles = []
        for y in range(self.pH):
            for x in range(self.pW):
                # + np.array([-50., 20., 25.])
                self.particles.append(Particle(
                    np.array([float(x * self.pSepX), float(- y * self.pSepY), 0.0]),
                    self.pM,
                    self.pD))

    def _create_constraints(self):
        self.constraints = []
        for k in range(self.pN):
            leftCol = k % self.pW == 0
            leftCols = leftCol or (k - 1) % self.pW == 0
            rightCol = (k + 1) % self.pW == 0
            topRow = k < self.pW
            topRows = k < self.pW * 2

            if not topRow:
                self.constraints.append(Constraint(self.particles[k], self.particles[k - self.pW]))
            if not topRows:
                self.constraints.append(Constraint(self.particles[k], self.particles[k - (self.pW * 2)]))
            if not leftCol:
                self.constraints.append(Constraint(self.particles[k], self.particles[k - 1]))
            if not leftCols:
                self.constraints.append(Constraint(self.particles[k], self.particles[k - 2]))
            if not topRow and not leftCol:
                self.constraints.append(Constraint(self.particles[k], self.particles[k - 1 - self.pW]))
            if not topRow and not rightCol:
                self.constraints.append(Constraint(self.particles[k], self.particles[k + 1 - self.pW]))

    def _step(self, timeStep):
        # gravity
        for i in range(self.pN):
            self.particles[i].force += np.array([0, -0.3, 0]) * timeStep

        # satisfy constraints
        for n in range(20):
            for i in range(len(self.constraints)):
                self.constraints[i]._satisfy()

        # apply forces
        for i in range(self.pN):
            self.particles[i]._step(timeStep)

        return np.array([self.particles[i].pos for i in range(self.pN)]).reshape(self.pN, 3)
