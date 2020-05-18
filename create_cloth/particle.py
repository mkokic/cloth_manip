from IPython import embed
import numpy as np


class Particle(object):
    def __init__(self, pos, mass, damp):
        self.pos = pos
        self.lastPos = pos
        self.mass = mass
        self.damp = damp

        self.force = [0, 0, 0]
        self.isStatic = False

    def _move(self, delta):
        if self.isStatic:
            pass
        else:
            self.pos += delta

    def _step(self, timeStep):
        if self.isStatic:
            pass
        else:
            temp = self.pos

            self.pos = self.pos + (
                (self.pos - self.lastPos) * self.damp + (np.array(self.force) / self.mass) * (timeStep / 100.))

            self.lastPos = temp
            self.force = [0, 0, 0]
