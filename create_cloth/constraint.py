import numpy as np


class Constraint(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

        self.dist = np.linalg.norm(p1.pos - p2.pos)

    def _satisfy(self):
        v = self.p1.pos - self.p2.pos
        v *= ((1.0 - (self.dist / np.linalg.norm(v))) * 0.8)

        self.p1._move(-v)
        self.p2._move(v)
