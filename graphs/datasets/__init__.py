'''Dataset generation functions.

mountain_car : the "Mountain Car" toy domain from reinforcement learning
shapes : various parameterized shapes
swiss_roll : the "Swiss Roll" toy domain from manifold learning
'''
from __future__ import absolute_import

from .mountain_car import mountain_car_trajectories
from .shapes import MobiusStrip, FigureEight, SCurve
from .swiss_roll import swiss_roll
