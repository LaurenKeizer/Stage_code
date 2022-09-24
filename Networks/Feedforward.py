from models import currentmodel
from brian2 import *
from foundationsmik.helpers import scale_input_theory


class FeedForward:

    def __init__(self, weights, input_theory, scales, dt):
        self.dt = dt
        self.dict = {}
        self.input_theory = input_theory
        self.make_input(scales)
        self.make_neurons()
        self.weights = weights



    def make_forward(self):


    self.network = network



    def make_backward(self):

        self.network = network

    def make_neurons(self):


    def make_input(self, scales):
        inj_input = []
        for scale in scales:
            self.inj_input = scale_input_theory(self.input_theory, 0, scale, self.dt)
        self.inj_input = inj_input

    def make_neurons(self):
        PC_ref = Barrel_PC(dt=dt, inj=inj_input[0])
        PC     = Barrel_PC(dt=dt, inj=inj_input[0])
        IN     = Barrel_IN(dt=dt, inj=inj_input[1])

        self.dict['PC_ref'], self.dict['PC'], self.dict['IN'] = PC_ref, PC, IN


def get_network(self):
        return self.network

class FeedBackward:
    def __init__(self, weights, input_theory, scales, dt):
        self.input_theory = input_theory
        self.make_input(scales)
        self.make_neurons()
        self.weights = weights
        self.dt = dt

    def make_forward(self):
        # this function makes a feed forward model

        PC = Barrel_PC(dt=dt, )
        PC2 = Barrel_PC(dt=dt)
        IN = Barrel_IN(dt=dt, Input=False)

        PC_neuron2 = PC2.getNeurongroup()
        PC_neuron = PC.getNeurongroup()
        IN_neuron = IN.getNeurongroup()

    self.network = network

    def make_backward(self):
        self.network = network

    def make_neurons(self):

    def make_input(self, scales):
        inj_input = []
        for scale in scales:
            self.inj_input = scale_input_theory(self.input_theory, 0, scale, self.dt)
        self.inj_input = inj_input

    def make_neurons(self):
        PC_ref = Barrel_PC(dt=dt)
        PC = Barrel_PC(dt=dt, )
        IN = Barrel_IN(dt=dt, Input=False)

    def get_network(self):
        return self.network