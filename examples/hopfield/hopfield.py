from pycann import *


class Hopfield(Network):
    def __init__(self, n):
        Network.__init__(self, n, 0, n)

        # configure learning
        self.set_learning_rate(0.01)
        self.set_gamma(0, 1.0)
        self.set_gamma(1, -0.1)
        self.set_gamma(2, 0.1)
        self.set_gamma(2, -0.1)

        # cofigure input layer
        for i in range(0, n):
            self.set_threshold(i, 0.0)
            self.set_activation_function(i, "LINEAR")

        # configure output and recurrent layer
        for i in range(n, 2*n):
            self.set_threshold(i, 0.0)
            self.set_activation_function(i, "SIGMOID_STEP")
            
            self.set_weight(i, i, 0.0)
            for j in range(i+1, 2*n):
                w = random.gauss(0.0, 1.0)
                self.set_weight(i, j, w)
                self.set_weight(j, i, w)

    def set_input(self, v):
        self.set_inputs(*v)
