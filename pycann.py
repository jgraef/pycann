# pycann - Neural network library
# A Python/C hybrid for fast neural networks in Python
# Copyright (C) 2010  Janosch Gräf <janosch.graef@gmx.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ctypes import CDLL, c_void_p, c_uint, c_int, c_float, c_char_p, POINTER, Structure


__all__ = ["PyCANNException", "Network"]


# utility function to check if variables are numeric
def is_numeric(*x):
    for y in x:
        try:
            x+1
        except TypeError:
            return False
    return True


# load library
__libpycann__ = CDLL("/usr/local/lib/libpycann.so")


# data types
pycann_t = c_void_p
pycann_float_t = c_float
pycann_activation_function_t = c_uint
pycann_embedded_format_t = c_uint


# load function prototypes
def __init_prototypes__(l):
    prototypes = [[l.pycann_get_error, c_char_p],
                  [l.pycann_reset_error, None],
                  [l.pycann_new, pycann_t, c_uint, c_uint, c_uint, c_uint],
                  [l.pycann_del, None, pycann_t],
                  [l.pycann_is_threading_enabled, c_uint],
                  [l.pycann_get_memory_usage, c_uint, pycann_t],
                  [l.pycann_get_size, c_uint, pycann_t],
                  [l.pycann_get_num_threads, c_uint, pycann_t],
                  [l.pycann_get_learning_rate, pycann_float_t, pycann_t],
                  [l.pycann_set_learning_rate, None, pycann_t, pycann_float_t],
                  [l.pycann_get_gamma, pycann_float_t, pycann_t, c_uint, POINTER(pycann_float_t)],
                  [l.pycann_set_gamma, None, pycann_t, c_uint, POINTER(pycann_float_t)],
                  [l.pycann_get_weight, pycann_float_t, pycann_t, c_uint, c_uint],
                  [l.pycann_set_weight, None, pycann_t, c_uint, c_uint, pycann_float_t],
                  [l.pycann_get_threshold, pycann_float_t, pycann_t, c_uint],
                  [l.pycann_set_threshold, None, pycann_t, c_uint, pycann_float_t],
                  [l.pycann_get_activation, pycann_float_t, pycann_t, c_uint],
                  [l.pycann_set_activation, None, pycann_t, c_uint, pycann_float_t],
                  [l.pycann_get_activation_function, pycann_activation_function_t, pycann_t, c_uint],
                  [l.pycann_set_activation_function, None, pycann_t, c_uint, pycann_activation_function_t],               
                  [l.pycann_get_mod_neuron, c_uint, pycann_t, c_uint],
                  [l.pycann_get_mod_weight, pycann_float_t, pycann_t, c_uint],
                  [l.pycann_set_mod, None, pycann_t, c_uint, c_uint, pycann_float_t],
                  [l.pycann_set_inputs, None, pycann_t, POINTER(pycann_float_t)],
                  [l.pycann_get_outputs, None, pycann_t, POINTER(pycann_float_t)],
                  [l.pycann_get_num_inputs, c_uint, pycann_t],
                  [l.pycann_get_num_outputs, c_uint, pycann_t],
                  [l.pycann_set_random_weights, None, pycann_t, pycann_float_t],
                  [l.pycann_step, None, pycann_t, c_uint],
                  [l.pycann_load_file, pycann_t, c_char_p, c_uint],
                  [l.pycann_save_file, c_int, c_char_p, pycann_t],
                  [l.pycann_export_embedded, c_int, c_char_p, pycann_t, pycann_embedded_format_t]]

    for p in prototypes:
        p[0].restype = p[1]
        p[0].argtypes = p[2:]


__init_prototypes__(__libpycann__)


# THREADING constant
THREADING = bool(__libpycann__.pycann_is_threading_enabled())


class PyCANNException(Exception):
    """ A pyCANN exception. Get error string from C library """
    def __init__(self, errstr = None):
        if (errstr==None):
            self.errstr = __libpycann__.pycann_get_error().decode()
        else:
            self.errstr = errstr

    def __str__(self):
        return self.errstr


class Network:
    """ Class wrapping C neural network library """
    l = __libpycann__
    net = None
    embedded_formats = {None: 0,
                        "NXT": 1}
    activation_functions = {None: 0,
                            "SIGMOID_STEP":   1,
                            "SIGMOID_EXP":    2,
                            "SIGMOID_APPROX": 3,
                            "LINEAR":         4}

    def __init__(self, *args):
        """ Contructor:
pycann.Network(num_inputs, num_interneurons, num_outputs [, num_threads])
pycann.Network(path [, num_threads]) """

        # check if threading is supported
        if (not THREADING):
            num_threads = 1

        # check whether to create or load
        num_args = len(args)
        if (num_args in (3, 4)):
            self.init_new(*args)
        elif (num_args in (1, 2)):
            self.init_load(*args)
        else:
            raise AttributeError("Unknown constructor with "+str(num_args)+" arguments.")

        # these values will never change, so we can hold them here too
        self.size = self.l.pycann_get_size(self.net)
        self.num_inputs = self.l.pycann_get_num_inputs(self.net)
        self.num_outputs = self.l.pycann_get_num_outputs(self.net)
        self.memory_usage = self.l.pycann_get_memory_usage(self.net)
        self.num_threads = self.l.pycann_get_num_threads(self.net)
        
    def init_new(self, num_inputs, num_interneurons, num_outputs, num_threads = 1):
        """ Creates a new neural network """
        
        # create neural network
        size = num_inputs + num_interneurons + num_outputs
        self.net = self.l.pycann_new(size, num_inputs, num_outputs, num_threads)
        if (not self.net):
            raise PyCANNException()

    def init_load(self, path, num_threads = 1):
        """ Loads a neural network from file """
        # load neural network from file
        self.net = self.l.pycann_load_file(path, num_threads)
        if (not self.net):
            raise PyCANNException()

    def __del__(self):
        """ Deletes the neural network """
        if (self.net!=None):
            self.l.pycann_del(self.net)

    def get_learning_rate(self):
        return self.l.pycann_get_learning_rate(self.net)

    def set_learning_rate(self, learning_rate):
        self.l.pycann_set_learning_rate(self.net, learning_rate)

    def get_gamma(self, i):
        gamma = (4*pycann_float_t)()
        self.l.pycann_get_gamma(self.net, i, gamma)
        return tuple(gamma)

    def set_gamma(self, i, gamma):
        if (len(gamma)==4 and is_numeric(*gamma)):
            TypeError("Gamma must be a 4-tuple of numbers.")
        gamma = (4*pycann_float_t)(*gamma)
        self.l.pycann_set_gamma(self.net, i, gamma)

    def get_weight(self, i, j):
        return self.l.pycann_get_weight(self.net, j, i)
    
    def set_weight(self, i, j, weight):
        self.l.pycann_set_weight(self.net, j, i, weight)

    def get_threshold(self, i):
        return __libpycann__.pycann_get_threshold(self.net, i)

    def set_threshold(self, i, threshold):
        self.l.pycann_set_threshold(self.net, i, threshold)

    def get_activation(self, i):
        return self.l.pycann_get_activation(self.net, i)

    def set_activation(self, i, activation):
        self.l.pycann_set_activation(self.net, i, activation)

    def get_activation_function(self, i):
        a = self.l.pycann_get_activation_function(self.net, i)
        for n in self.activation_functions:
            if (a==self.activation_functions[n]):
                return n
        return None

    def set_activation_function(self, i, activation_function = "SIGMOID_STEP"):
        self.l.pycann_set_activation_function(self.net, i, self.activation_functions[activation_function.upper()])

    def get_mod_connection(self, i):
        n = self.l.pycann_get_mod_neuron(self.net, i)
        w = self.l.pycann_get_mod_weight(self.net, i)
        return n, w

    def set_mod_connection(self, i, j, weight):
        self.l.pycann_set_mod(self.net, i, j, weight)

    def set_random_weights(self, connrate = 1.0):
        self.l.pycann_set_random_weights(self.net, connrate)

    def step(self, n = 1):
        self.l.pycann_step(self.net, n)

    def save(self, path, embedded = None):
        if (embedded==None):
            ret = self.l.pycann_save_file(path, self.net)
        else:
            ret = self.l.pycann_export_embedded(path, self.net, self.embedded_formats[embedded])
        if (ret==-1):
            raise PyCANNException()

    def set_inputs(self, *v):
        if (len(v)!=self.num_inputs):
            raise PyCANNException("Network has "+str(self.num_inputs)+" inputs, but only "+str(len(v))+" given")
        inputs = (self.num_inputs*c_float)(*v)
        self.l.pycann_set_inputs(self.net, inputs)

    def get_outputs(self):
        outputs = (self.num_outputs*c_float)()
        self.l.pycann_get_outputs(self.net, outputs)
        return tuple(outputs)


def logic_or():
    net = Network(2, 0, 1)
    net.set_threshold(0, 0.5)
    net.set_threshold(1, 0.5)
    net.set_threshold(2, 1.0)
    net.set_weight(0, 2, 1.0)
    net.set_weight(1, 2, 1.0)
    return net

def logic_and():
    net = Network(2, 0, 1)
    net.set_threshold(0, 0.5)
    net.set_threshold(1, 0.5)
    net.set_threshold(2, 2.0)
    net.set_weight(0, 2, 1.0)
    net.set_weight(1, 2, 1.0)
    return net

def logic_xor():
    net = Network(2, 1, 1)
    net.set_threshold(0, 0.5)
    net.set_threshold(1, 0.5)
    net.set_threshold(2, 2.0)
    net.set_threshold(3, 1.0)
    net.set_weight(0, 2, 1.0)
    net.set_weight(0, 3, 1.0)
    net.set_weight(1, 2, 1.0)
    net.set_weight(1, 3, 1.0)
    net.set_weight(2, 3, -2.0)
    return net

def test_logic_gate(gate):
    inputs = [(0.0, 0.0),
              (0.0, 1.0),
              (1.0, 0.0),
              (1.0, 1.0)]
    for i in inputs:
        gate.set_inputs(*i)
        gate.step(1)
        o = gate.get_outputs()
        print(repr(i)+" -> "+repr(o))        


if (__name__=="__main__"):
    print("Logic OR:")
    gate = logic_or()
    test_logic_gate(gate)

    print("Logic AND:")
    gate = logic_and()
    test_logic_gate(gate)

    print("Logic XOR:")
    gate = logic_xor()
    test_logic_gate(gate)
