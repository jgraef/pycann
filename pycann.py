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

from ctypes import CDLL, c_void_p, c_uint, c_int, c_float, c_char_p


# load library
__libpycann__ = CDLL("./libpycann.so")

# data types
pycann_t = c_void_p
pycann_float_t = c_float

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
                  [l.pycann_get_gamma, pycann_float_t, pycann_t, c_uint],
                  [l.pycann_set_gamma, None, pycann_t, c_uint, pycann_float_t],
                  [l.pycann_get_weight, pycann_float_t, pycann_t, c_uint, c_uint],
                  [l.pycann_set_weight, None, pycann_t, c_uint, c_uint, pycann_float_t],
                  [l.pycann_get_threshold, pycann_float_t, pycann_t, c_uint],
                  [l.pycann_set_threshold, None, pycann_t, c_uint, pycann_float_t],
                  [l.pycann_get_activation, pycann_float_t, pycann_t, c_uint],
                  [l.pycann_set_activation, None, pycann_t, c_uint, pycann_float_t],
                  [l.pycann_get_mod_neuron, c_uint, pycann_t, c_uint],
                  [l.pycann_get_mod_weight, pycann_float_t, pycann_t, c_uint],
                  [l.pycann_set_mod, None, pycann_t, c_uint, c_uint, pycann_float_t],
                  [l.pycann_set_inputs, None, pycann_t, pycann_float_t],
                  [l.pycann_get_outputs, None, pycann_t, pycann_float_t],
                  [l.pycann_get_num_inputs, c_uint, pycann_t],
                  [l.pycann_get_num_outputs, c_uint, pycann_t],
                  [l.pycann_set_random_weights, None, pycann_t, pycann_float_t],
                  [l.pycann_step, None, pycann_t, c_uint],
                  [l.pycann_load_file, pycann_t, c_char_p, c_uint],
                  [l.pycann_save_file, c_int, c_char_p, pycann_t]]

    for p in prototypes:
        p[0].restype = p[1]
        p[0].argtypes = p[2:]

__init_prototypes__(__libpycann__)


class PyCANNException(Exception):
    """ A pyCANN exception. Get error string from C library """
    def __init__(self, errstr = None):
        if (errstr==None):
            self.errstr = __libpycann__.pycann_get_error().value.decode()
        else:
            self.errstr = errstr

    def __str__(self):
        return self.errstr


THREADING = bool(__libpycann__.pycann_is_threading_enabled())

class Network:
    """ Class wrapping C neural network library """
    l = __libpycann__
    net = None

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
        self.memory_usage = self.l.pycann_get_memory_usage(self.net)
        self.num_inputs = self.l.pycann_get_num_inputs(self.net)
        self.num_outputs = self.l.pycann_get_num_outputs(self.net)
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
        self.net = self.l.pycann_load_file(c_char_p(path), num_threads)
        if (not self.net):
            raise PyCANNException()

    def __del__(self):
        """ Deletes the neural network """
        if (self.net!=None):
            self.l.pycann_del(self.net)

    def get_learning_rate(self):
        return self.l.pycann_get_learning_rate(self.net).value

    def set_learning_rate(self, learning_rate):
        self.l.pycann_set_learning_rate(self.net, pycann_float_t(learning_rate))

    def get_gamma(self, i):
        if (i not in range(0, 4)):
            raise AttributeError("Invalid index: "+repr(i))
        return self.l.pycann_get_gamma(self.net, c_uint(i)).value

    def set_gamma(self, i, gamma):
        if (i not in range(0, 4)):
            raise AttributeError("Invalid index: "+repr(i))
        self.l.pycann_set_gamma(self.net, c_uint(i), pycann_float_t(gamma))

    def get_weight(self, i, j):
        return self.l.pycann_get_weight(self.net, c_uint(i), c_uint(j)).value
    
    def set_weight(self, i, j, weight):
        self.l.pycann_set_weight(self.net, c_uint(i), c_uint(j), pycann_float_t(weight))

    def get_threshold(self, i):
        return __libpycann__.pycann_get_threshold(self.net, c_uint(i)).value

    def set_threshold(self, i, threshold):
        self.l.pycann_set_threshold(self.net, c_uint(i), pycann_float_t(threshold))

    def get_activation(self, i):
        return self.l.pycann_get_activation(self.net, c_uint(i)).value

    def set_activation(self, i, activation):
        self.l.pycann_set_activation(self.net, c_uint(i), pycann_float_t(activation))

    def get_mod_connection(self, i):
        n = self.l.pycann_get_mod_neuron(self.net, c_uint(i)).value
        w = self.l.pycann_get_mod_weight(self.net, c_uint(i)).value
        return n, w

    def set_mod_connection(self, i, j, weight):
        self.l.pycann_set_mod(self.net, c_uint(i), c_uint(j), pycann_float_t(weight))

    def set_random_weights(self, connrate = 1.0):
        self.l.pycann_set_random_weights(self.net, pycann_float_t(connrate))

    def step(self, n = 1):
        self.l.pycann_step(self.net, c_uint(n))

__all__ = [Network]


if (__name__=="__main__"):
    import random
    import time
    import pickle

    class Timer:
        def start(self, text = None):
            self.text = text
            if (text!=None):
                print(text, end=": ")
            self.t = self.time()

        def stop(self):
            t = self.time()-self.t
            if (self.text!=None):
                print(str(t)+"s")
            return t

        time = time.time

    #pickle.dump(random.getstate(), open("random.pkl", "wb"))
    random.setstate(pickle.load(open("random.pkl", "rb")))

    T = Timer()
    N = list(map(lambda x: int(10**(0.1*x)), range(10, 41)))
    #N = [1000]
    connrate = 0.75
    steps = 10
    threads = 1
    performance = []

    for neurons in N:
        net = Network(0, neurons, 0, threads)

        # connect neurons
        #T.start("Connecting neurons")
        #for i in range(neurons):
        #    for j in range(int(neurons*connrate)):
        #        net.set_weight(i, j, random.uniform(-1.0, 1.0))
        #T.stop()
        net.set_random_weights(0.75)

        # learning
        #net.set_learning_rate(1.0)
        #T.start("Connecting modularity neurons")
        #for i in range(neurons):
        #    w = random.uniform(-1.0, 1.0)
        #    if (abs(w)<0.05): # 95% neurons have a modularity neuron
        #        w = 0.0
        #    net.set_mod_connection(i, random.randrange(neurons), w)
        #T.stop()

        # run
        T.start()
        net.step(steps)
        t = T.stop()

        # print statistics
        # NOTE we need to get down to 1ms per step to have a realistic simulation.
        connections = int(neurons**2*connrate)
        print()
        print("Statistics:")
        print("Neurons:     "+str(net.size))
        print("Connections: "+str(connections))
        print("             "+str(connections/neurons)+" per neuron")
        print("Threads:     "+str(net.num_threads))
        print("Steps:       "+str(steps))
        print("Time:        "+str(t)+"s")
        print("             "+str(t/steps)+"s/step")
        print("             "+str(t/(neurons*steps))+"s/(neuron*step)")
        print("             "+str(t/(connections*steps))+"s/(connection*step)")
        print("Memory:      "+str(net.memory_usage/1024)+"kB")

        # record performace
        performance.append((neurons, connections, t/steps, net.memory_usage))

    fn = "performance_without_learning"
    pickle.dump(performance, open(fn+".pkl", "wb"))

    f = open(fn+".txt", "wt")
    for n in performance:
        t = n[2]/n[1]
        l = str(n[0])+":\t "+str(t*1000000000)+"ns"
        print(l)
        print(l, file = f)
    f.close()
