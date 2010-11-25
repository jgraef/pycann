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

import time
import random
from os.path import getsize as filesize
from optparse import OptionParser

from pycann import *


class Timer:
    time = time.time
    
    def start(self):
        self.t = self.time()

    def stop(self):
        return self.time()-self.t


class Benchmark:
    def __init__(self, netfiles, **options):
        self.netfiles = netfiles
        self.steps = options.get("steps", 1)
        self.threads = options.get("threads", 1)
        self.verbose = options.get("verbose", False)
        self.record = {}
        self.timer = Timer()

    def test_net(self, i):
        file = self.netfiles[i]
        if (self.verbose):
            print("Testing net file: "+str(file))
            
        fsize = filesize(file)
        self.timer.start()
        net = Network(file)
        tl = self.timer.stop()
        
        self.timer.start()
        net.step(self.steps)
        tr = self.timer.stop()/self.steps

        if (self.verbose):
            print("  Neurons:          "+str(net.size))
            print("  File size:        "+str(fsize))
            print("  Load time:        "+str(tl))
            print("  Execution time:   "+str(tr))
            print("  Execution memory: "+str(net.memory_usage))

        record = (tl, filesize, tr, net.memory_usage)
        self.record[net.size] = record

        del net
        return record

    def test_all(self):
        for i in range(len(self.netfiles)):
            self.test_net(i)
        return self.record


def create_test_network(output_file, neurons, connrate = 0.75, learning = False, verbose = False):
    num_inputs = int(neurons/3)
    num_inter = neurons-num_inputs
    #num_inputs = 0
    #num_inter = neurons
    if (verbose):
        print("Output file:     "+output_file)
        print("Neurons:         "+str(neurons))
        print("Inputs:          "+str(num_inputs))
        print("Interneurons:    "+str(num_inter))
        print("Outputs:         0")
        print("Connection rate: "+str(connrate))
        print("Learning:        "+str(learning))
    net = Network(num_inputs, num_inter, 0)
    net.set_random_weights(connrate)
    if (learning):
        net.set_learning_rate(1.0)
        for i in range(neurons):
            w = random.uniform(-1.0, 1.0)
            if (abs(w)<0.05): # 95% neurons have a modularity neuron
                w = 0.0
            net.set_mod_connection(i, random.randrange(neurons), w)
    net.save(output_file)


if (__name__=="__main__"):
    parser = OptionParser()
    parser.add_option("-s", "--steps", dest="steps", default="1", help="Define number of steps for each test")
    parser.add_option("-t", "--threads", dest="threads", default="1", help="Define number of threads")
    parser.add_option("-C", "--create", dest="create", action="store_true", default=False, help="Create network instead of testing networks")
    parser.add_option("-o", "--output", dest="output", default="benchmark.txt", help="Specify output file", metavar="FILE")
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print out more information")
    parser.add_option("-c", "--connrate", dest="connrate", default="0.75", help="Specify connection rate", metavar="RATE")
    parser.add_option("-l", "--learning", dest="learning", action="store_true", default=False, help="Use learning (when creating new network)")
    (options, args) = parser.parse_args()
    if (options.create):
        N = [int(10**(0.1*i)) for i in range(10, 41)]
        for neurons in N:
            filename = "testnets/net_"+str(neurons)+".net"
            if (options.verbose):
                print("Creating network with "+str(neurons)+" neurons: "+filename)
            create_test_network(filename, neurons, float(options.connrate), options.learning, options.verbose)
    else:
        b = Benchmark(args, steps = int(options.steps), threads = int(options.threads), verbose = options.verbose)
        results = b.test_all()
        f = open(options.output, "w")
        for r in results:
            l = str(r)+": "+repr(results[r][3])
            print(l)
            print(l, file = f)
        f.close()
