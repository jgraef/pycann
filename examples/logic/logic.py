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


# Example: Logic
#
# This examples demonstrates the use of neural networks as logic gates.
# There are pycann files for OR, AND and XOR gates.
# The program loads and tests them.
# The neural networks are also saved in the pycann embedded format. There is
# an examples program for the LEGO Mindstorms NXT to these networks with it.


from pycann import *


def test_gate(gate):
    inputs = [(0.0, 0.0),
              (0.0, 1.0),
              (1.0, 0.0),
              (1.0, 1.0)]
    for i in inputs:
        gate.set_inputs(*i)
        gate.step(1)
        o = gate.get_outputs()
        print(repr(i)+" -> "+repr(o))


gates = {"or": None,
         "and": None,
         "xor": None}

# load networks
for g in gates:
    # Note: '.pcn' is the file extension for the pycann file format.
    # Files with extension '.rnn' are specialized files for embedded devices.
    gates[g] = Network(g+".pcn")

# test networks
for g in gates:
    print("Logic "+g.upper()+":")
    test_gate(gates[g])
    print()
