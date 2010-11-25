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
 
from pycann import *

# exponential neuron numbers
# N = list(map(lambda x: int(10**(0.1*x)), range(10, 41)))

class Timer:
    def start(self):
        if (text!=None):
            print(text, end=": ")
        self.t = self.time()

    def stop(self):
        return self.time()-self.t


