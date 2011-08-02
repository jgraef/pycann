from pycann import *


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

print("Creating OR gate")
gate = logic_or()
gate.save("or.pcn")
gate.save("or.rnn", "NXT")

print("Creating AND gate")
gate = logic_and()
gate.save("and.pcn")
gate.save("and.rnn", "NXT")

print("Creating XOR gate")
gate = logic_xor()
gate.save("xor.pcn")
gate.save("xor.rnn", "NXT")
