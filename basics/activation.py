# -*- coding: utf-8 -*-
# @Author: aka
# @Date:   2019-03-16 20:17:28
# @Last Modified by:   aka
# @Last Modified time: 2019-03-16 21:22:39
# @Email: tenag_hirmb@hotmail.com
import numpy as np


def sgn(net):
    if net >= 0:
        return 1
    else:
        return -1


def bipolar_sigmoid(net):
    e_to_the_negative_xth_power = pow(np.e, -net)
    return (1 - e_to_the_negative_xth_power) / (1 + e_to_the_negative_xth_power)


def bipolar_sigmoid_derivative(net):
    out = bipolar_sigmoid(net)
    return 0.5 * (1 - out * out)
