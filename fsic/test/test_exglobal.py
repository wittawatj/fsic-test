"""
Module for testing exglobal module.
"""

__author__ = 'wittawat'

import numpy as np
import matplotlib.pyplot as plt
import fsic.data as data
import fsic.feature as fea
import fsic.ex.exglobal as exglo
import fsic.util as util
import fsic.kernel as kernel
import fsic.indtest as it
import fsic.glo as glo
import scipy.stats as stats

import unittest


class TestFunctions(unittest.TestCase):
    def setUp(self):
        pass 

    def test_parse_prob_label(self):
        def check_label_pat(label, name, ndx, ndy, is_c, is_std, is_h0, n):
            d_gen = prob_label_dict(name, ndx, ndy, is_c, is_std, is_h0, n)
            d_real = exglo.parse_prob_label(label)
            #print d_gen
            #print d_real
            return self.assertEqual(d_gen, d_real)

        check_label_pat('a.n2', 'a', 0, 0, False, False, False, 2)
        check_label_pat('ab_ndx3.n2', 'ab', 3, 0, False, False, False, 2)
        check_label_pat('ab_ndx3_ndy4.n2', 'ab', 3, 4, False, False, False, 2)
        check_label_pat('ab_ndx3_ndy4_std.n2', 'ab', 3, 4, False, True, False, 2)
        check_label_pat('ab_ndx3_ndy4_std_h0.n2', 'ab', 3, 4, False, True, True, 2)

        check_label_pat('a_std.n2', 'a', 0, 0, False, True, False, 2)
        check_label_pat('abc_de_c_std.n2', 'abc_de', 0, 0, True, True, False, 2)
        check_label_pat('a_h0.n2', 'a', 0, 0, False, False, True, 2)
        check_label_pat('a_c_h0.n2', 'a', 0, 0, True, False, True, 2)
        check_label_pat('a_ndx5.n2', 'a', 5, 0, False, False, False, 2)


        check_label_pat('a_ndy_ndy5.n2', 'a_ndy', 0, 5, False, False, False, 2)
        check_label_pat('a_h0_ndy5_std.n2', 'a_h0', 0, 5, False, True, False, 2)

    def tearDown(self):
        pass


def prob_label_dict(name, ndx, ndy, is_c, is_std, is_h0, n):
    return {'name': name, 'ndx': ndx, 'ndy': ndy, 'is_classification':is_c,
            'is_std': is_std, 'is_h0': is_h0, 'n': n}

