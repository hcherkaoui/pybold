""" Test the padd module.
"""
import unittest
import numpy as np
from pybold.padding import padd, custom_padd, unpadd


class TestPadding(unittest.TestCase):
    def test_power_2_padding_length(self):
        """ Test if the output of next_power_of_2_padding is a power of two.
        """
        for N in [100, 128, 200, 256, 250, 300, 500, 600, 1000]:
            signal = np.random.randn(N)
            padded_signal = custom_padd(signal)
            is_power_two = np.log2(len(padded_signal)).is_integer()
            assert(is_power_two)

    def test_zero_padd_unpadd(self):
        """ Test if the padded-unpadded array is equal to the orig.
        """
        for N in [100, 128, 200, 256, 250, 300, 500, 600, 1000]:
            s = np.random.randn(N)
            for p in [10, 20, 214]:
                for paddtype in ['left', 'right']:
                    padded_s = padd(s, p=p, paddtype=paddtype)
                    test_s = unpadd(padded_s, p=p, paddtype=paddtype)
                    if not np.allclose(s, test_s):
                        msg = ("test 'np.allclose(s, test_s)' "
                               "with p={0}, paddtype={1}".format(p, paddtype))
                        raise(AssertionError(msg))

            paddtype = 'center'
            for p in [(5, 10), (20, 20), (214, 145)]:
                padded_s = padd(s, p=p, paddtype=paddtype)
                test_s = unpadd(padded_s, p=p, paddtype=paddtype)
                if not np.allclose(s, test_s, atol=1.0e-7):
                    msg = ("test 'np.allclose(s, test_s)' "
                           "with p={0}".format(p))
                    raise(AssertionError(msg))

    def test_zero_mirror_zero_padd_unpadd(self):
        """ Test if the padded-unpadded array is equal to the orig.
        """
        for N in [100, 128, 200, 256, 250, 300, 500, 600, 900, 1000]:
            signal = np.random.randn(N)
            padded_signal, p = custom_padd(signal)
            test_signal = unpadd(padded_signal, p)
            assert(np.allclose(signal, test_signal))


if __name__ == '__main__':
    unittest.main()
