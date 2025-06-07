import unittest
from dosrixs.ylmexpansion import YlmExpansion

class TestYlmExpansion(unittest.TestCase):

    def test_multiplication(self):
        a = YlmExpansion(2, data={(2,0): 1.0, (2,1) : 1.0 })
        b = YlmExpansion(2, data={(2,0): 2.0, (2,1) : 2.0 })
        c = YlmExpansion(2, data={(2,0): 0.5, (2,1) : 0.5 })

        self.assertEqual(2*a, b)
        self.assertEqual(a, 0.5*b)
        self.assertEqual(a*0.5, c)

    def test_division(self):
        a = YlmExpansion(2, data={(2,0): 1.0, (2,1) : 1.0 })
        b = YlmExpansion(2, data={(2,0): 2.0, (2,1) : 2.0 })

        self.assertEqual(a, b/2)

    def test_inplace_division(self):
        a = YlmExpansion(2, data={(2,0): 1.0, (2,1) : 1.0 })
        b = YlmExpansion(2, data={(2,0): 2.0, (2,1) : 2.0 })
        b /= 2
        self.assertEqual(a, b)

    def test_negative(self):
        a = YlmExpansion(2, data={(2,0): 1.0, (2,1) : 1.0 })
        b = YlmExpansion(2, data={(2,0): -1.0, (2,1) : -1.0 })

        self.assertEqual(-a, b)
        self.assertEqual(-1*a, b)
        self.assertEqual(-(-1*a), -b)

    def test_addition_same_l(self):
        a = YlmExpansion(2, data={(2,0): 0.5, (2,1) :  1.0 })
        b = YlmExpansion(2, data={(2,0): 0.5, (2,1) : -1.0 })
        c = YlmExpansion(2, data={(2,0): 1.0, (2,1) :  0.0 })
        d = YlmExpansion(2, data={(2,0): 1.5, (2,1) :  1.0 })

        self.assertEqual(a + b, c)
        self.assertEqual(a + c, d)

    def test_addition_diff_l(self):
        a = YlmExpansion(2, data={(2,0): 0.5, (2,1) :  1.0 })
        b = YlmExpansion(1, data={(2,0): 0.5, (2,1) : -1.0 })
        with self.assertRaises(ValueError): a + b  # type: ignore

    def test_addition_with_const(self):
        a = YlmExpansion(2, data={(2,0): 0.5, (2,1) :  1.0 })
        with self.assertRaises(TypeError): a + 2.0 # type: ignore

    def test_subtraction(self):
        a = YlmExpansion(2, data={(2,0): 0.5, (2,1) :  1.0 })
        b = YlmExpansion(2, data={(2,0): 0.5, (2,1) : -1.0 })
        c = YlmExpansion(2, data={(2,0): 0.0, (2,1) :  2.0 })
        d = YlmExpansion(2, data={(2,0): -0.5, (2,1) :  1.0 })

        self.assertEqual(c - a, d)

    def test_equality(self):
        a = YlmExpansion(2, data={(2,0): 0.5, (2,1) :  1.0 })
        b = YlmExpansion(1, data={(2,0): 0.5, (2,1) :  1.0 })
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, 0.5*(2*b))
        with self.assertRaises(ValueError): a == 2.0 # type: ignore
        

    def test_repr(self):
        a = YlmExpansion(1, data={(1,0) : 1.0 + 0.j, (-1, 0): 0.0})
        self.assertIn("|1,1>", repr(a))

    def test_getitem(self):
        a = YlmExpansion(2, data = {(2,0): 1.0})
        self.assertEqual(a[(2,0)], 1.0)
        self.assertEqual(a[(1,0)], 0.0)

    def test_iter(self):
        a = YlmExpansion(2, data = {(2,0): 1.0, (1,1) : 2.0})
        expected = [(2, 0, 1.0), (1, 1, 2)]
        self.assertCountEqual(list(a), expected)

    def test_magnetic_quantum_numbers(self):
        a = YlmExpansion(2, data = {(2,1): 1.0, (-2,0) : 2.0, (0, 0): 1.0})
        self.assertCountEqual(a.magnetic_quantum_numbers, [2, -2, 0] )

if __name__ == "__main__":
    unittest.main(verbosity=2)
