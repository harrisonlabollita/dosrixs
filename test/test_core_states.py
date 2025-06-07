import unittest
import numpy as np
from dosrixs import build_core_states
from dosrixs.ylmexpansion import YlmExpansion

class TestCoreStates(unittest.TestCase):

    def test_l2_edge(self):
        ref_states:list[YlmExpansion] = [ 
            YlmExpansion(1,data={(-1,0): -complex(np.sqrt(2)/np.sqrt(3)), (0,1): complex(1/np.sqrt(3))}),
            YlmExpansion(1,data={(0,0): complex(-1/np.sqrt(3)), (1,1): complex(np.sqrt(2)/np.sqrt(3))}),
        ]
        core_states = build_core_states('L2')
        for (state, ref) in zip(core_states, ref_states): 
            for (m,s,c) in state: self.assertAlmostEqual(state[(m,s)], ref[(m,s)])

if __name__ == "__main__":
    unittest.main(verbosity=2)
