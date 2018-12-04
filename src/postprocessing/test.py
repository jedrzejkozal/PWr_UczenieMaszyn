import numpy as np
import scikit_posthocs as sp


x = np.array([[31,27,24],[31,28,31],[45,29,46],[21,18,48],[42,36,46],[32,17,40]])
print(sp.posthoc_nemenyi_friedman(x))
