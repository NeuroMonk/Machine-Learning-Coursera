import numpy as np
from sigmoid import sigmoid

print sigmoid(12000000) # ans 1
print sigmoid(-25000) # ans 0
print sigmoid(0) # ans 0.5
print sigmoid(np.array([4,5,6])) # ans 0.9820 0.9933 0.9975

