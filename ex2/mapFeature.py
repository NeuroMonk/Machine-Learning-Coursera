from __future__ import division
import numpy as np

# here is a little trick, which i was confused about for a while
# since python and octave has different array indexing 
# octave code would give us 28 dimensional array when
# python meantime would give 22 dimensional array 
# that happens because octave for loop would give as:
# degree = 6;
# for i = 1:degree  -- (1, 2, 3, 4, 5, 6)
#    for j = 0:i    -- (0, 1), (0, 1, 2), ... (0, 1, 2, 3, 4, 5, 6)
#
# when python instead, would give us different indexing result:
# for i in range(degree): -- (0, 1, 2, 3, 4, 5)
#    for j in range(0, i): -- (0), (0, 1), ...., (0, 1, 2, 3, 4, 5)
#
# the code bellow is written to give the same indexing as Octave gives
#

def mapFeature(X1, X2):
    degree = 6
    out = np.ones( (len(X1), 1) )
    for i in range(1, degree + 1):
        for j in range(0, i+1):
            out = np.concatenate ((out, X1**(i-j) * (X2**j)), axis =1 ) 
    return out