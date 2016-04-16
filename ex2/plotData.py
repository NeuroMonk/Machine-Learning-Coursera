import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):

    pos = X[np.squeeze(y == 1), :]  
    neg = X[np.squeeze(y == 0), :]  
    plt.plot(pos[:, 0], pos[:, 1], 'k+')
    plt.plot(neg[:, 0], neg[:, 1], 'ko')





