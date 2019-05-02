"""
  Bivariate normal distribution
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

means = [1,3]
covariance = [[5,0.3],[0.3,2]]
samples = int(1e5)

x,y = np.random.multivariate_normal(means, covariance, samples, 'raise').T

#Plottting
plt.hist2d(x, y, bins=(100, 100), cmap=plt.cm.jet)
plt.axis('equal')
plt.show()

sns.jointplot(x, y, kind='kde', color="skyblue")
plt.show()
