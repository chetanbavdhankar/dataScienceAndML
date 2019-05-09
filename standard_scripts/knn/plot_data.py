#!/usr/bin/env python
"""
  Simple script that shows how to load and plot data from
  iris_data.cvs
  Please change the FILENAME variable to point to the
  current location of the input file.

  Author: Wojtek Krzemien
  Date: 15.04 2018
  Usage: python plot_data.py
"""

import numpy as np
import matplotlib.pyplot as plt

FILENAME = 'iris_data.csv'


def loadData(filename, nbOfLinesToSkip=1):
  """
    Args:
      filename(string): name of the file to load.
      nbOfLinesToSkip: first nth number of lines from the filed that will be skipped.
  """
  labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
  data = np.loadtxt(filename, delimiter=',',
                    skiprows=nbOfLinesToSkip,
                    converters={4: lambda label: labels[label]}
                    )
  return data


def plotData(data, columnsToPlot=None, xLabel='x', yLabel='y'):
  """
    Args:
      columnsToPlot(list): list of numbers corresponding to columns that should be plotted.
                           e.g. if we want to plot 3rd column  vs 4th column columnsToPlot should be
                           a list [2,3]. By default it is 1rst vs 2nd so [0,1]

 """
  if not columnsToPlot:
    columnsToPlot = [0, 1]
  xdata = data[:, columnsToPlot[0]]
  ydata = data[:, columnsToPlot[1]]
  plt.scatter(xdata, ydata)
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  plt.show()
  # plt.savefig('plot.png') #save figure to file


def plotDatasets(datasets, dataSetsLabels, columnsToPlot=None, xLabel='x', yLabel='y', ):
  """
    Args:
      columnsToPlot(list): list of numbers corresponding to columns that should be plotted.
                           e.g. if we want to plot 3rd column  vs 4th column columnsToPlot should be
                           a list [2,3]. By default it is 1rst vs 2nd so [0,1]

 """
  colors = ['red','green','blue'] 
  if not columnsToPlot:
    columnsToPlot = [0, 1]
  for d, dlabel, color in zip(datasets, dataSetsLabels, colors):
    xdata = d[:, columnsToPlot[0]]
    ydata = d[:, columnsToPlot[1]]
    plt.scatter(xdata, ydata, label=dlabel, color = color)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
  plt.legend()
  plt.show()
  # plt.savefig('plot.png') #save figure to file


def plotAll(filename):
  dataAll = loadData(filename)
  # dataAll matrix has five columns,
  # the fifth column is the dataset label 0,1 or 2.
  # We want to select only a subset of data corresponding to given label.
  # dataAll[:,4] means select all rows of the third column.
  # dataAll[dataAll[:,4]== 0] means select all records for wich the third column is equal 0.
  dataset1 = dataAll[dataAll[:, 4] == 0]
  dataset2 = dataAll[dataAll[:, 4] == 1]
  dataset3 = dataAll[dataAll[:, 4] == 2]
  #plotData(dataset1, [0, 1], 'sepal_length', 'sepal_width')
  #plotData(dataset2, [0, 1], 'sepal_length', 'sepal_width')
  #plotData(dataset3, [0, 1], 'sepal_length', 'sepal_width')

  # we plot all 3 sets together but with labels class labels
  datasets = [dataset1, dataset2, dataset3]
  dLabels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
  plotDatasets(datasets, dLabels, [0, 1], 'sepal_length', 'sepal_width')


def plotBoundaries(x, y, predictValues, k):
  """
    Based on http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
    Args:
      x(list): of feature vectors which form our training sample e.g. [[0,1], [10,10]] -> two points [0,1] and [10,10] with two features each
      y(list): of class labels which assing given feature vector (point) to given class e.g. [0,1] the first point belongs to class 0 and the second to 1
      predictList(function): function that returns a list of predicted values, with the following signature: predictList(xObjects, Xtrain, Ytrain, k)
  """
  from matplotlib.colors import ListedColormap
  # Plot the decision boundary. For that, we will assign a color to each
  # point in the mesh [x_min, x_max]x[y_min, y_max].
  # Create color maps
  cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
  cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
  h = .02  # step size in the mesh
  #h = .005  # step size in the mesh
  X = np.array(x)
  Y = np.array(y)
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
  z= predictValues(np.c_[xx.ravel(), yy.ravel()].tolist(), x, y, k)
  Z = np.array(predictValues(np.c_[xx.ravel(), yy.ravel()].tolist(), x, y, k))
  # ravel() - to flatten array eg.[[1,2], [2,3]]  --> [1,2,2,3]

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.figure()
  plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

  # Plot also the training points
  plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
              edgecolor='k', s=20)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.show()
  # plt.savefig('plot.png') #save figure to file

def plotErrors(x, train,valid):
  plt.plot(x, train, label='training error')
  plt.plot(x, valid, label='test error')
  plt.xlabel('k')
  plt.legend()
  plt.show()
  plt.savefig('errors.png') #save figure to file
   

def main():
  plotAll(FILENAME)


if __name__ == "__main__":
  main()
