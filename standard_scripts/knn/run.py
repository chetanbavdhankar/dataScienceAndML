#!/usr/bin/env python
"""
  Template for implementation of the k-nearest neighbours classification.
  The code serves only for educational purposes so it
  is not optimized in view of speed or numerical calculations.
  For the real problems you should always use algorithms.
  from known,  well tested libraries, wherever possible.
  Author: Wojtek Krzemien
  Date: 10.04 2018
  Usage: python run.py
"""

import numpy as np
from plot_data import plotAll, loadData, plotBoundaries, plotErrors
from distance import distance
from getNeighbours import getNeighbours
from getKNNeighbours import getKNNeighbours
from majorityVote import majorityVote
from predict import predict
from predict import predictList
from error import meanSquaredError



def divideData(dataSet, fraction):

  nbTotal = dataSet.shape[0]
  nbTrain = int(nbTotal*fraction)
  return (dataSet[:nbTrain], dataSet[nbTrain:])



def runTests():
  # some tests of distance function
  np.testing.assert_almost_equal(distance([1, 1], [1, 1]), 0)
  np.testing.assert_almost_equal(distance([2, 0, 1], [5, 0, 1]), 3)
  np.testing.assert_almost_equal(distance([0, 0], [2, 2]), np.sqrt(8))

  # some tests of getNeighbours function
  Xtrain = [[2, 0], [0, 0],  [1, 0]]
  Ytrain = [1, 0, 1]
  x = [-1, 0]  # with respect to x we calculate the distance
  result = getNeighbours(x, Xtrain, Ytrain)
  expected = ((1, [0, 0], 0), (2, [1, 0], 1), (3, [2, 0], 1))
  np.testing.assert_equal(result, expected)

  # some tests of getKNNeighbours function
  data = [[1, [0, 0], 0], [2, [1, 0], 1], [3, [2, 0], 1]]
  np.testing.assert_equal(getKNNeighbours(data,  k=1), [[1, [0, 0], 0]])
  np.testing.assert_equal(getKNNeighbours(data,  k=2), [
                          [1, [0, 0], 0], [2, [1, 0], 1]])

  # some tests of majorityVote function
  data = [[1, [0, 0], 0], [2, [1, 0], 1], [3, [2, 0], 1]]
  np.testing.assert_equal(majorityVote(data), 2./3.)

  # some tests of predict function
  xTrain = [[0, 0], [1, 0], [2, 0]]
  yTrain = [0, 1, 1]
  x = [-1, 0]
  k = 1
  np.testing.assert_almost_equal(predict(x, xTrain, yTrain, k), 0)
  k = 2
  np.testing.assert_almost_equal(predict(x, xTrain, yTrain, k), 1)
  k = 3
  np.testing.assert_almost_equal(predict(x, xTrain, yTrain, k), 1)

  # some tests of predictList function
  xTrain = [[0, 0], [1, 0], [2, 0]]
  yTrain = [0, 1, 1]
  xToClassify = [[-1, 0], [3, 0]]
  k = 1
  np.testing.assert_almost_equal(predictList(
      xToClassify, xTrain, yTrain, k), [0., 1.])


def main():
  FILENAME = 'iris_data.csv'
  # First we plot the data
  plotAll(FILENAME)
  # Run all tests
  runTests()

  # We use our model to classify iris data
  dataAll = loadData(FILENAME)

  # dataAll matrix has five columns,
  # the fifth column is the dataset label 0,1 or 2.
  # We want to select only a subset of data corresponding to given label.
  # dataAll[:,4] means select all rows of the third column.
  # dataAll[dataAll[:,4]== 0] means select all records for which the third column is equal 0.
  #dataset1 = dataAll[dataAll[:, 4] == 0]
  #dataset2 = dataAll[dataAll[:, 4] == 1]
  #dataset3 = dataAll[dataAll[:, 4] == 2]

  classLabels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
  columnLabels = ['sepal_length', 'sepal_width',
                  'petal_length', 'petal_width', 'species']
  # First let's try  Iris-setosa vs Iris-versicolor, two features sepal-length vs sepal-width
  # we get rid of the third class Iris-viriginica
  dataWithoutVirignica = dataAll[dataAll[:, 4] != 2]
  # we leave only sepal_length, sepal_width and class label columns
  dataSepal = dataWithoutVirignica[:, [0, 1, 4]]

  # we can shuffle rows - if not they will be always first 0 then 1
  np.random.shuffle(dataSepal)

  # we  divide the content into the training and validation set
  fraction = 0.8
  (trainingSet, validationSet) = divideData(dataSepal, fraction)

  # changing array to standard lists
  xTrain = trainingSet[:, [0, 1]].tolist()
  yTrain = trainingSet[:, 2].tolist()
  xValid = validationSet[:, [0, 1]].tolist()
  yValid = validationSet[:, 2].tolist()

  #uncomment it if you want to plot boundaries
  #plotBoundaries(xTrain, yTrain, predictList, 1)
  #plotBoundaries(xTrain, yTrain, predictList, 3)
  #plotBoundaries(xTrain, yTrain, predictList, 11)

  if fraction < 1:
    #we calculate training error and validation error
    trainError=[]
    validError=[]
    kRange =xrange(1,80)
    trainPredictions = predictList(xTrain, xTrain, yTrain, 1)
    for k in kRange:
      trainPredictions = predictList(xTrain, xTrain, yTrain, k)
      validationPredictions = predictList(xValid, xTrain, yTrain, k)
      trainError.append(meanSquaredError(trainPredictions,yTrain))
      validError.append(meanSquaredError(validationPredictions,yValid))
    plotErrors(kRange, trainError,validError)  

if __name__ == "__main__":
  main()
