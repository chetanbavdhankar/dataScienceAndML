#!/usr/bin/env python
"""
  The code serves only for educational purposes so it
  is not optimized in view of speed or numerical calculations.
  For the real problems you should always use algorithms.
  from known,  well tested libraries, wherever possible.
  Author: Wojtek Krzemien
  Date: 10.04 2018
"""
from operator import itemgetter
from distance import distance


def getNeighbours(x, Xtrain, Ytrain, metric=distance):
  """
    Function should return neighbours of x, sorted by distance from x. 
    Xtrain and Ytrain form the training set e.g. Xtrain= [[0,1], [10,10]] Y=[0,1]
    can be interpreted in the following way: we have two points in the feature space: [0,1] and [10,10]
    First point belongs to the class 0 and the second to the class 1.

    In order to sort your results by distance one can use sorted construct.
    E.g. if we have a list of elements each of elements being the list with 2 elements, then we can sort the main list using e.g. the second element of every object:
    testList =[[0,1], [3,0], [2,-1]]   
    resultList = sorted(testList, key = itemgetter(1))  # 1 means second element of every object on the list
    We will end up with: [[2,-1], [3,0], [0,1]]

    Args: 
      x(list): list of features(numbers) which represents a point in the feature space with respect to which we calculate the distance.
      Xtrain(list): of feature vectors which form our training sample e.g. [[0,1], [10,10]] -> two points [0,1] and [10,10] with two features each
      Ytrain(list): of class labels which assing given feature vector (point) to given class e.g. [0,1] the first point belongs to class 0 and the second to 1
      metric(function): that calculates distance between two points from the feature space.
  """
  return (-1, [0, 0], 0)
