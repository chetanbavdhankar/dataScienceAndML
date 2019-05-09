#!/usr/bin/env python
"""
  The code serves only for educational purposes so it
  is not optimized in view of speed or numerical calculations.
  For the real problems you should always use algorithms.
  from known,  well tested libraries, wherever possible.
  Author: Wojtek Krzemien
  Date: 10.04 2018
"""
from majorityVote import majorityVote
from getKNNeighbours import getKNNeighbours


def predict(x, Xtrain, Ytrain, k):
  """
    Args:
      x(list): feature point that should be classify
      Xtrain(list): of feature vectors which form our training sample e.g. [[0,1], [10,10]] -> two points [0,1] and [10,10] with two features each
      Ytrain(list): of class labels which assing given feature vector (point) to given class e.g. [0,1] the first point belongs to class 0 and the second to 1
    Returns:
     int:  prediction
  """
  # TO IMPLEMENT
  return -1


def predictList(xObjects, Xtrain, Ytrain, k):
  """
    Args:
      xObjects(list): list of feature points that should be classify e.g. [[1,0], [2,2]] -> two points with 2 feature each
      Xtrain(list): of feature vectors which form our training sample e.g. [[0,1], [10,10]] -> two points [0,1] and [10,10] with two features each
      Ytrain(list): of class labels which assing given feature vector (point) to given class e.g. [0,1] the first point belongs to class 0 and the second to 1
    Returns:
     list:  list of int corresponding to predictions
  """
  # TO IMPLEMENT
  fakePredictions = [0] * len(xObjects)
  return fakePredictions
