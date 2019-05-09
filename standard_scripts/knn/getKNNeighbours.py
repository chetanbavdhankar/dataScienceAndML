#!/usr/bin/env python
"""
  The code serves only for educational purposes so it
  is not optimized in view of speed or numerical calculations.
  For the real problems you should always use algorithms.
  from known,  well tested libraries, wherever possible.
  Author: Wojtek Krzemien
  Date: 10.04 2018
"""


def getKNNeighbours(neighbours, k):
  """
    Function should return k nearest neighbours assuming that neighbours are already sorted by distance.

    Args:
      neighbours(list): neighbours sorted by distance e.g. [[1, [2,0],1],  [5, [2,0],1]],
                        where [5,[2,0],1] -> distance 5, closest feature point [2,0] which belongs to the class 1
      k(integer): number of neighbours to return
    Returns:
      list of k nearest neighbours assuming that neighbours are already sorted by distance.
  """
  return [-1, [0, 0], 0]
