from typing import List, Dict, Tuple
import numpy as np

# Path in a network as a list of nodes
Path = List[int]

SinglePathsDict = Dict[int, Dict[int, Path]]

PathLengthsDict = Dict[int, Dict[int, int]]

PathCountsDict = Dict[int, Dict[Tuple[int, int], Dict[int, int]]]

PCList = List[PathCountsDict]

# A dictionary of paths from source to destination
PathsDict = Dict[int, Dict[int, List[Path]]]

# The result of the run of an algorithm over multiple paths
#   This is in the form `result[src][dst] = src_dst_value`
PathsResult = Dict[int, Dict[int, List[float]]]

# Traffic Matrices
AllocationMatrices = Dict[int, np.ndarray]


def tm_from_list(lst: List[AllocationMatrices]) -> AllocationMatrices:
    dic = {}
    for tm in lst:
        k = tm.keys()[0]
        dic[k] = tm[k]
    return AllocationMatrices(dic)


def sp_from_list(lst: List[SinglePathsDict]) -> SinglePathsDict:
    dic = {}
    for sp in lst:
        k = sp.keys()[0]
        dic[k] = sp[k]
    return SinglePathsDict(dic)


def pl_from_list(lst: List[PathLengthsDict]) -> PathLengthsDict:
    dic = {}
    for pl in lst:
        k = pl.keys()[0]
        dic[k] = pl[k]
    return PathLengthsDict(dic)


def pc_from_list(lst: List[PathCountsDict]) -> PathCountsDict:
    dic = {}
    for pc in lst:
        k = pc.keys()[0]
        dic[k] = pc[k]
    return PathCountsDict(dic)
