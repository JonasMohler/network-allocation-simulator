"""Implementation of the improved GMA.

Algorithm developed by Mark Wyss.
"""
import numpy as np

from src.types import Path
from src.path_algorithms.path_algorithm import GMAPathAlgorithm

'''
from gma.gma_types import Path
from gma.path_algorithms.path_algorithm import GMAPathAlgorithm
'''


class GMAImproved(GMAPathAlgorithm):
    def compute_for_path(self, path: Path) -> float:
        if len(path) < 2:
            return 0
        cur_ij, cur_conv, cur_div = self.get_conv_div(path[0], None, path[1])
        prev_conv = cur_conv
        convs = [cur_conv]
        t_vals = [cur_ij]
        factor = 1
        for idx in range(1, len(path)):
            # Get the nodes for the current hop
            cur_node = path[idx]
            prev_node = path[idx - 1]
            next_node = path[idx + 1] if (idx + 1) < len(path) else None
            # Get matrix entry, convergent and divergent
            cur_ij, cur_conv, cur_div = self.get_conv_div(
                cur_node, prev_node, next_node
            )
            # Get the factors
            factor = min(prev_conv / cur_div, 1)
            prev_conv = cur_conv * factor
            # Gather the product of tval(i) / con(i-1)
            t_vals.append(cur_ij)
            convs.append(cur_conv)
        t_vals = np.asarray(t_vals)
        convs = np.asarray(convs)
        alloc = t_vals[0] * np.prod(t_vals[1:] / convs[:-1]) * factor
        return float(alloc)
