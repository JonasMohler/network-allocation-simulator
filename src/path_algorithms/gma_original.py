"""The original GMA algorithm."""

import numpy as np

from src.types import Path
from src.path_algorithms.path_algorithm import GMAPathAlgorithm

'''
from gma.gma_types import Path
from gma.path_algorithms.path_algorithm import GMAPathAlgorithm
'''

class GMAOriginal(GMAPathAlgorithm):
    """The original GMA path algorithm."""

    def compute_for_path(self, path: Path) -> float:
        """Compute the GMA-original algorithm for the given path."""
        if len(path) < 2:
            return 0
        t_vals = []
        divs = []
        convs = []
        prev_node = None
        for idx in range(0, len(path)):
            cur_node = path[idx]
            next_node = path[idx + 1] if (idx + 1) < len(path) else None
            cur_ij, cur_conv, cur_div = self.get_conv_div(
                cur_node, prev_node, next_node
            )
            t_vals.append(cur_ij)
            convs.append(cur_conv)
            divs.append(cur_div)
            prev_node = cur_node
        t_vals = np.asarray(t_vals)
        divs = np.asarray(divs)
        convs = np.asarray(convs)
        alloc = t_vals[0] * np.prod(t_vals[1:] / np.maximum(divs[1:], convs[:-1]))
        return alloc
