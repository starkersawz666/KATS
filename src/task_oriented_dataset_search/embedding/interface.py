from typing import List

import numpy as np


class BaseEmbedder:
    def dimension(self) -> int:
        raise NotImplementedError()

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError()
