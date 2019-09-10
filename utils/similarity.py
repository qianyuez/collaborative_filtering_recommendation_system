import numpy as np


def cosine_similarity(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0
    l = np.linalg.norm(a) * np.linalg.norm(b)
    if l == 0:
        return 0
    else:
        s = np.dot(a, b) / l
        s = s * 0.5 + 0.5
        return s