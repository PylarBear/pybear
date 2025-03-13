import numpy as n


def centroid(DATA):
    #THIS IS DESIGNED FOR LIST = EXAMPLE, NOT LIST = COLUMN OF DATA

    SEED = [0 for _ in range(len(DATA[0]))]

    for example in DATA:
        SEED = n.add(SEED, n.array(example), dtype=float)

    CENTROID = n.divide(SEED, len(DATA))

    return CENTROID


