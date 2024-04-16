import numpy as n, os
import sparse_dict

from general_sound import winlinsound as wls
import plotly
import plotly.graph_objects as go
import plotly.express as px
from debug import IdentifyObjectAndPrint as ioap
import sparse_dict as sd


def hits_and_misses(TARGET, DATA):

    TARGET = n.array(TARGET[0], dtype=object)

    HITS = n.argwhere(TARGET == 1).transpose()[0]
    MISSES = n.argwhere(TARGET == -1).transpose()[0]

    TARGET_HITS = TARGET[HITS]
    TARGET_MISSES = TARGET[MISSES]

    if not isinstance(DATA, dict):
        DATA = n.array(DATA, dtype=object)  # [ [] = COLUMNS ], MAKE SURE NP
        DATA = DATA.transpose()          # TRANSPOSE FOR EASY PICKINS
        DATA_HITS = DATA[HITS]
        DATA_MISSES = DATA[MISSES]
        DATA = DATA.transpose()  # TRANSPOSE BACK TO [ [] = COLUMNS ]

    elif isinstance(DATA, dict):
        DATA = sd.sparse_transpose(DATA)
        DATA_HITS = sd.multi_select_outer(DATA, HITS)
        DATA_HITS = sd.sparse_transpose(DATA_HITS)
        DATA_MISSES = sd.multi_select_outer(DATA, MISSES)
        DATA_MISSES = sd.sparse_transpose(DATA_MISSES)

    del HITS, MISSES

    return DATA_HITS, DATA_MISSES, TARGET_HITS, TARGET_MISSES


def svm_data_2d(TARGET, DATA):

    DATA_HITS, DATA_MISSES, TARGET_HITS, TARGET_MISSES = hits_and_misses(TARGET, DATA)

    print(f'\nBUILDING SCATTER PLOT OF DATA....')
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            line=None,
            marker={'color': 'green', 'size': 10},
            mode='markers',
            name='hits',
            opacity=0.75,
            x=[DATA_HITS[0] if not isinstance(DATA_HITS, dict) else sd.unzip_to_ndarray({0:DATA_HITS[0]})[0][0]][0],
            y=[DATA_HITS[1] if not isinstance(DATA_HITS, dict) else sd.unzip_to_ndarray({0:DATA_HITS[1]})[0][0]][0]
        )
    )

    fig.add_trace(
        go.Scatter(
            line=None,
            marker={'color': 'red', 'size': 10},
            mode='markers',
            name='miss',
            opacity=0.75,
            x=[DATA_MISSES[0] if not isinstance(DATA_MISSES, dict) else sd.unzip_to_ndarray({0:DATA_MISSES[0]})[0][0]][0],
            y=[DATA_MISSES[1] if not isinstance(DATA_MISSES, dict) else sd.unzip_to_ndarray({0:DATA_MISSES[1]})[0][0]][0]
        )
    )

    fig.update_layout(
        title_text='Data',  # title of plot
        xaxis_title_text='X',  # xaxis label
        yaxis_title_text='Y',  # yaxis label
    )

    plotly.offline.plot(fig)
    # wls.winlinsound(888, 100)
    print(f'Done.')


def svm_data_and_results_2d(TARGET, DATA, SUPPORT_VECTORS, SUPPORT_ALPHAS, SUPPORT_TARGETS, b):

    TARGET = n.array(TARGET, dtype=object)

    DATA_HITS, DATA_MISSES, TARGET_HITS, TARGET_MISSES = hits_and_misses(TARGET, DATA)

    print(f'\nBUILDING SCATTER PLOT OF DATA AND DECISION BOUNDARY....')
    if isinstance(DATA, dict): _min, _max = sd.min_({0:DATA[0]}), sd.max_({0:DATA[0]})
    elif not isinstance(DATA, dict): _min, _max = n.min(DATA[0]), n.max(DATA[0])

    X_BOUNDARY = [_ for _ in range(int(n.floor(_min)), int(n.ceil(_max) + 1))]
    '''
    w.x + b = 0
    w_x * x + w_y * y + b = 0
    (-b - w_x * x) / w_y = y'''
    w = n.matmul(SUPPORT_TARGETS.astype(float) * SUPPORT_ALPHAS.astype(float), SUPPORT_VECTORS.astype(float),
                 dtype=float)
    Y_BOUNDARY = [(-b - w[0] * x) / w[1] for x in X_BOUNDARY]  # FOR LINEAR KERNEL

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            marker={'color': 'green', 'size': 10},
            mode='markers',
            name='hit',  # name used in legend and hover labels
            opacity=0.75,
            x=[DATA_HITS[0] if not isinstance(DATA_HITS, dict) else sd.unzip_to_ndarray({0: DATA_HITS[0]})[0][0]][0],
            y=[DATA_HITS[1] if not isinstance(DATA_HITS, dict) else sd.unzip_to_ndarray({0: DATA_HITS[1]})[0][0]][0]
        )
    )

    fig.add_trace(
        go.Scatter(
            marker={'color': 'red', 'size': 10},
            mode='markers',
            name='miss',
            opacity=0.75,
            x=[DATA_MISSES[0] if not isinstance(DATA_MISSES, dict) else sd.unzip_to_ndarray({0: DATA_MISSES[0]})[0][0]][0],
            y=[DATA_MISSES[1] if not isinstance(DATA_MISSES, dict) else sd.unzip_to_ndarray({0: DATA_MISSES[1]})[0][0]][0]
        )
    )

    fig.add_trace(
        go.Scatter(
            marker={'color': 'black', 'size': 1},
            mode='markers+lines',
            name='decision\nboundary',  # name used in legend and hover labels
            opacity=0.75,
            line={'color': 'black', 'width': 5},
            x=X_BOUNDARY,
            y=Y_BOUNDARY
        )
    )

    fig.update_layout(
        title_text='Data & Boundary',  # title of plot
        xaxis_title_text='X',  # xaxis label
        yaxis_title_text='Y',  # yaxis label
    )

    plotly.offline.plot(fig)
    if os.name == 'nt': wls.winlinsound(888, 100)

























