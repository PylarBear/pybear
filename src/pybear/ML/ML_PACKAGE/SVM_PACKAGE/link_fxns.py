import numpy as np


def define_links():
    return ['SVM_Perceptron'
    ]


def link_fxns(ARG, select_link_fxn):
    # ARG is a numpy array aka [[ ]]
    #None
    if select_link_fxn in ['None']:
        pass

    elif select_link_fxn in ['SVM_Perceptron']:
        ARG = np.sign(ARG)


    return ARG



