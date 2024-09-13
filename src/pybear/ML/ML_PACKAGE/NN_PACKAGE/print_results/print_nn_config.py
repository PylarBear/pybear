import pandas as p


#CALLED BY array_of_nodes, nn_config, select_link_fxn
def print_nn_config(SELECT_LINK_FXN, NEURONS):

    __ = lambda text, width: str(text).center(width)

    print(f'\nCurrent node / link / neuron configuration:')
    print(f"{__('NODE', 10)}{__('LINK', 10)}{__('NEURONS', 10)}")

    for link in range(len(SELECT_LINK_FXN)):
        if SELECT_LINK_FXN[link] != 'Multi-out':
            print(f"{__(link,10)}{__(SELECT_LINK_FXN[link],10)}{__(NEURONS[link],10)}")
        else:
            print(f"{__(' ',10)}{__('Multi-out',10)}{__(' ',10)}")
    print()





