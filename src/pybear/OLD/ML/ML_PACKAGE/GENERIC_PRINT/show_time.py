import datetime

#CALLED BY NN
def show_start_time(name):
    now = datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")
    print(f'{name} start time: ' + f'{now}'.rjust(12))

#CALLED BY NN
def show_end_time(name):
    now = datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")
    print(f'\n{name} end time: ' + f'{now}'.rjust(12))






