import random
from matplotlib.colors import ListedColormap
import datetime


white = (1, 1, 1)

"""

"""


def rand_color() -> tuple:
    return (random.random(), random.random(), random.random())


def get_rand_cmap():
    return ListedColormap((white, rand_color()))


def get_datetime_str(style='dt'):
    cur_time = datetime.datetime.now()
    date_str = cur_time.strftime('%y_%m_%d_')
    time_str = cur_time.strftime('%H_%M_%S')
    if style == 'data':
        return date_str
    elif style == 'time':
        return time_str
    return date_str + time_str
