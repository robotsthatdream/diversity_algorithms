#!/usr/bin/python -w

from diversity_algorithms.analysis.data_utils import *

def plot_points(points, bg="maze_hard.pbm", title=None):
    x,y = zip(*points)
    fig1, ax1 = plt.subplots()
    ax1.set_xlim(0,600)
    ax1.set_ylim(600,0) # Decreasing
    ax1.set_aspect('equal')
    if(bg):
        img = plt.imread(bg)
        ax1.imshow(img, extent=[0, 600, 600, 0])
    if(title):
        ax1.set_title(title)
    ax1.scatter(x, y, s=2)
    plt.show()
    
def plot_points_lists(lpoints, bg="maze_hard.pbm", title=None):
    fig1, ax1 = plt.subplots()
    ax1.set_xlim(0,600)
    ax1.set_ylim(600,0) # Decreasing
    ax1.set_aspect('equal')
    if(bg):
        img = plt.imread(bg)
        ax1.imshow(img, extent=[0, 600, 600, 0])
    if(title):
        ax1.set_title(title)
    for points in lpoints:
        x,y = zip(*points)
        ax1.scatter(x, y, s=2)
    plt.show()
