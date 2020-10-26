# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:33:07 2020

@author: remib
"""

#Import modules
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import animation

#User-defined variables
initial_p_tree = 0.2 #Probability of each site having a tree initially
p_growing_tree = 0.02

empty, tree, burning = 0,1,2

#Generate intial set of trees
tree_array = np.random.choice([0,1], size=(50,50), p=[1-initial_p_tree, initial_p_tree])

def update_tree_array(tree_array):
    row_counter = -1
    col_counter = 0
    
    for row in tree_array:
        row_counter = row_counter + 1
        col_counter = 0
    
        for site in row:

            if site == empty:
                #Randomly populate empty site with tree
                tree_array[row_counter][col_counter] = np.random.choice([0,1], size=1, p=[1-p_growing_tree, p_growing_tree])[0]
            
            col_counter = col_counter + 1
    return tree_array

print(np.count_nonzero(tree_array == 1))

def animate(i):
    im.set_data(animate.X)
    animate.X = update_tree_array(animate.X)

"""
#Assign colours. Green-tree, red-burning tree, black-empty site
fig = plt.figure()
color_array =["green", "red", "black"] 
cmap_test = colors.ListedColormap(color_array) 

#Animation
im = plt.pcolormesh(tree_array, cmap=cmap_test)
# Interval between frames (ms).
interval = 100
anim = animation.FuncAnimation(fig, animate, interval=interval)
plt.show()
"""

color_array =["black", "red", "green"] 
cmap_test = colors.ListedColormap(color_array) 

fig = plt.figure(figsize=(25/3, 6.25))
ax = fig.add_subplot(111)
im = ax.imshow(tree_array, cmap=cmap_test)#, interpolation='nearest')

animate.X = tree_array

# Interval between frames (ms).
interval = 100
anim = animation.FuncAnimation(fig, animate, interval=interval)
plt.draw()
plt.show()
