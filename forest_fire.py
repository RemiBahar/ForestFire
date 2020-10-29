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
initial_p_tree = 1 #Probability of each site having a tree initially
p = 0.1 #Probability of a tree being grown in an empty site
f = 0.01 #Probability of a tree burning even if there are no nearby burning trees
sidelength = 50

empty, tree, burning = 0,1,2

#Generate intial set of trees
tree_array = np.random.choice([0,1], size=(sidelength,sidelength), p=[1-initial_p_tree, initial_p_tree])

def update_tree_array(tree_array):
    tree_array_old = tree_array
    tree_array = np.zeros((sidelength,sidelength))
    row_counter = -1
    col_counter = 0

    for row in tree_array_old:
        row_counter = row_counter + 1
        col_counter = 0
    
        for site in row:
            initial_state = site
            if site == empty:
                #Randomly populate empty site with tree
                tree_array[row_counter][col_counter] = np.random.choice([empty,tree], size=1, p=[1-p, p])[0]
                if tree_array[row_counter][col_counter] == burning:
                    print("burning randomly again")
            elif site == tree:
                #If one of direct neighbour burning then burn this tree
                #If tree above burning
                if row_counter != 0 and tree_array_old[row_counter-1][col_counter] == burning:
                    tree_array[row_counter][col_counter] = burning
                #If tree below burning
                elif row_counter != (sidelength -1) and tree_array_old[row_counter+1][col_counter] == burning:
                    tree_array[row_counter][col_counter] = burning
                #If tree to the left burning
                elif col_counter != 0 and tree_array_old[row_counter][col_counter-1] == burning:
                    tree_array[row_counter][col_counter] = burning
                #If tree to the right burning
                elif col_counter != (sidelength-1) and tree_array_old[row_counter][col_counter+1] == burning:
                    tree_array[row_counter][col_counter] = burning
                #Otherwise randomly burn trees
                else:
                 tree_array[row_counter][col_counter] = np.random.choice([tree,burning], size=1, p=[1-f, f])[0]
                 
            elif site==burning:
                #Burning tree become empty site
                tree_array[row_counter][col_counter] = empty
            
            col_counter = col_counter + 1
    print("burning, tree, empty")        
    print(np.count_nonzero(tree_array==burning))
    print(np.count_nonzero(tree_array==tree))
    print(np.count_nonzero(tree_array==empty))
    return tree_array

def animate(i):
    im.set_data(animate.X)
    animate.X = update_tree_array(animate.X)

color_array =["black", "green", "red"] 
cmap_test = colors.ListedColormap(color_array) 

colors_list = [(0.2,0,0), (0,0.5,0), (1,0,0), 'orange']
cmap_test = colors.ListedColormap(colors_list)
bounds = [0,1,2,3]
norm_test = colors.BoundaryNorm(bounds, cmap_test.N)

fig = plt.figure(figsize=(25/3, 6.25))
ax = fig.add_subplot(111)
im = ax.imshow(tree_array, cmap=cmap_test, norm=norm_test)#, interpolation='nearest')

animate.X = tree_array

# Interval between frames (ms).
interval = 1000
anim = animation.FuncAnimation(fig, animate, interval=interval)
plt.draw()
plt.show()