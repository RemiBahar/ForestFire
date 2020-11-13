# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:43:20 2020

@author: remib
"""
#Import modules 
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk #for user forms
from random import choices

#Executed when user clicks run button
def fetch(entries):
    global window, user_input_array #Extends functions scope
    
    user_input_array = np.array([])
    
    #Saves input data to an array
    for entry in entries:
        text  = entry[1].get()
        user_input_array = np.append(user_input_array, text)
        
    window.destroy() #Exits form and runs main python file

#Populates form
def makeform(window, fields):
    entries = []
    i=0
    #Loops through form
    for field in fields:
        row = tk.Frame(window) 
        label = tk.Label(row, width=30, text=field, anchor='w') #Adds form label
        entry = tk.Entry(row) #Adds form input
        entry.insert(0, predefined_values[i]) #Adds pre-filled text
        
        #Form layout
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5) 
        label.pack(side=tk.LEFT)
        entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, entry))#
        
        i=i+1
        
    return entries
        
#Generate user pop-up window
fields = 'Mode (animation or analysis)', 'Iterations to run', 'Initial p_tree', 'p', 'f', 'tree rows', 'tree columns', 'interval'
predefined_values = np.array(["analysis",3000,1,0.1,0.001,50,50,2000])
#predefined_values = np.array(["animation",100,1,0.1,0.001,10,10,2000])

if __name__ == '__main__':
    #Generate pop-up window
    window = tk.Tk() #creates window
    ents = makeform(window, fields) #makes form
    window.bind('<Return>', (lambda event, e=ents: fetch(e))) #closes window
    b1 = tk.Button(window, text='Run',
                  command=(lambda e=ents: fetch(e))) #run button
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(window, text='Cancel', command=window.quit) #cancel button
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    window.mainloop() #goes to main python file
    
#User-defined variables
mode= user_input_array[0] #animation or analysis
run_iteration = int(user_input_array[1]) #Number of times to run simulation. 15s for 100 iterations. Need 3000 for cluster analysis
p_initial = float(user_input_array[2]) #Probability of each site having a tree initially
p = float(user_input_array[3]) #Probability of a tree being grown in an empty site
f = float(user_input_array[4]) #Probability of a tree burning even if there are no nearby burning trees
rows = int(user_input_array[5])
columns = int(user_input_array[6])
interval = float(user_input_array[7]) #Interval between animation frames

iteration = 0
#Randomly generates initial forest
total_iterations = (rows*columns*run_iteration)+1
tree_array = np.random.choice([0,1], size=(rows,columns), p=[1-p_initial, p_initial]) 
empty_p_array = choices([0,1], [1-p, p], k=total_iterations)
tree_f_array =  choices([1,2], [1-f, f], k=total_iterations)
site_counter = 0
cluster_frequency_array = np.zeros(int(rows*columns)+1)
burning_array = np.array([])
normal_tree_array = np.array([])
empty_array = np.array([])

def hk(site_matrix, site_matrix_rows, site_matrix_columns):
    """
    Labels the sites in an array using the Hoshen-Kopelman algorithm.
    
    When merging two clusters, the equivalence class of the left cluster is changed 
    to that of the above cluster, e.g. 2->4. A raster scan is then done of the grid, and the equivalence
    class is used to assign the correct label, e.g. for a site with a label of 2, A[2-1] = 4 so a 4 is assigned 
    to the site.
    
    Parameters:
    site_matrix (2D Array) - Array of sites
    site_matrix_rows (int) - Number of rows of sites
    site_matrix_columns (int) - Number of columns of sites
    
    Returns:
    hk_result_array - Array of results
        Index 0 - cluster_matrix (2D Array) - Array of site labels 
        Index 1 - equivalence_array (1D Array) - Array of equivalence classes
    """
    
   
    equivalence_array = np.array([])  #For equivalence classes
    largest_equivalence_class = 0 #Largest equivalence class
    cluster_matrix = np.zeros((site_matrix_rows,site_matrix_columns)) #Initialize cluster array filled with zeros initially
    
    #Raster scan grid
    for i in range(site_matrix_rows): #loop through rows
        for j in range(site_matrix_columns): #loop through columns
            if site_matrix[i][j] == 1: #only label occupied sites

                #Get left and above sites
                left = 0 if j == 0 else site_matrix[i][j-1] 
                above = 0 if i == 0 else site_matrix[i-1][j]
    
                #If there are no occupied sites above or on the left, create a new equivalence class
                if left != 1 and above != 1: 
                    largest_equivalence_class = largest_equivalence_class + 1
                    cluster_matrix[i][j] = largest_equivalence_class
                    equivalence_array = np.append(equivalence_array, largest_equivalence_class)
                #If there is an occupied site on the left only, use left site's equivalence class    
                elif left == 1 and above != 1: #Occupied site on the left
                    cluster_matrix[i][j] = cluster_matrix[i][j-1]
                #If there is an occupied site above only, use the above site's equivalence class
                elif left != 1 and above == 1: 
                    cluster_matrix[i][j] = cluster_matrix[i-1][j]
                #If there are occupied sites above and on the left, merge these two equivalence classes    
                else:
                    cluster_matrix[i][j] = cluster_matrix[i-1][j]
                    if cluster_matrix[i-1][j] != cluster_matrix[i][j-1]:
                        left_root_node = equivalence_array[int(cluster_matrix[i][j-1]-1)]
                        above_root_node = equivalence_array[int(cluster_matrix[i-1][j]-1)]
                       
                        c=-1
                        for e in equivalence_array:
                            c=c+1
                            if equivalence_array[c] == left_root_node:
                                equivalence_array[c] = above_root_node

    
    #Raster scan grid and assign correct labels to sites                
    for i in range(site_matrix_rows):
        for j in range(site_matrix_columns):
            if cluster_matrix[i][j] != 0:
                #Use the equivalence class to label the site
                cluster_matrix[i][j] = equivalence_array[int(cluster_matrix[i][j]-1)]
    
    hk_result_array = [cluster_matrix, equivalence_array]       
    
    return hk_result_array

def update_tree_array(matrix):
    global x, y, f, p, rows, columns, site_counter, ax, cluster_frequency_array
    global burning_array, empty_array, normal_tree_array
    #Create variable to store updated forest
    tree_array_old = matrix 
    matrix = np.zeros((rows, columns)) 
                
    hk_result_array = hk(tree_array_old, rows, columns)
    cluster_array = hk_result_array[0]
    label_array = hk_result_array[1]            
    """         
    for t in ax.texts:
        t.set_visible(False)
    
    for i in range(rows):
        for j in range(columns):
            text = ax.text(x=j, y=i,s= cluster_array[i][j],ha="center", va="center", color="w")  
     
    """
    unique_label_array = list(set(label_array))
    for label in unique_label_array:
        cluster_size = np.count_nonzero(cluster_array==label)
        cluster_frequency_array[cluster_size] = cluster_frequency_array[cluster_size] + 1
    
    #Raster scan grid
    for i in range(rows):
        for j in range(columns):
            site_counter += 1
            if tree_array_old[i][j] == 0: #site empty
                #Randomly populate empty site with tree
                #matrix[i][j] = np.random.choice([0,1], size=1, p=[1-p, p])[0]
                matrix[i][j] = empty_p_array[site_counter]
            elif tree_array_old[i][j] == 1: #site occupied by tree
                #Get neighbouring sites
                above = 0 if i == 0 else tree_array_old[i-1][j]
                right = 0 if j == (columns-1) else tree_array_old[i][j+1]
                below = 0 if i == (rows-1) else tree_array_old[i+1][j]
                left = 0 if j == 0 else tree_array_old[i][j-1]
                
                #Burn site if neighbouring site is burning
                if above == 2 or right == 2 or below == 2 or left == 2:
                    matrix[i][j] = 2
                #Otherwise site has a probability f of burning    
                else:
                    matrix[i][j] = tree_f_array[site_counter]
            #If site burning becomes empty
            else: 
                matrix[i][j] = 0
                
    #After updating sites 
    burning_array = np.append(burning_array, np.count_nonzero(tree_array==2))
    normal_tree_array = np.append(normal_tree_array, np.count_nonzero(tree_array==1))
    empty_array = np.append(empty_array, np.count_nonzero(tree_array==0))
           
    return matrix
    
def analyse_data():
    global rows, columns, normal_tree_array, cluster_frequency_array, p, f, p_initial, burning_array
    global empty_array
    tree_density = np.mean(tree_array[0:-1])/(rows*columns)
    
    #Update log files
    file = open("Data/cluster_array.txt", "w")
    np.savetxt(file, cluster_frequency_array)
    file.close()

    summary_array = np.array([p,f,run_iteration,rows,columns,p_initial, tree_density])
    file = open("Data/summary_data.txt", "w")
    np.savetxt(file, summary_array)
    file.close()
    
    file = open("Data/burning_array.txt", "w")
    np.savetxt(file, burning_array)
    file.close()
    
    file = open("Data/tree_array.txt", "w")
    np.savetxt(file, normal_tree_array)
    file.close()
    
    file = open("Data/empty_array.txt", "w")
    np.savetxt(file, empty_array)
    file.close()
    
    #Plot graph of site evolution
    plt.figure()
    plt.title("Evolution of grid")
    plt.plot(burning_array, label="Burning", color="orange")
    plt.plot(normal_tree_array, label="Trees", color="green")
    plt.plot(empty_array, label="Empty", color="black")
    plt.xlabel("Time")
    plt.ylabel("Number")
    plt.legend()
    plt.minorticks_on()
    plt.grid(which="both")
    #plt.savefig("GridEvolution.png", dpi=1200, format="png")
    plt.show()
    
    #Plot histogram of cluster sizes
    xsize = np.size(cluster_frequency_array)
    bins_array = np.linspace(start=1, stop=xsize, num=xsize)
    plt.figure()
    plt.title("Cluster Size Frequency")
    plt.hist(cluster_frequency_array, bins=bins_array)
    #plt.xlim((0, 80)) 
    plt.xlabel("Cluster Size")
    plt.ylabel("Frequency")
    plt.minorticks_on()
    plt.grid(which="both")
    #plt.savefig("ClusterSizeFrequency.png", dpi=1200, format="png")
    plt.show()
        

def animate(i):
    
    global iteration #Allows function to access iteration variable
    
    if iteration < run_iteration:
        iteration = iteration + 1 #Increment iteration
        im.set_data(animate.X) #Updates animation
        animate.X = update_tree_array(animate.X) #Calculates new animation data
    else:
        print("Finished Animation. Producing plots")
        analyse_data()
        
#Runs animation
if mode == "analysis":
    
    while iteration < run_iteration:
        iteration = iteration +1
        print(iteration)
        tree_array = update_tree_array(tree_array) #Calculates new site data
    print("Finished calculating data")
    analyse_data()
#Analyses data
else: 
    #Sets colours of animation        
    colors_list = [(0.2,0,0), (0,0.5,0), (1,0,0), 'orange']
    cmap_test = colors.ListedColormap(colors_list)
    bounds = [0,1,2,3]
    norm_test = colors.BoundaryNorm(bounds, cmap_test.N)
    
    #Animation
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(tree_array, cmap=cmap_test, norm=norm_test)#, interpolation='nearest')
    animate.X = tree_array
    anim = animation.FuncAnimation(fig, animate, frames=run_iteration, interval=interval, repeat=False)
    plt.draw()
    plt.axis('off')
    plt.show()