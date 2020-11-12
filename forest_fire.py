#Import modules
#%matplotlib notebook #needed for animations in jupyter
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk #for user forms

#Arrays of form labels and predefined values
fields = 'Mode (animation or analysis)', 'Iterations to run', 'Initial p_tree', 'p', 'f', 'tree rows', 'tree columns', 'interval'
#Array of pre-defined values for running program
#predefined_values = np.array(["animation",100,1,0.1,0.01,10,10,2000])
predefined_values = np.array(["analysis",3000,1,0.1,0.001,200,200,2000])

#Functions

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
                    left_root_node_index = int(cluster_matrix[i][j-1]-1) 
                    right_root_node_index = int(cluster_matrix[i-1][j]-1)
                    equivalence_array[left_root_node_index] = equivalence_array[right_root_node_index]
                    cluster_matrix[i][j] = cluster_matrix[i][j-1]
    
    #Raster scan grid and assign correct labels to sites                
    for i in range(site_matrix_rows):
        for j in range(site_matrix_columns):
            if cluster_matrix[i][j] != 0:
                #Use the equivalence class to label the site
                cluster_matrix[i][j] = equivalence_array[int(cluster_matrix[i][j]-1)]
     
    hk_result_array = [cluster_matrix, equivalence_array]       
    
    return hk_result_array


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
initial_p_tree = float(user_input_array[2]) #Probability of each site having a tree initially
p = float(user_input_array[3]) #Probability of a tree being grown in an empty site
f = float(user_input_array[4]) #Probability of a tree burning even if there are no nearby burning trees
#Forest dimensions
x = int(user_input_array[5])
y = int(user_input_array[6])
interval = float(user_input_array[7]) #Interval between animation frames

#Other variables
iteration = 0
empty, tree, burning = 0,1,2 #State of a site
tree_array = np.random.choice([0,1], size=(x,y), p=[1-initial_p_tree, initial_p_tree]) #Generates initial forest
cluster_frequency_array = np.zeros(int((x*y)+1))
burning_tree_array, normal_tree_array, empty_tree_array = np.array([]), np.array([]), np.array([])
cluster_array = np.zeros((x,y))

def update_tree_array(tree_array):
    
    #Allows function to access variables outside its scope
    global burning_tree_array, normal_tree_array, empty_tree_array, cluster_array
    global cluster_frequency_array, largest_label, iteration
    cluster_array = np.zeros((x,y))
    #Define variables
    tree_array_old = tree_array
    tree_array = np.zeros((x,y)) #Generate new tree array to replace previous one
    row_counter = -1 #y position of site
    col_counter = 0 #x position of site
    largest_label = 0 #For cluster labelling
    
    #For cluster label testing
    """
    for t in ax.texts:
        t.set_visible(False)
    """
    
    #Loop through rows and update sites
    for row in tree_array_old:
        row_counter = row_counter + 1 #Increment x position
        col_counter = 0 #Reset y position
        
        #Loop through columns in a row
        for site in row:
            if site == empty:
                #Randomly populate empty site with tree
                tree_array[row_counter][col_counter] = np.random.choice([empty,tree], size=1, p=[1-p, p])[0]
            elif site == tree:
                #If one of direct neighbour burning then burn this tree
                #If tree above burning
                if row_counter != 0 and tree_array_old[row_counter-1][col_counter] == burning:
                    tree_array[row_counter][col_counter] = burning
                #If tree below burning
                elif row_counter != (y -1) and tree_array_old[row_counter+1][col_counter] == burning:
                    tree_array[row_counter][col_counter] = burning
                #If tree to the left burning
                elif col_counter != 0 and tree_array_old[row_counter][col_counter-1] == burning:
                    tree_array[row_counter][col_counter] = burning
                #If tree to the right burning
                elif col_counter != (x-1) and tree_array_old[row_counter][col_counter+1] == burning:
                    tree_array[row_counter][col_counter] = burning
                #Otherwise randomly burn trees
                else:
                     tree_array[row_counter][col_counter] = np.random.choice([tree,burning], size=1, p=[1-f, f])[0]
            elif site==burning:
                #Burning tree become empty site
                tree_array[row_counter][col_counter] = empty
            
            col_counter = col_counter + 1
    
    #After updating sites 
    burning_tree_array = np.append(burning_tree_array, np.count_nonzero(tree_array==burning))
    normal_tree_array = np.append(normal_tree_array, np.count_nonzero(tree_array==tree))
    empty_tree_array = np.append(empty_tree_array, np.count_nonzero(tree_array==empty))
   
    row_counter = -1 #y position of site
    col_counter = 0 #x position of site
    
    largest_label = 0
    id_array = np.arange((x*y))
    label_array = id_array
    
    hk_result_array = hk(tree_array_old, x, y)
    cluster_array = hk_result_array[0]
    label_array = hk_result_array[1]
    """
    for row in cluster_array:
        row_counter = row_counter + 1
        col_counter = 0
        for site in cluster_array:
            
            #For testing cluster labelling
            text = ax.text(x=col_counter, y=row_counter,s= cluster_array[row_counter][col_counter],ha="center", va="center", color="w")  
            col_counter = col_counter + 1
    """
    #Count size of each cluster and increment counters for each cluster size
    unique_label_array = list(set(label_array))
    for label in unique_label_array:
        cluster_size = np.count_nonzero(cluster_array==label)
        cluster_frequency_array[cluster_size] = cluster_frequency_array[cluster_size] + 1
 
    return tree_array

def animate(i):
   
    global iteration #Allows function to access iteration variable
    
    if iteration < run_iteration:
        iteration = iteration + 1 #Increment iteration
        im.set_data(animate.X) #Updates animation
        animate.X = update_tree_array(animate.X) #Calculates new animation data
    else:
        print("Finished Animation. Producing plots")
        analyse_data()
              
        
def analyse_data(): 
    print("Hello")
    global burning_tree_array, normal_tree_array, empty_tree_array, x, y, cluster_frequency_array, tree_density
    #Plot graph of site evolution
    plt.figure()
    plt.title("Evolution of grid")
    plt.plot(burning_tree_array, label="Burning", color="orange")
    plt.plot(normal_tree_array, label="Trees", color="green")
    plt.plot(empty_tree_array, label="Empty", color="black")
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
    
    #Calculate important stats
    total_sites=x*y
    mean_trees = np.mean(normal_tree_array[0:-1])
    tree_density = mean_trees/total_sites
    print("tree density: ", tree_density)
    
    file = open("Data/ClusterArray.txt", "w")
    np.savetxt(file, cluster_frequency_array)
    file.close()

    summary_array = np.array([p,f,run_iteration,x,y,initial_p_tree,tree_density])
    file = open("Data/SummaryData.txt", "w")
    np.savetxt(file, summary_array)
    file.close()
    
    file = open("Data/BurningTreeArray.txt", "w")
    np.savetxt(file, burning_tree_array)
    file.close()
    
    file = open("Data/NormalTreeArray.txt", "w")
    np.savetxt(file, normal_tree_array)
    file.close()
    
    file = open("Data/EmptyTreeArray.txt", "w")
    np.savetxt(file, empty_tree_array)
    file.close()
    
    print("Finished")

#Runs animation
if mode=="animation":
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
#Analyses data
elif mode=="analysis":
    while iteration < run_iteration:
        iteration = iteration +1
        print(iteration)
        tree_array = update_tree_array(tree_array) #Calculates new site data
    print("Finished calculating data")
    analyse_data()
    