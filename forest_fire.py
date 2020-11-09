#Import modules
#%matplotlib notebook #needed for animations in jupyter
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk #for user forms


#Arrays of form labels and predefined values
fields = 'Mode (animation or analysis)', 'Iterations to run', 'Initial p_tree', 'p', 'f', 'tree rows', 'tree columns', 'interval'
predefined_values = np.array(["animation",50,1,0.1,0.01,50,50,500])
#predefined_values = np.array(["animation",100,1,0.1,0.01,10,10,2000])

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
        
def get_cluster_members(A, key, sidelength):
    row=np.array([])    
    col = np.array([])
    
    for i in range(0, sidelength):
        for j in range(0, sidelength):
            if A[i,j] == key:
                row = np.append(row, i)
                j = np.append(col, j)
                
    return [[row], [col]]
  
def union(A, x, y, sidelength):
    #Get smaller label
    left = A[x][y-1]
    above = A[x-1][y]
    
    #Set members of larger cluster to smaller cluster 
    row_counter = -1
    col_counter = 0
    
    for row in A:
        
        row_counter = row_counter + 1
        col_counter = 0
        
        for site in row:
            
            if site == left:
                A[row_counter][col_counter] = above
                
            col_counter = col_counter + 1
        
    return A

if __name__ == '__main__':
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
    
    for t in ax.texts:
        t.set_visible(False)
    
    
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
            
            #Cluster code
            
            if site == tree:
                if  row_counter == 0:
                    above = empty 
                else:
                    above = int(tree_array_old[row_counter-1][col_counter])
                    
                if col_counter == 0:
                    left = empty
                else:
                    left = int(tree_array_old[row_counter][col_counter-1])
                    
                #If no tree above or to the left of current site then assign a new cluster label
                if left != tree and above != tree:
                    largest_label = largest_label + 1
                    cluster_array[row_counter][col_counter] = largest_label 
                #One neighbour to the left 
                elif left == tree and above != tree:
                    cluster_array[row_counter][col_counter] = cluster_array[row_counter][col_counter-1]
                #One neighbour above
                elif left != tree and above == tree:
                    cluster_array[row_counter][col_counter] = cluster_array[row_counter-1][col_counter] #old?
                else:
                    #Merge left and above clusters based on lowest value
                    cluster_array = union(cluster_array, row_counter, col_counter, x)
                    #Arbitarily set site label equal to left neighbour
                    cluster_array[row_counter][col_counter] = cluster_array[row_counter][col_counter-1]
                    """
                    #Arbitarily sets site label equal to left neighbour
                    left_label = cluster_array[row_counter][col_counter-1]
                    above_label = cluster_array[row_counter-1][col_counter]
                    if left_label < above_label:
                        cluster_array[row_counter-1][col_counter] = cluster_array[row_counter][col_counter-1]
                    else:
                        cluster_array[row_counter][col_counter-1] = cluster_array[row_counter-1][col_counter]
                        
                    cluster_array[row_counter][col_counter] = cluster_array[row_counter][col_counter-1]
                    """
            else:
                cluster_array[row_counter][col_counter] = 0 #Ensures burning and empty sites have no label
            
            
            
            
            
            col_counter = col_counter + 1
    
    #After updating sites 
    burning_tree_array = np.append(burning_tree_array, np.count_nonzero(tree_array==burning))
    normal_tree_array = np.append(normal_tree_array, np.count_nonzero(tree_array==tree))
    empty_tree_array = np.append(empty_tree_array, np.count_nonzero(tree_array==empty))
    """
    left_column = tree_array_old[:,0]
    i=0
    for site in left_column:
        row = tree_array_old[i,:]
        max_index = np.where(row != tree)[0] 
        print(max_index)
                
        i=i+1
    
    print("hello")
    """
    row_counter = -1 #y position of site
    col_counter = 0 #x position of site
    
    for row in cluster_array:
        row_counter = row_counter + 1
        col_counter = 0
        for site in cluster_array:
            
            #For testing cluster labelling
            text = ax.text(x=col_counter, y=row_counter,s= cluster_array[row_counter][col_counter],ha="center", va="center", color="w")  
            col_counter = col_counter + 1

    #Gets number of trees for each cluster label
    i=0
    cluster_label_frequency_array = np.array([])
    while i <= largest_label:
        i=i+1
        tree_count = np.count_nonzero(cluster_array==i)
        cluster_label_frequency_array = np.append(cluster_label_frequency_array,tree_count)
    
    #Adds frequency of each cluster size to array
    for e in cluster_label_frequency_array:
        e = int(e)
        cluster_frequency_array[e] = cluster_frequency_array[e]+1
    
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
    