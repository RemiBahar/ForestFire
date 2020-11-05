#Import modules
#%matplotlib notebook #needed for animations in jupyter
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import animation

#User-defined variables
mode="animation" #animation or analysis
run_iteration = 20 #Number of times to run simulation. 15s for 100 iterations. Need 3000 for cluster analysis

initial_p_tree = 1 #Probability of each site having a tree initially
p = 0.1 #Probability of a tree being grown in an empty site
f = 0.01 #Probability of a tree burning even if there are no nearby burning trees
#Forest dimensions
x = 50
y = 50
interval = 500 #Interval between animation frames

#Other variables
iteration = 0
empty, tree, burning = 0,1,2 #State of a site
tree_array = np.random.choice([0,1], size=(x,y), p=[1-initial_p_tree, initial_p_tree]) #Generates initial forest
cluster_frequency_array = np.zeros(int((x*y)))

burning_tree_array, normal_tree_array, empty_tree_array = np.array([]), np.array([]), np.array([])
cluster_array = np.zeros((x,y))

def update_tree_array(tree_array):
    #Allows function to access variables outside its scope
    global burning_tree_array, normal_tree_array, empty_tree_array, cluster_array
    global cluster_frequency_array, largest_label, iteration
    
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
            
            #Cluster code
            if site == tree:
                if y == 0:
                    above = empty 
                else:
                    above = tree_array_old[row_counter-1][col_counter]
                    
                if col_counter == 0:
                    left = empty
                else:
                    left = tree_array_old[row_counter][col_counter-1]
                    
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
                    cluster_array[row_counter][col_counter] = cluster_array[row_counter][col_counter-1]
                    #Need code to stop double counting of clusters
            else:
                cluster_array[row_counter][col_counter] = 0 #Ensures burning and empty sites have no label
            
            #For testing cluster labelling
            #text = ax.text(x=col_counter, y=row_counter,s= cluster_array[row_counter][col_counter],ha="center", va="center", color="w")  
            
            col_counter = col_counter + 1
    
    #After updating sites 
    burning_tree_array = np.append(burning_tree_array, np.count_nonzero(tree_array==burning))
    normal_tree_array = np.append(normal_tree_array, np.count_nonzero(tree_array==tree))
    empty_tree_array = np.append(empty_tree_array, np.count_nonzero(tree_array==empty))

    #Gets number of trees for each cluster label
    i=0
    cluster_label_frequency_array = np.array([])
    while i <= largest_label:
        i=i+1
        cluster_label_frequency_array = np.append(cluster_label_frequency_array,np.count_nonzero(tree_array==i))
    
    #Adds frequency of each cluster size to array
    for e in cluster_label_frequency_array:
        e=int(e)
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
    global burning_tree_array, normal_tree_array, empty_tree_array, x, y, cluster_frequency_array
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
    plt.xlim((0, 80)) 
    plt.xlabel("Cluster Size")
    plt.ylabel("Frequency")
    plt.minorticks_on()
    plt.grid(which="both")
    #plt.savefig("ClusterSizeFrequency.png", dpi=1200, format="png")
    plt.show()
    
    #Calculate important stats
    total_sites=x*y
    mean_trees = np.mean(normal_tree_array[200:-1])
    tree_density = mean_trees/total_sites
    print("tree density: ", tree_density)

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
    plt.show()
elif mode=="analysis":
    while iteration < run_iteration:
        iteration = iteration +1
        print(iteration)
        tree_array = update_tree_array(tree_array) #Calculates new site data
    print("Finished calculating data")
    analyse_data()