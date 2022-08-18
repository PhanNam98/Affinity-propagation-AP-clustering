import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import imageio
from io import BytesIO
import csv
import pandas as pd
import time
from tqdm import tqdm
from sklearn.datasets import make_blobs
from sklearn import metrics

##utils
def make_gif(figures, filename, fps=10, **kwargs):
    '''Make gif result
    figures: fig array
    filename
    fps: default is 10
    '''
    print('Creating gif')
    try:
        images = []
        for fig in figures:
            output = BytesIO()
            fig.savefig(output)
            plt.close(fig)  
            output.seek(0)
            images.append(imageio.imread(output))
        imageio.mimsave(filename, images, fps=fps, **kwargs)
        print(filename, ' gif created successfully!')
    except:
        print('Can not create gif file!')

def read_file(FILE_PATH, FEATURE, INDEX_COL=None):
    global data

    try:
        data = pd.read_csv(FILE_PATH, names=FEATURE, index_col=INDEX_COL)
        print('Read csv file successfully')
        print(data)
        print('-------------------------------------------------------------------')
        x = data.values
        return x
    except:
        print('Can not read csv file!')

def create_csv(FILE_NAME):
    print('Creating result.csv')
    try:
        data2 = data.sort_values(by='label', ascending=True)
        #print(data2.head())
        data2.to_csv(FILE_NAME)
        print(FILE_NAME, ' created successfully!')

    except:
        print('Can not create csv file!')

def plot_result(labels, exemplars, iter):
    colors = dict(zip(exemplars, cycle('bgrcmyk')))
    #colors = dict(zip(exemplars, cycle('0.1, 0.2, 0.5')))
    figure3 = plt.figure(figsize=(12, 6))
    figure3 = plt.axes()
    figure3.set_xlabel('Age')
    figure3.set_ylabel('Spending Score (1-100)')

    figure4 = plt.figure(figsize=(12, 6))
    figure4 = plt.axes()
    figure4.set_xlabel('Age')
    figure4.set_ylabel('Spending Score (1-100)')
    for i in range(len(labels)):
        X = x[i][1] #age selected
        Y = x[i][3] #spending_score selected
        
        if i in exemplars:
            #plot the exemplar
            exemplar = i
            edge = 'k'
            ms = 10
            figure3.plot(X, Y, '^', markersize=ms,  markeredgecolor=edge, c=colors[exemplar])
            figure4.plot(X, Y, '^', markersize=ms,  markeredgecolor=edge, c=colors[exemplar])
        else:
            #plot the member
            exemplar = labels[i]
            ms = 3
            edge = None
            figure3.plot([X, x[exemplar][1]], [Y, x[exemplar][3]], markersize=ms, markeredgecolor=edge, c=colors[exemplar]) 
            figure4.plot([X, x[exemplar][1]], [Y, x[exemplar][3]], 'o', markersize=ms, markeredgecolor=edge, c=colors[exemplar])

    figure3.set_title('Number of exemplars: %s, iteration: %d' % (len(exemplars),iter))
    figure4.set_title('Number of exemplars: %s, iteration: %d' % (len(exemplars),iter))
    plt.show()

#for samples data
def plot_iteration(labels, exemplars, iter, show=False):
    fig = plt.figure(figsize=(12, 6))
    colors = dict(zip(exemplars, cycle('bgrcmyk')))
    
    for i in range(len(labels)):
        X = x[i][0]
        Y = x[i][1]
        
        if i in exemplars:
            exemplar = i
            edge = 'k'
            ms = 10
        else:
            exemplar = labels[i]
            ms = 3
            edge = 'r'
            plt.plot([X, x[exemplar][0]], [Y, x[exemplar][1]], c=colors[exemplar])
        plt.plot(X, Y, 'g^', markersize=ms,  markeredgecolor=edge, c=colors[exemplar])
        
    plt.title('Number of exemplars: %s, iteration: %d' % (len(exemplars),iter))
    if show:
        plt.show()
    
    return fig

##AP
def update_r(damping=0.9, slow=False):
    '''
responsibility messages:
r(i,k) <-- s(i,k) - MAX {a(i,k') + s(i,k')} with k's.t.k' ≠ k
We could implement this with a nested for loop where we iterate over every row i and then determine the 
max(A+S) (of that row) for every index not equal to k or i (The index should not be equal to i as it would be sending messages to itself). 

The damping factor is just there for nummerical stabilization and can be regarded as a slowly converging learning rate. 
Sklearn advised to choose a damping factor within the range of 0.5 to 1.
'''
    global R
    if (slow == True):
        for i in range(x.shape[0]):
            for k in range(x.shape[0]):
                v = S[i, :] + A[i, :]
                v[k] = -np.inf
                v[i]= -np.inf
                R[i, k] = R[i, k] * damping + (1 - damping) * (S[i, k] - np.max(v))
                
    #optimize it by vectorizing the loops
    else:
        # For every column k, except for the column with the maximum value the max is the same.
        # So we can subtract the maximum for every row, and only need to do something different for k == argmax
        
        v = S + A
        rows = np.arange(x.shape[0])
        # We only compare the current point to all other points, so the diagonal can be filled with -infinity
        np.fill_diagonal(v, -np.inf)

        # max values
        idx_max = np.argmax(v, axis=1)
        first_max = v[rows, idx_max]

        # Second max values. For every column where k is the max value.
        v[rows, idx_max] = -np.inf
        second_max = v[rows, np.argmax(v, axis=1)]

        # Broadcast the maximum value per row over all the columns per row.
        max_matrix = np.zeros_like(R) + first_max[:, None]
        max_matrix[rows, idx_max] = second_max

        new_val = S - max_matrix

        R = R * damping + (1 - damping) * new_val

def update_a(damping=0.9, slow=False):
    '''
availability messages:
a(i,k) <-- min{0,r(k,k) + ∑ max{0,r(i',k)}} with i's.t.i' ∉ {i,k}
a(k,k) <-- ∑ max(0,r(i',k)) with i' ≠ k

For all points not on the diagonal of A (all the messages going from one data point to all other points), 
the update is equal to the responsibility that point k assigns to itself and the sum of the responsibilities that other data points (except the current point) assign to k. 
Note that, due to the min function, this holds only true for negative values.

The damping factor is just there for nummerical stabilization and can be regarded as a slowly converging learning rate. 
The authors advised to choose a damping factor within the range of 0.5 to 1.
'''
    global A
    
    if (slow == True):
        for i in range(x.shape[0]):
            for k in range(x.shape[0]):
                v = np.array(R[:, k])
                if i != k:
                    v[i] = -np.inf
                    v[k] = - np.inf
                    v[v < 0] = 0

                    A[i, k] = A[i, k] * damping + (1 - damping) * min(0, R[k, k] + v.sum())

                else:
                    v[k] = -np.inf
                    v[v < 0] = 0
                    A[k, k] = A[k, k] * damping + (1 - damping) * v.sum()

    #optimize it by vectorizing the logic
    else:
        k_k_idx = np.arange(x.shape[0])
        # set a(i, k)
        v = np.array(R)
        v[v < 0] = 0
        np.fill_diagonal(v, 0)
        v = v.sum(axis=0) # columnwise sum
        v = v + R[k_k_idx, k_k_idx]

        # broadcasting of columns 'r(k, k) + sum(max(0, r(i', k))) to rows.
        v = np.ones(A.shape) * v

        # For every column k, subtract the positive value of k. 
        # This value is included in the sum and shouldn't be
        v -= np.clip(R, 0, np.inf)
        v[v > 0] = 0
        
        # set(a(k, k))
        v_ = np.array(R)
        np.fill_diagonal(v_, 0)

        v_[v_ < 0] = 0

        v[k_k_idx, k_k_idx] = v_.sum(axis=0) # column wise sum
        A = A * damping + (1 - damping) * v

def similarity(xi, xj):
    '''
The first messages sent per iteration, are the responsibilities. These responsibility values are based on a similarity function s

+The negative euclidean  distance squared
s(i,k) = -||xi - xk||^2
'''
    return -((xi - xj)**2).sum()

def init_matrices():
    '''
We can simply implement this similarity function and define a similarity matrix S, 
which is a graph of the similarities between all the points. We also initialize the R and A matrix to zeros.
'''
    S = np.zeros((x.shape[0], x.shape[0])) #shape[0] get the number of rows
    R = np.array(S)
    A = np.array(S)
    # when looking in row i, the value means you should compare to column i - value
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            S[i, j] = similarity(x[i], x[j])
    
    return A, R, S

def extract_result(AR):
    #every data point i chooses the maximum index k
    labels = np.argmax(AR, axis=1)
    exemplars = np.unique(labels)

    return labels, exemplars
    
'''samples data

#Generate sample data
centers = [[1, 3], [1, 4], [1, 2]] #cluster overlapped
#centers = [[1, 1], [4, 4], [7, 7]] #cluster separated
n = 100
predefined_iter = 1500 #predefined number of loops

x, labels_true = make_blobs(n_samples=n*3, centers=centers, cluster_std=0.5, random_state=0)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

A, R, S = init_matrices()


#A value close to min(s) can produce fewer classes, while a value close to or greater than max(s) can produce more classes. 
#It is usually initialized with the mean median(s) or min(s).

#preference = np.median(S)
#preference = np.min(s)
preference = -50

np.fill_diagonal(S, preference)

#Damping is a factor for numerical stability and can slow down the convergence learning rate. 
#According to sklearn, choose damping in the range from 0.5 to 1

damping = 0.5

#figures = []
last_AR = np.ones(A.shape)
i=0

print('Affinity Propagation begins')
start_time = time.time()
while True:
    update_r(damping, slow=False)
    update_a(damping, slow=False)
    
    AR = A + R

    #Uncomment if make_gif
    #make_gif could affect program speed
    #if i % 5 == 0:
    #    labels, exemplars = extract_result(AR)
    #    fig = plot_iteration(labels, exemplars, i)
    #    figures.append(fig)

    if np.allclose(last_AR, AR) or i == predefined_iter:
        labels, exemplars = extract_result(AR)
        print('\n', exemplars, end='\n')
        time_taken = time.time() - start_time
        print('time taken: %.4f (s)' % time_taken)
        
        fig = plot_iteration(labels, exemplars, i, show=True)
        #figures.append(fig)
        break

    last_AR = AR
    i += 1
    print (i," iteration(s) completed", end='\r')

print('#########')
print('Estimated number of clusters: %d' % len(exemplars))
print("Homogeneity (Tính đồng nhất): %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness (Tính đầy đủ): %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure (Phân phối V): %0.3f" % metrics.v_measure_score(labels_true, labels))
print('#########')
#make_gif(figures, 'test2.gif', 5) #uncomment if make_gif
'''

#real data
filePath = os.getcwd() + '\data\Mall_Customers_edited.csv'
feature = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
x = read_file(filePath, feature, 'CustomerID')

#plot original data
figure1 = plt.figure(figsize=(10,6))
figure1 = plt.axes()
figure1.set_xlabel('Age')
figure1.set_ylabel('Spending Score (1-100)')
figure1.set_title('Spending Score based on Age')
figure1.plot(x[:,1], x[:,3], 'b.')  #[0,:] first row and all column | [:,0] first column and all row

figure2 = plt.figure(figsize=(6,6))
figure2 = plt.axes(projection='3d')
figure2.set_xlabel('Age')
figure2.set_ylabel('Annual Income (k$)')
figure2.set_zlabel('Spending Score (1-100)')
#da.set_title('Spending Score based on Age')
figure2.scatter3D(x[:,1], x[:,2], x[:,3], c=x[:,3], cmap='viridis')

plt.show()

A, R, S = init_matrices()

predefined_iter = 1500

#preference = -50
#preference = np.min(S)
preference = np.median(S)
np.fill_diagonal(S, preference)
damping = 0.5
last_AR = np.ones(A.shape) #empty
i=0

print('Affinity Propagation begins')
start_time = time.time()
while True:
    update_r(damping, slow=False)
    update_a(damping, slow=False)

    AR = A + R
    
    if np.allclose(last_AR, AR) or i == predefined_iter:
        labels, exemplars = extract_result(AR)
        time_taken = time.time() - start_time
        print('\ntime taken: %.4f (s)' % time_taken)
        plot_result(labels, exemplars, i)
        
        #customerId starts with 1
        labels = labels + 1
        exemplars = exemplars + 1
        data['label'] = labels
        break

    last_AR = AR
    i+=1
    print (i," iteration(s) completed", end='\r')

create_csv('result1.csv')
print('Number of Groups: %s' % len(exemplars))

##AP from sklearn
from sklearn.cluster import AffinityPropagation
clustering = AffinityPropagation(random_state=None).fit(x)
labels_true = clustering.labels_
#exemplars_true = np.unique(labels_true)
#iter = clustering.n_iter_
#plot_result(labels_true, exemplars_true, iter)

print('#########')
print("Homogeneity (Tính đồng nhất): %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness (Tính đầy đủ): %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure (Phân phối V): %0.3f" % metrics.v_measure_score(labels_true, labels))
print('#########')

#print('Gender', '| Age', '| Annual Income (k$)', '| Spending Score (1-100)', '| Gender: 0 = Male, 1 = Female')
#for i in range(len(exemplars)):
#    print('Group %d: ' %(i+1))
#    for j in range(len(labels)):
#        if labels[j] == exemplars[i]:
#            print(x[j])