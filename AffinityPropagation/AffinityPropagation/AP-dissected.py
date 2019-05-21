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


def make_gif(figures, filename, fps=10, **kwargs):
    '''Make gif result'''
    images = []
    for fig in figures:
        output = BytesIO()
        fig.savefig(output)
        plt.close(fig)  
        output.seek(0)
        images.append(imageio.imread(output))
    imageio.mimsave(filename, images, fps=fps, **kwargs)

def similarity(xi, xj):
    '''
The first messages sent per iteration, are the responsibilities. These responsibility values are based on a similarity function s

+The negative euclidian distance squared
s(i,k) = -||xi - xk||^2
'''
    return -((xi - xj)**2).sum()

def create_matrices():
    '''
We can simply implement this similarity function and define a similarity matrix S, 
which is a graph of the similarities between all the points. We also initialize the R and A matrix to zeros.
'''
    S = np.zeros((x.shape[0], x.shape[0]))
    R = np.array(S)
    A = np.array(S)
    # when looking in row i, the value means you should compare to column i - value
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            S[i, j] = similarity(x[i], x[j])
    
    #print("Done create_matrices!")
    return A, R, S

def update_r(damping=0.9, slow=False):
    '''
responsibility messages:
r(i,k) <-- s(i,k) - MAX {a(i,k') + s(i,k')} with k's.t.k' ≠ k
We could implement this with a nested for loop where we iterate over every row i and then determine the 
max(A+S) (of that row) for every index not equal to k or i (The index should not be equal to i as it would be sending messages to itself). 

The damping factor is just there for nummerical stabilization and can be regarded as a slowly converging learning rate. 
The authors advised to choose a damping factor within the range of 0.5 to 1.
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

def read_file(FILE_PATH, FEATURE):
    data = pd.read_csv(FILE_PATH, names=FEATURE)
    print('Done read csv file')
    print(data.head())
    print('-----------------------------------------------------------------------')
    data = data.drop('CustomerID', 1)
    x = data.values
    return x

def plot_result(A, R):
    sol = A + R
    
    labels = np.argmax(sol, axis=1)

    exemplars = np.unique(labels)
    colors = dict(zip(exemplars, cycle('bgrcmyk')))
    fig2 = plt.figure(figsize=(12, 6))
    re = plt.axes()
    re.set_xlabel('Age')
    re.set_ylabel('Spending Score (1-100)')
    for i in range(len(labels)):
        X = x[i][1]
        Y = x[i][3]
        
        if i in exemplars:
            exemplar = i
            edge = 'k'
            ms = 10
        else:
            exemplar = labels[i]
            ms = 3
            edge = None
            re.plot([X, x[exemplar][1]], [Y, x[exemplar][3]], c=colors[exemplar])
        re.plot(X, Y, 'o', markersize=ms,  markeredgecolor=edge, c=colors[exemplar])
        

    re.set_title('Number of exemplars: %s' % len(exemplars))
    plt.show()
    #return fig, labels, exemplars
    return labels, exemplars

#for genarated data
def plot_iteration(A, R):
    fig = plt.figure(figsize=(12, 6))
    sol = A + R
    # every data point i chooses the maximum index k
    labels = np.argmax(sol, axis=1)
    exemplars = np.unique(labels)
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
            edge = None
            plt.plot([X, x[exemplar][0]], [Y, x[exemplar][1]], c=colors[exemplar])
        plt.plot(X, Y, 'o', markersize=ms,  markeredgecolor=edge, c=colors[exemplar])
        

    plt.title('Number of exemplars: %s' % len(exemplars))
    return fig, labels, exemplars

def create_csv():
    print('Creating result.csv')
    try:
        with open('result.csv', mode='w') as data:
            data = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data.writerow(['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
            for i in range(len(exemplars)):
                for j in range(len(labels)):
                    if labels[j] == exemplars[i]:
                        data.writerow(x[j])
                data.writerow(['---', '---', '---', '---'])

        print('Successfully create result.csv!')
    except:
        print('Something went wrong!')


#==================================================
#n = 50
#size = (n, 2)
#np.random.seed(1)
#x = np.random.normal(0, 1, size)
#x = np.append(x, np.random.normal(5, 1, size), axis=0)

#c = ['r' for _ in range(n)] + ['b' for _ in range(n)]
#plt.scatter(x[:, 0], x[:, 1], c=c)
#plt.show()

#A, R, S = create_matrices()
#preference = np.median(S)
##preference =  -1000

#np.fill_diagonal(S, preference)
#damping = 0.5

#figures = []
#start_time = time.time()
#for i in tqdm(range(n)):
#    update_r(damping, slow=False)
#    update_a(damping, slow=False)
    
#    if i % 5 == 0:
#        fig, labels, exemplars = plot_iteration(A, R)
#        figures.append(fig)

#make_gif(figures, 'test.gif', 2)
#time_taken = time.time() - start_time
#print('time taken: %.4f (s)' % time_taken)

#==================================================
filePath = os.getcwd() + '\data\Mall_Customers_edited.csv'
feature = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
x = read_file(filePath, feature)

#plot original data
fig1 = plt.figure(figsize=(12, 6))
da = plt.axes()
da.set_xlabel('Age')
da.set_ylabel('Spending Score (1-100)')
da.set_title('Spending Score based on Age')
da.plot(x[:, 1], x[:, 3], 'b.')

fig2 = plt.figure(figsize=(12, 6))
da = plt.axes()
da.set_xlabel('Age')
da.set_ylabel('Spending Score (1-100)')
da.set_title('Spending Score based on Age')
da.bar(x[:, 1], x[:, 3])

plt.show()

#create matrix
A, R, S = create_matrices()
preference = np.median(S)
np.fill_diagonal(S, preference)
damping = 0.5

print('Affinity Propagation begins')
start_time = time.time()
for i in tqdm(range(len(x))):
    update_r(damping, slow=False)
    update_a(damping, slow=False)

labels, exemplars = plot_result(A, R)
create_csv()
time_taken = time.time() - start_time
print('time taken: %.4f (s)' % time_taken)
print('-----------------------------------------------------------------------')
print('Number of Groups: %s' % len(exemplars))
print('Gender', '| Age', '| Annual Income (k$)', '| Spending Score (1-100)', '| Gender: 0 = Male, 1 = Female')
for i in range(len(exemplars)):
    print('Group %d: ' %(i+1))
    for j in range(len(labels)):
        if labels[j] == exemplars[i]:
            print(x[j])
            
    print('-------------')