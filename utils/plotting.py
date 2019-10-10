import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import sklearn.decomposition

# Import local files
from .stats import distribution_per_age


def plot_distribution_per_woman(matrix1, matrix2, sparse=False, title_postfix=""):
    fig = plt.figure(figsize=(12, 12))

    if scipy.sparse.issparse(matrix1):
        data1 = np.sum(matrix1 == 3, axis=1).A1
        data3 = np.sum(matrix1 == 4, axis=1).A1
        data5 = np.sum(matrix1 != 0, axis=1).A1
        
    else:
        data1 = np.sum(matrix1 == 3, axis=1)
        data3 = np.sum(matrix1 == 4, axis=1)
        data5 = np.sum(matrix1 != 0, axis=1)
        
    data2 = np.sum(matrix2 == 3, axis=1).A1
    data4 = np.sum(matrix2 == 4, axis=1).A1
    data6 = np.sum(matrix2 != 0, axis=1).A1
    
    plt.subplot(3, 2, 1)
    plt.title("Synthetic" + title_postfix)
    plt.ylabel("All risk levels")
    plt.hist(data5, bins = np.linspace(0, 40).astype(np.int), edgecolor='k')
    
    plt.subplot(3, 2, 2)
    plt.title("Original" + title_postfix)
    plt.ylabel("All risk levels")
    plt.hist(data6, bins = np.linspace(0, 40).astype(np.int), edgecolor='k')
    
    plt.subplot(3, 2, 3)
    plt.ylabel("risk level 3")
    plt.hist(data1, bins = np.linspace(0, 20).astype(np.int), edgecolor='k')

    plt.subplot(3, 2, 4)
    plt.hist(data2, bins = np.linspace(0, 20).astype(np.int), edgecolor='k')
    
    plt.subplot(3, 2, 5)
    plt.ylabel("risk level 4")
    plt.hist(data3, bins = np.linspace(0, 4).astype(np.int), edgecolor='k')

    plt.subplot(3, 2, 6)
    plt.hist(data4, bins = np.linspace(0, 4).astype(np.int), edgecolor='k')
    plt.show()
    
def plot_distribution_per_age(matrix1, matrix2, sparse=False, title_postfix=""):
    fig = plt.figure(figsize=(12, 10))

    data1, data3, data5 = distribution_per_age(matrix1)
    data2, data4, data6 = distribution_per_age(matrix2)
    
    plt.subplot(3, 2, 1)
    plt.title("Synthetic" + title_postfix)
    plt.ylabel("All risk levels")
    plt.scatter(np.linspace(0, 320, 321), data5, marker="x")
    
    plt.subplot(3, 2, 2)
    plt.title("Original" + title_postfix)
    plt.ylabel("All risk levels")
    plt.scatter(np.linspace(0, 320, 321), data6, marker="x")
    
    plt.subplot(3, 2, 3)
    plt.title("Synthetic" + title_postfix)
    plt.ylabel("risk level 3")
    plt.scatter(np.linspace(0, 320, 321), data1, marker="x")

    plt.subplot(3, 2, 4)
    plt.title("Original")
    plt.scatter(np.linspace(0, 320, 321), data2, marker="x")
    
    plt.subplot(3, 2, 5)
    plt.ylabel("risk level 4")
    plt.scatter(np.linspace(0, 320, 321), data3, marker="x")

    plt.subplot(3, 2, 6)
    plt.scatter(np.linspace(0, 320, 321), data4, marker="x")
    plt.show()
        
def plot_sparsity(matrix1, matrix2):
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Synthetic masked")
    out = plt.spy(matrix1, aspect="auto", markersize=0.06)

    plt.subplot(1, 2, 2)
    plt.title("Original")
    out = plt.spy(matrix2, aspect="auto", markersize=0.06)
    plt.show()
    
def plot_visual(matrix1, matrix2, titles=("Synthetic", "Original")):
    fig = plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title(titles[0])
    if scipy.sparse.issparse(matrix1):
        out = plt.imshow(matrix1.todense(), aspect="auto")
    else:
        out = plt.imshow(matrix1, aspect="auto")
    out = plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title(titles[1])
    if scipy.sparse.issparse(matrix2):
        out = plt.imshow(matrix2.todense(), aspect="auto")
    else:
        out = plt.imshow(matrix2, aspect="auto")
    out = plt.colorbar()
    plt.show()
    
def plot_compare_svd_decomp_with_basis(S, path_to_basis):

    original_singular_components = np.load(path_to_basis)
    n_components = original_singular_components.shape[0]        

    pca = sklearn.decomposition.PCA(n_components=n_components, copy=True, whiten=True,
        svd_solver='auto', tol=0.0, iterated_power='auto',
        random_state=None)
    pca.fit(S)

    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.title("Recovered")
    for sc in pca.components_[:n_components]:
        plt.plot(sc)
        
    plt.subplot(1,2,2)
    plt.title("Original")
    for sc in original_singular_components:
        plt.plot(sc)
    plt.show()
