import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

def plot_distribution_per_woman(matrix_synthetic, matrix_original, sparse=False, title_postfix=""):
    fig = plt.figure(figsize=(12, 12))

    if scipy.sparse.issparse(matrix_synthetic):
        data1 = np.sum(matrix_synthetic == 3, axis=1).A1
        data3 = np.sum(matrix_synthetic == 4, axis=1).A1
        data5 = np.sum(matrix_synthetic != 0, axis=1).A1
        
    else:
        data1 = np.sum(matrix_synthetic == 3, axis=1)
        data3 = np.sum(matrix_synthetic == 4, axis=1)
        data5 = np.sum(matrix_synthetic != 0, axis=1)
        
    data2 = np.sum(matrix_original == 3, axis=1).A1
    data4 = np.sum(matrix_original == 4, axis=1).A1
    data6 = np.sum(matrix_original != 0, axis=1).A1
    
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
    
def plot_distribution_per_age(matrix_synthetic, matrix_original, sparse=False, title_postfix=""):
    fig = plt.figure(figsize=(12, 10))

    if scipy.sparse.issparse(matrix_synthetic):
        data1 = np.sum(matrix_synthetic == 3, axis=0).A1 / np.sum(matrix_synthetic == 3)
        data3 = np.sum(matrix_synthetic == 4, axis=0).A1 / np.sum(matrix_synthetic == 4)
        data5 = np.sum(matrix_synthetic != 0, axis=0).A1 / matrix_synthetic.count_nonzero()
        
    else:
        data1 = np.sum(matrix_synthetic == 3, axis=0) / np.sum(matrix_synthetic == 3)
        data3 = np.sum(matrix_synthetic == 4, axis=0) / np.sum(matrix_synthetic == 4)
        data5 = np.sum(matrix_synthetic != 0, axis=0) / np.sum(matrix_synthetic != 0)
        
    data2 = np.sum(matrix_original == 3, axis=0).A1 / np.sum(matrix_original == 3)
    data4 = np.sum(matrix_original == 4, axis=0).A1 / np.sum(matrix_original == 4)
    data6 = np.sum(matrix_original != 0, axis=0).A1 / matrix_original.count_nonzero()
    
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
        
def plot_sparsity(matrix_synthetic, matrix_original):
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Synthetic masked")
    out = plt.spy(matrix_synthetic, aspect="auto", markersize=0.06)

    plt.subplot(1, 2, 2)
    plt.title("Original")
    out = plt.spy(matrix_original, aspect="auto", markersize=0.06)
    
def plot_visual(matrix_synthetic, matrix_original, titles=("Synthetic", "Original")):
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(titles[0])
    if scipy.sparse.issparse(matrix_synthetic):
        out = plt.imshow(matrix_synthetic.todense(), aspect="auto")
    else:
        out = plt.imshow(matrix_synthetic, aspect="auto")

    plt.subplot(1, 2, 2)
    plt.title(titles[1])
    out = plt.imshow(matrix_original.todense(), aspect="auto")
    