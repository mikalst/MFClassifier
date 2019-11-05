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
    plt.hist(data5, bins=np.linspace(0, 40).astype(np.int), edgecolor='k')

    plt.subplot(3, 2, 2)
    plt.title("Original" + title_postfix)
    plt.ylabel("All risk levels")
    plt.hist(data6, bins=np.linspace(0, 40).astype(np.int), edgecolor='k')

    plt.subplot(3, 2, 3)
    plt.ylabel("risk level 3")
    plt.hist(data1, bins=np.linspace(0, 20).astype(np.int), edgecolor='k')

    plt.subplot(3, 2, 4)
    plt.hist(data2, bins=np.linspace(0, 20).astype(np.int), edgecolor='k')

    plt.subplot(3, 2, 5)
    plt.ylabel("risk level 4")
    plt.hist(data3, bins=np.linspace(0, 4).astype(np.int), edgecolor='k')

    plt.subplot(3, 2, 6)
    plt.hist(data4, bins=np.linspace(0, 4).astype(np.int), edgecolor='k')
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


def plot_visual(matrices, vrange=(1, 4), titles=None, output_file=None):

    if titles is None:
        titles = ['matrix' + str(i) for i in range(len(matrices))]

    if len(matrices) % 2 == 0:
        fig, axes = plt.subplots(
            figsize=(14, 6*(len(matrices)//2)),
            nrows=len(matrices)//2,
            ncols=2
        )

        for i, matrix in enumerate(matrices):
            if scipy.sparse.issparse(matrix):
                out = axes[i].imshow(matrix.todense(), aspect="auto",
                                     vmin=vrange[0], vmax=vrange[1])
            else:
                out = axes[i].imshow(matrix, aspect="auto",
                                     vmin=vrange[0], vmax=vrange[1])
            axes[i].title.set_text(titles[i])

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
        fig.colorbar(out, cax=cbar_ax)

    elif len(matrices) % 3 == 0:
        fig, axes = plt.subplots(
            figsize=(14, 6*(len(matrices)//3)),
            nrows=len(matrices)//3,
            ncols=3
        )

        for i, matrix in enumerate(matrices):
            if scipy.sparse.issparse(matrix):
                out = axes[i].imshow(matrix.todense(), aspect="auto",
                                     vmin=vrange[0], vmax=vrange[1])
            else:
                out = axes[i].imshow(matrix, aspect="auto",
                                     vmin=vrange[0], vmax=vrange[1])
            axes[i].title.set_text(titles[i])

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
        fig.colorbar(out, cax=cbar_ax)

    else:
        fig, axes = plt.subplots(
            figsize=(14, 6*(len(matrices)//3 + 1)),
            nrows=len(matrices)//3 + 1,
            ncols=3
        )

        for i, matrix in enumerate(matrices):
            if scipy.sparse.issparse(matrix):
                out = axes[i].imshow(matrix.todense(), aspect="auto",
                                     vmin=vrange[0], vmax=vrange[1])
            else:
                out = axes[i].imshow(matrix, aspect="auto",
                                     vmin=vrange[0], vmax=vrange[1])
            axes[i].title.set_text(titles[i])

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
        fig.colorbar(out, cax=cbar_ax)

    # Save output
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
        plt.show()


def plot_compare_svd_with_basis(S, path_to_basis):
    original_singular_components = np.load(path_to_basis)
    n_components = original_singular_components.shape[0]

    pca = sklearn.decomposition.PCA(n_components=n_components, copy=True, whiten=True,
                                    svd_solver='auto', tol=0.0, iterated_power='auto',
                                    random_state=None)
    pca.fit(S)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Recovered")
    for sc in pca.components_[:n_components]:
        plt.plot(sc)

    plt.subplot(1, 2, 2)
    plt.title("Original")
    for sc in original_singular_components:
        plt.plot(sc)
    plt.show()


def plot_compare_svd_with_svd(X1, X2, components):
    try:
        first_comp = components[0]
        last_comp = components[1]
    except TypeError:
        first_comp = 0
        last_comp = components

    pca1 = sklearn.decomposition.PCA(n_components=last_comp, copy=True, whiten=True,
                                     svd_solver='auto', tol=0.0, iterated_power='auto',
                                     random_state=None)
    pca1.fit(X1)
    pca2 = sklearn.decomposition.PCA(n_components=last_comp, copy=True, whiten=True,
                                     svd_solver='auto', tol=0.0, iterated_power='auto',
                                     random_state=None)
    pca2.fit(X2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Recovered")
    for sc in pca1.components_[first_comp:last_comp]:
        plt.plot(sc)

    plt.subplot(1, 2, 2)
    plt.title("Original")
    for sc in pca2.components_[first_comp:last_comp]:
        plt.plot(sc)
    plt.show()
