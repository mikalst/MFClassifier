import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import scipy.sparse
import sklearn.decomposition
import copy

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

    data2 = np.sum(matrix2 == 3, axis=1)
    data4 = np.sum(matrix2 == 4, axis=1)
    data6 = np.sum(matrix2 != 0, axis=1)

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
    plt.spy(matrix1, aspect="auto", markersize=0.06)

    plt.subplot(1, 2, 2)
    plt.title("Original")
    plt.spy(matrix2, aspect="auto", markersize=0.06)
    plt.show()


def plot_single(
    matrix,
    vrange=(1, 4),
    figsize=(4.74, 4.74),
    title=None,
    output_file=None,
    show_missing=None,
    cmap=plt.get_cmap('viridis')
):
    cmap_copy = copy.copy(cmap)
    # make locations under vmin translucent black
    if show_missing:
        cmap_copy.set_under('white', alpha=0.8)

    if show_missing is None:
        show_missing = False

    if title is None:
        title = 'matrix'

    NROWS = 300
    NCOLS = 321

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(figsize[0], figsize[1])
    spec = gs.GridSpec(ncols=4, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[0, 1:3])

    idc_rows = np.linspace(0, matrix.shape[0]-1, NROWS, dtype=np.int)
    idc_cols = np.linspace(0, matrix.shape[1]-1, NCOLS, dtype=np.int)
    
    if scipy.sparse.issparse(matrix):
        matrix_show = matrix[:, idc_cols][idc_rows].todense()
    else:
        matrix_show = matrix[:, idc_cols][idc_rows]

    out = ax.imshow(matrix_show, aspect="auto", 
                    vmin=vrange[0]-0.5, vmax=vrange[1]+0.5, cmap=cmap_copy)

    ax.set_xlabel("T ({})".format(matrix.shape[1]))
    ax.set_ylabel("N ({})".format(matrix.shape[0]))
    ax.tick_params(axis='both', labelbottom=False, labelleft=False)
    ax.set_title(title)

    cbar_ax = fig.add_axes([0.76, 0.10, 0.015, 0.80])
    fig.colorbar(out, cax=cbar_ax, ticks=np.arange(vrange[0], vrange[1]+1), orientation='vertical')

    # Save output
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
        plt.show()


def plot_visual_continuous(
        matrices, 
        vrange=(1, 4),
        figsize=(4.74, 3.00),
        fig_arrangement=(1,2),
        titles=None,
        output_file=None,
        show_missing=None,
        cmap=plt.get_cmap('viridis'),
        suptitle=None
):

    NROWS = 100
    NCOLS = 200

    if type(matrices) not in (tuple, list):
        plot_single(
            matrices,
            vrange,
            figsize,
            titles,
            output_file,
            show_missing,
            cmap
        )
        return

    nonmasked_cmap = copy.copy(cmap)
    masked_cmap = copy.copy(cmap)
    # make locations under vmin translucent black
    masked_cmap.set_under('white', alpha=0.8)

    if show_missing is None:
        show_missing = [False]*len(matrices)

    get_cmap = lambda i: masked_cmap if show_missing[i] else nonmasked_cmap

    if titles is None:
        titles = ['matrix' + str(i) for i in range(len(matrices))]

    fig, axes = plt.subplots(
        nrows=fig_arrangement[0],
        ncols=fig_arrangement[1]
    )

    fig.set_size_inches(figsize[0], figsize[1])

    for i, ax in enumerate(axes.flat):

        idc_rows = np.linspace(0, matrices[i].shape[0]-1, NROWS, dtype=np.int)
        idc_cols = np.linspace(0, matrices[i].shape[1]-1, NCOLS, dtype=np.int)

        if scipy.sparse.issparse(matrices[i]):
            matrix = matrices[i][:, idc_cols][idc_rows].todense()
        else:
            matrix = matrices[i][:, idc_cols][idc_rows]

        out = ax.imshow(matrix, aspect="auto", 
                        vmin=vrange[0], vmax=vrange[1], cmap=get_cmap(i))

        ax.set_ylabel("N")
        ax.set_xlabel("T ({})".format(matrices[i].shape[1]))
        ax.tick_params(axis='both', labelbottom=False, direction='inout', color='k', labelleft=False)
        ax.set_ylabel("N ({})".format(matrices[i].shape[0]))
        ax.title.set_text(titles[i])
    
    if fig_arrangement[1] == 2:
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.125, 0.015, 0.755])
        fig.colorbar(out, cbar_ax, orientation='vertical')
    elif fig_arrangement[1] == 3:
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.125, 0.015, 0.755])
        fig.colorbar(out, cbar_ax, orientation='vertical')
        #plt.subplots_adjust(wspace=0.12, hspace=0.12)
        #plt.tight_layout()
    if not(suptitle is None):
        plt.suptitle(suptitle)

    # Save output
    if output_file is None:
        plt.show()
    else:
        plt.savefig(
            output_file,
            dpi=1000, 
            # Plot will be occupy a maximum of available space
            bbox_inches='tight'
        )


def plot_visual(
        matrices, 
        vrange=(1, 4),
        figsize=(4.74, 4.74),
        fig_arrangement=(1,2),
        titles=None,
        output_file=None,
        show_missing=None,
        cmap=plt.get_cmap('viridis')
):

    NROWS = 100
    NCOLS = 200

    if type(matrices) not in (tuple, list):
        plot_single(
            matrices,
            vrange,
            figsize,
            titles,
            output_file,
            show_missing,
            cmap
        )
        return

    nonmasked_cmap = copy.copy(cmap)
    masked_cmap = copy.copy(cmap)
    # make locations under vmin translucent black
    masked_cmap.set_under('white', alpha=0.8)

    if show_missing is None:
        show_missing = [False]*len(matrices)

    get_cmap = lambda i: masked_cmap if show_missing[i] else nonmasked_cmap

    if titles is None:
        titles = ['matrix' + str(i) for i in range(len(matrices))]

    fig, axes = plt.subplots(
        nrows=fig_arrangement[0],
        ncols=fig_arrangement[1]
    )

    fig.set_size_inches(figsize[0], figsize[1])

    for i, ax in enumerate(axes.flat):

        idc_rows = np.linspace(0, matrices[i].shape[0]-1, NROWS, dtype=np.int)
        idc_cols = np.linspace(0, matrices[i].shape[1]-1, NCOLS, dtype=np.int)

        if scipy.sparse.issparse(matrices[i]):
            matrix = matrices[i][:, idc_cols][idc_rows].todense()
        else:
            matrix = matrices[i][:, idc_cols][idc_rows]

        out = ax.imshow(matrix, aspect="auto", 
                        vmin=vrange[0]-0.5, vmax=vrange[1]+0.5, cmap=get_cmap(i))

        ax.set_ylabel("N")
        ax.set_xlabel("T ({})".format(matrices[i].shape[1]))
        ax.tick_params(axis='both', labelbottom=False, labelleft=False)
        ax.tick_params(axis='both', length=0)
        ax.set_ylabel("N ({})".format(matrices[i].shape[0]))
        ax.title.set_text(titles[i])

    if fig_arrangement[1] == 2:
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.125, 0.015, 0.755])
        fig.colorbar(out, cbar_ax, ticks=np.arange(vrange[0], vrange[1]+1), orientation='vertical')
    elif fig_arrangement[1] == 3:
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.125, 0.015, 0.755])
        fig.colorbar(out, cbar_ax, ticks=np.arange(vrange[0], vrange[1]+1), orientation='vertical')
        #plt.subplots_adjust(wspace=0.12, hspace=0.12)
        #plt.tight_layout()

    # Save output
    if output_file is None:
        plt.show()
    else:
        plt.savefig(
            output_file,
            # Plot will be occupy a maximum of available space
            bbox_inches='tight'
        )


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
