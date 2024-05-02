import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import numpy as np
from matplotlib import cm
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def biplot(score, coeff, y=None, labels:list=None, 
           figsize=(12,9), xlim=None, ylim=None, scale=False):
    """Plot a biplot 

    A biplot is plot which aims to represent both the observations and variables of a matrix of multivariate 
    data on the same plot. 
    Args:
        score (np.ndarray): Score of PCAs.
        coeff (np.ndarray): Component of PCAs.
        y (pd.core.series.Series, optional): sample labels. Defaults to None.
        labels (list, optional): sample properties. Defaults to None.
        figsize (tuple, optional): Size of created figure. Defaults to (12,9).
        xlim (tuple, optional): X limits of created plot. Defaults to None.
        ylim (tuple, optional): Y limits of creataed plot. Defaults to None.
        scale(bool, optional): 
    """
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    if scale:
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        scale_xs = xs * scalex
        scale_ys = ys * scaley
    else:
        scale_xs = xs 
        scale_ys = ys 
        
    # df = pd.DataFrame({'x':xs * scalex, 'y':ys * scaley})
    # df['c'] = y.values
    _, ax = plt.subplots(figsize=figsize)
    # if labels is None:
    sns.scatterplot(x=scale_xs * 1.15, y=scale_ys * 1.15, alpha=0)
    # else:
    #     sns.scatterplot(x='x', y='y', hue='c', data=df)
    if y is not None:
        try:
            for i, txt in enumerate(y.values):
                ax.annotate(txt[0], (scale_xs[i], scale_ys[i]))
        except:
            for i, txt in enumerate(y):
                ax.annotate(txt[0], (scale_xs[i], scale_ys[i]))
            
    for i in range(n):
        plt.arrow(0, 0, np.abs(scale_xs).max() * coeff[i,0], np.abs(scale_ys).max() * coeff[i,1],color = 'r')
        if labels is None:
            plt.text(np.abs(scale_xs).max() * coeff[i,0] * 1.15, np.abs(scale_ys).max() * coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(np.abs(scale_xs).max() * coeff[i,0] * 1.15, np.abs(scale_ys).max() * coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

def plot_clusters(df, labels, feature1:str=None, feature2:str=None, centers=None, 
                  show_curves:bool=False, figsize=(12, 9), alpha_curves:float=0.5, scale=False):
    """Plot silhouette and cluster of samples 
    
    Create two plots, one with silhouette score and other with a 2D view of the cluster of df.
    If there are more than 2 features in df, 2D view is made on the first 2 PCAs space.
    labels must have the same samples as df. If there are more than 2 features in df, show_curves 
    argument make a plot assuming you are trying to cluster curve samples, indexed by rows. 
    Args:
        df (pd.core.frame.DataFrame): data.frame containing samples to be clustered
        labels (np.ndarray): cluster labels
        feature1 (str, optional): Name of the x-axis variable to create plot. Defaults to None.
        feature2 (str, optional): Name of the y-axis variable to create plot. Defaults to None.
        centers (np.ndarray, optional): cluster centers. Defaults to None.
        show_curves (bool, optional): show curve plot. Defaults to False.
        figsize (tuple, optional): Size of created figure. Defaults to (12, 9).
        alpha_curves (float, optional): alpha property of curve plot. Defaults to 0.5.
        scale (bool, optional): Scale axis in scatter plot in range [0, 1]. Defaults to False.

    Raises:
        ValueError: feature1 and feature2 must be df column names
    """
    if type(df) is not pd.core.frame.DataFrame:
        raise TypeError(f"df argument must be a pandas DataFrame, object passed is  {type(df)}")
    if df.shape[1] == 1:
        # Plot silhouette 
        # Create a subplot with 1 row and 1 columns
        _, (ax1) = plt.subplots(1, 1, figsize=figsize)
    elif df.shape[1] == 2 or not show_curves:
        # Plot silhouette and real clusters
        # Create a subplot with 1 row and 2 columns
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 2, 4)
    
    if len(labels.shape) > 1:
        labels = np.squeeze(labels)
    # Obtain number of clusters
    n_clusters = len(np.unique(labels))
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, df.shape[0] + (n_clusters + 1) * 10])
    y_lower = 10
    
    # Compute the silhouette scores for each sample
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X=df.values)
    pca = PCA(n_components=2,)
    X_pca = pd.DataFrame(pca.fit_transform(X_transformed), columns=['PC1','PC2'])
    sample_silhouette_values = silhouette_samples(X_transformed, labels)
    if scale:
        scalex = 1.0/(X_pca.iloc[:,0].max() - X_pca.iloc[:,0].min())
        scaley = 1.0/(X_pca.iloc[:,1].max() - X_pca.iloc[:,1].min())
        X_pca.iloc[:,0] = X_pca.iloc[:,0] * scalex
        X_pca.iloc[:,1] = X_pca.iloc[:,1] * scaley
        
        
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        samples_cluster = labels == i
        ith_cluster_silhouette_values = \
            sample_silhouette_values[samples_cluster]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        if df.shape[1] == 2:
            if feature1 is None:
                feature1 = df.columns[0]
            if feature2 is None:
                feature2 = df.columns[1]
            
            if feature1 not in df.columns:
                raise ValueError(f'feature {feature1} is not in provided data')
            if feature2 not in df.columns:
                raise ValueError(f'feature {feature2} is not in provided data') 
            # 2nd Plot showing the actual clusters formed
            df_plot = df.iloc[samples_cluster,:]
            sns.scatterplot(x='X1', y='X2', data=df_plot, color=color, edgecolor='black', legend='full', ax=ax2)

            ax2.set_title("Visualization of the clustered data.")
            ax2.set_xlabel(f"Feature space for {feature1}")
            ax2.set_ylabel(f"Feature space for {feature2}")
            
            if centers is not None:
                # Draw white circles at cluster centers
                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')
        if df.shape[1] > 2:
            
            df_plot = X_pca.iloc[samples_cluster,:]
            sns.scatterplot(x='PC1', y='PC2', data=df_plot, color=color, edgecolor='black', legend='full', ax=ax2)

            ax2.set_title("Visualization of the clustered data.")
            ax2.set_xlabel(f"Feature space for PC1")
            ax2.set_ylabel(f"Feature space for PC2")
            
            if centers is not None:
                center_pca = pca.transform(scaler.transform(centers))
                if scale:
                    center_pca[:,0] = center_pca[:,0] * scalex
                    center_pca[:,1] = center_pca[:,1] * scaley
                    
                # Draw white circles at cluster centers
                ax2.scatter(center_pca[:, 0], center_pca[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for i, c in enumerate(center_pca):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')

    ax1.set_title(f"Silhouette plot. Mean silhouette: {round(sample_silhouette_values.mean(),3)}")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=sample_silhouette_values.mean(), color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    if df.shape[1] > 2 and show_curves:
        ## Multivariate plots
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            samples_cluster = labels == i
            ith_cluster_silhouette_values = \
                sample_silhouette_values[samples_cluster]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            df.iloc[samples_cluster,:].T.plot(legend=None, color=color, ax=ax3, alpha=alpha_curves)
        
    plt.suptitle(("Silhouette analysis for hierarchical clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()   