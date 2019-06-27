import pandas as pd
import natsort
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .Handshape import handshape
import numpy as np
from sklearn.cluster import KMeans
np.seterr(divide='ignore', invalid='ignore')
from sklearn.cluster import KMeans
from bokeh.plotting import figure, output_notebook, show
from bokeh.palettes import brewer
from bokeh.palettes import mpl
from bokeh.palettes import d3

def doKmeans(X, nclust=10):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

def preprocessing(json_directory, frames_directory):
        
        df = pd.DataFrame(columns=['JSON', 'FRAME'])
        # find json files
        json_files = [f for f in listdir(json_directory) if isfile(join(json_directory, f))]
        sorted_json_files = natsort.natsorted(json_files)
        del sorted_json_files[0]
        
        
        # find frame files
        frame_files = [f for f in listdir(frames_directory) if isfile(join(frames_directory, f))]
        sorted_frame_files = natsort.natsorted(frame_files)
        

        
        i=0
        for index, keyfile in enumerate(sorted_json_files):
            js_file = json_directory+str(keyfile)
            fr_file = frames_directory+str(sorted_frame_files[index])

            df.loc[i] = [js_file,fr_file]
            i = i+1
            
        if (len(sorted_frame_files) - len(sorted_json_files) == 1):
            df = df.drop(df.index[-1])
        
        master_array = np.empty((len(sorted_frame_files),21,2), dtype='float64')
        for index, row in df.iterrows():
            coo, cf = handshape(row['JSON']).get_right_fingers_from_json
            master_array[index] = coo
                
        train_dataset = master_array.reshape((len(sorted_frame_files),21*2))
        train_dataset = np.nan_to_num(train_dataset)
        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(train_dataset)
        principalDf = pd.DataFrame(data = principalComponents
                         , columns = ['principal component 1', 'principal component 2'])
        
        
        
        
        return(df,sorted_json_files,sorted_frame_files, train_dataset, principalDf)
#         self._sorted_frame_files = sorted_frame_files
#         self._sorted_frame_files = sorted_frame_files
        


class visualization(object):
    """
    Object to visualize pca, kmeans
    from a directory with jsons from the OpenPose output
    """
    def __init__(self, json_directory, frame_directory,  *args, **kwargs):
        """    
        parameters:
        -----------
        json_directory (ex. /data/),
        frame_directory,
        
        optional:
        elbow: True/False,
        range_n_clusters = [] | default = 15
        """
        self._directory = json_directory
        self._frame_directory = frame_directory
        self._elbow_range = kwargs.get('elbow_range', None)
        self._range_n_clusters = kwargs.get('range_n_clusters', 15)
        self._pca_show = True
        
        self._df,self._sorted_json_files,self._sorted_frame_files, self._train_dataset, self._principalDf = preprocessing(self._directory, self._frame_directory)
        
        
                
        
        
    @property
    def pca(self):
        
        if (self._pca_show):
            
            fig = plt.figure(figsize = (12,12))
            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel('Principal Component 1', fontsize = 15)
            ax.set_ylabel('Principal Component 2', fontsize = 15)
            ax.set_title('2 component PCA', fontsize = 20)
            ax.scatter(self._principalDf['principal component 1'],self._principalDf['principal component 2'],s=4.8)

            ax.grid()
        #return(principalDf)
        
    @property
    def elbow_method(self):
        
        plt.figure(figsize=(10, 8))
        
        wcss = []
        for i in range(1,self._elbow_range):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(self._train_dataset)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1,self._elbow_range), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
    
    @property
    def run_k_means(self):
       
        clust_labels, cent = doKmeans(self._train_dataset, self._range_n_clusters)
        kmeans = pd.DataFrame(clust_labels)        
        
        return(kmeans)
    
    @property
    def visualize_kmeans(self):
        kmeans = self.run_k_means
        # output to static HTML file
        output_notebook()

        TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)")
        ]

        p1 = figure(plot_width=500, plot_height=500, tooltips=TOOLTIPS,)

        # Get the number of colors we'll need for the plot.
        colors = d3["Category20"][len(kmeans[0].unique())]

        # Create a map between factor and color.
        colormap = {i: colors[i] for i in kmeans[0].unique()}

        # Create a list of colors for each value that we will be looking at.
        colors = [colormap[x] for x in kmeans[0]]
        # add a square renderer with a size, color, and alpha
        p1.circle(self._principalDf['principal component 1'],self._principalDf['principal component 2'], size=2, color=colors, alpha=0.7)

        # show the results
        show(p1)
        
        
    @property
    def silhouette_analysis(self):
        from sklearn.datasets import make_blobs
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_samples, silhouette_score

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        print(__doc__)
        
        X = self._train_dataset

        range_n_clusters = self._range_n_clusters

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

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

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        plt.show()