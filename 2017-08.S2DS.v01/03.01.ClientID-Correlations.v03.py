# WHAT
# This is to compute the correlations of variables: this can be used with log if needed
# ----
# AUTHOR = Daniele Scopece, 2017-08-29 (adapted)
#
#
#---------------------------------------------------------------------------------

# ---- Imports various
import csv
import scipy
print('scipy: {}'.format(scipy.__version__))
from scipy.cluster.hierarchy import dendrogram, linkage as lkg, cophenet
from scipy.spatial.distance import pdist

import pandas as pd
from pandas.tools.plotting import scatter_matrix

import numpy as np
print('numpy: {}'.format(np.__version__))

import matplotlib
import matplotlib.pyplot as plt
print('matplotlib: {}'.format( matplotlib.__version__))
#from mpl_toolkits.mplot3d import Axes3D

import sklearn
print ('sklearn: {}'.format(sklearn.__version__))
#from sklearn import manifold
from sklearn.cluster import KMeans, AgglomerativeClustering as AggC, DBSCAN
#from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import seaborn as sns
#--------------


## PARAMETERS TO TWEAK ===============================================
#----- Data input

from datetime import datetime
from datetime import timedelta

# create directory
import os


# What to do
dokMeans = False
dokElbow = False #used to optimise number of clusters in kMeans
doHierarchy = False
doDendro = False #doHierarchy must be true
maxDays = 45
doScatterPlots = True

if dokElbow:
    dokMeans = True

if doDendro:
    doHierarchy = True



## END of PARAMETERS TO TWEAK ===============================================

#-----------------------------------------------------------------------
def removeNANs(POI, clust_df):
    #tmp = clust_df.dropna(axis=0,how='any')
    for par in POI:
        if par != 'click_days_last_visit_max':
            clust_df[par].fillna(0, inplace=True)
        elif par == 'click_days_last_visit_max':
            clust_df[par].fillna(maxDays+1, inplace=True)

    tmp = clust_df
    return tmp

#-----------------------------------------------------------
def correlations():
    corr = clean_clust_df.corr(method='spearman')

    cmap =sns.diverging_palette(5, 250, as_cmap=True)


    #labels = (np.asarray(["{0} \n {1:.2f}".format(symb,value) for symb, value in zip(symbol.flatten(),perchange.flatten())]))

    sns.heatmap(corr,square=True, annot=True , fmt=".2f", annot_kws={"size":5}, vmin = -1.0 , vmax = 1.0,cmap=cmap,cbar=True)
    #plt.imshow(corr,cmap=plt.cm.bwr_r, vmin=-1.0, vmax =1.0)

    tick_marks = list(range(len(corr.columns)))
    plt.xticks(tick_marks, corr.columns, rotation=90)
    plt.yticks(tick_marks,corr.columns, rotation=0)

#    plt.colorbar()
    plt.savefig('plots/corrmatrix.png')
    plt.clf()

def scattermatrix():
    scatter_matrix(clean_clust_df)
    plt.show()

def clust_DBscan():
    print('DB Scan')

    DB = DBSCAN(eps = 5.00, min_samples = 4)
    DB.fit(clean_clust_df)

    plt.scatter(clean_clust_df[POI[0]],
                clean_clust_df[POI[1]],
                c=DB.labels_
                )
    plt.xlabel(POI[0])
    plt.ylabel(POI[1])
    print('plots/dbscan_'+''.join('_'.join([str(POI[0]),str(POI[1]),'.png'])))
    plt.savefig('plots/dbscan_'+''.join('_'.join([str(POI[0]),str(POI[1]),'.png'])))
    plt.clf()

def plotKdistortion(Kmax = 10): #elbow
    print("Starting plotKdistortion...")
    # determine n_clusters
    distortions = []
    K = range(1, Kmax)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(clean_clust_df)
        kmeanModel.fit(clean_clust_df)

        # calculate distortion
        each_rows_distance_to_each_centroid = cdist(clean_clust_df, kmeanModel.cluster_centers_, 'euclidean')
        shortest_distance_for_each_row_to_a_centroid = np.min(each_rows_distance_to_each_centroid, axis=1)
        distrotion = sum(shortest_distance_for_each_row_to_a_centroid) / clean_clust_df.shape[0]
        distortions.append(distrotion)

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    print("Finished plotKdistortion")

def clust_KMeansClustering():
    print("Starting kMeansClustering...")

    if dokElbow:
        plotKdistortion(10) #function to obtain elbow shape: maximum K as input

    kmeans_model = KMeans(n_clusters=10,random_state=1).fit(clean_clust_df)

    if doScatterPlots:
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # ax.scatter(clean_clust_df[POI[0]],
        #             clean_clust_df[POI[1]],
        #             clean_clust_df[POI[2]],
        #             c = kmeans_model.predict(clean_clust_df))
        # ax.scatter(kmeans_model.cluster_centers_[:, 0],
        #             kmeans_model.cluster_centers_[:, 1],
        #             kmeans_model.cluster_centers_[:, 2],
        #             s=1000,
        #             alpha=0.4
        #             )
        # ax.set_xlabel(POI[0])
        # ax.set_ylabel(POI[1])
        # ax.set_zlabel(POI[2])
        # plt.show()
        # plt.clf()

        for x, par in enumerate(POI):
            for y, par2 in enumerate(POI):

                if x == y:
                    continue
                if y < x: #removes duplicate plots where axes are reversed
                    continue

                plt.scatter(clean_clust_df[POI[x]],
                            clean_clust_df[POI[y]],
                            c=kmeans_model.predict(clean_clust_df),
                            cmap=plt.get_cmap('gist_ncar'))
                plt.scatter(kmeans_model.cluster_centers_[:, x],
                            kmeans_model.cluster_centers_[:, y],
                            s=1000,
                            alpha=0.4
                            )
                plt.xlabel(POI[x])
                plt.ylabel(POI[y])
                print('plots/kmeans_'+''.join('_'.join([str(POI[x]),str(POI[y]),'.png'])))
                plt.savefig('plots/kmeans_'+''.join('_'.join([str(POI[x]),str(POI[y]),'.png'])))
                plt.clf()

    print("K-means finished...")

def clust_hierarchyClustering(linkage):
    print("Starting Hierarchical Clustering......")

    if doDendro:
        plotDendrogram()

    clustering = AggC(linkage=linkage, n_clusters=6, memory='tmp/')
    #clustering = AggC(linkage=linkage, memory='tmp/')
    #clean_clust_df_red = manifold.SpectralEmbedding(n_components=3).fit_transform(clean_clust_df)
    clustering.fit(clean_clust_df)

    for x, par in enumerate(POI):
        for y, par2 in enumerate(POI):

            if x == y:
                continue
            if y < x: #removes duplicate plots where axes are reversed
                continue

            plt.scatter(clean_clust_df[POI[x]],
                        clean_clust_df[POI[y]],
                        c=clustering.labels_
                        )
            plt.xlabel(POI[x])
            plt.ylabel(POI[y])
            print('plots/hier_'+''.join('_'.join([str(POI[x]),str(POI[y]),'.png'])))
            plt.savefig('plots/hier_'+''.join('_'.join([str(POI[x]),str(POI[y]),'.png'])))
            plt.clf()

    print("Plotting dendrogram")

   # plt.title("Hierarchical clustering dendrogram")
   # plot_dendrogram(clustering,
#                labels=clustering.labels_,
  #                  leaf_rotation=90.,  # rotates the x axis labels
   #                 leaf_font_size=8,  # font size for the x axis labels
    #                truncate_mode='lastp',  # show only the last p merged clusters
     #               p=24,  # show only the last p merged clusters
      #              show_leaf_counts=False,
       #             show_contracted=True
        #            )
    #plt.show()
    #plt.clf()

    print("Hierarchical finished...")

def plot_dendrogram(model, **kwargs):
    children = model.children_

    distance = np.arange(children.shape[0])

    no_of_observations = np.arange(2, children.shape[0]+2)

    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    dendrogram(linkage_matrix,**kwargs)

def plotDendrogram():
    Z = lkg(clean_clust_df,'ward')
    c, coph_dists = cophenet(Z,pdist(clean_clust_df))

    print(c)
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
        truncate_mode = 'lastp', # show only the last p merged clusters
        p = 24,  # show only the last p merged clusters
        show_leaf_counts = False,
        show_contracted = True
    )
    plt.show()


#----------------------------------------------------------------------------------
def main():

    print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (' Computing Correlations of the features')

    print ('')
    print ('------------------------------------------------------------')
    print (' Reading the start, end date and the period:')
    input_file = '00.00.PARAMETERS.txt'
    # input_file = '00.00.PARAMETERS.txt'
    f_in = open(input_file, 'r')
    lines = f_in.readlines()[1:]  # reads starting from the second line and stores the line in a string
    print (lines)
    # print (lines[0].split('=', 1)[0])
    start_date = lines[0].split(' ', 1)[0]
    stop_date = lines[1].split(' ', 1)[0]
    period = int(lines[2].split(' ', 1)[0])

    print (' Using the following:')
    print (' start_date (closest to us) = ' + start_date)
    print (' stop_date (the most in the past) = ' + stop_date)
    print (' period (days) = ' + str(period))
    print ('')

    if period != 30 and period != 7 and period != 1:
        print (' ERROR: period must be 1, 7 or 30')
        quit()

    f_in.close()


    start = datetime.strptime(start_date, "%Y-%m-%d")
    stop = datetime.strptime(stop_date, "%Y-%m-%d")
    day = start.strftime('%Y-%m-%d')


    folder_csv = '01.01.RES-CSV-TABLES'
    csv_file_type = folder_csv + '/Data.'  # + day + .csv


    print ('')
    print ('------------------------------------------------------------')
    print (' Folder of the Outputs : ')
    folder_RES = '03.01.RES-Correlations'
    print (folder_RES)
    # creating the folder if it does not exist
    if not os.path.exists(folder_RES):
        os.makedirs(folder_RES)

    print ('')
    print ('---------------------------------------------------------------')
    print (' Parameters of potential interest')

    # TODO: Write here the parameters of interest
    POI = ['audience',
           'sum_revenue_euro',
           'reach',
           'CPC',
           'ratio_displays_acc_over_pot',
           #'exposed_users',
           'potential_displays',
           #'sum_displays',
           'sum_clicks'
           ]
    print (POI)


    print ('')
    print ('---------------------------------------------------------------')
    print (' Starting the cycle')


    while start > stop:
        # print (day)  # start.strftime('%Y-%m-%d'))
        #tbl = 'Data.' + day  # name of the table to be written
        print ('///////////////////////////////////////////////////////////////////////////')

        MYFILE = csv_file_type + day + '.csv'  # 'TABLE_v02.02_No-duplicate-funnel.csv'
        print('Table now (MYFILE) = ' + MYFILE)

        print ('')
        print ('  ---------------------------------')
        print ('  Reading the dataframe')

        df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True, thousands=',')

        # if doHierarchy:  # !!hierarchy memory intensive O(n^2), 30k rows is slow, with no limit it crashes
        #     df = pd.read_csv(MYFILE, na_values=['None'], nrows=20000, skip_blank_lines=True)
        #     # df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True)
        # else:
        #     df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True, thousands=',')


        print ('')
        print ('  ---------------------------------')
        print ('  Computing the new variables in the dataframe')

        # Define the new entries in the table
        df['reach'] = df['exposed_users'] / df['audience']
        df['CPC'] = df['sum_revenue_euro'] / df['sum_clicks']
        df['ratio_displays_acc_over_pot'] = df['sum_displays'] / df['potential_displays']

        clust_df = df[POI].copy()
        print(clust_df.head())
        clust_df.apply(pd.to_numeric)
        # Remove NaN's in this column -- need to find a better way to automate it
        clean_clust_df = removeNANs(POI, clust_df)

        del df
        del clust_df

        print(clean_clust_df.shape)


        # --------------------- start correlations
        #print('Starting Correlations:')
        # define the dataframe with logs
        var_df = clean_clust_df[POI].copy()

        print ('')
        print ('  ---------------------------------')
        print ('  Computing log of the variables on the dataframe var_df:')


        # change the name of the column -> with log: audience
        print ('log_audience')
        var_df.rename(columns={'audience': 'log_audience'}, inplace=True)
        var_df['log_audience'] = np.log10(1.0 + clean_clust_df.audience)

        # change sum_revenue_euro
        print ('log_sum_revenue_euro')
        var_df.rename(columns={'sum_revenue_euro': 'log_sum_revenue_euro'}, inplace=True)
        var_df['log_sum_revenue_euro'] = np.log10(1.0 + clean_clust_df.sum_revenue_euro)

        # change reach
        print ('reach left without log')
        #var_df.rename(columns={'reach': 'log_reach'}, inplace=True)
        #var_df['log_reach'] = np.log10(1.0 + clean_clust_df.reach)

        # change CPC
        print ('log_CPC')
        var_df.rename(columns={'CPC': 'log_CPC'}, inplace=True)
        var_df['log_CPC'] = np.log10(1.0 + clean_clust_df.CPC)

        # change ratio_displays_acc_over_pot
        print ('log_ratio_displays_acc_over_pot')
        var_df.rename(columns={'ratio_displays_acc_over_pot': 'log_ratio_displays_acc_over_pot'}, inplace=True)
        var_df['log_ratio_displays_acc_over_pot'] = np.log10(1.0 + clean_clust_df.ratio_displays_acc_over_pot)

        # change potential_displays
        print ('log_potential_displays')
        var_df.rename(columns={'potential_displays': 'log_potential_displays'}, inplace=True)
        var_df['log_potential_displays'] = np.log10(1.0 + clean_clust_df.potential_displays)

        # change sum_clicks
        print ('log_sum_clicks')
        var_df.rename(columns={'sum_clicks': 'log_sum_clicks'}, inplace=True)
        var_df['log_sum_clicks'] = np.log10(1.0 + clean_clust_df.sum_clicks)
        #print clean_clust_df.head()
        #print var_df.head()

        #quit()

        print ('')
        print ('  ---------------------------------')
        print ('  Computing the correlation of the log and writing the heatmap:')

        corr = var_df.corr(method='spearman')
        cmap = sns.diverging_palette(5, 250, as_cmap=True)
        # labels = (np.asarray(["{0} \n {1:.2f}".format(symb,value) for symb, value in zip(symbol.flatten(),perchange.flatten())]))
        sns.heatmap(corr, square=True, annot=True, fmt=".2f", annot_kws={"size": 5}, vmin=-1.0, vmax=1.0, cmap=cmap,
                    cbar=True)
        # plt.imshow(corr,cmap=plt.cm.bwr_r, vmin=-1.0, vmax =1.0)
        tick_marks = list(range(len(corr.columns)))
        plt.xticks(tick_marks, corr.columns, rotation=90)
        plt.yticks(tick_marks, reversed(corr.columns), rotation=0)
        plt.title('Correlations: Day = ' + day + ', Period = ' + str(period))
        #    plt.colorbar()
        fig_name = folder_RES + '/Correlations.' + day + '.Period-' + str(period) + '.png'
        print ('   ==> Figure = ' + fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.clf()

        #quit()
        # --------------------- end correlations

        #correlations()

        #scattermatrix()

        #  clust_DBscan()

        # if dokMeans:
        #     clust_KMeansClustering()
        #
        # if doHierarchy:
        #     clust_hierarchyClustering('ward') #choose which agglomerative clustering you want to use http://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py


        # cycles
        start = start - timedelta(days=period)
        day = start.strftime('%Y-%m-%d')

        # clean the df
        del clean_clust_df
        del var_df



#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()