# WHAT
# This is to compute the hierachical clustering aggregation with log if needed
# used 4 variables only: log_audience, log_sum_revenue_euro, log_ratio_displays_acc_over_pot, log_CPC
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
from mpl_toolkits.mplot3d import Axes3D

import sklearn
print ('sklearn: {}'.format(sklearn.__version__))
from sklearn import manifold
from sklearn.cluster import KMeans, AgglomerativeClustering as AggC, DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import seaborn as sns
#--------------


from datetime import datetime
from datetime import timedelta

# create directory
import os

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


#-------------------------------------------------------------------------
def clust_hierarchyClustering(dataframe_here, POI_here, linkage):
    print("Starting Hierarchical Clustering......")

    #if doDendro:
    #    plotDendrogram()

    clustering = AggC(linkage=linkage, n_clusters=6, memory='tmp/')
    #clustering = AggC(linkage=linkage, memory='tmp/')
    #clean_clust_df_red = manifold.SpectralEmbedding(n_components=3).fit_transform(clean_clust_df)
    clustering.fit(dataframe_here)

    for x, par in enumerate(POI_here):
        for y, par2 in enumerate(POI_here):

            if x == y:
                continue
            if y < x: #removes duplicate plots where axes are reversed
                continue

            plt.scatter(dataframe_here[POI_here[x]],
                        dataframe_here[POI_here[y]],
                        c=clustering.labels_
                        )
            plt.xlabel(POI_here[x])
            plt.ylabel(POI_here[y])
            print('plots/hier_'+''.join('_'.join([str(POI_here[x]),str(POI_here[y]),'.png'])))
            plt.savefig('PLOTS-P-04/hier_'+''.join('_'.join([str(POI_here[x]),str(POI_here[y]),'.png'])))
            plt.clf()

    print("Plotting dendrogram")

    # plt.title("Hierarchical clustering dendrogram")
    # plot_dendrogram(clustering,
    #                labels=clustering.labels_,
    #                  leaf_rotation=90.,  # rotates the x axis labels
    #                 leaf_font_size=8,  # font size for the x axis labels
    #               truncate_mode='lastp',  # show only the last p merged clusters
    #               p=24,  # show only the last p merged clusters
    #              show_leaf_counts=False,
    #             show_contracted=True
    #            )
    #plt.show()
    #plt.clf()

    print("Hierarchical finished...")


#---------------------------------------------------------------------------------
def plotDendrogram(dataframe_here, day_here, period_here, fig_name):
    Z = lkg(dataframe_here,'ward')
    c, coph_dists = cophenet(Z,pdist(dataframe_here))

    print(c)
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram: Day = ' + day_here + ', Period = '+ str(period_here))
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
    #plt.show()
    plt.savefig(fig_name)
    plt.clf()

#----------------------------------------------------------------------------------
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
    folder_RES = '03.02.RES-HierarchicalClustering'
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
           #'reach',
           'CPC',
           'ratio_displays_acc_over_pot',
           #'exposed_users',
           #'potential_displays',
           #'sum_displays',
           #'sum_clicks'
           ]
    print (POI)
    #start_date = "2017-08-01"  # the closest to ours
    #stop_date = "2017-06-20"  # the furthest in the past
    #stop_date = "2016-01-01"  # the furthest in the past
    #stop_date = "2017-06-01"  # the furthest in the past
    #period = 30  # can be 1 or 7 or 30 (in days)

    #start = datetime.strptime(start_date, "%Y-%m-%d")
    #stop = datetime.strptime(stop_date, "%Y-%m-%d")
    #day = start.strftime('%Y-%m-%d')

    #csv_file_type = 'Data.'  # + day + .csv


    # create the lists necessary for the plot of sums
    #dates = []
    #num_clients_active = []
    #sumall_revenues = []
    #sumall_audience = []
    #sumall_clicks = []
    #sumall_displays = []
    #sumall_potdisplays = []
    #sumall_actualdisplays = []


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
        print (df.head())

        # if doHierarchy:  # !!hierarchy memory intensive O(n^2), 30k rows is slow, with no limit it crashes
        #     df = pd.read_csv(MYFILE, na_values=['None'], nrows=20000, skip_blank_lines=True)
        #     # df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True)
        # else:
        #     df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True, thousands=',')


        print ('')
        print ('  ---------------------------------')
        print ('  Computing the new variables in the dataframe')
        # Define the new entries in the table
        #df['reach'] = df['exposed_users'] / df['audience']
        df['CPC'] = df['sum_revenue_euro'] / df['sum_clicks']
        df['ratio_displays_acc_over_pot'] = df['sum_displays'] / df['potential_displays']

        clust_df = df[POI].copy()

        print ('')
        print ('  ---------------------------------')
        print ('  Converting to numeric')
        print(clust_df.head())
        clust_df.apply(pd.to_numeric)


        print ('')
        print ('  ---------------------------------')
        print ('  Removing the lines with NaN')
        # Remove NaN's in this column -- need to find a better way to automate it
        clean_clust_df = removeNANs(POI, clust_df)

        del df
        del clust_df

        print(clean_clust_df.head())

        # remove lines with potential_displays = 0 => gives inf
        # done below



        # --------------------- start hierarchical: change variables = log
        print ('')
        print ('  ---------------------------------')
        print ('  Compute the log of the variables on the dataframe var_df')

        # define the dataframe with logs
        var_df = clean_clust_df[POI].copy()

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
        print ('potential_displays ignored')
        #var_df.rename(columns={'potential_displays': 'log_potential_displays'}, inplace=True)
        #var_df['log_potential_displays'] = np.log10(1.0 + clean_clust_df.potential_displays)

        # change sum_clicks
        print ('sum_clicks ignored')
        #var_df.rename(columns={'sum_clicks': 'log_sum_clicks'}, inplace=True)
        #var_df['log_sum_clicks'] = np.log10(1.0 + clean_clust_df.sum_clicks)
        #print clean_clust_df.head()
        #print var_df.head()


        print ('')
        print ('  ---------------------------------')
        print ('  List of the new log variables dataframe var_df')

        POI_here = ['log_audience', 'log_sum_revenue_euro', 'log_CPC', 'log_ratio_displays_acc_over_pot']
        print (POI_here)


        # replace inf with NaN -> then remove NaN : inf can come from the divisions of the displays
        print ('')
        print ('  ---------------------------------')
        print ('  Replacing inf with NaN:')
        #print ('    Check Before = ' + str(var_df.ix[124]))
        var_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        #print ('    Check After = ' + str(var_df.ix[124]))
        #var_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        var_df_OK = removeNANs(POI_here, var_df)
        #var_df.dropna(axis=0, how="any", inplace=True)
        #print ('    Check After = ' + str(var_df_OK.ix[124]))


        #print (var_df.head())

        # find NaNs
        #inds = pd.isnull(var_df).any(1).nonzero()[0]
        ##inds = pd.isnull(var_df).any(0).nonzero()[0]
        ##inds = pd.isnull(var_df).any('log_audience').nonzero()[0]
        #print (' List of indeces with NaN:')
        #print (inds)

        #index = var_df['log_audience'].index[var_df['log_audience'].apply(np.isnan)]
        #print ('Column = ' + 'log_audience' + ', index NaN = ' + index)

        ## look if large number (errors)
        #for col in POI_here:
        #    print ('Locating the max in each column:')
        #   print (col)
        #    max_here = var_df_OK.loc[var_df[col].idxmax()]
        #    print (max_here)
        #    print (var_df_OK[col].idxmax())
        #    print

        #for col in POI_here:
        #    print ('Locating the min in each column:')
        #    print (col)
        #    min_here = var_df_OK.loc[var_df[col].idxmin()]
        #    print (min_here)
        #    print (var_df_OK[col].idxmin())
        #    print


        #max_here = var_df.loc[var_df['log_audience'].idxmax()]
        #max_here = var_df.max()
        #print ('max_here (log_audience) = ' + max_here)



        # Plot Dendrogram --> see the distances
        fig_name = folder_RES + '/Hierarchical-Dendro.' + day + '.Period-' + str(period) + '.png'
        print ('   ==> Figure Dendrogram = ' + fig_name)
        plotDendrogram(var_df_OK, day, period, fig_name)

        #quit()


        #clust_hierarchyClustering(var_df, POI_here, 'ward')
        # choose which agglomerative clustering you want to use
        # http://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py


        #corr = var_df.corr(method='spearman')
        #cmap = sns.diverging_palette(5, 250, as_cmap=True)
        # labels = (np.asarray(["{0} \n {1:.2f}".format(symb,value) for symb, value in zip(symbol.flatten(),perchange.flatten())]))
        #sns.heatmap(corr, square=True, annot=True, fmt=".2f", annot_kws={"size": 5}, vmin=-1.0, vmax=1.0, cmap=cmap,
        #            cbar=True)
        ## plt.imshow(corr,cmap=plt.cm.bwr_r, vmin=-1.0, vmax =1.0)
        #tick_marks = list(range(len(corr.columns)))
        #plt.xticks(tick_marks, corr.columns, rotation=90)
        #plt.yticks(tick_marks, reversed(corr.columns), rotation=0)
        #plt.title('Correlations: Day = ' + day + ', Period = ' + str(period))
        ##    plt.colorbar()

        #fig_name = 'Hierarchical-Dendro.' + day + '.Period-' + str(period) + '.png'
        #print (' ==> Figure Dendrogram = ' + fig_name)
        #plt.tight_layout()
        #plt.savefig('PLOTS-P-04/'+fig_name)
        #plt.clf()

        #quit()
        # --------------------- end correlations

        #correlations()

        #scattermatrix()

        #  clust_DBscan()

        #if dokMeans:
        #    clust_KMeansClustering()

        #if doHierarchy:
        #    clust_hierarchyClustering('ward') #choose which agglomerative clustering you want to use http://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py


        # cycles
        start = start - timedelta(days=period)
        day = start.strftime('%Y-%m-%d')

        # clean the df
        del clean_clust_df
        del var_df
        del var_df_OK


#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()