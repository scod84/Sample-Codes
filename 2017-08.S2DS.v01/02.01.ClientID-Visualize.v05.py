# WHAT
# This is to visualize the data of the tables
# ----
# AUTHOR = Daniele Scopece, 2017-08-25 (adapted) / 2017-09-01
#
#
#---------------------------------------------------------------------------------

# ---- Imports various
import csv
import scipy
print('scipy: {}'.format(scipy.__version__))
#from scipy.cluster.hierarchy import dendrogram, linkage as lkg, cophenet
#from scipy.spatial.distance import pdist

import pandas as pd
#from pandas.tools.plotting import scatter_matrix

import numpy as np
print('numpy: {}'.format(np.__version__))

import matplotlib
import matplotlib.pyplot as plt
print('matplotlib: {}'.format( matplotlib.__version__))
#from mpl_toolkits.mplot3d import Axes3D

import sklearn
print ('sklearn: {}'.format(sklearn.__version__))
#from sklearn import manifold
#from sklearn.cluster import KMeans, AgglomerativeClustering as AggC, DBSCAN
#from sklearn.decomposition import PCA
#from scipy.spatial.distance import cdist

#import seaborn as sns
#--------------

#----- Data input
from datetime import datetime
from datetime import timedelta

# to create directory
import os, errno


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

#-----------------------------------------------------------------------
# to plot the log log figures
def plot_figure_log_log(title_here, name_fig_here, x_lab, x_here, x_lim_here, y_lab, y_here, y_lim_here, color_lab, color_here): # pass the vectors
    fig = plt.figure()
    #plt.title('Clients: End Date ' + day_here + ' & Period of ' + str(period_here) + ' days (color=Reach=Exposed/Audience)')
    plt.title(title_here)
    #x = clean_clust_df[POI[0]]
    #y = clean_clust_df[POI[1]]
    #plt.xlabel(POI[0])
    #plt.ylabel(POI[1])
    #x = clean_clust_df[POI[entry_POI_x]]
    #y = clean_clust_df[POI[entry_POI_y]]
    x = x_here
    y = y_here
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    ax = fig.add_subplot(1, 1, 1)
    # plt.colorbar(ax.imshow(image, interpolation='nearest'))
    #plt.scatter(x, y, c=clean_clust_df[POI[entry_POI_color]], cmap='rainbow', vmin=0.0, vmax=1.0)
    plt.scatter(x, y, c=color_here, cmap='rainbow', vmin=0.0, vmax=1.0)
    plt.colorbar()
    #plt.ylim([1e0, 1e6])
    plt.ylim(y_lim_here)
    plt.xlim([1e1, 1e9])
    plt.xlim(x_lim_here)
    plt.grid(True)
    # log scale
    ax.set_yscale('log')
    ax.set_xscale('log')
    # cbar = plt.colorbar()
    # cbar.set_label('Reach', rotation=270)
    # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
    # cmap = sns.diverging_palette(5, 250, as_cmap=True)
    #name_fig = folder_name + '/Fig.' + str(fig_num) + '-Revenue-vs-Audience-colorReach.' + day + '-period-' + str(period) + '.png'
    plt.savefig(name_fig_here, format='png')
    print ('     ==> figure = ' + name_fig_here)
    # plt.show() # caution: it stops the flow of the program
    plt.draw()  # draws without stopping the flow of the program
    plt.clf()  # clear figure
    plt.close()  # close the figure window and continue


#----------------------------------------------------------------------------------
def main():

    print ('')
    print ('----------------------------------------------------------')
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


    print ('')
    print ('----------------------------------------------------------')
    print (' The parameters to be plotted are the following:')

    # TODO: Write here the parameters of interest to be plotted
    POI = ['audience',
           'sum_revenue_euro',
           'reach',
           'CPC',
           'ratio_displays_acc_over_pot',
           ]

    print (POI)


    print ('')
    print ('----------------------------------------------------------')
    print (' The figures will be plotted in this folder:')
    folder_figures_out = "02.01.RES-Plots"
    print (folder_figures_out)
    # creating the folder if it does not exist
    if not os.path.exists(folder_figures_out):
        os.makedirs(folder_figures_out)


    print ('')
    print ('----------------------------------------------------------')
    print (' Cycling on the csv files:')

    folder = '01.01.RES-CSV-TABLES'
    print (' The folder in which the csv files are is: ' + folder)

    # saving the limits in the right format of dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    stop = datetime.strptime(stop_date, "%Y-%m-%d")
    day = start.strftime('%Y-%m-%d')

    csv_file_type = folder + '/Data.'  # + day + .csv

    print ('')


    while start > stop:
        # print (day)  # start.strftime('%Y-%m-%d'))
        #tbl = 'Data.' + day  # name of the table to be written

        MYFILE = csv_file_type + day + '.csv'  # 'TABLE_v02.02_No-duplicate-funnel.csv'
        print ('---')
        print('Table now (MYFILE) = ' + MYFILE)

        df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True, thousands=',')

        print (df.head())


        # Define the new entries in the table
        print (' Defining the other derived features in the table...')
        df['reach'] = df['exposed_users'] / df['audience']
        df['CPC'] = df['sum_revenue_euro'] / df['sum_clicks']
        df['ratio_displays_acc_over_pot'] = df['sum_displays'] / df['potential_displays']

        print (' The dataframe used for only the POI: ')
        clust_df = df[POI].copy()
        print(clust_df.head())

        print (' Converting dataframe to numeric ... ')
        clust_df.apply(pd.to_numeric)

        print (' Removing the lines with Nan in the colums of interest and creating a cleaned dataframe...')
        # Remove NaN's in this column -- need to find a better way to automate it
        clean_clust_df = removeNANs(POI, clust_df)
        print (clean_clust_df.head())

        del df
        del clust_df

        print ('')

        # ----------------------------------------------------------------------------------------
        # plot 2D # 1: revenue vs audience (color = reach)

        x_lab = POI[0]
        x_now = clean_clust_df[POI[0]]
        x_lim_now = [1e1, 1e9]

        y_lab = POI[1]
        y_now = clean_clust_df[POI[1]]
        y_lim_now = [1e0, 1e6]

        color_lab = POI[2]
        color_now = clean_clust_df[POI[2]]

        title_now = 'Clients: End Date ' + day + ' & Period of ' + str(period) + ' days (color=' + color_lab + ')'


        subfolder = '/01.01' # + y_lab + '-vs-' + x_lab + '-color-' + color_lab
        # create the directory if it does not exist
        directory = folder_figures_out + subfolder
        if not os.path.exists(directory):
            os.makedirs(directory)
        name_fig_now = folder_figures_out + subfolder + '/Fig.LogLog.' + y_lab + '-vs-' + x_lab + '-color-' + color_lab + '.' + day + '-period-' + str(period) + '.png'

        #print (title_now, name_fig_now)

        plot_figure_log_log(title_now, name_fig_now, x_lab, x_now, x_lim_now, y_lab, y_now, y_lim_now, color_lab, color_now)


        # ----------------------------------------------------------------------------------------
        # plot 2d: # 2: CPC vs ratio_displays_acc_over_pot & color = reach
        x_lab = POI[3]
        x_now = clean_clust_df[POI[3]]
        x_lim_now = [1e-2, 1e2]

        y_lab = POI[4]
        y_now = clean_clust_df[POI[4]]
        y_lim_now = [1e-5, 1e0]

        color_lab = POI[2]
        color_now = clean_clust_df[POI[2]]

        title_now = 'Clients: End Date ' + day + ' & Period of ' + str(period) + ' days (color=' + color_lab + ')'

        subfolder = '/02.01' # + y_lab + '-vs-' + x_lab + '-color-' + color_lab
        # create the directory if it does not exist
        directory = folder_figures_out + subfolder
        if not os.path.exists(directory):
            os.makedirs(directory)

        name_fig_now = folder_figures_out + subfolder + '/Fig.LogLog.' + y_lab + '-vs-' + x_lab + '-color-' + color_lab + '.' + day + '-period-' \
                       + str(period) + '.png'

        plot_figure_log_log(title_now, name_fig_now, x_lab, x_now, x_lim_now, y_lab, y_now, y_lim_now, color_lab, color_now)


        # ----------------------------------------------------------------------------------------
        # plot 2d: # 3: sum revenue vs ratio & color = reach
        x_lab = POI[4]
        x_now = clean_clust_df[POI[4]]
        x_lim_now = [1e-5, 1e0]

        y_lab = POI[1]
        y_now = clean_clust_df[POI[1]]
        y_lim_now = [1e-5, 1e6]

        color_lab = POI[2]
        color_now = clean_clust_df[POI[2]]

        title_now = 'Clients: End Date ' + day + ' & Period of ' + str(period) + ' days (color=' + color_lab + ')'

        subfolder = '/03.01' #  + y_lab + '-vs-' + x_lab + '-color-' + color_lab
        # create the directory if it does not exist
        directory = folder_figures_out + subfolder
        if not os.path.exists(directory):
            os.makedirs(directory)

        name_fig_now = folder_figures_out + subfolder + '/Fig.LogLog.' + y_lab + '-vs-' + x_lab + '-color-' + color_lab + '.' + day + '-period-' \
                       + str(period) + '.png'

        plot_figure_log_log(title_now, name_fig_now, x_lab, x_now, x_lim_now, y_lab, y_now, y_lim_now, color_lab,
                            color_now)


        # ----------------------------------------------------------------------------------------
        # plot 2d: # 4: sum revenue vs CPC & color = reach

        x_lab = POI[3]
        x_now = clean_clust_df[POI[3]]
        x_lim_now = [1e-2, 1e2]

        y_lab = POI[1]
        y_now = clean_clust_df[POI[1]]
        y_lim_now = [1e0, 1e6]

        color_lab = POI[2]
        color_now = clean_clust_df[POI[2]]

        title_now = 'Clients: End Date ' + day + ' & Period of ' + str(period) + ' days (color=' + color_lab + ')'

        subfolder = '/04.01' # + y_lab + '-vs-' + x_lab + '-color-' + color_lab
        # create the directory if it does not exist
        directory = folder_figures_out + subfolder
        if not os.path.exists(directory):
            os.makedirs(directory)

        name_fig_now = folder_figures_out + subfolder + '/Fig.LogLog.' + y_lab + '-vs-' + x_lab + '-color-' + color_lab + '.' + day + '-period-' \
                       + str(period) + '.png'

        plot_figure_log_log(title_now, name_fig_now, x_lab, x_now, x_lim_now, y_lab, y_now, y_lim_now, color_lab,
                            color_now)



        # ----------------------------------------------------------------------------------------
        # plot 2d: # 5: Audience vs CPC & color = reach
        x_lab = POI[3]
        x_now = clean_clust_df[POI[3]]
        x_lim_now = [1e-2, 1e2]

        y_lab = POI[0]
        y_now = clean_clust_df[POI[0]]
        y_lim_now = [1e1, 1e9]

        color_lab = POI[2]
        color_now = clean_clust_df[POI[2]]

        title_now = 'Clients: End Date ' + day + ' & Period of ' + str(period) + ' days (color=' + color_lab + ')'

        subfolder = '/05.01' # + y_lab + '-vs-' + x_lab + '-color-' + color_lab
        # create the directory if it does not exist
        directory = folder_figures_out + subfolder
        if not os.path.exists(directory):
            os.makedirs(directory)

        name_fig_now = folder_figures_out + subfolder + '/Fig.LogLog.' + y_lab + '-vs-' + x_lab + '-color-' + color_lab + '.' + day + '-period-' \
                       + str(period) + '.png'

        plot_figure_log_log(title_now, name_fig_now, x_lab, x_now, x_lim_now, y_lab, y_now, y_lim_now, color_lab,
                            color_now)


        # ------------------------------------------------------------------------------
        print (' Cycling date')
        # cycles
        start = start - timedelta(days=period)
        day = start.strftime('%Y-%m-%d')

        # clean the df
        del clean_clust_df


#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()