# WHAT
# This is to visualize the data of the tables
# ----
# AUTHOR = Daniele Scopece, 2017-08-25 (adapted)
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

import matplotlib.dates as mdates # for dates as axis


#import sklearn
#print ('sklearn: {}'.format(sklearn.__version__))
#from sklearn import manifold
#from sklearn.cluster import KMeans, AgglomerativeClustering as AggC, DBSCAN
#from sklearn.decomposition import PCA
#from scipy.spatial.distance import cdist

import seaborn as sns

import matplotlib.ticker as ticker # needed to change the scale

from datetime import datetime
from datetime import timedelta

# to create directory
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


#----------------------------------------------------------------------------------
def main():
    print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (' Plotting trends with time')

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
    folder_RES = '02.03.RES-Plots-Trend-Time'
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
           'exposed_users',
           'potential_displays',
           'sum_displays',
           'sum_clicks',
           'is_tier_1'
           ]
    print (POI)

    #start_date = "2017-08-01"  # the closest to ours
    #stop_date = "2017-06-20"  # the furthest in the past
    #stop_date = "2016-01-01"  # the furthest in the past
    #stop_date = "2017-06-01"  # the furthest in the past
    #period = 30  # can be 1 or 7 or 30 (in days)


    print ('')
    print ('---------------------------------------------------------------')
    print (' Starting to sum in the cycle')


    # create the lists necessary for the plot of sums
    dates = []
    num_clients_active = []
    num_active_tier_1 = []
    sumall_revenues = []
    sumall_audience = []
    sumall_clicks = []
    #sumall_displays = []
    sumall_potdisplays = []
    sumall_actualdisplays = []



    while start > stop:
        # print (day)  # start.strftime('%Y-%m-%d'))
        #tbl = 'Data.' + day  # name of the table to be written

        MYFILE = csv_file_type + day + '.csv'  # 'TABLE_v02.02_No-duplicate-funnel.csv'
        print('Table computing now now (MYFILE) = ' + MYFILE)

        df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True, thousands=',')

        # if doHierarchy:  # !!hierarchy memory intensive O(n^2), 30k rows is slow, with no limit it crashes
        #     df = pd.read_csv(MYFILE, na_values=['None'], nrows=20000, skip_blank_lines=True)
        #     # df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True)
        # else:
        #     df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True, thousands=',')

        #for row in df:
        #    print row

        # Define the new entries in the table
        df['reach'] = df['exposed_users'] / df['audience']
        df['CPC'] = df['sum_revenue_euro'] / df['sum_clicks']
        df['ratio_displays_acc_over_pot'] = df['sum_displays'] / df['potential_displays']

        # needed for plot sum
        #df['exposed_users'] =
        #'potential_displays',
        #'sum_displays',
        #'sum_clicks'


        clust_df = df[POI].copy()
        #print(clust_df.head())
        clust_df.apply(pd.to_numeric)
        # Remove NaN's in this column -- need to find a better way to automate it
        clean_clust_df = removeNANs(POI, clust_df)

        del df
        del clust_df


        #print(clean_clust_df.shape)


        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Operate the sums & co

        # Count number of entries in the clean dataframe -> put in a vector
        total_rows = clean_clust_df.shape[0]
        #print (day, total_rows)



        dates.append(day)
        num_clients_active.append(total_rows)
        num_active_tier_1.append(clean_clust_df['is_tier_1'].sum())
        sumall_revenues.append(clean_clust_df['sum_revenue_euro'].sum())
        sumall_audience.append(clean_clust_df['audience'].sum())
        sumall_clicks.append(clean_clust_df['sum_clicks'].sum())
        sumall_actualdisplays.append(clean_clust_df['sum_displays'].sum())
        sumall_potdisplays.append(clean_clust_df['potential_displays'].sum())

        # cycles
        start = start - timedelta(days=period)
        day = start.strftime('%Y-%m-%d')

        # clean the df
        del clean_clust_df


    # out of the while => plot at the end
    print ('While finished')


    # ----------------------------------------------------------------------------------------
    print ('')
    print ('---------------------------------------------------')
    print (' Plotting the graphs:')
    # plot 2d: # 1: num active clients vs time
    fig1 = plt.figure()
    plt.title('Clients: Period of ' + str(period) + ' days: [No NaN or Null]')
    ax1 = fig1.add_subplot(1, 1, 1)
    x = mdates.datestr2num(dates)
    plt.xlabel('Date')
    #plt.xlim([1e-2, 1e2])
    #ax1.set_xscale('log')  # log scale
    # y = clean_clust_df[POI[4]]*100
    # plt.ylabel(POI[4] + '*100')
    # log => no x 100
    y = num_clients_active
    plt.ylabel('Number of Active Clients in this period')
    # plt.ylim([-2, 40])
    plt.ylim([0, 700])
    #ax5.set_yscale('log')  # log scale
    # plt.colorbar(ax.imshow(image, interpolation='nearest'))
    #plt.scatter(x, y)

    y_tier_1 = num_active_tier_1

    ax1.plot(x, y, linestyle='-', marker='o', color='r', label="All Clients")
    ax1.plot(x, y_tier_1, linestyle='-', c='black', marker='v', label="Is Tier 1")

    plt.legend(loc='upper left')

    #plt.plot(x, y, linestyle='-', marker='o', color='b') #,  fmt="bo", tz=None, xdate=True)
    xfmt = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(xfmt)
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #plt.colorbar()
    plt.grid(True)
    # cbar = plt.colorbar()
    # cbar.set_label('Reach', rotation=270)
    # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
    # cmap = sns.diverging_palette(5, 250, as_cmap=True)
    name_fig1 = folder_RES + '/Fig.NumActiveClients-vs-Date.period-' + str(period) + '.png'
    plt.savefig(name_fig1, format='png')
    print (' ===> figure 1 = ' + name_fig1)
    # plt.show() # caution: it stops the flow of the program
    plt.draw()  # draws without stopping the flow of the program
    plt.clf()  # clear figure
    plt.close()  # close the figure window and continue

    # ----------------------------------------------------------------------------------------
    # plot 2d: # 2: sumall revenue vs time
    fig2 = plt.figure()
    plt.title('Clients: Period of ' + str(period) + ' days: [no NaN or Null]')
    ax2 = fig2.add_subplot(1, 1, 1)
    x = mdates.datestr2num(dates)
    plt.xlabel('Date')
    # plt.xlim([1e-2, 1e2])
    # ax1.set_xscale('log')  # log scale
    # y = clean_clust_df[POI[4]]*100
    # plt.ylabel(POI[4] + '*100')
    # log => no x 100
    y = sumall_revenues
    # change scale
    scale_y = 1e6
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y / scale_y))
    ax2.yaxis.set_major_formatter(ticks)
    plt.ylabel('Sum of All Criteo\'s Revenues in this period [M euros]')
    # plt.ylim([-2, 40])
    # plt.ylim([1e1, 1e9])
    #ax2.set_yscale('log')  # log scale
    # plt.colorbar(ax.imshow(image, interpolation='nearest'))
    # plt.scatter(x, y)
    plt.plot(x, y, linestyle='-', marker='o', color='b')  # ,  fmt="bo", tz=None, xdate=True)
    xfmt = mdates.DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(xfmt)
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.colorbar()
    plt.grid(True)
    # cbar = plt.colorbar()
    # cbar.set_label('Reach', rotation=270)
    # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
    # cmap = sns.diverging_palette(5, 250, as_cmap=True)
    name_fig2 = folder_RES + '/Fig.SumAllRevenue-vs-Date.period-' + str(period) + '.png'
    plt.savefig(name_fig2, format='png')
    print (' ===> figure 2 = ' + name_fig2)
    # plt.show() # caution: it stops the flow of the program
    plt.draw()  # draws without stopping the flow of the program
    plt.clf()  # clear figure
    plt.close()  # close the figure window and continue

    # ----------------------------------------------------------------------------------------
    # plot 2d: # 3: sumall audience vs time
    fig3 = plt.figure()
    plt.title('Clients: Period of ' + str(period) + ' days: [no NaN or Null]')
    ax3 = fig3.add_subplot(1, 1, 1)
    x = mdates.datestr2num(dates)
    plt.xlabel('Date')
    # plt.xlim([1e-2, 1e2])
    # ax1.set_xscale('log')  # log scale
    # y = clean_clust_df[POI[4]]*100
    # plt.ylabel(POI[4] + '*100')
    # log => no x 100
    y = sumall_audience
    # change scale
    scale_y = 1e6
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y / scale_y))
    ax3.yaxis.set_major_formatter(ticks)
    plt.ylabel('Sum of Audience in this period [M]')
    # plt.ylim([-2, 40])
    # plt.ylim([1e1, 1e9])
    # ax2.set_yscale('log')  # log scale
    # plt.colorbar(ax.imshow(image, interpolation='nearest'))
    # plt.scatter(x, y)
    plt.plot(x, y, linestyle='-', marker='o', color='b')  # ,  fmt="bo", tz=None, xdate=True)
    xfmt = mdates.DateFormatter('%Y-%m-%d')
    ax3.xaxis.set_major_formatter(xfmt)
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.colorbar()
    plt.grid(True)
    # cbar = plt.colorbar()
    # cbar.set_label('Reach', rotation=270)
    # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
    # cmap = sns.diverging_palette(5, 250, as_cmap=True)
    name_fig3 = folder_RES + '/Fig.SumAllAudience-vs-Date.period-' + str(period) + '.png'
    plt.savefig(name_fig3, format='png')
    print (' ===> figure 3 = ' + name_fig3)
    # plt.show() # caution: it stops the flow of the program
    plt.draw()  # draws without stopping the flow of the program
    plt.clf()  # clear figure
    plt.close()  # close the figure window and continue


    # ----------------------------------------------------------------------------------------
    # plot 2d: # 4: sumall clicks vs time
    fig4 = plt.figure()
    plt.title('Clients: Period of ' + str(period) + ' days: [no NaN or Null]')
    ax4 = fig4.add_subplot(1, 1, 1)
    x = mdates.datestr2num(dates)
    plt.xlabel('Date')
    # plt.xlim([1e-2, 1e2])
    # ax1.set_xscale('log')  # log scale
    # y = clean_clust_df[POI[4]]*100
    # plt.ylabel(POI[4] + '*100')
    # log => no x 100
    y = sumall_clicks
    # change scale
    scale_y = 1e6
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y / scale_y))
    ax4.yaxis.set_major_formatter(ticks)
    plt.ylabel('Sum of Clicks in this period [M]')
    # plt.ylim([-2, 40])
    # plt.ylim([1e1, 1e9])
    # ax2.set_yscale('log')  # log scale
    # plt.colorbar(ax.imshow(image, interpolation='nearest'))
    # plt.scatter(x, y)
    plt.plot(x, y, linestyle='-', marker='o', color='b')  # ,  fmt="bo", tz=None, xdate=True)
    xfmt = mdates.DateFormatter('%Y-%m-%d')
    ax4.xaxis.set_major_formatter(xfmt)
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.colorbar()
    plt.grid(True)
    # cbar = plt.colorbar()
    # cbar.set_label('Reach', rotation=270)
    # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
    # cmap = sns.diverging_palette(5, 250, as_cmap=True)
    name_fig4 = folder_RES + '/Fig.SumAllClicks-vs-Date.period-' + str(period) + '.png'
    plt.savefig(name_fig4, format='png')
    print (' ===> figure 4 = ' + name_fig4)
    # plt.show() # caution: it stops the flow of the program
    plt.draw()  # draws without stopping the flow of the program
    plt.clf()  # clear figure
    plt.close()  # close the figure window and continue

    # ----------------------------------------------------------------------------------------
    # plot 2d: # 5: sumall actual displays vs time
    fig5 = plt.figure()
    plt.title('Clients: Period of ' + str(period) + ' days: [no NaN or Null]')
    ax5 = fig5.add_subplot(1, 1, 1)
    x = mdates.datestr2num(dates)
    plt.xlabel('Date')
    # plt.xlim([1e-2, 1e2])
    # ax1.set_xscale('log')  # log scale
    # y = clean_clust_df[POI[4]]*100
    # plt.ylabel(POI[4] + '*100')
    # log => no x 100
    y = sumall_actualdisplays
    # change scale
    scale_y = 1e6
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y / scale_y))
    ax5.yaxis.set_major_formatter(ticks)
    plt.ylabel('Sum of Actual Displays in this period [M]')
    # plt.ylim([-2, 40])
    # plt.ylim([1e1, 1e9])
    # ax2.set_yscale('log')  # log scale
    # plt.colorbar(ax.imshow(image, interpolation='nearest'))
    # plt.scatter(x, y)
    plt.plot(x, y, linestyle='-', marker='o', color='b')  # ,  fmt="bo", tz=None, xdate=True)
    xfmt = mdates.DateFormatter('%Y-%m-%d')
    ax5.xaxis.set_major_formatter(xfmt)
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.colorbar()
    plt.grid(True)
    # cbar = plt.colorbar()
    # cbar.set_label('Reach', rotation=270)
    # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
    # cmap = sns.diverging_palette(5, 250, as_cmap=True)
    name_fig5 = folder_RES + '/Fig.SumAllActualDisplays-vs-Date.period-' + str(period) + '.png'
    plt.savefig(name_fig5, format='png')
    print (' ===> figure 5 = ' + name_fig5)
    # plt.show() # caution: it stops the flow of the program
    plt.draw()  # draws without stopping the flow of the program
    plt.clf()  # clear figure
    plt.close()  # close the figure window and continue

    # ----------------------------------------------------------------------------------------
    # plot 2d: # 6: sumall potential displays vs time
    fig6 = plt.figure()
    plt.title('Clients: Period of ' + str(period) + ' days: [no NaN or Null]')
    ax6 = fig6.add_subplot(1, 1, 1)
    x = mdates.datestr2num(dates)
    plt.xlabel('Date')
    # plt.xlim([1e-2, 1e2])
    # ax1.set_xscale('log')  # log scale
    # y = clean_clust_df[POI[4]]*100
    # plt.ylabel(POI[4] + '*100')
    # log => no x 100
    y = sumall_potdisplays
    # change scale
    scale_y = 1e6
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y / scale_y))
    ax6.yaxis.set_major_formatter(ticks)
    plt.ylabel('Sum of Potential Displays in this period [M]')
    # plt.ylim([-2, 40])
    # plt.ylim([1e1, 1e9])
    # ax2.set_yscale('log')  # log scale
    # plt.colorbar(ax.imshow(image, interpolation='nearest'))
    # plt.scatter(x, y)
    plt.plot(x, y, linestyle='-', marker='o', color='b')  # ,  fmt="bo", tz=None, xdate=True)
    xfmt = mdates.DateFormatter('%Y-%m-%d')
    ax6.xaxis.set_major_formatter(xfmt)
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.colorbar()
    plt.grid(True)
    # cbar = plt.colorbar()
    # cbar.set_label('Reach', rotation=270)
    # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
    # cmap = sns.diverging_palette(5, 250, as_cmap=True)
    name_fig6 = folder_RES + '/Fig.SumAllPotDisplays-vs-Date.period-' + str(period) + '.png'
    plt.savefig(name_fig6, format='png')
    print (' ===> figure 6 = ' + name_fig6)
    # plt.show() # caution: it stops the flow of the program
    plt.draw()  # draws without stopping the flow of the program
    plt.clf()  # clear figure
    plt.close()  # close the figure window and continue


    #--------------- PLOTTING FNISHED ----------------

#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()