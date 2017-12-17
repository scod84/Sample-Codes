# WHAT
# This takes the table created and plots the files with highlighted the points in each group -> check if OK
#
#  ----
# AUTHOR = Daniele Scopece, 2017-08-30 (adapted)
#
# VERSIONS:
# v01.01 = it works only with 4 sectors
#
#
#---------------------------------------------------------------------------------

# ---- Imports various
#import csv
#import scipy
#print('scipy: {}'.format(scipy.__version__))
#from scipy.cluster.hierarchy import dendrogram, linkage as lkg, cophenet
#from scipy.spatial.distance import pdist

import pandas as pd
from pandas import DataFrame # to create a dataframe
#from pandas.tools.plotting import scatter_matrix

import numpy as np
print('numpy: {}'.format(np.__version__))

import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.lines as mlines # to draw lines
print('matplotlib: {}'.format( matplotlib.__version__))
#from mpl_toolkits.mplot3d import Axes3D

#import sklearn
#print ('sklearn: {}'.format(sklearn.__version__))
#from sklearn import manifold
#from sklearn.cluster import KMeans, AgglomerativeClustering as AggC, DBSCAN
#from sklearn.decomposition import PCA
#from scipy.spatial.distance import cdist

#import math

import seaborn as sns

from datetime import datetime
from datetime import timedelta

# create directory
import os


#--------------


#----------------------------------------------------------------------------------
def main():
    print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (' Plotting the classification based on Quartiles of File 04.01')

    print ('')
    print ('------------------------------------------------------------')
    print (' IMPORTANT MEMO: Dependencies:')
    print (' Have you run:')
    print ('   1) the file 01.01 to get the csv files?')
    print ('   2) the file 04.01 to get the classification?')

    print ('')
    print ('------------------------------------------------------------')
    print (' Reading the start, end date and the period:')
    input_file = '00.00.PARAMETERS.txt'

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




    print ('')
    print ('------------------------------------------------------------')
    print (' Folder of the Outputs : ')
    folder_RES = '04.02.RES-Quartiles-Plots-e-Graphs'
    print (folder_RES)
    # creating the folder if it does not exist
    if not os.path.exists(folder_RES):
        os.makedirs(folder_RES)

    #quit()


    print ('')
    print (' ---------------------------------')
    print (' Where taking the csv file from')
    folder_input = '04.01.RES-Quartiles'
    print (folder_input)
    csv_file_type = folder_input


    print ('')
    print (' ---------------------------------')
    print (' Saving the columns x = log10 audience; y = log10 revenues , color = reach')
    # name of x, y, color to take from var_df
    List_name_col = {
        'x': 'log_audience',
        'y': 'log_sum_revenue_euro',
        'color': 'reach'
    }
    print (List_name_col)


    while start > stop:
        # print (day)  # start.strftime('%Y-%m-%d'))
        # tbl = 'Data.' + day  # name of the table to be written


        print ('')
        print ('/////////////////////////////////////////////////////////////////////////////////////////////////////')
        #MYFILE = csv_file_type + day + '.csv'  # 'TABLE_v02.02_No-duplicate-funnel.csv'
        #print ('Table now (MYFILE) = ' + MYFILE)
        print (' +++ Date now = ' + day)
        name_df = folder_input + '/RES.Date-' + day + '.period-' + str(period) + '.clients_id-idxColorReach-idxYRev.01-Unsorted.csv'
        print (' name_df = ' + name_df)


        print ('')
        print (' -------------------------------------')
        print (' Reading the database of 04.01 and changing names of columns')
        df_input = pd.read_csv(name_df, na_values=['None'], skiprows=3, skip_blank_lines=True, thousands=',', index_col=False)
        print(df_input.head())
        #quit()

        print ('')
        print (' -------------------------------------')
        print (' Changing the names of the columns for practical purposes')
        df_input.rename(columns={'index_'+List_name_col['color']: 'index_quartile_reach'}, inplace=True)
        df_input.rename(columns={'index_'+List_name_col['y']: 'index_quartile_log_rev_eur'}, inplace=True)
        print(df_input.head())

        #quit()

        # ---------------------------------------------------------------------------------------------------
        print ('')
        print ('-------------------------------------')
        print (' Selecting the clients that are Tier1 according to Criteo')

        # save all coord for plots of all points
        x_all = df_input['log_audience']
        y_all = df_input['log_sum_revenue_euro']


        x_tier_1 = []
        y_tier_1 = []

        l_print_tier1 = False

        for index, row in df_input.iterrows():
            if l_print_tier1: print (index),
            if df_input['is_tier_1'][index] == 1:
                if l_print_tier1: print (df_input['is_tier_1'][index]),
                # here we are ok
                x_tier_1.append(df_input['log_audience'][index])
                y_tier_1.append(df_input['log_sum_revenue_euro'][index])
            if l_print_tier1: print ('')

        print ('Total number of clients now = ' + str(len(x_tier_1)))
        print ('Out of which are Tier 1 = ' + str(len(x_all)))

        print ('')
        print ('-------------------------------------')
        print (' Making a figure to with all without Tier 1')

        # plot 2d: # 5: Audience vs CPC & color = reach
        fig1 = plt.figure()
        plt.title('Clients Segmentation: Date ' + day + ', Period ' + str(period) +
                  '\n number of total clients = ' + str(len(x_all)) )
                  #+ ' (' + str(int(len(x_tier_1) * 100 / len(x_all))) + '%)')
        ax1 = fig1.add_subplot(1, 1, 1)

        plt.xlabel('log_audience')
        plt.ylabel('log_sum_revenue_euro')

        plt.xlim([2, 8])
        plt.ylim([-2, 6.5])

        # select the x and y -> x2, y2 -> made above: ATTENTION: for this excample only

        # multiple series
        # all points
        ax1.scatter(x_all, y_all, c=df_input['reach'], cmap='rainbow', vmin=0.0, vmax=1.0, marker="o",
                    label="All Points")
        ax1.colorbar()
        # points that are tier 1
        #ax1.scatter(x_tier_1, y_tier_1, c='black', marker="1", label="Tier1", s=100)
        plt.legend(loc='upper left')

        plt.grid(True)

        subfolder_now = '/01.01.graph-All-vs-Tier1'
        # creating the folder if it does not exist
        directory = folder_RES + subfolder_now
        if not os.path.exists(directory):
            os.makedirs(directory)

        name_fig1 = folder_RES + subfolder_now + '/Fig.Date-' + day + '.period-' + str(period) + '.All.png'
        plt.savefig(name_fig1, format='png')
        print ('  ==> Figure now = ' + name_fig1)
        # plt.show() # caution: it stops the flow of the program
        plt.draw()  # draws without stopping the flow of the program
        plt.clf()  # clear figure
        plt.close()  # close the figure window and continue


        print ('')
        print ('-------------------------------------')
        print (' Making a figure to compare the Tier 1 vs non-tier 1')

        # plot 2d: # 5: Audience vs CPC & color = reach
        fig1 = plt.figure()
        plt.title('Clients Segmentation: Date ' + day + ', Period ' + str(period) +
                  '\n number of total clients = '+str(len(x_all)) + ', of which tier 1 = ' + str(len(x_tier_1))
                  + ' (' + str(int(len(x_tier_1)*100/len(x_all))) + '%)')
        ax1 = fig1.add_subplot(1, 1, 1)

        plt.xlabel('log_audience')
        plt.ylabel('log_sum_revenue_euro')

        plt.xlim([2, 8])
        plt.ylim ([-2, 6.5])

        # select the x and y -> x2, y2 -> made above: ATTENTION: for this excample only

        # multiple series
        # all points
        ax1.scatter(x_all, y_all, c=df_input['reach'], cmap='rainbow', vmin=0.0, vmax=1.0, marker="o",
                     label="All Points")
        # points that are tier 1
        ax1.scatter(x_tier_1, y_tier_1, c='black', marker="1", label="Tier1", s=100)
        plt.legend(loc='upper left')

        plt.grid(True)

        subfolder_now = '/01.01.graph-All-vs-Tier1'
        # creating the folder if it does not exist
        directory = folder_RES + subfolder_now
        if not os.path.exists(directory):
            os.makedirs(directory)

        name_fig1 = folder_RES + subfolder_now + '/Fig.Date-' + day + '.period-' + str(period) + '.All-vs-Tier-1.png'
        plt.savefig(name_fig1, format='png')
        print ('  ==> Figure now = ' + name_fig1)
        # plt.show() # caution: it stops the flow of the program
        plt.draw()  # draws without stopping the flow of the program
        plt.clf()  # clear figure
        plt.close()  # close the figure window and continue






        #---------------------------------------------------------------------------------------------------
        print ('')
        print ('-------------------------------------')
        print (' Plotting the Clients highlighted on the graph for each sector')

        # it works for 4 groups

        # for test purposes
        #coord_chosen = [df_input['log_audience'][0], df_input['log_sum_revenue_euro'][0]]
        #print (coord_chosen)
        #x_chosen = coord_chosen[0]
        #x2 = x_chosen
        #y_chosen = coord_chosen[1]
        #y2 = y_chosen



        # cycle on the points
        for i_reach_now in range(0,4):
            # sum 1 because it starts form 1
            index_reach_now = i_reach_now + 1
            for i_rev_now in range (0,4):
                index_rev_now = i_rev_now + 1

                #print ('++++ reach, rev = ' + str(index_reach_now) + ', ' + str(index_rev_now))

                # initialize the vectors
                # declartion of the coordinates of the points in the group
                x_chosen = []
                y_chosen = []

                l_print = False

                #print ('---')
                #print (' Analyzing line ny line for idx_reach = ' + str(index_reach_now) + ', index_log_rev = ' + str(index_rev_now))
                # searching in the lines
                for index, row in df_input.iterrows():
                    if l_print: print (index),
                    if df_input['index_quartile_reach'][index] == index_reach_now:
                        if l_print: print (df_input['index_quartile_reach'][index]),
                        if df_input['index_quartile_log_rev_eur'][index] == index_rev_now:
                            if l_print: print (df_input['index_quartile_log_rev_eur'][index])
                            # here we are ok
                            x_chosen.append(df_input['log_audience'][index])
                            y_chosen.append(df_input['log_sum_revenue_euro'][index])
                    if l_print: print ('')




                #print ('---')
                #print (' Plotting the Clients highlighted on the graph')
                # check
                #print (x_chosen)
                #print (y_chosen)
                x2 = x_chosen
                y2 = y_chosen

                number_clients = len(x_chosen)

                # plot 2d: # 5: Audience vs CPC & color = reach
                fig10 = plt.figure()
                plt.title('Clients Segmentation: Date ' + day + ', Period ' + str(period) + '\nIdxReach' + str(index_reach_now) + '-IdxRev' + str(index_rev_now) + '; num clients = ' + str(number_clients))
                ax10 = fig10.add_subplot(1, 1, 1)

                plt.xlabel('log_audience')
                # plt.xlim([-2, 2])
                # ax5.set_xscale('log')  # log scale
                # y = clean_clust_df[POI[4]]*100
                # plt.ylabel(POI[4] + '*100')
                # log => no x 100

                plt.ylabel('log_sum_revenue_euro')

                plt.xlim([2, 8])
                plt.ylim([-2, 6.5])

                # select the x and y -> x2, y2 -> made above: ATTENTION: for this excample only

               # multiple series
                # all points
                ax10.scatter(x_all, y_all, c=df_input['reach'], cmap='rainbow', vmin=0.0, vmax=1.0, marker="o", label="All Points")
                # points in the group
                #ax10.scatter(x2, y2, c='black', marker="s", label="In the sector", s=100)

                ax10.scatter(x2, y2, facecolors = 'none', edgecolors = 'black', marker="s", label="In the sector", s=100)
                # points that are tier 1
                ax10.scatter(x_tier_1, y_tier_1, c='black', marker="1", label="Tier1", s=100)
                plt.legend(loc='upper left')

                # plt.ylim([-2, 40])
                # plt.ylim([1, 9])
                # ax5.set_yscale('log')  # log scale
                # plt.colorbar(ax.imshow(image, interpolation='nearest'))
                # plt.scatter(x, y, c=var_df[POI_here[2]], cmap='rainbow', vmin=0.0, vmax=1.0)
                # ax10.colorbar()
                plt.grid(True)
                # cbar = plt.colorbar()
                # cbar.set_label('Reach', rotation=270)
               # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
                # cmap = sns.diverging_palette(5, 250, as_cmap=True)
                subfolder_now = '/02.01.graph-All-vs-Groups'
                # creating the folder if it does not exist
                directory = folder_RES + subfolder_now
                if not os.path.exists(directory):
                    os.makedirs(directory)
                name_fig10 = folder_RES + subfolder_now + '/Fig.Date-'+ day + '.period-' + str(period) +'.IdxReach' + str(index_reach_now) \
                             + '-IdxRev' + str(index_rev_now) + '.png'
                plt.savefig(name_fig10, format='png')
                print ('  ==> Figure now = ' + name_fig10)
                # plt.show() # caution: it stops the flow of the program
                plt.draw()  # draws without stopping the flow of the program
                plt.clf()  # clear figure
                plt.close()  # close the figure window and continue

                del x_chosen
                del y_chosen

                #print (' End of cycle on the rows')

        print ('')
        print ('----------------------------------------------------------------------------')
        print (
        ' Initialization for counting of how many elements in each of the groups and how many of these are tier 1')

        num_sectors_reach = 4  # x
        num_sectors_log_rev = 4  # y

        # number of clients in each sector: first index = x = reach, second index = y = log_rev
        num_client_in_group = [[0] * num_sectors_log_rev for x in xrange(num_sectors_reach)]
        print (' Check: must be 4x4 null: '),
        print (num_client_in_group)

        # number of clients in each sector that are tier_1: first index = x = reach, second index = y = log_rev
        num_client_in_group_tier1 = [[0] * num_sectors_log_rev for x in xrange(num_sectors_reach)]
        print (' Check: must be 4x4 null: '),
        print (num_client_in_group_tier1)


        print ('')
        print ('------------------------------------------')
        print (' Reading the elements in each group from RES-04.01 *03-Count')
        #print (' +++ Date now = ' + day)
        name_df = folder_input + '/RES.Date-' + day + '.period-' + str(
            period) + '.clients_id-idxColorReach-idxYRev.03-Count.csv'
        print (' name_df = ' + name_df)
        df_input_3 = pd.read_csv(name_df, na_values=['None'], skiprows=6, skip_blank_lines=True, thousands=',',
                               index_col=False)
        print(df_input_3.head())

        #quit()

        for index, row in df_input_3.iterrows():
            index_of_reach_in_vector  = df_input_3['index_reach'][index] - 1
            index_of_logrev_in_vector = df_input_3['index_revenues'][index] - 1
            num_client_in_group[index_of_reach_in_vector][index_of_logrev_in_vector] = df_input_3['number_clients_in_sector'][index]
            num_client_in_group_tier1[index_of_reach_in_vector][index_of_logrev_in_vector] = df_input_3['number_clients_tier1'][index]

        # for index, row in df_input.iterrows():
        #     # counting number of elements in each group and how many are tier 1 -> plot of heatmap
        #     # make plus 1
        #     index_of_reach_in_vector = df_input['index_quartile_reach'][index] - 1
        #     index_of_logrev_in_vector = df_input['index_quartile_log_rev_eur'][index] - 1
        #     num_client_in_group[index_of_reach_in_vector][index_of_logrev_in_vector] += 1
        #     # check
        #     if df_input['is_tier_1'][index] == 1:
        #         num_client_in_group_tier1[index_of_reach_in_vector][index_of_logrev_in_vector] += 1
        #
        #     # print ('print (index, index_of_reach_in_vector, index_of_logrev_in_vector, df_input[is_tier_1][index]) = '),
        #     # print (index, index_of_reach_in_vector, index_of_logrev_in_vector, df_input['is_tier_1'][index])
        #     # print ('num_client_in_group       : '),
        #     # print (num_client_in_group)
        #     #  print ('num_client_in_group_tier1 : '),
        #     # print (num_client_in_group_tier1)
        #     # print ('----')

        print ('Final count:')
        print ('num_client_in_group       : '),
        print (num_client_in_group)
        print ('num_client_in_group_tier1 : '),
        print (num_client_in_group_tier1)

        del df_input_3


        print ('')
        print ('------------------------------------------')
        print (' Plotting the Heatmap of number of clients')
        # create a dataframe to plot
        # https://stackoverflow.com/questions/12286607/python-making-heatmap-from-dataframe
        dfplot_index = [1, 2, 3, 4]  #
        dfplot_cols = [1, 2, 3, 4]  #
        dfplot = DataFrame(num_client_in_group, index=dfplot_index, columns=dfplot_cols)
        print (dfplot.head())

        ax3 = sns.heatmap(dfplot, annot=True, fmt='g')
        # for t in ax3.texts: t.set_text(t.get_text() + " AllClients")  # add percentage in notation

        plt.title('Clients: Date ' + day + ', Period ' + str(period) + '\nHow many clients in each sector?')
        plt.xlabel('Index of Log Revenue Euro')
        plt.yticks(rotation=0)
        plt.ylabel('Index of Reach')
        plt.yticks(rotation=0)

        plt.tight_layout()
        subfolder_now = '/03.01.heatmap-groups-all'
        # creating the folder if it does not exist
        directory = folder_RES + subfolder_now
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig_name = folder_RES + subfolder_now + '/Heatmap-01-AllClients.Date-' + day + '.period-' + str(period) + '.png'
        print ('   ==> Figure = ' + fig_name)
        plt.savefig(fig_name)
        plt.clf()
        del dfplot # clean dataframe per plot


        print ('')
        print ('----------------------------------------------------------------------------')
        print (' Plotting Graph with numbers of clients that are classified as tier1')

        # create a dataframe to plot
        # https://stackoverflow.com/questions/12286607/python-making-heatmap-from-dataframe
        dfplot2_index = [1, 2, 3, 4]  #
        dfplot2_cols = [1, 2, 3, 4]  #
        dfplot2 = DataFrame(num_client_in_group_tier1, index=dfplot2_index, columns=dfplot2_cols)
        print (dfplot2.head())

        ax2 = sns.heatmap(dfplot2, annot=True, fmt='g')
        # for t in ax2.texts: t.set_text(t.get_text() + " Tier1") # add percentage in notation

        # NOTE: x and y are reversed here wrt to the graphs
        plt.title('Clients: Date ' + day + ', Period ' + str(period) + '\nHow many clients in each sector are classified as Tier1 by Criteo?')
        plt.xlabel('Index of Log Revenue Euro')
        plt.yticks(rotation=0)
        plt.ylabel('Index of Reach')
        plt.yticks(rotation=0)

        plt.tight_layout()
        subfolder_now = '/04.01.heatmap-groups-tier-1-absolute'
        # creating the folder if it does not exist
        directory = folder_RES + subfolder_now
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig_name2 = folder_RES + subfolder_now + '/Heatmap-02-Tier1Abs.Date-' + day + '.period-' + str(period) + '.png'
        print (' ==> Figure = ' + fig_name2)
        plt.savefig(fig_name2)
        plt.clf()

        del dfplot2


        print ('')
        print ('----------------------------------------------------------------------------')
        print (' Plotting Graph with % of clients that are tier 1 in each sector')

        # create a dataframe to plot
        # https://stackoverflow.com/questions/12286607/python-making-heatmap-from-dataframe
        dfplot3_index = [1, 2, 3, 4]  #
        dfplot3_cols = [1, 2, 3, 4]  #

        # compute percentage
        perc_clients_tier_1 = [[0] * num_sectors_log_rev for x in xrange(num_sectors_reach)]
        #print (perc_clients_tier_1)
        for i_reach in range(0, 4):
            for j_rev in range(0, 4):
                if num_client_in_group[i_reach][j_rev] == 0:
                    perc_clients_tier_1[i_reach][j_rev] = 0.0
                else:
                    perc_clients_tier_1[i_reach][j_rev] = float(
                        num_client_in_group_tier1[i_reach][j_rev]) * 100.0 / float(num_client_in_group[i_reach][j_rev])
        #print (perc_clients_tier_1)

        # quit()

        dfplot3 = DataFrame(perc_clients_tier_1, index=dfplot3_index, columns=dfplot3_cols)
        print (dfplot3.head())

        ax3 = sns.heatmap(dfplot3, annot=True, fmt='.0f',vmin=0, vmax=100)  # fmt='g')
        for t in ax3.texts: t.set_text(t.get_text() + " %")  # add percentage in notation

        # NOTE: x and y are reversed here wrt to the graphs
        plt.title('Clients: Date ' + day + ', Period ' + str(period) + '\nHow much percentage of clients in each sector are classified as Tier1 by Criteo?')
        plt.xlabel('Index of Log Revenue Euro')
        plt.yticks(rotation=0)
        plt.ylabel('Index of Reach')
        plt.yticks(rotation=0)

        plt.tight_layout()
        subfolder_now = '/05.01.heatmap-groups-tier-1-percentage'
        # creating the folder if it does not exist
        directory = folder_RES + subfolder_now
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig_name3 = folder_RES + subfolder_now + '/Heatmap-03-Tier1Perc.Date-' + day + '.period-' + str(period) + '.png'
        print (' ==> Figure = ' + fig_name3)
        plt.savefig(fig_name3)
        plt.clf()

        del dfplot3

        print ('')
        print ('----------------------------------------')
        print (' Cleaning the dataframe for this period')
        del df_input


        #quit() # 1 date only

        # cycles
        start = start - timedelta(days=period)
        day = start.strftime('%Y-%m-%d')

        # clean the df
        #del clean_clust_df

    quit()




#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()