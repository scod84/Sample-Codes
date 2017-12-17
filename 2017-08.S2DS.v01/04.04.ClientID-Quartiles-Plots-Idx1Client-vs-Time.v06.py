# WHAT
# This takes the table created and plots the indexes vs time of a given set of client_id
#
#  ----
# AUTHOR = Daniele Scopece, 2017-08-30 (adapted)
#
# VERSIONS:
# v01.01 = it works only with 4 sectors
#
#
# ---------------------------------------------------------------------------------

# ---- Imports various
#import csv
#import scipy
#print('scipy: {}'.format(scipy.__version__))
#from scipy.cluster.hierarchy import dendrogram, linkage as lkg, cophenet
#from scipy.spatial.distance import pdist

import pandas as pd
#from pandas import DataFrame # to create a dataframe
#from pandas.tools.plotting import scatter_matrix

import numpy as np
print('numpy: {}'.format(np.__version__))

import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.lines as mlines # to draw lines
print('matplotlib: {}'.format( matplotlib.__version__))
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates # for dates as axis

#import sklearn
#print ('sklearn: {}'.format(sklearn.__version__))
#from sklearn import manifold
#from sklearn.cluster import KMeans, AgglomerativeClustering as AggC, DBSCAN
#from sklearn.decomposition import PCA
#from scipy.spatial.distance import cdist

#import math

#import seaborn as sns

from datetime import datetime
from datetime import timedelta

# create directory
import os
#--------------


#----------------------------------------------------------------------------------
def main():
    print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (' Plotting the trend with dates of the indexes for a given set of client_id based on File 04.01')

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

    day_start = start.strftime('%Y-%m-%d')
    day_stop = stop.strftime('%Y-%m-%d')



    print ('')
    print ('------------------------------------------------------------')
    input_file_clientid = '04.04.LIST-Client_id.txt'
    print (' Reading the list of the client_id from the file : ' + input_file_clientid)

    # first line = number of client_id that I want
    # second... n+1 lines = the number of the client_id one per line


    f_in_client = open(input_file_clientid, 'r')
    lines_id = f_in_client.readlines()[1:]  # reads starting from the first line and stores the line in a string
    #print (lines_id)
    list_clientid = []
    for i in range(0, len(lines_id)):
        list_clientid.append(int(lines_id[i].split(' ', 1)[0]))
    num_clientid_2_analyse = len(list_clientid)
    print (' Found '+ str(len(list_clientid)) + ' client_id to analyse:'),
    print (list_clientid)



    print ('')
    print ('------------------------------------------------------------')
    print (' Folder of the Outputs : ')
    folder_RES = '04.04.RES-Quartiles-Plot-Idx1Client-vs-Time'
    print (folder_RES)
    # creating the folder if it does not exist
    if not os.path.exists(folder_RES):
        os.makedirs(folder_RES)

    #quit()


    print ('')
    print (' ---------------------------------')
    print (' Where taking the csv file from ')
    folder_input = '04.01.RES-Quartiles'
    print (folder_input)
    csv_file_type = folder_input



    print ('')
    print (' ---------------------------------')
    print (' Opening the files output to write:')
    name_file_RES = []  # '' * num_clientid_2_analyse ] # name_file_RES[0][1] = index_reach = 0+1, index_rev = 1+1
    file_RES_csv = []


    #print (name_file_RES)
    #quit()

    for i in range(0,num_clientid_2_analyse):
        client_id_now = list_clientid[i]
        print client_id_now
        name_here = folder_RES + '/RES.Date-' + day_start + '.' + day_stop + '.period-' + str(period) + '.client_id-' \
                    + str(client_id_now) + '-Indexes-vs-date'
        name_file_RES.append(name_here)
        file_RES_csv.append(client_id_now)
        file_RES_csv[i] = open(name_here + '.csv', 'w')
        file_RES_csv[i].write('client_id,date,period,index_reach,index_logrev,merchant_id,is_tier_1\n')
        print (file_RES_csv[i])


    print ('')
    print (' ---------------------------------')
    print (' Cycling on the date:')

    while start > stop:
        # print (day)  # start.strftime('%Y-%m-%d'))
        # tbl = 'Data.' + day  # name of the table to be written


        print ('')
        print ('/////////////////////////////////////////////////////////////////////////////////////////////////////')
        #MYFILE = csv_file_type + day + '.csv'  # 'TABLE_v02.02_No-duplicate-funnel.csv'
        #print ('Table now (MYFILE) = ' + MYFILE)
        print (' +++ Date now = ' + day)
        name_df = folder_input + '/RES.Date-' + day + '.period-' + str(period) + '.clients_id-idxColorReach-idxYRev.02-with-Date.csv'
        print (' name_df = ' + name_df)


        print ('')
        print (' -------------------------------------')
        print (' Reading the database of 04.01 and changing names of columns')
        df_input = pd.read_csv(name_df, na_values=['None'], skiprows=1, skip_blank_lines=True, thousands=',', index_col=False)
        print(df_input.head())
        #quit()


        # ---------------------------------------------------------------------------------------------------
        print ('')
        print ('-------------------------------------')
        print (' Cycling on the rows of this dataframe for this date and writing the csv file')

        # find the index of client_id =

        #for i_reach in range(0, 4):
        #    for i_rev in range(0, 4):
        #        index_reach_now = i_reach + 1 #df_input['index_reach'][index]
        #        index_rev_now = i_rev + 1 #df_input['index_log_sum_revenue_euro'][index]
        #        print (' +++ Doing: idx reach = ' + str(index_reach_now) + ', idx rev = ' + str(index_rev_now))

                #print (' Cycling on the client_id to find it')
        for i in range(0, num_clientid_2_analyse):
                    client_id_now = list_clientid[i]
                    #print (client_id_now)
                    #print ('client_id_now = ' + str(client_id_now))

                    # find the index where df_input['client_id'] = client_id_now, by handling when not in list
                    #print (Index(df_input['client_id']).get_loc(client_id_now)) #
                    #index_in_df = list(df_input['client_id']).index(client_id_now)
                    # initialize to -1: it is so if not found
                    index_in_df = -1
                    for index, row in df_input.iterrows():
                        if df_input['client_id'][index] == client_id_now:
                            index_in_df = index
                            break
                    #print ('index_in_df = ' + str(index_in_df))

                    #print ('ERRORE QUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII')
                    #print ('TODO: check if the element is in the list: -> element, else = index = -1')
                    #quit()

                    #if client_id_now in df_input['client_id']:
                    #    index_in_df = list(df_input['client_id']).index(client_id_now)
                    #else:
                    #    index_in_df = -1
                    #index_in_df = list(df_input['client_id'].index(client_id_now)) if client_id_now in df_input['client_id'] else -1
                    #print (list(df_input['client_id']).index(client_id_now))

                    if index_in_df != -1:
                        # here if found in the list
                        idx_reach_now = df_input['index_reach'][index_in_df]
                        idx_rev_now = df_input['index_log_sum_revenue_euro'][index_in_df]
                        merchant_id_now = df_input['merchant_id'][index_in_df]
                        is_tier1_now = df_input['is_tier_1'][index_in_df]
                    else:
                        # not present in the list
                        idx_reach_now = 0
                        idx_rev_now = 0
                        merchant_id_now = -1
                        is_tier1_now = -1

                    # client_id,date,period,index_reach,index_logrev
                    # write the correct file
                    file_RES_csv[i].write(str(client_id_now) + ',' + day + ',' + str(period) + ',' + \
                        str(idx_reach_now) + ',' + str(idx_rev_now) + ',' + \
                        str(merchant_id_now) + ',' + str(is_tier1_now) + '\n')


        print ('')
        print ('----------------------------------------')
        print (' Cleaning the dataframe for this period')
        del df_input


        #quit() # 1 date only

        # cycles
        start = start - timedelta(days=period)
        day = start.strftime('%Y-%m-%d')


    print ('')
    print ('----------------------------------------')
    print (' Closing the files csv')
    for i in range(0,num_clientid_2_analyse):
        file_RES_csv[i].close()
        print ('Closing : ' + str(name_file_RES[i]) + '.csv')

    #quit()


    print ('')
    print ('----------------------------------------')
    print (' Plotting the Trends for each client with the csv just created')
    for i in range(0, num_clientid_2_analyse):
            client_id_now = list_clientid[i]
            name_png_now = name_file_RES[i] + '.png'
            name_csv_now = name_file_RES[i] + '.csv'

            #print (' -- Opening the dataframe just created: '),
            #print (name_csv_now)
            df_now = pd.read_csv(name_csv_now, na_values=['None'], skip_blank_lines=True, skiprows=0, thousands=',')
            #print (df_now.head())
            #quit()

            #print (' -- Writing the image file = '),
            #print (name_png_now)

            # save in a vector
            date_now = []
            lst_idx_reach = []
            lst_idx_rev = []
            lst_is_tier_1 =[]
            merchant_id_now = df_now['merchant_id'][0]
            #is_tier1_now = df_now['is_tier_1'][0]
            for index, row in df_now.iterrows():
                date_now.append(df_now['date'][index])
                lst_idx_reach.append(df_now['index_reach'][index])
                lst_idx_rev.append(df_now['index_logrev'][index])
                lst_is_tier_1.append(df_now['is_tier_1'][index]/2.0)



            fig1 = plt.figure()
            plt.title('Date in ' + day_start + '.' + day_stop + ' Period ' + str(period) \
                      + '\nClient_id = ' + str(client_id_now) +', merchant_id = ' + str(merchant_id_now))
            #+ \
            #          ', is_tier_1 = ' + str(is_tier1_now))
            ax1 = fig1.add_subplot(1, 1, 1)
            #day_now = df_now['day']
            #print (date_now)
            #quit()
            x = mdates.datestr2num(date_now)
            plt.xlabel('Date')
            xfmt = mdates.DateFormatter('%Y-%m-%d')
            ax1.xaxis.set_major_formatter(xfmt)

            y_reach = lst_idx_reach
            y_rev = lst_idx_rev
            plt.ylabel('Sector Index (0 = not present in list)')
            plt.ylim([-0.2, 5.5])

            y_tier1 = lst_is_tier_1

            ax1.plot(x, y_rev, c='black', marker='^', linestyle='-', label="Index LogRevenue")
            ax1.plot(x, y_reach, c='red', marker="o", linestyle='-', label="Index Reach")
            ax1.plot(x, y_tier1, c='green', marker='v', linestyle='-', label="Is Tier 1")

            start_x_axis, end_x_axis = ax1.get_xlim()
            #plt.xticks(rotation=-30, np.arange(start_x_axis, end_x_axis, 30.0))
            plt.xticks(rotation = -30)
            plt.xticks(np.arange(start_x_axis, end_x_axis, 90.0))
            # plt.xlim([1e-2, 1e2])
            # ax1.set_xscale('log')  # log scale
            # y = clean_clust_df[POI[4]]*100
            # plt.ylabel(POI[4] + '*100')
            # log => no x 100


            # plt.ylim([1e1, 1e9])
            # ax5.set_yscale('log')  # log scale
            # plt.colorbar(ax.imshow(image, interpolation='nearest'))
            # plt.scatter(x, y)
            #ax1.scatter(x, y_in_sector, c='red', marker="o", linestyle='-', label="Clients Inside")

            plt.legend(loc='upper right')
            #plt.plot(x, y, linestyle='-', marker='o', color='b')  # ,  fmt="bo", tz=None, xdate=True)

            # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            # plt.colorbar()
            plt.grid(True)
            # cbar = plt.colorbar()
            # cbar.set_label('Reach', rotation=270)
            # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
            # cmap = sns.diverging_palette(5, 250, as_cmap=True)
            #name_fig1 = folder_RES + '/Fig.NumActiveClients-vs-Date.period-' + str(period) + '.png'
            plt.tight_layout()
            plt.savefig(name_png_now, format='png')
            print ('  ==> figure now = ' + name_png_now)
            # plt.show() # caution: it stops the flow of the program
            plt.draw()  # draws without stopping the flow of the program
            plt.clf()  # clear figure
            plt.close()  # close the figure window and continue

            #quit() # test



    print ('')
    print (' ### THE END ###')
    #quit()




#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()