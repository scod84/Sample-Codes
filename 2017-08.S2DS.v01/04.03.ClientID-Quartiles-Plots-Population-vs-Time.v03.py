# WHAT
# This takes the table created and plots the plot population of each sector vs time in
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
import matplotlib.dates as mdates # for dates as axis

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


#create directory
import os
#--------------


#----------------------------------------------------------------------------------
def main():
    print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (' Plotting the Number of clients in each sector vs Date based on File 04.01')

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
    print (' Folder of the Outputs : ')
    folder_RES = '04.03.RES-Quartiles-Plots-Population-vs-Time'
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
    print (' Opening the files output to write:')
    #dates = []
    name_file_RES = [[''] * 4 for x in xrange(4)] # name_file_RES[0][1] = index_reach = 0+1, index_rev = 1+1
    file_RES_csv = [['']* 4 for x in xrange(4) ]
    #file_RES_png = [[''] * 4 for x in xrange(4)]
    print (name_file_RES)
    for i_reach in range(0,4):
        for i_rev in range(0,4):
            name_file_RES[i_reach][i_rev] = folder_RES + '/RES.Date-' + day_start + '.' + day_stop + '.period-' + str(period) \
                                        + '.GroupReach' + str(i_reach+1) + '.GroupRev' + str(i_rev+1)
            print ('name_file_RES [' + str(i_reach) + '][' + str(i_rev) +'] = ' + name_file_RES[i_reach][i_rev])
            # csv files
            #file_RES_csv[i_reach][i_rev] = 'f_csv_' + str(i_reach+1) + str(i_rev+1)
            file_RES_csv[i_reach][i_rev] = open(name_file_RES[i_reach][i_rev]+'.csv', 'w')
            file_RES_csv[i_reach][i_rev].write('# Index Reach = ' + str(i_reach+1) + ' -- Index Rev = ' + str(i_rev+1) + '\n')
            file_RES_csv[i_reach][i_rev].write('day,period,num_clients_inside,num_clients_tier1,num_clients_total,max_reach,max_rev\n')
            # png
            #file_RES_png[i_reach][i_rev] = 'f_png_' + str(i_reach + 1) + str(i_rev + 1)
            #file_RES_png[i_reach][i_rev] = open(name_file_RES[i_reach][i_rev] + '.png', 'w')
            #file_RES_png[i_reach][i_rev].write('# ' + str(i_reach + 1) + str(i_rev + 1))
    print (file_RES_csv) # how to call the right file in the program
    #print (file_RES_png)
    #quit()



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
        name_df = folder_input + '/RES.Date-' + day + '.period-' + str(period) + '.clients_id-idxColorReach-idxYRev.03-Count.csv'
        print (' name_df = ' + name_df)


        print ('')
        print (' -------------------------------------')
        print (' Reading the database of 04.01 and changing names of columns')
        df_input = pd.read_csv(name_df, na_values=['None'], skiprows=6, skip_blank_lines=True, thousands=',', index_col=False)
        print(df_input.head())
        #quit()

        print ('')
        print (' -------------------------------------')
        print (' Computing the total number of clients in this period')
        num_clients_total = df_input['number_clients_in_sector'].sum()
        print (num_clients_total)

        #print ('')
        #print (' -------------------------------------')
        #print (' Initializing the list of the number_clients[][]')
        #list_number_clients_inside = [[]] # first index = reach, second index = rev
        # test


        # ---------------------------------------------------------------------------------------------------
        print ('')
        print ('-------------------------------------')
        print (' Cycling on the rows of this dataframe for this date and writing the csv file')


        for index, row in df_input.iterrows():
            index_reach_now = df_input['index_reach'][index]
            index_rev_now = df_input['index_revenues'][index]
            print (' +++ idx reach = ' + str(index_reach_now) + ', idx rev = ' + str(index_rev_now))

            # reference of the file to write to now
            file_to_write = file_RES_csv[index_reach_now-1][index_rev_now-1] # -1 needed to be ok: in file is 1-4, in vector 0-3
            print (file_to_write)
            #day,period,num_clients_inside,num_clients_tier1,num_clients_total,max_reach,max_rev
            file_to_write.write(day + ',' + str(period) + ',' )
            file_to_write.write(str(df_input['number_clients_in_sector'][index]) + ',')
            file_to_write.write(str(df_input['number_clients_tier1'][index]) + ',')
            file_to_write.write(str(num_clients_total) + ',')
            file_to_write.write(str(df_input['max_sector_reach'][index]) + ',')
            file_to_write.write(str(df_input['max_sector_revenues'][index]))
            file_to_write.write('\n')
        #quit()

        print ('')
        print ('----------------------------------------')
        print (' Cleaning the dataframe for this period')
        del df_input


        # #quit() # 1 date only

        # cycles
        start = start - timedelta(days=period)
        day = start.strftime('%Y-%m-%d')


    print ('')
    print ('----------------------------------------')
    print (' Closing the files csv')
    for i_reach in range(0,4):
        for i_rev in range(0,4):
            print ('Closing : ' + str(name_file_RES[i_reach][i_rev]) + '.csv') #+ ' ??? TODO: how')
            file_RES_csv[i_reach][i_rev].close()


    print ('')
    print ('----------------------------------------')
    print (' Plotting the Trends with time of the number of clients in each sector')
    for i_reach in range(0, 4):
        for i_rev in range(0, 4):
            #print ('')
            # names of the files to open (csv) and to write (png)
            name_png_now = name_file_RES[i_reach][i_rev] + '.png'
            name_csv_now = name_file_RES[i_reach][i_rev] + '.csv'

            #print (' -- Opening the dataframe just created: '),
            #print (name_csv_now)
            df_now = pd.read_csv(name_csv_now, na_values=['None'], skip_blank_lines=True, skiprows=1, thousands=',')
            #print (df_now.head())

            #print (' -- Writing the image file = '),
            #print (name_png_now)

            # save in a vector
            date_now = []
            clients_in_sector = []
            clients_in_sector_tier1 = []
            for index, row in df_now.iterrows():
                date_now.append(df_now['day'][index])
                clients_in_sector.append(df_now['num_clients_inside'][index])
                clients_in_sector_tier1.append(df_now['num_clients_tier1'][index])



            fig1 = plt.figure()
            plt.title('Clients: Date in ' + day_start + '.' + day_stop + ' Period ' + str(period) \
                      + '\nSector: Index Reach = ' + str(i_reach+1) + ', Index LogRev = ' + str(i_rev+1))
            ax1 = fig1.add_subplot(1, 1, 1)
            #day_now = df_now['day']
            #print (date_now)
            #quit()
            x = mdates.datestr2num(date_now)
            plt.xlabel('Date')
            # plt.xlim([1e-2, 1e2])
            # ax1.set_xscale('log')  # log scale
            # y = clean_clust_df[POI[4]]*100
            # plt.ylabel(POI[4] + '*100')
            # log => no x 100
            y_in_sector = clients_in_sector
            y_in_sector_tier1 = clients_in_sector_tier1
            plt.ylabel('Number of Clients in this sector')
            # plt.ylim([-2, 40])
            # plt.ylim([1e1, 1e9])
            # ax5.set_yscale('log')  # log scale
            # plt.colorbar(ax.imshow(image, interpolation='nearest'))
            # plt.scatter(x, y)
            #ax1.scatter(x, y_in_sector, c='red', marker="o", linestyle='-', label="Clients Inside")
            ax1.plot(x, y_in_sector, c='red', marker="o", linestyle='-', label="Clients Inside")
            # points that are tier 1
            #ax1.scatter(x, y_in_sector_tier1, c='black', marker="1", linestyle='-', label="Of which Tier1", s=100)
            ax1.plot(x, y_in_sector_tier1, c='black', marker='v', linestyle='-', label="Of which Tier1")
            plt.legend(loc='upper right')
            #plt.plot(x, y, linestyle='-', marker='o', color='b')  # ,  fmt="bo", tz=None, xdate=True)
            xfmt = mdates.DateFormatter('%Y-%m-%d')
            ax1.xaxis.set_major_formatter(xfmt)
            # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            # plt.colorbar()
            plt.grid(True)
            # cbar = plt.colorbar()
            # cbar.set_label('Reach', rotation=270)
            # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
            # cmap = sns.diverging_palette(5, 250, as_cmap=True)
            #name_fig1 = folder_RES + '/Fig.NumActiveClients-vs-Date.period-' + str(period) + '.png'
            plt.savefig(name_png_now, format='png')
            print ('  ==> figure now = ' + name_png_now)
            # plt.show() # caution: it stops the flow of the program
            plt.draw()  # draws without stopping the flow of the program
            plt.clf()  # clear figure
            plt.close()  # close the figure window and continue




    print ('')
    print (' ### THE END ###')
    #quit()




#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()