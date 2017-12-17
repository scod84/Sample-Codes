# WHAT
# This is to compute a different classification of clients, based on Reach quantile and revenues quantile
# Purpose & Utility: new classification with two numbers -> more precise than Tier1/MMS
# --> table: client_id, date, period, index_reach, index_revenues
# --> see also 1 client how it evolves
#
# Method: with quantile divide into 4: 4 = values higher, ..., 1 = lower values
# diagram: x = log10 audience; y = log10 revenues_euro; color = reach
# divide first in reach and then in revenue inside each
#
#  ----
# AUTHOR = Daniele Scopece, 2017-08-30 (adapted)
#
# VERSIONS:
# v02 = ready for reach = color, log_sum_revenue_euro = y
#
#---------------------------------------------------------------------------------

# ---- Imports various
import csv
#import scipy
#print('scipy: {}'.format(scipy.__version__))
#from scipy.cluster.hierarchy import dendrogram, linkage as lkg, cophenet
#from scipy.spatial.distance import pdist

import pandas as pd
#from pandas.tools.plotting import scatter_matrix

import numpy as np
print('numpy: {}'.format(np.__version__))

#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.lines as mlines # to draw lines
#print('matplotlib: {}'.format( matplotlib.__version__))
#from mpl_toolkits.mplot3d import Axes3D

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
    print (' Classifying the Clients into Quartiles')

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

    folder_csv = '01.01.RES-CSV-TABLES'
    csv_file_type = folder_csv + '/Data.'  # + day + .csv


    print ('')
    print ('------------------------------------------------------------')
    print (' Folder of the Outputs : ')
    folder_RES = '04.01.RES-Quartiles'
    print (folder_RES)
    # creating the folder if it does not exist
    if not os.path.exists(folder_RES):
        os.makedirs(folder_RES)


    print ('')
    print ('---------------------------------------------------------------')
    print (' Parameters of to be saved in the dataframe')

    # TODO: Write here the parameters of interest
    POI = ['audience',
           'sum_revenue_euro',
           'reach',
           'CPC',
           'ratio_displays_acc_over_pot',
           #'exposed_users',
           'potential_displays',
           #'sum_displays',
           'sum_clicks',
           'client_id',
           'merchant_id',
           'is_tier_1'
           #'merchant_name'
           ]
    print (POI)


    print ('')
    print ('---------------------------------------------------------------')
    print (' Starting the cycle')

    while start > stop:
        # print (day)  # start.strftime('%Y-%m-%d'))
        #tbl = 'Data.' + day  # name of the table to be written


        print ('')
        print ('/////////////////////////////////////////////////////////////////////////////////////////////////////')
        MYFILE = csv_file_type + day + '.csv'  # 'TABLE_v02.02_No-duplicate-funnel.csv'
        print ('Table now (MYFILE) = ' + MYFILE)


        print ('')
        print (' ---------------------------------')
        print (' Reading the original dataframe')
        df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True, thousands=',')
        print (df.head())

        #quit()


        print ('')
        print (' ---------------------------------')
        print (' Computing the new variables in the dataframe')
        # Define the new entries in the table
        df['reach'] = df['exposed_users'] / df['audience']
        df['CPC'] = df['sum_revenue_euro'] / df['sum_clicks']
        df['ratio_displays_acc_over_pot'] = df['sum_displays'] / df['potential_displays']
        print(df.head())


        print ('')
        print (' ---------------------------------')
        print (' Creating a dataframe with only the variables wanted')
        clust_df = df[POI].copy()
        print(clust_df.head())

        #quit()

        print ('')
        print (' ---------------------------------')
        print (' Applying to numeric')
        clust_df.apply(pd.to_numeric)

        print ('')
        print (' ---------------------------------')
        print (' Removing the lines with NaN and removing the old dataframes (no more useful)')
        # Remove NaN's in this column -- need to find a better way to automate it
        clean_clust_df = removeNANs(POI, clust_df)

        del df
        del clust_df

        print(clean_clust_df.head())


        print ('')
        print (' ---------------------------------')
        print (' Computing the log of this dataframe')
        # --------------------- start compute log
        print('Starting compute log:')
        # define the dataframe with logs
        var_df = clean_clust_df[POI].copy()

        # change the name of the column -> with log: audience
        var_df.rename(columns={'audience': 'log_audience'}, inplace=True)
        #var_df['log_audience'] = np.log10(1.0 + clean_clust_df.audience)
        var_df['log_audience'] = np.log10(clean_clust_df.audience)
        # change sum_revenue_euro
        var_df.rename(columns={'sum_revenue_euro': 'log_sum_revenue_euro'}, inplace=True)
        #var_df['log_sum_revenue_euro'] = np.log10(1.0 + clean_clust_df.sum_revenue_euro)
        var_df['log_sum_revenue_euro'] = np.log10(clean_clust_df.sum_revenue_euro)
        # change reach
        #var_df.rename(columns={'reach': 'log_reach'}, inplace=True)
        #var_df['log_reach'] = np.log10(1.0 + clean_clust_df.reach)
        # change CPC
        var_df.rename(columns={'CPC': 'log_CPC'}, inplace=True)
        #var_df['log_CPC'] = np.log10(1.0 + clean_clust_df.CPC)
        var_df['log_CPC'] = np.log10(clean_clust_df.CPC)
        # change ratio_displays_acc_over_pot
        var_df.rename(columns={'ratio_displays_acc_over_pot': 'log_ratio_displays_acc_over_pot'}, inplace=True)
        #var_df['log_ratio_displays_acc_over_pot'] = np.log10(1.0 + clean_clust_df.ratio_displays_acc_over_pot)
        var_df['log_ratio_displays_acc_over_pot'] = np.log10(clean_clust_df.ratio_displays_acc_over_pot)
        # change potential_displays
        var_df.rename(columns={'potential_displays': 'log_potential_displays'}, inplace=True)
        #var_df['log_potential_displays'] = np.log10(1.0 + clean_clust_df.potential_displays)
        var_df['log_potential_displays'] = np.log10(clean_clust_df.potential_displays)
        # change sum_clicks
        var_df.rename(columns={'sum_clicks': 'log_sum_clicks'}, inplace=True)
        #var_df['log_sum_clicks'] = np.log10(1.0 + clean_clust_df.sum_clicks)
        var_df['log_sum_clicks'] = np.log10(clean_clust_df.sum_clicks)
        #print clean_clust_df.head()

        del clean_clust_df

        print (var_df.head())
        # --------------------- end compute log

        #quit()

        #------------------------------------------------------------------------------------------------------
        print ('')
        print (' ---------------------------------')
        print (' Saving the columns x = log10 audience; y = log10 revenues , color = reach')
        # name of x, y, color to take from var_df
        List_name_col = {
                    'x' : 'log_audience',
                    'y' : 'log_sum_revenue_euro',
                    'color' : 'reach'
                    }
        print (List_name_col)

        # tuple: (index, x, y, color, index_color, index_y, client_id, merchant_id, is_tier_1), declare list
        # index_color and index_y initialized to -1
        points_in_log_log = []

        # columns to divide and find max min
        column_color = []
        column_y = []

        # saving the columns from var_df, taking just the finite numbers, not inf, not NaN
        print ('')
        print (' ---------------------------------')
        print (' Filling the column of the index with -1 = initialization')
        index_tuple = -1
        for index, row in var_df.iterrows():
            if np.isfinite(var_df[List_name_col['x']][index]) and np.isfinite(var_df[List_name_col['y']][index]) and np.isfinite(var_df[List_name_col['color']][index]):
                index_tuple += 1 # increase -> index needed to correlate
                #points_in_log_log.append (( index_tuple, var_df[List_name_col['x']][index], var_df[List_name_col['y']][index], var_df[List_name_col['color']][index], -1, -1 ))
                points_in_log_log.append([index_tuple, var_df[List_name_col['x']][index], var_df[List_name_col['y']][index],
                                          var_df[List_name_col['color']][index], -1, -1,
                                          var_df['client_id'][index], int(var_df['merchant_id'][index]), var_df['is_tier_1'][index] ])
                column_color.append(var_df[List_name_col['color']][index])
                column_y.append(var_df[List_name_col['y']][index])

        #print ('column_color')
        #print (column_color)
        #quit()

        # check if ok
        print (' Check: ')
        #for i in range(0, len(points_in_log_log)): # on number of elements = rows
        for i in range(0, 5):  # on number of elements = rows
            print (points_in_log_log[i])
        #quit()
        #print ('ARRIVATO QUI A SISTEMAE LISTS OF LISTS')


        # ------------------------------------------------------------------------------------------------------
        print ('')
        print (' ---------------------------------')
        print (' Computing the Sectors according to color = reach')

        # TODO: I want 4 groups containing
        # I compute the max and min of the reach, then I divide in a given number of sectors
        num_sectors_color = 4
        #num_sectors_y = 4

        print ('Using ' + str(num_sectors_color) + ' Different classes in color')

        #print ('Classify t')
        #points_in_log_log
        #column_color = zip(*points_in_log_log)[3]
        print (' Check column_color: '),
        print (column_color)
        #print (column_y)
        #print(np.percentile(column_color, 50 / num_sectors_color))  # reports the 50th percentile = median

        upper_bounds_sectors_color = [] # 0, 1, 2, ... num_sectors_color-1
        size_one_interval_color = (max(column_color)-min(column_color)) / float(num_sectors_color)
        print ('size_one_interval_color = ' + str(size_one_interval_color))
        lower_lim_color_now = min(column_color)
        print ('Color min, max = ' + str(min(column_color)) + ' ,  ' + str(max(column_color)))

        for i in range(0,num_sectors_color):
            #upper_bounds_sectors_color.append(np.percentile(column_color, 100 * (i+1) / num_sectors_color))
            upper_bounds_sectors_color.append(lower_lim_color_now + size_one_interval_color)
            print ('upper bound' + str(i+1) + ' = ' + str(upper_bounds_sectors_color[i]) + '; '),
            lower_lim_color_now = upper_bounds_sectors_color[i]
        print ('')

        #print (len(points_in_log_log))

        #quit()

        # now fill the entry of column_color (entry = sector+1)
        for entry in range(0,len(points_in_log_log)):
            for sector_color in range(0,len(upper_bounds_sectors_color)):
                # start from the minimum and then grow => ok no overlap
                if points_in_log_log[entry][4] == -1: # take only the ones not filled yet
                    if points_in_log_log[entry][3] <= upper_bounds_sectors_color[sector_color]:
                        points_in_log_log[entry][4] = sector_color + 1 # sector_color starts from 0, but I want index from 1

        print (' Check: index, x, y, color, index_color, index_y, client_id, merchant_id, is_tier_1 ')
        for entry in range(0,5):
            print (points_in_log_log[entry])

        # ------------------------------------------------------------------------------------------------------
        print ('')
        print (' ---------------------------------')
        print (' Computing the Sectors according to y = log_sum_revenue_euro')

        # TODO: I want 4 groups containing
        # I compute the max and min of the reach, then I divide in a given number of sectors
        num_sectors_y = 4

        print ('Using ' + str(num_sectors_y) + ' Different classes in y')

        print (' Check column_y: '),
        print (column_y)


        upper_bounds_sectors_y = [] # 0, 1, 2, ... num_sectors_color-1
        size_one_interval_y = (max(column_y)-min(column_y)) / float(num_sectors_y)
        print ('size_one_interval_y = ' + str(size_one_interval_y))
        lower_lim_y_now = min(column_y)
        print ('y min, max = ' + str(min(column_y)) + ' ,  ' + str(max(column_y)))

        for i in range(0,num_sectors_y):
            #upper_bounds_sectors_color.append(np.percentile(column_color, 100 * (i+1) / num_sectors_color))
            upper_bounds_sectors_y.append(lower_lim_y_now + size_one_interval_y)
            print ('upper bound' + str(i+1) + ' = ' + str(upper_bounds_sectors_y[i]) + '; '),
            lower_lim_y_now = upper_bounds_sectors_y[i]
        print ('')

        # now fill the entry of column_y (entry = sector+1)
        for entry in range(0,len(points_in_log_log)):
            for sector_y in range(0,len(upper_bounds_sectors_y)):
                # start from the minimum and then grow => ok no overlap
                if points_in_log_log[entry][5] == -1: # take only the ones not filled yet
                    if points_in_log_log[entry][2] <= upper_bounds_sectors_y[sector_y]:
                        points_in_log_log[entry][5] = sector_y + 1 # sector_color starts from 0, but I want index from 1

        print (' Check: index, x, y, color, index_color, index_y, client_id, merchant_id, is_tier_1 ')
        for entry in range(0, 5):
            print (points_in_log_log[entry])



        #quit()

        # ------------------------------------------------------------------------------------------------------
        print ('')
        print (' ---------------------------------')
        print (' Printing the csv file with the index ')
        #folder_RES = ''

        name_file_1 = folder_RES + '/RES.Date-' + day + '.period-' + str(period) + '.clients_id-idxColorReach-idxYRev.01-Unsorted.csv'
        f = open(name_file_1, 'w')
        # f = open('Test-clients_id-inside.txt', 'w')
        f.write('# Data from Date = ')
        f.write(day)
        f.write(', period = ')
        f.write(str(period))
        f.write('\n')
        f.write('# classified by color ('+List_name_col['color'] + ' - ' + str(num_sectors_color)+ ' sectors) and by y ('
                + List_name_col['y']+' - ' + str(num_sectors_y)+' sectors)\n' )
        f.write('# Features =  client_id,  x,     y,      color,    index_color,      index_y \n')
        f.write('client_id,' + List_name_col['x'] + ',' + List_name_col['y'] +
                ',' + List_name_col['color'] + ',index_' + List_name_col['color'] + ',index_' + List_name_col['y'] +',merchant_id,is_tier_1\n')

        #points_in_log_log = tuple: (index, x, y, color, index_color, index_y, client_id), declare list

        for entry in range(0,len(points_in_log_log)):
            f.write(str(points_in_log_log[entry][6]) + ',')
            f.write(str(points_in_log_log[entry][1]) + ',')
            f.write(str(points_in_log_log[entry][2]) + ',')
            f.write(str(points_in_log_log[entry][3]) + ',')
            f.write(str(points_in_log_log[entry][4]) + ',')
            f.write(str(points_in_log_log[entry][5]) + ',')
            f.write(str(points_in_log_log[entry][7]) + ',') # merchant_id
            f.write(str(points_in_log_log[entry][8]))  # is_tier_1
            f.write('\n')

        f.close()


        print (' ==> FILE = ' + name_file_1)


        # ------------------------------------------------------------------------------------------------------
        print ('')
        print (' ---------------------------------')
        print (' Printing the csv file with client_id, date, period, index_reach, index_revenues, merchant_id, is_tier_1 ')

        name_file_2 = folder_RES + '/RES.Date-' + day + '.period-' + str(period) + '.clients_id-idxColorReach-idxYRev.02-with-Date.csv'
        f2 = open(name_file_2, 'w')
        f2.write('# RETAIL - GB - groups of reach = ' + str(num_sectors_color) + ' - groups of log_revenue = ' + str(num_sectors_y) + '\n')
        f2.write('client_id,date,period,index_'+ List_name_col['color'] + ',index_' +  List_name_col['y'] +',merchant_id,is_tier_1\n')
        #f.write(day)
        #f.write(', period = ')
        #f.write(str(period))
        #f.write('\n')
        #f.write('# classified by color (' + List_name_col['color'] + ' - ' + str(
        #    num_sectors_color) + ' sectors) and by y (' + List_name_col['y'] + ' - ' + str(
        #    num_sectors_y) + ' sectors)\n')
        #f.write('# Features =  client_id,  x,     y,      color,    index_color,      index_y \n')
        #f.write('# Features =  client_id, ' + List_name_col['x'] + ', ' + List_name_col['y'] + ', ' + List_name_col[
        #    'color'] + ', index_color,   index_y \n ')

        # points_in_log_log = tuple: (index, x, y, color, index_color, index_y, client_id), declare list

        for entry in range(0, len(points_in_log_log)):
            f2.write(str(points_in_log_log[entry][6]) + ',') # client_id
            f2.write(day + ',') #
            f2.write(str(period) + ',')
            f2.write(str(points_in_log_log[entry][4]) + ',')
            f2.write(str(points_in_log_log[entry][5]) + ',')
            f2.write(str(points_in_log_log[entry][7]) + ',') # merchant_id
            f2.write(str(points_in_log_log[entry][8]))  # is_tier_1
            f2.write('\n')
        f2.close()
        print (' ==> FILE = ' + name_file_2)
        #quit()


        # ------------------------------------------------------------------------------------------------------
        print ('')
        print (' ----------------------------')
        print (' Writing number of Clients in each subgroup')

        #print ('Determining number of points in each group and writing file')
        # first index = row = color = color, second index = column = log_revenues = y
        #num_client_in_group = [[0] * num_col for x in xrange(num_row)]
        num_client_in_group = [[0]*num_sectors_y for x in xrange(num_sectors_color)]
        print (' Check initialization to 0')
        print (num_client_in_group)

        # count num clients in groups that is tier 1
        num_client_in_group_tier1 = [[0] * num_sectors_y for x in xrange(num_sectors_color)]
        print (' Check initialization to 0')
        print (num_client_in_group_tier1)

        name_file_3 = folder_RES + '/RES.Date-'+day+'.period-'+str(period)+'.clients_id-idxColorReach-idxYRev.03-Count.csv'
        f3 = open(name_file_3, 'w')
        # f = open('Test-clients_id-inside.txt', 'w')
        f3.write('# Data from Date = ')
        f3.write(day)
        f3.write(', period = ')
        f3.write(str(period))
        f3.write('\n')
        f3.write('# classified by color (' + List_name_col['color'] + ' - ' + str(
            num_sectors_color) + ' sectors) and by y (' + List_name_col['y'] + ' - ' + str(
            num_sectors_y) + ' sectors)\n')
        f3.write('# Features =  client_id,  x,     y,      color,    index_color,      index_y\n')
        f3.write('#client_id,' + List_name_col['x'] + ',' + List_name_col['y'] + ',' + List_name_col[
            'color'] + ',index_'+List_name_col['color']+',index_'+List_name_col['y']+',merchant_id,is_tier_1\n ')

        # points_in_log_log = tuple: (index, x, y, color, index_color, index_y, client_id)
        # points_in_log_log = (index, x, y, color, index_color, index_y, client_id, merchant_id, is_tier_1), declare list
        for entry in range(0, len(points_in_log_log)):
            num_client_in_group[points_in_log_log[entry][4]-1][points_in_log_log[entry][5]-1] += 1
            num_client_in_group_tier1[points_in_log_log[entry][4]-1][points_in_log_log[entry][5]-1] += points_in_log_log[entry][8]
        print (num_client_in_group)

        f3.write('# max of sector color = "')
        f3.write(str(upper_bounds_sectors_color))
        f3.write('"')
        f3.write('\n')
        f3.write('# max of sector y = "')
        f3.write(str(upper_bounds_sectors_y))
        f3.write('"')
        f3.write('\n')

        f3.write('index_reach,index_revenues,number_clients_in_sector,number_clients_tier1,max_sector_reach,max_sector_revenues\n')
        print('index_reach,index_revenues,number_clients_in_sector,number_clients_tier1,max_sector_reach,max_sector_revenues')
        for i_color in range(0, num_sectors_color):
            for i_y in range(0, num_sectors_y):
                print (i_color+1, i_y+1, num_client_in_group[i_color][i_y])
                f3.write(str(i_color+1) + ', ' + str(i_y+1) + ', ' + str(num_client_in_group[i_color][i_y]) +
                         ',' + str(num_client_in_group_tier1[i_color][i_y]) + ','
                         + str(upper_bounds_sectors_color[i_color]) + ','
                         + str(upper_bounds_sectors_y[i_y]) + '\n')
        f3.close()

        print ('   ==> FILE = ' + name_file_3)

        print ('FINISHED THIS VERSION')

        #quit() # does only 1 date

        # cycles
        start = start - timedelta(days=period)
        day = start.strftime('%Y-%m-%d')

        # clean the df
        #del clean_clust_df


#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()