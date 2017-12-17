# WHAT
# This is to compute the clients with the specified variables in a given interval with log if needed
# convention
#  ----
# AUTHOR = Daniele Scopece, 2017-08-29 (adapted)
#
# VERSIONS:
# v01.03 = post Aliki
# v01.04 = made with tuple to sort
# v01.05 = understood how to convert the log accurately: the average change (shrink of the scale)
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
#import matplotlib.lines as mlines # to draw lines
import matplotlib.dates as mdates # for dates as axis
print('matplotlib: {}'.format( matplotlib.__version__))
#from mpl_toolkits.mplot3d import Axes3D

import sklearn
print ('sklearn: {}'.format(sklearn.__version__))
#from sklearn import manifold
from sklearn.cluster import KMeans, AgglomerativeClustering as AggC, DBSCAN
#from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import math

import seaborn as sns
#--------------


## PARAMETERS TO TWEAK ===============================================
#----- Data input

from datetime import datetime
from datetime import timedelta

# to create directory
import os


#---------------------------------------------
# function needed to compute the average in log scale, given the average and error bar in the real space
# average_log = (log(ave+err) + log(ave-err)) / 2.0
def average_log(average_normal, error_normal):
    computing_ave_log = np.log10(average_normal+error_normal) +  np.log10(average_normal-error_normal)
    computing_ave_log = computing_ave_log / 2.0
    return computing_ave_log

#---------------------------------------------------------------------------
# function to compute the error bar in the log scale, given the average and error in the real scale
def error_log(average_normal, error_normal):
    ave_log = average_log(average_normal, error_normal)
    error_log_here = abs(ave_log - np.log10(average_normal-error_normal))
    return error_log_here

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
    plt.xlim([1e3, 1e8])
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

    print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (' Searching the clients in an interval')

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


    print ('')
    print ('------------------------------------------------------------')
    print (' Features (potentially) considered: ')

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
           'client_id'
           ]
    print (POI)

    #start_date = "2017-08-01"  # the closest to ours
    #stop_date = "2017-06-20"  # the furthest in the past
    #stop_date = "2016-01-01"  # the furthest in the past
    #stop_date = "2017-06-01"  # the furthest in the past
    #period = 30  # can be 1 or 7 or 30 (in days)


    print ('')
    print ('------------------------------------------------------------')
    print (' Folder of the Outputs : ')
    folder_RES = '02.02.RES-Plots-Interval'
    print (folder_RES)
    # creating the folder if it does not exist
    if not os.path.exists(folder_RES):
        os.makedirs(folder_RES)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    stop = datetime.strptime(stop_date, "%Y-%m-%d")
    day_start = start.strftime('%Y-%m-%d')
    day_stop = stop.strftime('%Y-%m-%d')

    # ++++++++++++++++++++ START wanted values in human readable
    print ('')
    print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('Reading the wanted features and tolerance')
    # insert the value of the variables here and the code transforms them into log, when needed
    POI_input = ['audience',
                 'sum_revenue_euro',
                 'reach',
                 'CPC',
                 'ratio_displays_acc_over_pot',
                 # 'log_exposed_users',
                 'potential_displays',
                 # 'log_sum_displays',
                 'sum_clicks']

    # Put np.nan if the variable is not used for the selection
    # input in human readable = no log
    Wanted_center_input = {'audience': 5e5,
                           'sum_revenue_euro': 5e2,
                           'reach': np.nan,
                           'CPC': np.nan,
                           'ratio_displays_acc_over_pot': np.nan,
                           # 'log_exposed_users' : np.nan,
                           'potential_displays': np.nan,
                           # 'log_sum_displays' : np.nan,
                           'sum_clicks': np.nan
                           }
    # error bars +/-
    # NOTE: put positive values !!!
    Wanted_radius_input = {'audience': 3e5,
                           'sum_revenue_euro': 3e2,
                           'reach': np.nan,
                           'CPC': np.nan,
                           'ratio_displays_acc_over_pot': np.nan,
                           # 'exposed_users' : xx,
                           'potential_displays': np.nan,
                           # 'sum_displays': XX,
                           'sum_clicks': np.nan
                           }

    for i in POI_input:  # range(0, len(Wanted_center_input)):
        # print (i)
        print (str(i) + ': ' + str(Wanted_center_input[i]) + ' +/- ' + str(Wanted_radius_input[i]))

    # quit()


    print ('')
    print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('Computing the log of the inserted quantities after converted and rescaled')
    # -------- start wanted values converting in log
    # list of the entries of var_df
    POI_here = ['log_audience',
                'log_sum_revenue_euro',
                'reach',
                'log_CPC',
                'log_ratio_displays_acc_over_pot',
                # 'log_exposed_users',
                'log_potential_displays',
                # 'log_sum_displays',
                'log_sum_clicks']
    # add client_id ?

    # for the correct error (due to conversion to log) I need to compute the value-error and value+error
    # real
    # compute max - min = (ave+error) - (ave-error)
    # the distance in the real numbers can be different when log is applied
    # --> average in log scale = (log(ave+error) + log(ave-error))/2 = this is the average point in log scale
    # --> error in log scale = abs(log_center - np.log10(x_ave-x_err))
    # Wanted_max_min_input



    # list of the values in the center of the interval: np.nan = not used for the selection
    # made via dictionary

    # tests conversions log
    # print ('Test conversions')
    # print (Wanted_center_input['audience'], Wanted_radius_input['audience'])
    # print (np.log10(Wanted_center_input['audience']-Wanted_radius_input['audience']), np.log10(Wanted_center_input['audience']+Wanted_radius_input['audience']))
    # print (average_log(Wanted_center_input['audience'], Wanted_radius_input['audience']))
    # print (error_log(Wanted_center_input['audience'], Wanted_radius_input['audience']))
    # quit()


    Wanted_center = {'log_audience': average_log(Wanted_center_input['audience'], Wanted_radius_input['audience']),
                     # np.log10(Wanted_center_input['audience']),
                     'log_sum_revenue_euro': average_log(Wanted_center_input['sum_revenue_euro'],
                                                         Wanted_radius_input['sum_revenue_euro']),
                     'reach': Wanted_center_input['reach'],
                     'log_CPC': average_log(Wanted_center_input['CPC'], Wanted_radius_input['CPC']),
                     'log_ratio_displays_acc_over_pot': average_log(Wanted_center_input['ratio_displays_acc_over_pot'],
                                                                    Wanted_radius_input['ratio_displays_acc_over_pot']),
                     # 'log_exposed_users' : average_log(Wanted_center_input['exposed_users'], Wanted_radius_input['exposed_users']),
                     'log_potential_displays': average_log(Wanted_center_input['potential_displays'],
                                                           Wanted_radius_input['potential_displays']),
                     # 'log_sum_displays' : average_log(Wanted_center_input['sum_display'], Wanted_radius_input['sum_display']),
                     'log_sum_clicks': average_log(Wanted_center_input['sum_clicks'], Wanted_radius_input['sum_clicks'])
                     }

    # print ('Wanted_center_input :'),
    # print (Wanted_center_input)
    # print ('Wanted_center : '),
    # print (Wanted_center)

    # quit()

    # list of the values in the radius of the interval: np.nan = not used for the selection
    # NOTE: put positive values !!!
    Wanted_radius = {'log_audience': error_log(Wanted_center_input['audience'], Wanted_radius_input['audience']),
                     # np.log10(Wanted_center_input['audience']),
                     'log_sum_revenue_euro': error_log(Wanted_center_input['sum_revenue_euro'],
                                                       Wanted_radius_input['sum_revenue_euro']),
                     'reach': Wanted_radius_input['reach'],
                     'log_CPC': error_log(Wanted_center_input['CPC'], Wanted_radius_input['CPC']),
                     'log_ratio_displays_acc_over_pot': error_log(Wanted_center_input['ratio_displays_acc_over_pot'],
                                                                  Wanted_radius_input['ratio_displays_acc_over_pot']),
                     # 'log_exposed_users' : error_log(Wanted_center_input['exposed_users'], Wanted_radius_input['exposed_users']),
                     'log_potential_displays': error_log(Wanted_center_input['potential_displays'],
                                                         Wanted_radius_input['potential_displays']),
                     # 'log_sum_displays' : error_log(Wanted_center_input['sum_display'], Wanted_radius_input['sum_display']),
                     'log_sum_clicks': error_log(Wanted_center_input['sum_clicks'], Wanted_radius_input['sum_clicks'])
                     }

    for i in POI_here:
        print (str(i) + ': ' + str(Wanted_center[i]) + ' +/- ' + str(Wanted_radius[i]))

    # print ('Wanted_radius_input :'),
    # print (Wanted_radius_input)
    # print ('Wanted_radius : '),
    # print (Wanted_radius)
    # print (Wanted_center)
    # print (Wanted_radius)
    # ------------ end wanted values


    print ('')
    print ('------------------------------------------------------------')
    print (' Opening the file for the date, number of clients, list of clients: ')
    name_file_resume = folder_RES + '/RESUME-Clients_inside.dates.' + str(day_start) + '.' + str(day_stop) + '.period-'+str(period) + '.csv'
    print (name_file_resume)
    f_resume = open(name_file_resume, 'w')

    f_resume.write('# Data from Date = ')
    f_resume.write(day_start)
    f_resume.write(' and ')
    f_resume.write(day_stop)
    f_resume.write(', period = ')
    f_resume.write(str(period))
    f_resume.write('\n')
    f_resume.write('# POI = X')
    for column_write in POI_here:
        f_resume.write(',')
        f_resume.write(str(column_write))
    f_resume.write('\n')
    f_resume.write('# Wanted_center = X')
    for column_write in POI_here:
        f_resume.write(',')
        f_resume.write(str(Wanted_center[column_write]))
    f_resume.write('\n')
    f_resume.write('# Wanted_radius = X')
    for column_write in POI_here:
        f_resume.write(',')
        f_resume.write(str(Wanted_radius[column_write]))
    f_resume.write('\n')

    f_resume.write('date,num_clients_inside,list_client_sorted_by_log,distance_in_log\n')

    # initialize for the resume
    resume_day = []
    resume_numClient = []

    #quit()

    print ('')
    print ('------------------------------------------------------------')
    print (' Starting the Cycles: ')

    day = start.strftime('%Y-%m-%d')

    folder_csv = '01.01.RES-CSV-TABLES'
    csv_file_type = folder_csv + '/Data.'  # + day + .csv





    while start > stop:
        # print (day)  # start.strftime('%Y-%m-%d'))
        #tbl = 'Data.' + day  # name of the table to be written

        MYFILE = csv_file_type + day + '.csv'  # 'TABLE_v02.02_No-duplicate-funnel.csv'
        print ('')
        print ('/////////////////////////////////////////////////////////////////////////////////////////////////////')
        print ('Table now (MYFILE) = ' + MYFILE)

        # initializing the list for the resume
        list_inside_sorted = []  # initialized, sorted by log distance
        list_inside_sorted_logdistance = []
        print (' check list_inside_sorted: must be blank: '),
        print (list_inside_sorted)

        print ('')
        print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Reading the dataframe')

        df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True, thousands=',')

        #if doHierarchy:  # !!hierarchy memory intensive O(n^2), 30k rows is slow, with no limit it crashes
        #    df = pd.read_csv(MYFILE, na_values=['None'], nrows=20000, skip_blank_lines=True)
        #    # df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True)
        #else:
        #    df = pd.read_csv(MYFILE, na_values=['None'], skip_blank_lines=True, thousands=',')


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


        print ('')
        print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('Computing the log of the dataframe for the columns wanted only (var_df)')
        # --------------------- start compute log
        print('Starting compute log (Please IGNORE the warnings of division by zero, we do not consider them later):')
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

        print var_df.head()
        # --------------------- end compute log




        #quit()
        # --------------------- START plot log audience vs log CPC as done here
        print ('')
        print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('Visualize initial data on a log-log plot')
        # plot 2d: # 5: Audience vs CPC & color = reach
        x_lab = POI[0]
        x_now = clean_clust_df[POI[0]]
        x_lim_now = [1e1, 1e9]

        y_lab = POI[1]
        y_now = clean_clust_df[POI[1]]
        y_lim_now = [1e0, 1e6]

        color_lab = POI[2]
        color_now = clean_clust_df[POI[2]]

        title_now = 'Clients: End Date ' + day + ' & Period of ' + str(period) + ' days (color=' + color_lab + ')'

        #name_fig_now = folder_RES + '/Fig.LogLog.' + y_lab + '-vs-' + x_lab + '-color-' + color_lab + '.' + day + '-period-' + str(
        #    period) + '.png'

        name_fig_now = folder_RES + '/Fig-' + str(day) + '.period-' + str(period) + '.' + y_lab + '-vs-' + x_lab + '-color-' + color_lab + '.png'


        # print (title_now, name_fig_now)
        plot_figure_log_log(title_now, name_fig_now, x_lab, x_now, x_lim_now, y_lab, y_now, y_lim_now, color_lab,
                            color_now)

        #quit()

        # fig5 = plt.figure()
        # plt.title('Clients: End Date ' + day + ' & Period of ' + str(period) + ' days (color=Reach=Exposed/Audience)')
        # ax5 = fig5.add_subplot(1, 1, 1)
        # x = var_df[POI_here[0]]
        # plt.xlabel(POI_here[0])
        # #plt.xlim([-2, 2])
        # #ax5.set_xscale('log')  # log scale
        # # y = clean_clust_df[POI[4]]*100
        # # plt.ylabel(POI[4] + '*100')
        # # log => no x 100
        # y = var_df[POI_here[1]]
        # plt.ylabel(POI_here[1])
        # # plt.ylim([-2, 40])
        # #plt.ylim([1, 9])
        # #ax5.set_yscale('log')  # log scale
        # # plt.colorbar(ax.imshow(image, interpolation='nearest'))
        # plt.scatter(x, y, c=var_df[POI_here[2]], cmap='rainbow', vmin=0.0, vmax=1.0)
        # plt.colorbar()
        # plt.grid(True)
        # # cbar = plt.colorbar()
        # # cbar.set_label('Reach', rotation=270)
        # # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
        # # cmap = sns.diverging_palette(5, 250, as_cmap=True)
        # name_fig5 = folder_RES + '/Fig.New-Revenue-vs-Audience-colorReach.' + day + '-period-' + str(period) + '.png'
        # plt.savefig(name_fig5, format='png')
        # print (' ===> figure 5 = ' + name_fig5)
        # #plt.show() # caution: it stops the flow of the program
        # plt.draw()  # draws without stopping the flow of the program
        # plt.clf()  # clear figure
        # plt.close()  # close the figure window and continue



        #------------ START to fill is_inside columns
        print ('')
        print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Making the new columns for inside the interval -> 0 (outside), 1 (inside)')
        for col in POI_here:
            name_col = 'is_inside_' + col
            print ('--------- Column checked now = ' + name_col + ', Wanted_center = ' + str(Wanted_center[col]))
            #var_df[name_col] = np.nan
            # fill the columns
            #print ('Center Wanted = ' + Wanted_center[col])
            if math.isnan(Wanted_center[col]):
            #if Wanted_center[col] == np.nan:
                # if the value is NaN --> not importnat --> all are inside
                print ('Column Is NaN -> setting is_inside to 1')
                var_df[name_col] = 1
            else:
                # initialize to create the column
                var_df[name_col] = ""
                # here if not NaN -> compute the distance and put 0 or 1
                print('Column is Not NaN -> Computing the distance and filling with 0 or 1: PLEASE WAIT and IGNORE the warnings')
                # cycle on all rows and compute the distance
                for index, row in var_df.iterrows():
                    #print ('index = ' + str(index) + ', row[' + col +'] = ' + str(row[col]))
                    #print (row)
                    #print (row[col])
                    #print (Wanted_center[col])
                    distance = abs(float(row[col]) - float(Wanted_center[col]))
                    #print (distance)
                    if distance <= float(Wanted_radius[col]):
                        # the point is inside in this column
                        #print ('index = ' + str(index) + ', row[' + col +'] = ' + str(row[col]) + ': The distance is ' + str(distance) + ' <= ' + str(Wanted_radius[col]) + ' => set to 1')
                        var_df[name_col][index] = 1
                    else:
                        # the point is outside in this column
                        #print ('  The distance is ' + str(distance) + ' > ' + str(Wanted_radius[col]) + ' => set to 0')
                        var_df[name_col][index] = 0

        print (var_df.head())

        #-------------- now make a list of client_id that have all is_inside = 1 & count them
        # tuple that contains: index_in_var_df, client_id, distance from center --> easier to sort
        # made as a list of tuples => I can append
        inside_overall_tuple = []

        #list_inside_overall = []
        #index_inside_overall = [] # needed to print later
        #distance_from_wanted_center = [] # distance in order to sort
        #num_inside_overall = 0
        print ('')
        print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('Making a list of the clients inside overall in the range given & writing to a file:')
        name_file_1 = folder_RES + '/RES-' + str(day) + '.period-' + str(period) + '.clients_id-inside.01-Unsorted.csv'
        f = open(name_file_1, 'w')
        #f = open('Test-clients_id-inside.txt', 'w')
        f.write('# Data from Date = ')
        f.write(day)
        f.write(', period = ')
        f.write(str(period))
        f.write('\n')
        f.write('# Wanted_center = X')
        for column_write in POI_here:
            f.write(', ')
            f.write(str(Wanted_center[column_write]))
        f.write('\n')
        f.write('# Wanted_radius = X')
        for column_write in POI_here:
            f.write(', ')
            f.write(str(Wanted_radius[column_write]))
        f.write('\n')
        f.write('client_id')
        for column_write in POI_here:
            f.write(',')
            f.write(column_write)
        f.write(',')
        f.write('log_distance_from_wanted_center')
        f.write('\n')


        l_check = False

        # cycles on the rows => find the atoms inside and fill the vector index_inside_overall & distance
        for index, row in var_df.iterrows():
            if l_check: print ('------------------ index = ' + str(index)), # + ', row[' + str(col) +'] = ' + str(row[col]))
            # multiple all inside: if only 1 is 0 --> excluded
            inside_multiplied = 1
            #print ('inside_multiplied Pre = ' + str(inside_multiplied))
            for col in POI_here:
                name_col = 'is_inside_' + col
                inside_multiplied *= var_df[name_col][index]
                if int(var_df[name_col][index]) == 0:
                    if l_check: print (' 0 is due to ' + name_col + ' = ' + str(var_df[col][index]) + ', '),
                #print (index, name_col, var_df[name_col][index], inside_multiplied)
            #print ('inside_multiplied Final = ', str(inside_multiplied))
            if int(inside_multiplied) == 1:
                #list_inside_overall.append(var_df['client_id'][index])
                #index_inside_overall.append(index)
                f.write(str(var_df['client_id'][index]))
                distance_here = 0.0 # initialization of the distance
                for col in POI_here:
                    f.write(', ')
                    f.write(str(var_df[col][index]))
                    #compute also distance if the Wanted_center is not np.nan
                    # distance computed in log scale
                    if np.isfinite(Wanted_center[col]):
                        if l_check: print (' Wanted_center['+col+'] = ' + str(Wanted_center[col])),
                        distance_here += (float(var_df[col][index]) - float(Wanted_center[col]))**2.0
                        if l_check: print ('Inside distance now = ' + str(distance_here)),
                if l_check: print ()
                # compute distance total and append to the vector
                #distance_here = math.sqrt(distance_here)
                #distance_from_wanted_center.append(distance_here)
                # save the tuple of the points inside
                inside_overall_tuple.append((index, var_df['client_id'][index], distance_here))
                f.write(', ')
                f.write(str(distance_here))
                f.write('\n')
            #print ('Elements inside so far = ' + str(len(list_inside_overall)))
            if l_check: print ('Elements inside so far = ' + str(len(inside_overall_tuple)))
        print (' CONCLUSION: this group contains ' + str(len(inside_overall_tuple)) + ' Elements')
        print (' ==> FILE = ' + name_file_1)
        f.close()

        # print to see
        #print ('inside_overall_tuple = ')
        #print (inside_overall_tuple)

        #print ('index_inside_overall = '), # index in the original list -> needed to print
        #print (index_inside_overall)
        #print ('list_inside_overall = '),
        #print (list_inside_overall)
        #print ('distance_from_wanted_center = '),
        #print (distance_from_wanted_center)


        #---------------------------------------------------------------------------------------------------
        print ('')
        print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('Sorting the values by distance and printing to a file')

        # tuple that contains: index_in_var_df, client_id, distance from center
        inside_overall_tuple_sorted = sorted(inside_overall_tuple, key=lambda distance: distance[2])
        #for i in range(0,len(inside_overall_tuple_sorted)):
        #    print (inside_overall_tuple_sorted[i])

        name_file_2 = folder_RES + '/RES-' + str(day) + '.period-' + str(period) + '.clients_id-inside.02-Sorted-w-LogDistance.csv'
        f_2 = open(name_file_2, 'w')
        f_2.write('# Data from Date = ')
        f_2.write(day)
        f_2.write(', period = ')
        f_2.write(str(period))
        f_2.write('\n')
        f_2.write('# Wanted_center = X, X')
        for column_write in POI_here:
            f_2.write(', ')
            f_2.write(str(Wanted_center[column_write]))
        f_2.write('\n')
        f_2.write('# Wanted_radius = X, X')
        for column_write in POI_here:
            f_2.write(', ')
            f_2.write(str(Wanted_radius[column_write]))
        f_2.write('\n')
        f_2.write('client_id,log_distance_from_wanted_center')
        for column_write in POI_here:
            f_2.write(',')
            f_2.write(column_write)
        f_2.write('\n')


        #to plot
        # for this test x2 = var_df['log_audience'][index]
        # for this test y2 = var_df['log_sum_revenue_euro']
        x2 = []
        y2 = []

        for i in range(0,len(inside_overall_tuple_sorted)):
            # select the index in var_df
            index_now = inside_overall_tuple_sorted[i][0]
            # the client_id_now
            client_id_now = inside_overall_tuple_sorted[i][1]
            f_2.write(str(client_id_now))

            # saving to the resume list
            list_inside_sorted.append(client_id_now)

            # distance_now
            distance_now = inside_overall_tuple_sorted[i][2]
            f_2.write(', ')
            f_2.write(str(distance_now))

            # append to the resume list
            list_inside_sorted_logdistance.append(distance_now)

            # write the properties
            for col in POI_here:
                f_2.write(', ')
                f_2.write(str(var_df[col][index_now]))
            f_2.write('\n') # end the line
            x2.append(var_df['log_audience'][index_now])
            y2.append(var_df['log_sum_revenue_euro'][index_now])
        f_2.close()

        print(' ==> File sorted by distance = ' + name_file_2)


        #--------------------------------------------------------------------------------------------------
        print ('')
        print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('Making the plot with the points taken to highlight:')
        # plot 2d: # 5: Audience vs CPC & color = reach
        fig10 = plt.figure()
        plt.title('Clients: End Date ' + day + ' & Period of ' + str(period) + ' days (color=Reach=Exposed/Audience)')
        ax10 = fig10.add_subplot(1, 1, 1)
        x1 = var_df[POI_here[0]]
        plt.xlabel(POI_here[0])
        # plt.xlim([-2, 2])
        # ax5.set_xscale('log')  # log scale
        # y = clean_clust_df[POI[4]]*100
        # plt.ylabel(POI[4] + '*100')
        # log => no x 100
        y1 = var_df[POI_here[1]]
        plt.ylabel(POI_here[1])

        # select the x and y -> x2, y2 -> made above: ATTENTION: for this excample only

        # multiple series
        ax10.scatter(x1, y1, c=var_df[POI_here[2]], cmap='rainbow', vmin=0.0, vmax=1.0, marker = "o", label = "All Points")
        ax10.scatter(x2, y2, c='black', marker="1", label="Selected")
        plt.legend(loc='upper left')

        # plt.ylim([-2, 40])
        # plt.ylim([1, 9])
        # ax5.set_yscale('log')  # log scale
        # plt.colorbar(ax.imshow(image, interpolation='nearest'))
        #plt.scatter(x, y, c=var_df[POI_here[2]], cmap='rainbow', vmin=0.0, vmax=1.0)
        #ax10.colorbar()
        plt.grid(True)
        # cbar = plt.colorbar()
        # cbar.set_label('Reach', rotation=270)
        # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
        # cmap = sns.diverging_palette(5, 250, as_cmap=True)
        name_fig10 = folder_RES + '/Fig-' + str(day) + '.period-' + str(period) + '.Revenue-vs-Audience-colorReach.InRange.png'
        plt.savefig(name_fig10, format='png')
        print (' ===> figure 10 = ' + name_fig10)
        # plt.show() # caution: it stops the flow of the program
        plt.draw()  # draws without stopping the flow of the program
        plt.clf()  # clear figure
        plt.close()  # close the figure window and continue

        # --------------------------------------------------------------------------------------------------
        print ('')
        print ('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print (' Adding line to the Resume file: ')
        f_resume.write(str(day))
        resume_day.append(day) # for the resume plot
        f_resume.write(',')
        f_resume.write(str(len(inside_overall_tuple)))
        resume_numClient.append(len(inside_overall_tuple))  # for the resume plot
        f_resume.write(',')
        f_resume.write('"')
        f_resume.write(str(list_inside_sorted))
        f_resume.write('"')
        f_resume.write(',')
        f_resume.write('"')
        f_resume.write(str(list_inside_sorted_logdistance))
        f_resume.write('"')
        f_resume.write('\n')

        #quit()

        # cycles
        start = start - timedelta(days=period)
        day = start.strftime('%Y-%m-%d')

        # clean the df
        del clean_clust_df
        del var_df

    #------------------------------------------------
    print ('')
    print ('------------------------------------------------------------')
    print (' Closing the file of the resume and plotting it')
    f_resume.close()
    print (' ==> FILE = ' + name_file_resume)

    name_fig_resume = folder_RES + '/RESUME-Clients_inside.dates.' + str(day_start) + '.' + str(day_stop) + '.period-'+str(period) + '.numClients-vs-date.png'

    fig_resume = plt.figure()
    plt.title('Number of Clients in the range specified: Period of ' + str(period) + ' days: [No NaN or Null]')
    ax1 = fig_resume.add_subplot(1, 1, 1)
    x = mdates.datestr2num(resume_day)
    plt.xlabel('Date')
    # plt.xlim([1e-2, 1e2])
    # ax1.set_xscale('log')  # log scale
    # y = clean_clust_df[POI[4]]*100
    # plt.ylabel(POI[4] + '*100')
    # log => no x 100
    y = resume_numClient
    plt.ylabel('Number of Clients inside the range in this period')
    # plt.ylim([-2, 40])
    # plt.ylim([1e1, 1e9])
    # ax5.set_yscale('log')  # log scale
    # plt.colorbar(ax.imshow(image, interpolation='nearest'))
    # plt.scatter(x, y)
    print ('x : '),
    print (x)
    print ('y : '),
    print (y)
    plt.plot(x, y, linestyle='-', marker='o', color='b')  # ,  fmt="bo", tz=None, xdate=True)
    xfmt = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(xfmt)
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.colorbar()
    plt.grid(True)
    # cbar = plt.colorbar()
    # cbar.set_label('Reach', rotation=270)
    # plt.scatter(x, y, c=clean_clust_df[POI[2]], cmap=plt.cm.bwr_r)
    # cmap = sns.diverging_palette(5, 250, as_cmap=True)
    #name_fig1 = 'Fig-P02.NumActiveClients-vs-Date.-period-' + str(period) + '.png'
    plt.savefig(name_fig_resume, format='png')
    print ('  ==> FIGURE Resume = ' + name_fig_resume)
    # plt.show() # caution: it stops the flow of the program
    plt.draw()  # draws without stopping the flow of the program
    plt.clf()  # clear figure
    plt.close()  # close the figure window and continue


#-----------------------------------------------------------------------
if __name__ == '__main__':
    main()