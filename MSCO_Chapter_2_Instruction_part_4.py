import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# # reg model from fig 2.15
    # df = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.4_figure_2.15', header=0, index_col=None, nrows=24, usecols=[23,24,25,26,27,28])

    # plt.ylabel('Widget Demand (000s)');
    # plt.axes().yaxis.grid(linestyle=':');
    # plt.xlabel('Month');
    # plt.xlim(0,25);
    # plt.title('Widget Demand Data');
    # plt.plot(df['Month'], df['Monthly Demand'], marker='D', color='red');
    # plt.plot(df['Month'], df['Initial Forecast'], marker='o', color='orange');
    # plt.plot(df['Month'], df['Updated Regression'], marker='o', color='green');
    # plt.plot(df['Month'], df['Updated Using Holt\'s'], marker='o', color='blue');
    # plt.legend();
    # plt.show();

# # 2.16 
    # # This ingests the optimized Holt's table - period, demand, and forecast
    # df_holts = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.4_figure_2.16', header=27, index_col=None, nrows=13, usecols=[0,1,4])

    # # This drops the extra row we have in that table in the Excel file
    # df_holts = df_holts[1:]

    # # This shifts the Ft+1 column down one, and renames it Ft, so the forecast lines up with the actual period it forecasts
    # df_holts['Ft+1'] = df_holts['Ft+1'].shift(1)
    # df_holts_map = {'Ft+1':'Ft'}
    # df_holts = df_holts.rename(columns=df_holts_map)

    # # This ingests the linear regression table - period, demand, and calculated forecast
    # df_linreg = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.4_figure_2.16', header=0, index_col=None, nrows=12, usecols=[7,8,9])

    # # This merges the two dataframes into one
    # df = pd.merge(df_holts, df_linreg, how='outer', on=['Period t','Dt'])

    # # This code is cosmetic. It renames the columns first to numbers 0-3 using df.columns, then creates and uses a dictionary to use df.rename() 
    # df.columns = ['0','1','2','3']
    # df_map = {'0':'Period','1':'Demand','2':'Holt\'s forecast','3':'Excel regression forecast'}
    # df = df.rename(columns=df_map)

    # # This code trains a regression model and creates a column with the models predictions, to show that it comes out the same as the Excel regression forecast
    # import statsmodels.api as sm 
    # X = sm.add_constant(df['Period'])
    # results = sm.OLS(df['Demand'], X).fit()
    # df['sm.OLS regression forecast'] = pd.DataFrame(results.predict(X))

    # plt.ylabel('Widget Demand (000s)');
    # plt.axes().yaxis.grid(linestyle=':');
    # plt.xlabel('Month');
    # plt.title('Widget Demand Data');
    # plt.plot(df['Period'], df['Demand'], marker='D', color='red');
    # plt.plot(df['Period'], df['Holt\'s forecast'], marker='.', color='blue');
    # plt.plot(df['Period'], df['Excel regression forecast'], marker='o', color='orange');
    # plt.plot(df['Period'], df['sm.OLS regression forecast'], linestyle='--', marker=None, color='black');
    # plt.legend();
    # plt.show();

# # 2.16 twice

    # # This ingests slightly more data than before, by including the St and Gt columns
    # df = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.4_figure_2.16', header=27, index_col=None, nrows=13, usecols=[0,1,2,3,4])

    # # This drops the same row, shifts demand, renames the columns, and resets the index as before
    # df = df[1:]
    # df['Ft+1'] = df['Ft+1'].shift(1)
    # df_map = {'Period t':'Period','Dt':'Demand','Ft+1':'Holt\'s forecast'}
    # df = df.rename(columns=df_map)
    # df = df.reset_index(drop=True)

    # # This grabs the important values from the extra columns we ingested - the new base and new growth - then drops the columns
    # holts_base = df['St'][11]
    # holts_growth = df['Gt'][11]
    # df = df.drop(columns=['St','Gt'])

    # # This creates a dataframe with new periods ranging from one above the previous max to twelve beyond that
    # new_periods = pd.DataFrame([i for i in range (max(df['Period'])+1, max(df['Period'])+1+12)],columns=['Period'])

    # # This defines a function that we will be using to create new predictions using Holt's model
    # def holtsPrediction(period):
    #     ypred = holts_base + holts_growth*(period-12)
    #     return ypred 

    # # This actual creates those new predictions
    # new_periods['Holt\'s forecast'] = [holtsPrediction(i) for i in range( min(new_periods['Period']), max(new_periods['Period'])+1 )]

    # # This merges the new data on the first dataframe we created
    # df = pd.merge(df, new_periods, how='outer', on=['Period','Holt\'s forecast'])

    # # This creates the OLS predictions for all periods using StatsModels - note how we train it only on periods we have data for
    # import statsmodels.api as sm 
    # X = sm.add_constant(df['Period'][0:12])
    # results = sm.OLS(df['Demand'][0:12], X).fit()
    # Xfull = sm.add_constant(df['Period'])
    # df['sm.OLS regression forecast'] = pd.DataFrame(results.predict(Xfull))

    # # This plots it as before 
    # plt.ylabel('Widget Demand (000s)');
    # plt.axes().yaxis.grid(linestyle=':');
    # plt.xlabel('Month');
    # plt.title('Widget Demand Data');
    # plt.plot(df['Period'], df['Demand'], marker='D', color='red');
    # plt.plot(df['Period'], df['Holt\'s forecast'], marker='.', color='blue');
    # plt.plot(df['Period'], df['sm.OLS regression forecast'], linestyle='--', marker=None, color='black');
    # plt.legend();
    # plt.show();

# # Plotting the new exercise I made up
    # # This ingests slightly more data than before, by including the St and Gt columns
    # df = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.4_figure_2.15_new_exercise', header=0, index_col=None, nrows=12, usecols=[0,1,2,6])

    # # This plots it as before 
    # plt.ylabel('Widget Demand (000s)');
    # plt.axes().yaxis.grid(linestyle=':');
    # plt.xlabel('Period');
    # plt.title('Widget Demand Data');
    # plt.plot(df['Period'], df['Demand'], marker='D', color='#B22222');
    # plt.plot(df['Period'], df['Holts Forecast'], marker='.', color='#0000FF');
    # plt.plot(df['Period'], df['LinReg Forecast'], linestyle='--', marker=None, color='#228B22');
    # plt.legend();
    # plt.show();

    # # This ingests slightly more data than before, by including the St and Gt columns
    # df2 = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.4_figure_2.15_new_exercise', header=33, index_col=None, nrows=13, usecols=[0,1,2,6])

    # plt.ylabel('Widget Demand (000s)');
    # plt.axes().yaxis.grid(linestyle=':');
    # plt.xlabel('Period');
    # plt.title('Widget Demand Data');
    # plt.plot(df2['Period'], df2['Demand'], marker='D', color='#B22222');
    # plt.plot(df2['Period'], df2['Holts Forecast'], marker='.', color='#6495ED');
    # plt.plot(df2['Period'], df2['LinReg Forecast'], linestyle='--', marker=None, color='#32CD32');
    # plt.legend();
    # plt.show();

    # plt.ylabel('Widget Demand (000s)');
    # plt.axes().yaxis.grid(linestyle=':');
    # plt.xlabel('Period');
    # plt.title('Widget Demand Data');
    # plt.plot(df2['Period'], df2['Demand'], marker='D', color='#B22222');
    # plt.plot(df['Period'], df['Holts Forecast'], marker='.', color='#0000FF');
    # plt.plot(df['Period'], df['LinReg Forecast'], linestyle='--', marker=None, color='#228B22');
    # plt.plot(df2['Period'], df2['Holts Forecast'], marker='.', color='#6495ED');
    # plt.plot(df2['Period'], df2['LinReg Forecast'], linestyle='--', marker=None, color='#32CD32');
    # plt.legend();
    # plt.show();

# # 
# This starts a timer
import time
start = time.time()

# # This sets up the logger
# import logging.handlers
# import os
# handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "D:/Keep/Learning/Supply Chain/Managing Supply Chain Operations/py/log/holtsoptimizer.log"))
# formatter = logging.Formatter(logging.BASIC_FORMAT)
# handler.setFormatter(formatter)
# root = logging.getLogger()
# root.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
# root.addHandler(handler)
 
# This ingests just the period and demand data, for the 13 periods
df = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.4_figure_2.15_new_exercise', header=33, index_col=None, nrows=13, usecols=[0,1])

# import statsmodels.api as sm 
# X = sm.add_constant(df['Period'])
# results = sm.OLS(df['Demand'], X).fit()
# df['sm.OLS regression forecast'] = pd.DataFrame(results.predict(X))

holts_base = (df['Demand'][0]+df['Demand'][1])/2
holts_growth = holts_base - df['Demand'][0]

f_call_count = 0

import sys

def holtsForecast(period, alpha, beta):
    # logging.debug((sys._getframe().f_code.co_name,": ",period))
    holts_forecast = holtsBase(period, alpha, beta) + holtsGrowth(period, alpha, beta)
    global f_call_count
    f_call_count += 1
    return holts_forecast

def holtsBase(period, alpha, beta):
    # logging.debug((sys._getframe().f_code.co_name,": ",period))
    global f_call_count
    if period == 1:
        base = holts_base
        f_call_count += 1
        return base
    else:
        base = alpha*df['Demand'][period-1]+(1-alpha)*holtsForecast(period-1, alpha, beta)
        f_call_count += 1
        return base

def holtsGrowth(period, alpha, beta):
    # logging.debug((sys._getframe().f_code.co_name,": ",period))
    global f_call_count
    if period == 1:
        growth = holts_growth
        f_call_count += 1
        return growth
    else:
        growth = beta*(holtsBase(period, alpha, beta)-holtsBase(period-1, alpha, beta))+(1-beta)*holtsGrowth(period-1, alpha, beta)
        f_call_count += 1
        return growth

def holtsMSE(x):
    # logging.debug((sys._getframe().f_code.co_name,": ",x))
    sse_val = 0
    sse_count = -1
    for i in range(1,len(df['Demand'])):
        sse_val = (holtsForecast(i, x[0], x[1])-df['Demand'][i])**2 + sse_val
        sse_count += 1
    mse_val = sse_val/sse_count
    global f_call_count
    f_call_count += 1
    return mse_val

test_alpha = 0.1
test_beta = 0.2
initial_guess = [test_alpha, test_beta]

mse_result = holtsMSE(initial_guess)
print("\n MSE: \n",mse_result)

# forecast_results = holtsForecast(2, test_alpha, test_beta)
# print("\n Forecast: \n",forecast_results)

# # This is the actual optimization
# from scipy.optimize import minimize
# result = minimize(holtsMSE, initial_guess, bounds=((0,1),(0,1)))
# print (result.x)

print("\n Function call count: \n",f_call_count)

#This ends the timer
end = time.time()
print("\n Seconds:\n",end - start)