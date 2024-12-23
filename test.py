import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def file_path_converter(directory):
    '''Accesses the folder directory and creates a dictionary 
    of the file path directories for all files in the folder, 
    where the key is the year and the value is the file path.
    '''
    file_list = os.listdir(directory)  # shows list of files in folder
    file_path_dict = {}  # initialize dictionary to store key and values 
    
    for file in file_list:  # go through every file in directory list 
        file_path = directory + "/" + file  # create file paths 
        # extract file name
        file_name = os.path.splitext(os.path.basename(file))[0]  
        file_path_dict[file_name] = file_path   
             
    return file_path_dict


def read_file(filename):
    '''Reads the CSV files and returns a dictionary for company data,
    and a separate DataFrame for announcement dates if present.
    '''    
    data = {}  # dictionary for storing company data
    acquisitions = None  # variable to store announcement dates 
    
    for keys, path in filename.items():  # Loop through each file's data
        # check if file is announcement dates
        if 'announcement_dates' in keys.lower():  
            acquisitions = pd.read_csv(path, index_col='Name')
            acquisitions['Date'] = pd.to_datetime(acquisitions['Date'])
        else:
            df = pd.read_csv(path, index_col='Date', 
                             usecols=['Date', 'Adj Close'])
            df.index = pd.to_datetime(df.index)  # convert to datetime
            df = df.sort_values(by='Date')
            df = df.dropna()  # drop NA value rows (dividends)
            data[keys] = df  # Add each company data to dictionary
    
    return data, acquisitions


def pct_change(filename):
    '''Adds % Change column for each dataframe for each company'''
    for keys, df in filename.items():  # loop through each company's data  
        # calculate % change in adj close for each day and storing in column
        df['% Change'] = round(df['Adj Close'].pct_change() * 100, 2)
        df['% Change'].iloc[0] = 0  # replacing NA in 1st rows with 0
        
    return filename


def prepare_data(data, acquisitions):
    '''Merges the company data with the announcement dates 
    and calculates days since the announcement for regression.
    '''
    # Ensure that announcement dates are properly aligned with company data
    announcement_dates = acquisitions.loc[data.keys(), 'Date']
    
    # Filter data for each company, keeping only data after the announcement date
    filtered_data = {
        company: df[df.index >= announcement_dates[company]] 
        for company, df in data.items()
        if company in acquisitions.index
    }

    # Add 'Days Since Announcement' column for each company
    for company, df in filtered_data.items():
        announcement_date = acquisitions.loc[company, 'Date']
        
        # Ensure both the index and announcement date are datetime objects
        df.index = pd.to_datetime(df.index)
        
        # Add 'Days Since Announcement' column: subtract the announcement date from each stock data date
        df['Days Since Announcement'] = (df.index - announcement_date).days

    # Merge the filtered data into one DataFrame
    merged_data = list(filtered_data.values())
    
    # Concatenate merged_data if not empty
    merged_df = pd.concat(merged_data) if len(merged_data) > 0 else pd.DataFrame()
    
    return merged_df, filtered_data



def perform_individual_regression(df_dict):
    '''Generates regression plots for individual companies.'''
      
    for company, df in df_dict.items():
        # flattening data to create 1-dimensional format
        # extracting independent and dependent variables
        x = df[['Days Since Announcement']].values.flatten()
        y = df['% Change'].values
        
        group = df['Group'].iloc[0]
        group_label = 'Smooth Acquisition' if group == 1 else 'TakeOver'
        
        
        # Plotting regression line 
        plt.figure(figsize=(10, 6)) 
        sns.regplot(x=x, y=y, color = "blue", 
                    line_kws={"color": "red"}, 
                    scatter_kws={"alpha": 0.5},
                    label='Stock Price Change (%)')
        
        result = stats.linregress(x, y) # getting statistics for regression 
        slope = result.slope
        intercept = result.intercept
        
        print(f"{company} Regression Coefficient: {slope:.4f}")
        print(f"{company} Intercept: {intercept:.4f}")
        
        
        plt.xlabel('Days Since Announcement')
        plt.ylabel('% Change')
        plt.title(f'Regression: % Change vs Days Since Announcement for {company}')
        plt.grid()
        plt.legend()
        plt.savefig(f"{company} {group_label} Plot.png")
        plt.show()
    

def sector_regression(df, title):
    '''Performing regression on a pool of companies'''
    
    # flattening data to create 1-dimensional format 
    # extracting independent and dependent variables
    x = df[['Days Since Announcement']].values.flatten() 
    y = df['% Change'].values

    # Plotting regression line 
    plt.figure(figsize=(10, 6)) 
    sns.regplot(x=x, y=y, color = "blue", 
                line_kws={"color": "red"},
                scatter_kws={"alpha": 0.5},
                label='Stock Price Change (%)')

    result = stats.linregress(x, y) # getting statistics for regression 
    slope = result.slope
    intercept = result.intercept
    
    print(f"{title} Regression Coefficient: {slope:.4f}")
    print(f"{title} Intercept: {intercept:.4f}")
    
    plt.xlabel('Days Since Announcement')
    plt.ylabel('% Change')
    plt.title(f'Regression: % Change vs Days Since Announcement ({title})')
    plt.legend()
    plt.grid()
    plt.savefig(f"{title} Plot.png")
    plt.show()

def combined_regression(df1, df2):
    ''' Performing regression combining both smooth acquisition 
    and hostile takeover company sectors'''
    
    # combining dataframes 
    combined = pd.concat([df1,df2])
    
    # extracting independent and dependent variables 
    x = combined[['Days Since Announcement', 'Group']].values
    y = combined['% Change'].values
    
    # performing linear regression 
    model = LinearRegression() 
    model.fit(x, y)
    y_pred = model.predict(x) # getting predicted y values 
    
    residuals = y - y_pred # calculating residuals
    
    coefficients = model.coef_
    intercept = model.intercept_

    print(f"Combined Regression Coefficients: {coefficients}")
    print(f"Combined Intercept: {intercept:.4f}")
    
    # Plotting regression line 
    plt.figure(figsize=(10, 6)) 
    

    sns.regplot(x=combined['Days Since Announcement'], y=residuals, 
                color='blue', 
                line_kws={"color": "red"}, 
                scatter_kws={"alpha": 0.5},
                label='Stock Price Change (%)')
    
    plt.xlabel('Days Since Announcement')
    plt.ylabel('Residuals (% Change)')
    plt.title(f'Regression: % Change vs Days Since Announcement Combined')
    plt.legend()
    plt.grid()
    plt.savefig("Combined Plot.png")
    plt.show()
    
def main():
    
    acquisitions_dir = 'company_data'
    hostile_takeovers_dir = 'hostile_takeover'

    # Data set for smooth acquisitions
    list_of_filepaths_acq = file_path_converter(acquisitions_dir)
    companies_data_aq, announcement_dates_aq = read_file(list_of_filepaths_acq)
    companies_data_aq = pct_change(companies_data_aq)

    # Data set for companies that experienced hostile takeovers
    list_of_paths_takeover = file_path_converter(hostile_takeovers_dir)
    companies_data_takeover, announcement_dates_takeover = read_file(list_of_paths_takeover)
    companies_data_takeover = pct_change(companies_data_takeover)

    # Prepare data for regression
    combined_data_aq, individual_data_aq = prepare_data(companies_data_aq, announcement_dates_aq)
    combined_data_takeover, individual_data_takeover = prepare_data(companies_data_takeover, announcement_dates_takeover)
    
    
    # Assigning binary values to perform regression Acq vs Takeovers
    for company, df in individual_data_aq.items():
        df['Group'] = 1 # adding Group column to each company with binary value 1

    for company, df in individual_data_takeover.items():
        df['Group'] = 0  # adding Group column to each company with binary value 0
    
    
    # Perform individual regressions
    print("Performing individual regressions for smooth acquisitions:")
    perform_individual_regression(individual_data_aq)

    print("Performing individual regressions for hostile takeovers:")
    perform_individual_regression(individual_data_takeover)

    # Perform regressions for each sectored companies 
    print("\nPerforming combined regression for smooth acquisitions:")
    sector_regression(combined_data_aq, 'Smooth Acquisitions')

    print("\nPerforming combined regression for hostile takeovers:")
    sector_regression(combined_data_takeover, 'Hostile Takeovers')
    
    # Assigning binary values to perform regression Acq vs Takeovers
    combined_data_aq['Group'] = 1
    combined_data_takeover['Group'] = 0     
    
    # Perform combined regressions
    print("\nPerforming Overall Combined Regression")
    combined_regression(combined_data_aq, combined_data_takeover)
    
    

if __name__ == "__main__":
    main()
