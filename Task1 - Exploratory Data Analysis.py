

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn import set_config
from scipy.stats import ttest_ind
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

from config import * 

# from config import DATE_FORMAT, TIME_PERIOD_LABELS, TIME_FORMAT, RENAME_COLUMNS, COLUMN_ARRANGEMENT, SORT_DATA
# from config import DROP_CLEAN_FLIGHTS, TRANSFORM_DATATYPE, DROP_TIME_COL


# #### Defining dummy dataframe before the function calling

# In[2]:


df =[]


# ## Data Collection and Data Preprocessing

# In[3]:


def concat_flights(df):
    
    # Read the csv files from the directory
    df_business = pd.read_csv('business.csv')
    df_economy = pd.read_csv('economy.csv')
    
    # Assigns the values based on the Journey Class
    df_business['class']='Business'
    df_economy['class']='Economy'
    
    # Concatenate the two datasets into one and store them in a dataframe
    df = pd.concat([df_business,df_economy], axis =0)
    
    # This shows the Dataframe rows after the concatenation
    df.head(8)
    return df


# In[4]:


def clean_flights(df):
    
    # Replacing the string to number of stops and assigning it to new column
    df['Number of Stops']= df['stop'].str.slice(start=0, stop=1).replace('n','0')
    
    # Changing the Date format and assigning it to new column
    df['Date of Journey']= df['date'].str.replace('-','/')
    
    # Replacing the ' , ' from the price string and assigning it to new column
    df['Ticket Price']= df['price'].str.replace(',','')
    
    # Concatenating the two columns and assigning it to new column
    df['Flight Code'] =  df['ch_code'] + "-" + df['num_code'].astype(str)
    
    # Dropping the old columns, 'DROP_CLEAN_FLIGHTS' from config file
    df = df.drop(DROP_CLEAN_FLIGHTS, axis = 1)
    
    return df


# #### Transformation Functions

# In[5]:


def transform_date(df):
    
    # Changing the datatype from string to Datetime, Format from 'DATE_FORMAT' from config file
    df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], format=DATE_FORMAT)
    
    return df



def transform_days_left(df):
    
    # Calculating the days left from the date of journey
    df['Days Left'] = (df['Date of Journey'] + timedelta(days=1)) - min(df['Date of Journey'])
    df['Days Left'] = df['Days Left'].astype(str).str.slice(start=0, stop=2)
    
    # Changing the datatypes, "TRANSFORM_DATATYPE" from config file
    df[TRANSFORM_DATATYPE] = df[TRANSFORM_DATATYPE].astype(int)
    return df



def transform_day_of_week(df):
    
    # Assigning Days Name to a new column based on Day of Journey
    df['Day of Week'] = df['Date of Journey'].dt.day_name()
    return df



def transform_departure_time(df):
    
    # Converting the departure time to hours and 
    # then assigning them based on 'TIME_PERIOD_LABELS' from config file
    
    dep_time = (pd.to_datetime(df['dep_time'], format=TIME_FORMAT).dt.hour % 24 + 4) // 4
    dep_time.replace(TIME_PERIOD_LABELS, inplace=True) 
    
    # Assigning dep_time column to New Column
    df['Departure Time'] = dep_time 
    return df



def transform_arrival_time(df):
    
    # Converting the arrival time to hours and 
    # then assigning them based on 'TIME_PERIOD_LABELS' from config file
    
    arr_time = (pd.to_datetime(df['arr_time'], format=TIME_FORMAT).dt.hour % 24 + 4) // 4
    arr_time.replace(TIME_PERIOD_LABELS, inplace=True) 
    
    # Assigning arr_time column to New Column
    df['Arrival Time'] = arr_time 
    return df



def transform_trip_duration(df):
    
    # Coverting the string from time_taken column to hour and minutes
    hour = df['time_taken'].str.split('h ', n=1, expand=True)
    df['hour'] = hour[0]
    df['minute'] = hour[1].str.replace('m', '')
    df['minute'] = np.where(df['minute'] == "", 0, df['minute'])
    
    # Calculating the duration in float values for better analysis in models
    df['Trip Duration'] = (df['hour'].astype(float) + (df['minute'].astype(float))/60).round(2)
    
    # Dropping the temporary time columns, 'DROP_TIME_COL' from config file
    df = df.drop(DROP_TIME_COL, axis=1)
    return df



def transform_rename_columns(df):
    
    # Renaming the columns, RENAME_COLUMNS from the config file
    df.rename(columns= RENAME_COLUMNS, inplace=True)
    
    #Arranging the columns in desired format, 'COLUMN_ARRANGEMENT' from config file
    df = df[COLUMN_ARRANGEMENT]
    return df



def transform_sort(df):
    
    #Sorting the data, SORT_DATA from config file
    df = df.sort_values(by= SORT_DATA)
    return df



def drop_airlines(df):
    
    # Dropping some rows in Airline, due to less number of available data
    df = df[df['Airline'] != 'StarAir']
    df = df[df['Airline'] != 'Trujet']
    return df


def transform_flights(df):
    
    # Calling all the functions and assigning them to original dataframe
    df = transform_date(df)
    df = transform_days_left(df)
    df = transform_day_of_week(df)
    df = transform_departure_time(df)
    df = transform_arrival_time(df)
    df = transform_trip_duration(df)
    df = transform_rename_columns(df)
    df = transform_sort(df)
    df = drop_airlines(df)
    
    return df


# #### Pipeline Creation to visualize the preprocessing effectively

# In[6]:


# Creating Pipeline to help in visualizing the tranformation in the columns

# First step deals with tansformation around date
transform_day = ColumnTransformer([
    ('transform_date', FunctionTransformer(transform_date), ['date_column']),
    ('transform_days_left', FunctionTransformer(transform_days_left), ['days_left_column']),
    ('transform_day_of_week', FunctionTransformer(transform_day_of_week), ['day_of_week_column'])
])

# Second steps deals with tansformation around time
transform_time = ColumnTransformer([
    ('transform_departure_time', FunctionTransformer(transform_departure_time), ['departure_time_column']),
    ('transform_arrival_time', FunctionTransformer(transform_arrival_time), ['arrival_time_column']),
    ('transform_trip_duration', FunctionTransformer(transform_trip_duration), ['trip_duration_column'])
])

# Third steps deals with tansformation around columns

transform_columns = ColumnTransformer([
    ('transform_rename_columns', FunctionTransformer(transform_rename_columns), None),
    ('transform_sort', FunctionTransformer(transform_sort), None)
])

# Preprocess Pipeline Creations and assigning all the steps
preprocess_pipeline = Pipeline([
    ('transform_day', transform_day),
    ('transform_time', transform_time),
    ('transform_columns', transform_columns),
])

# Visualizing the Pipeline
set_config(display="diagram")
preprocess_pipeline


# #### Pipeline function

# In[7]:


def preprocess_pipeline(df):
    
    #Preprocess function to call all the functions one after another
    df = concat_flights(df)
    df = clean_flights(df)
    df = transform_flights(df)
    df = drop_airlines(df)

    return df


# In[8]:


concat_flights(df)


# ### Calling the function for final Preprocessing

# In[9]:


get_ipython().run_cell_magic('time', '', 'preprocessed_df = preprocess_pipeline(df)\npreprocessed_df.head()')


# #### Exporting the final processed dataframe to CSV file in working directory

# In[10]:


preprocessed_df.to_csv('cleaned_flight_dataset.csv', index=False)


# #### Assigning the processed file to new dataframe

# In[11]:


df = preprocessed_df


# In[12]:


df.info()


# ##### No Null Values in the Dataset

# In[13]:


df.isnull().sum()


# In[14]:


df.describe()


# #### Defining png save function to save plots

# In[15]:


def save_plot_as_png(plot, title):
    # Replace spaces in the title with underscores and add the file extension
    filename = title.replace(' ', '_') + '.png'
    
    # Save the plot as a PNG file
    plot.savefig(filename, dpi=600, bbox_inches='tight')
    
    print(f"Plot saved as {filename}")


# ## Exploratory Data Analysis (EDA)

# #### Plotting the Categorical values and there counts

# In[16]:


# Plotting all the plots in the same line
fig, axes = plt.subplots(1, len(list1), figsize=(20, 5))

# Adjusting the space in between
fig.subplots_adjust(wspace=0.4)

# Running loop to iterate for each column, list1 from config file
for i, l in enumerate(list1):
    counts = df[[l]].value_counts().reset_index(name='count')
    counts.plot(x=l, y='count', kind='bar', ax=axes[i], color=list1_colors[i % len(list1_colors)])
    axes[i].set_title(l)
    axes[i].set_xlabel(l)
    axes[i].set_ylabel('Count')
    
# Showing the Plots
plt.show()

# Saving the plots to .png in the working directory
save_plot_as_png(fig, 'Bar Plots of Categorical Columns')


# #### Plotting the Ticket Prices with their respective counts

# In[17]:


# Plotting the Ticket Prices with their respective counts

fig = plt.figure(figsize=(10,4))

# Histogram plot using seaborn with KDE
sns.histplot(data=df, x='Ticket Price', color="green", label="Airline", kde=True)

# Saving the plots to .png in the working directory
save_plot_as_png(fig, 'Ticket Price vs Flight Counts')


# #### Seperating values for business and economy for better visualisation

# In[18]:


# Select rows where Journey class is Business or Economy
business_flights = df[df['Journey Class'] == 'Business']
economy_flights = df[df['Journey Class'] == 'Economy']


# #### Defining plot function to plot various graphs

# In[19]:


def plot_flight_prices(data, x_col, y_col, hue_col, palette, title, layout):
    
    # Sorting the Business flights dataframe by the order of departure times
    business_flights_sorted = business_flights.sort_values(x_col)

    # Sorting the Economy flights dataframe by the order of departure times
    economy_flights_sorted = economy_flights.sort_values(x_col)
    
    # Set up the figure with two subplots
    if (layout =='H'):
        fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))
        
    # Plot the distribution of flight prices across different departure times for Business flights
    sns.lineplot(data=business_flights_sorted, x=x_col, y=y_col, hue=hue_col, ax=axs[0], palette=palette)
    axs[0].set_title('Business Flights', fontsize =14)
    axs[0].get_legend().remove()
    axs[0].grid(color='lightgrey', linestyle='--', linewidth=0.9)

    # Plot the distribution of flight prices across different departure times for Economy flights
    sns.lineplot(data=economy_flights_sorted, x=x_col, y=y_col, hue=hue_col, ax=axs[1], palette=palette)
    axs[1].set_title('Economy Flights', fontsize =14)
    axs[1].get_legend().remove()
    axs[1].grid(color='lightgrey', linestyle='--', linewidth=0.9)
    
    # Plotting the common legend for both plots on right center
    handles, labels = axs[1].get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    fig.legend(unique_handles, unique_labels,loc='center right', bbox_to_anchor=(1.0, 0.5), fancybox= True)

    # Set the title of the figure
    fig.suptitle(title, fontsize =18)
    plt.grid(color = 'lightgrey', linestyle = '--', linewidth = 0.9)
    plt.show()
    save_plot_as_png(fig, title)


# ### Plotting Flight Prices based on the Departure Time with the function

# In[20]:


# Plotting Flight Prices based on the Departure Time with the function

plot_flight_prices(df, 'Departure Time', 'Ticket Price', 'Airline',
                   airline_colors, 'Flight Prices based on Different Departure Time', 'H')


# ### Plotting Flight Prices based on the Source City with the function

# In[21]:


# Plotting Flight Prices based on the Source City with the function

plot_flight_prices(df, 'Source City', 'Ticket Price', 'Airline',
                   airline_colors, 'Flight Prices based on Different Source City', 'H')


# ### Plotting Flight Prices based on the Destination City with the function

# In[22]:


# Plotting Flight Prices based on the Destination City with the function

plot_flight_prices(df, 'Destination City', 'Ticket Price', 'Airline', 
                   airline_colors, 'Flight Prices based on Different Destination City', 'H')


# ### Plotting Flight Prices based on the Number of Stops with the defined function

# In[23]:


# Plotting Flight Prices based on the Number of Stops with the defined function

plot_flight_prices(df, 'Number of Stops', 'Ticket Price', 'Airline', 
                   airline_colors, 'Flight Prices based on Number of Stops', 'H')


# ### Plotting Flight Prices based on the Trip Duration with the defined function

# In[24]:


# Plotting Flight Prices based on the Trip Duration with the defined function

plot_flight_prices(df, 'Trip Duration', 'Ticket Price', 'Airline', 
                   airline_colors, 'Flight Prices based on Trip Duration', 'V')


# ### Plotting Flight Prices based on the Simplified Trip Duration

# In[25]:


# Plotting Flight Prices based on the Simplified Trip Duration

fig = plt.figure(figsize=(20,8))
sns.lineplot(data=df, x='Trip Duration', y='Ticket Price', hue='Journey Class')
plt.xticks(range(0, 52, 5))
plt.show()

# Saving the plots to .png in the working directory
save_plot_as_png(fig, 'Trip Duration vs Ticket Price')


# ### Plotting Flight Prices based on the Day of Week with the defined function

# In[26]:


# Plotting Flight Prices based on the Day of Week with the defined function

plot_flight_prices(df, 'Day of Week', 'Ticket Price', 'Airline', 
                   airline_colors, 'Flight Prices based on Day of Week', 'H')


# ### Plotting Flight Prices based on the Days left

# In[27]:


# Plotting Flight Prices based on the Days left

fig = plt.figure(figsize=(10,4))
sns.lineplot(data=df, x='Days Left', y='Ticket Price', hue='Journey Class')
plt.xticks(range(0, 52, 5))

# Shaded area for Economy Flight Price Change
plt.axhspan(5000, 10000, 0.305, 0.37, alpha=0.2, color='dodgerblue')

# Adding a label to the shaded area with an arrow pointing
plt.annotate('Economy Class Price Change', xy=(18, 10000), xytext=(27, 25000), 
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.annotate('Business Class Price Change', xy=(11, 55000), xytext=(27, 60000), 
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Shaded area for Business Flight Price Change
plt.axhspan(50000, 55000, 0.175, 0.24, alpha=0.2, color='orange')
plt.show()

# Saving the plot witht the function
save_plot_as_png(fig, 'Price change window based on Journey Class')


# ### Plotting the Violin Plots for Flight Price Variation based on Number of Stops

# In[28]:


# Plotting the Violin Plots for Flight Price Variation based on Number of Stops

fig, axs = plt.subplots(1, 2, figsize=(20, 7))

# Plot for Business Flights
sns.violinplot(x='Number of Stops', y='Ticket Price', ax=axs[0], data=business_flights)
axs[0].set_title('Business Flights', fontsize =14)

# Plot for Economy Flights
sns.violinplot(x='Number of Stops', y='Ticket Price', ax=axs[1], data=economy_flights)
axs[1].set_title('Economy Flights', fontsize =14)

fig.suptitle('Flight Price Variation based on Stops', fontsize =18)
plt.show()

# Saving the plot with the function
save_plot_as_png(fig, 'Flight Price Variation based on Stops')


# ### Defining function for plotting pie Charts

# In[29]:


def pie_plots(column, title):
    counts = df[column].value_counts()

    # Create the pie chart
    fig, ax = plt.subplots()
    pie = ax.pie(counts,labels= counts.index, autopct='%1.1f%%', 
                 wedgeprops={ 'linewidth' : 2, 'edgecolor' : 'white' },
                 startangle=150,labeldistance=1.1)
    # Add a circle at the center of the chart to create the donut shape
    center_circle = plt.Circle((0, 0), 0.8, fc='white')
    fig.gca().add_artist(center_circle)

    plt.text(0, 0, column, ha='center', va='center', fontsize=10)
    plt.title(title, fontsize=18)
    # Show the plot
    plt.show()
    
    # Save the plot with the function
    save_plot_as_png(fig, title)


# ### Pie Plots

# In[30]:


pie_plots('Airline', 'Airline Market Share')
pie_plots('Source City', 'Flights from Source City')


# In[31]:


pie_plots('Destination City', 'Flights to Destination City')
pie_plots('Arrival Time', 'Arrival Time')


# ### Top Popular and Least Popular fights

# In[32]:


top_flights = df['Flight Code'].value_counts().head(5)
least_flights = df['Flight Code'].value_counts().tail(5)

print('Top Popular Flights: \n')
print(top_flights)
print('\n')
print('Least Popular Flights: \n')
print(least_flights)
print('\n')


# ### Correlation Matix of Numerial Columns

# In[33]:


corr = df.corr()

# Visualize the correlation matrix using a heatmap
fig = sns.heatmap(corr, annot=True, cmap='coolwarm').get_figure()
plt.title('Correlation matrix of Flight Data')
plt.show()
save_plot_as_png(fig,'Correlation matrix of Flight Data')


# ### T-Test for Departure and Arrival Time

# In[34]:


# Data Preparation
df_temp = df[['Departure Time', 'Arrival Time', 'Ticket Price']]

# Visualization
fig1 = sns.boxplot(x='Departure Time', y='Ticket Price', data=df)
plt.title('Ticket price based on Departure Time')
plt.show()

fig2 = sns.boxplot(x='Arrival Time', y='Ticket Price', data=df)
plt.title('Ticket price based on Arrival Time')
plt.show()


# Statistical Analysis
night_prices = df_temp[df_temp['Departure Time'] == 'Night']['Ticket Price']
early_morning_prices = df_temp[df_temp['Arrival Time'] == 'Early Morning']['Ticket Price']
t_statistic, p_value = ttest_ind(night_prices, early_morning_prices)

print('T-Statistic: {:.2f}, P-Value: {:.4f}'.format(t_statistic, p_value))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




