

# Datetime transformation
DATE_FORMAT = '%d/%m/%Y'

#TimePeriod labels for DepartureTime and Arrival Time
TIME_PERIOD_LABELS = {
    1: 'Late Night',
    2: 'Early Morning',
    3: 'Morning',
    4: 'Afternoon',
    5: 'Evening',
    6: 'Night'
}

#Time Format for Hour minutes to calculate  trip duration
TIME_FORMAT = '%H:%M'


#Rename column list
RENAME_COLUMNS = {
    'airline': 'Airline', 
    'from': 'Source City', 
    'to': 'Destination City', 
    'class': 'Journey Class', 
    'price': 'Ticket Price'
}

#Arranging the columns as per the requirement
COLUMN_ARRANGEMENT = [
    'Airline', 'Source City', 'Departure Time', 
    'Number of Stops', 'Destination City', 'Arrival Time', 
    'Trip Duration', 'Date of Journey', 'Days Left', 
    'Day of Week', 'Flight Code', 'Journey Class', 'Ticket Price'
]

#Sort the data accordingly
SORT_DATA = [
    'Days Left',
    'Ticket Price'
]

#Dropping the non required columns
DROP_CLEAN_FLIGHTS = [
    'ch_code',
    'num_code',
    'stop',
    'date',
    'price'
]

#Transform datatype of the columns
TRANSFORM_DATATYPE = [
    'Days Left', 
    'Ticket Price', 
    'Number of Stops'
]

#Dropping the non required columns
DROP_TIME_COL = [
    'hour', 
    'minute', 
    'time_taken', 
    'dep_time', 
    'arr_time'
]

#List for the plots visualisation function
list1 = ['Source City', 'Destination City', 'Airline', 'Arrival Time', 'Departure Time']

#colour for the plots visualisation function
list1_colors = ['coral', 'orange', 'green', 'darkseagreen', 'darkslategrey']



#colour for the airline line plots visualisation
airline_colors = {
    'Vistara': 'coral',
    'Indigo': 'orange', 
    'Air India': 'green', 
    'SpiceJet': 'brown',
    'AirAsia': 'darkslategrey',
    'GO FIRST': 'blue'
}

#parameter grid for grid seach cv in decision tree
param_grid = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2]
}