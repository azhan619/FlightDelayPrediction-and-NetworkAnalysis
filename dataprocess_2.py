import pandas as pd
from datetime import datetime

def main():
   process_data()


def process_data(airportList=None):

    file = open("model_result2.txt","w+")

    now = datetime.now()

    # timer to check how much time it took the method to run  
    current_time = now.strftime("%H:%M:%S")
    file.write("Start time" + str(current_time) + " \n \n")



    

    flights_data = pd.read_csv('flights2018_processed.csv', index_col=0)

    fl_st = []

    # Converting delays value to with 0 or 1, 0 means delay < 10 minutes , 1 means delay > 10 minutes

    for y in flights_data['ARR_DELAY']:
        
        if y < 0:
        
            fl_st.append(0)
        
        else:
        
            fl_st.append(1)
    
    flights_data['DELAY_BIN'] = fl_st

    value=[]
    

    # Columns for top airlines, converting the values to 0 or 1
    cols = ['UA_BIN','WN_BIN','YV_BIN','AS_BIN','EV_BIN'	,'JB_BIN','SA_BIN','PSA_BIN','SW_BIN','AL_BIN','EJ_BIN','FA_BIN',
    'EA_BIN','VA_BIN','RA_BIN','AA_BIN','HA_BIN','DL_BIN']

    flights_data[cols] = 0

    my_dict = {'United Airlines':1,'Southwest Airlines':0,'Mesa Airline':0,'Alaska Airlines':0,'Endeavor Air':0,'JetBlue Airways':0,'ExpressJet':0,'Frontier Airlines':0,
    'Spirit Airlines':0,'PSA Airlines':0,'SkyWest Airlines':0,'Allegiant Air':0,'Envoy Air':0,
    'Virgin America':0,'Republic Airways':0,'American Airlines':0,'Hawaiian Airlines':0,
    'Delta Airlines':0}

    # converting airlines names from values to column names and if the flight belong to a airline then it is else 0

    flights_data.loc[flights_data.OP_CARRIER == 'United Airlines', 'UA_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Southwest Airlines', 'WN_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Mesa Airline', 'YV_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Alaska Airlines', 'AS_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Endeavor Air', 'EV_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'JetBlue Airways', 'JB_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'ExpressJet', 'EJ_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Frontier Airlines', 'FA_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'SkyWest Airlines', 'SW_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Spirit Airlines', 'SA_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'PSA Airlines', 'PSA_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Allegiant Air', 'AL_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Envoy Air', 'EA_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Virgin America', 'VA_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Republic Airways', 'RA_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'American Airlines', 'AA_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Hawaiian Airlines', 'HA_BIN'] = 1

    flights_data.loc[flights_data.OP_CARRIER == 'Delta Airlines', 'DL_BIN'] = 1

    flights_data.info()

    
    airport_mod =   ['ATL','ORD','DFW','DEN','MSP','DTW','ANC','SLC','IAH','SEA','PHX','SFO',
                'LAX','MCO','JFK','EWR','JNU','FLL','HNL','LAS','CLT','PHL','LGA','BOS','BGR','BFL',
                'DCA'
                
                
                ]


    flights_data_b = flights_data.DEST.isin(airport_mod)
    
    flights_data_2 = flights_data[flights_data_b]
    
    flights_data[cols] = 0
    # similar to airlines , converting airport values to column names as well
    for x in airport_mod:
        flights_data_2['DEST_'+ x] = 0

    for y in airport_mod:

        flights_data_2.loc[flights_data_2.DEST == y, 'DEST_' + y] = 1      

    flights_data_2['WEEK_DAY']= pd.DatetimeIndex(flights_data_2['FL_DATE']).dayofweek

    month_mod = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    # conveting month value to column
    for x in month_mod:
        flights_data_2['MONTH_'+ str(x)] = 0

    for y in month_mod:

        flights_data_2.loc[flights_data_2.MONTH == y, 'MONTH_' + str(y)] = 1  

    weekday_mod = [0,1,2,3,4,5,6]
    # converting weekdat value to column
    for x in weekday_mod:
        flights_data_2['WEEKDAY_'+ str(x)] = 0

    for y in weekday_mod:

        flights_data_2.loc[flights_data_2.WEEK_DAY == y, 'WEEKDAY_' + str(y)] = 1  


    DEP_mod = [1,2,3,4,5,6]

    for x in DEP_mod:
        flights_data_2['CRS_DEP_'+ str(x)] = 0

    for y in DEP_mod:

        flights_data_2.loc[flights_data_2.CRS_DEP_TIME == y, 'CRS_DEP_' + str(y)] = 1

    ARR_mod = [1,2,3,4,5,6]

    for x in ARR_mod:
        flights_data_2['CRS_ARR_'+ str(x)] = 0

    for y in ARR_mod:

        flights_data_2.loc[flights_data_2.CRS_ARR_TIME == y, 'CRS_ARR_' + str(y)] = 1

    #print(str(flights_data_2.DEST.value_counts() ))

    #print(str(flights_data_2.shape))



    flights_data.to_csv('2018-flData-NEW.csv')

    flights_data_2.to_csv('2018-flData_2.csv')   

if __name__ == "__main__":
    main()      