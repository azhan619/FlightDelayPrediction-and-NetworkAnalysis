import pandas as pd
import numpy as np



def main():
    
    flights_data = pd.read_csv('2018.csv')
    # filtering data for selected airlines
    airline_names = {
    'UA':'United Airlines','WN':'Southwest Airlines','YV':'Mesa Airline','AS':'Alaska Airlines','9E':'Endeavor Air','B6':'JetBlue Airways','EV':'ExpressJet','F9':'Frontier Airlines',
    'NK':'Spirit Airlines','OH':'PSA Airlines','OO':'SkyWest Airlines','G4':'Allegiant Air','MQ':'Envoy Air',
    'VX':'Virgin America','YX':'Republic Airways','AA':'American Airlines','HA':'Hawaiian Airlines',
    'DL':'Delta Airlines'}

    # replacing airline code with airline name
    flights_data['OP_CARRIER'].replace( airline_names, inplace=True)


    flights_data = flights_data.drop(["Unnamed: 27"], axis=1)

   

    # remove all the cancelled flights
   
    flights_data = flights_data[(flights_data['CANCELLED'] == 0)]

    


    

    #flights_data.info()



    airports = pd.read_csv('airports.csv')

    #print(len(airports.IATA_CODE.unique()))

    #print(len(flights_data.ORIGIN.unique()))

    airport_IATA_CODE = list(airports['IATA_CODE'])

    flights_ORIGIN = flights_data.ORIGIN.unique().tolist()
    
    lights_DEST = flights_data.DEST.unique().tolist()

    #print( len( airport_IATA_CODE  )  )
    #print( len( flights_ORIGIN )  )
    #print( len( flights_DEST )  )

    # airport codes that exists in airport.csv but not in flights_data.
    
    airport_diff = list( set(airport_IATA_CODE) - set(flights_ORIGIN)   )

    
    flights_diff = list(  set(flights_ORIGIN) - set(airport_IATA_CODE)  )
    
    #print( str(len(flights_diff) ))
    
    #print(str(flights_data.shape))

    # remove the airports with missing information
    for x in flights_diff:

        flights_data = flights_data[flights_data.ORIGIN != x ]
        flights_data = flights_data[flights_data.DEST != x]
    

    for x in airport_diff:

        airports = airports[airports.IATA_CODE != x ]

    
    airports = airports[airports.IATA_CODE != 'STC' ]
    
    flights_ORIGIN = flights_data.ORIGIN.unique().tolist()
    
    flights_DEST = flights_data.DEST.unique().tolist()
    
    airport_IATA_CODE = list(airports['IATA_CODE'])
    
    airport_diff = list( set(airport_IATA_CODE) - set(flights_ORIGIN)   )
    
    airport_diff2 = list( set(airport_IATA_CODE) - set(flights_DEST)   )

    flights_ORIGIN = flights_data.ORIGIN.unique().tolist()
    flights_DEST = flights_data.DEST.unique().tolist()        
    
    flights_diff = list(  set(flights_ORIGIN) - set(airport_IATA_CODE)  )

    df2_diff = list(  set(flights_ORIGIN) - set(airport_IATA_CODE)  )

    
    #print( len( flights_ORIGIN )  )
    #print( len( flights_DEST )  )
    #print(airport_diff)
    #print(airport_diff2)
 
    #print(flights_diff)
    #print(df2_diff)

    # drop un-used columns from data
    drop_columns = ['AIR_TIME','ACTUAL_ELAPSED_TIME','ARR_TIME','DEP_TIME',
            'DIVERTED','WHEELS_OFF','WHEELS_ON','CANCELLATION_CODE','TAXI_OUT','TAXI_IN',
            'CANCELLED','CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 
            'LATE_AIRCRAFT_DELAY']
    
    
    print(str(flights_data.shape))
    #flights_data = flights_data.drop(drop_columns, axis=1)
    flights_data = flights_data.dropna()


    flights_data['DAY'] = pd.DatetimeIndex(flights_data['FL_DATE']).day
    
    flights_data['YEAR'] = pd.DatetimeIndex(flights_data['FL_DATE']).year
    
    flights_data['MONTH'] = pd.DatetimeIndex(flights_data['FL_DATE']).month

 
    

    #print('Total number of years:', flights_data.YEAR.nunique())
    
    flights_data['CRS_DEP_TIME'] = np.ceil(flights_data['CRS_DEP_TIME']/400).apply(int)
    
    flights_data['CRS_ARR_TIME'] = np.ceil(flights_data['CRS_ARR_TIME']/400).apply(int)

    #print(str(flights_data.shape))
    #print(str(flights_data.isna().sum()))
    
    flights_data.to_csv('flights2018_network.csv')

if __name__ == "__main__":
    main()      
