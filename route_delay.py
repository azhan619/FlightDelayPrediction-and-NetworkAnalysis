import pandas as pd

import networkx as nx

from mpl_toolkits.basemap import Basemap as Basemap

from nn_model import predict_model




def main():

    #flights data

    df1 = pd.read_csv('flights2018_processed.csv')

    #flights_d1 = pd.read_csv('2017.csv')
    #flights_d1 = flights_d1.dropna()
    
    #print(str(flights_d.shape[0]))

    #airports data
    airports_d = pd.read_csv('airports.csv')

    # selecting the columns we will be using.

    #df1 = flights_d[['YEAR','MONTH','DAY','AIRLINE','FLIGHT_NUMBER','TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','SCHEDULED_DEPARTURE'
    #	,'DEPARTURE_TIME','DEPARTURE_DELAY']]

    #  Merging the 3 columns to 1

    #df11 = flights_d1[['FL_DATE',	'OP_CARRIER','OP_CARRIER_FL_NUM','ORIGIN','DEST','CRS_DEP_TIME'
    #	,'DEP_TIME','DEP_DELAY']]

    df1 = df1.rename(columns={'FL_DATE': 'Date', 'OP_CARRIER': 'AIRLINE', 'OP_CARRIER_FL_NUM': 'FLIGHT_NUMBER' ,'ORIGIN': 'ORIGIN_AIRPORT' ,'DEST': 'DESTINATION_AIRPORT'
                  , 'CRS_DEP_TIME': 'SCHEDULED_DEPARTURE' ,  'DEP_DELAY':'DEPARTURE_DELAY' })

    #df1['Date']= pd.to_datetime(df1[['YEAR', 'MONTH', 'DAY']])

    df1.to_csv('2018_data.csv')
    
    #
    




    ##################################################

    # Dataframe containing routes, ORIGIN,Destination,counts ( number of flights between them)
    routes_total =  pd.DataFrame(df1.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']).size().reset_index(name='flight_cnt'))

    routes_mean =  pd.DataFrame(df1.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'],as_index=False)['ARR_DELAY'].mean() )
    
    #routes_total1 =  pd.DataFrame(df1_new.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']).size().reset_index(name='flight_cnt'))

    # removing empty and incorrect columns , e.g only using airport with 3 letter IATA Code.
    routes_total = routes_total[routes_total.ORIGIN_AIRPORT.str.contains(r'[A-Z][A-Z][A-Z]', na=False)]
    #routes_total1 = routes_total1[routes_total1.ORIGIN_AIRPORT.str.contains(r'[A-Z][A-Z][A-Z]', na=False)]
    routes_mean = routes_mean[routes_mean.ORIGIN_AIRPORT.str.contains(r'[A-Z][A-Z][A-Z]', na=False)]


    # convert data
    routes_total.to_csv('routes_total.csv')
    routes_mean.to_csv('routes-MEAN.csv')
    #routes_total1.to_csv('routes_total1.csv')


    flight_cnt = routes_total['ORIGIN_AIRPORT'].append(routes_total.loc[routes_total['ORIGIN_AIRPORT'] != routes_total['DESTINATION_AIRPORT'], 'DESTINATION_AIRPORT']).value_counts()
    flight_cnt = pd.DataFrame({'IATA_CODE': flight_cnt.index, 'flight-count': flight_cnt})
    flight_cnt.to_csv("flight_cnt-2018.csv")
    pos_data = flight_cnt.merge(airports_d, on = 'IATA_CODE')


    #flight_cnt1 = routes_total1['ORIGIN_AIRPORT'].append(routes_total1.loc[routes_total1['ORIGIN_AIRPORT'] != routes_total1['DESTINATION_AIRPORT'], 'DESTINATION_AIRPORT']).value_counts()
    #flight_cnt1 = pd.DataFrame({'IATA_CODE': flight_cnt1.index, 'flight-count': flight_cnt1})
    
    #pos_data1 = flight_cnt1.merge(airports_d, on = 'IATA_CODE')
    

    
    #flight_cnt1 = flight_cnt1.merge(pos_data1,on='IATA_CODE')
    #pos_data1.to_csv("pos1.csv")
    #flight_cnt1 = flight_cnt1.rename(columns={'flight-count_x':'flight-count'})
    #flight_cnt1 =flight_cnt1.drop(columns=['flight-count_y'])
    #flight_cnt1.to_csv("flight_cnt1.csv")

    #pos_data1.to_csv("pos1.csv")



    # with weights
    graph = nx.from_pandas_edgelist(routes_total, source = 'ORIGIN_AIRPORT', target = 'DESTINATION_AIRPORT',edge_attr = 'flight_cnt',create_using = nx.DiGraph())
    # flight network , with delays as edge weights.
    graph_delay = nx.from_pandas_edgelist(routes_mean, source = 'ORIGIN_AIRPORT', target = 'DESTINATION_AIRPORT',edge_attr = 'ARR_DELAY',create_using = nx.DiGraph())
    # without weights
    graph_un = nx.from_pandas_edgelist(routes_total, source = 'ORIGIN_AIRPORT', target = 'DESTINATION_AIRPORT')
    #print(str(  graph_delay['ATL']['ABE']  ))
    #print(str(  graph_delay['ABE']['ATL']  ))

    #route_path = nx.all_shortest_paths(graph_delay,source='BNA',target='BOI')

    #print(list(route_path))

    #route_path_2 = nx.all_shortest_paths(graph_delay,source='TVC',target='ELP')
    #print(list(route_path_2))

    #route_path_3 = nx.all_shortest_paths(graph_delay,source='BGR',target='RNO')

    route_path_4 = nx.all_shortest_paths(graph_delay,source='BGR',target='BFL')

    route_list = list(route_path_4)

    print(route_list)
    print(len(route_list))

    airport_li =[]

    # calculating all the unique connected airports and storing them in list.

    for x in route_list:

        for j in x:

            if j not in airport_li:

                airport_li.append(j)

    print("----------------------------------------------------")
    print("----------------------------------------------------")

    print(airport_li)
    print(len(airport_li))

    airport_delay_dict = {}

    # for all the connected airports, getting the predicted value.
    for k in airport_li:

       perc = predict_model(airport=k) 

       airport_delay_dict[k] = perc

    result_dict={}

    # calculating the avg delay prediction on each route.

    for x in route_list:
        my_str=''
        my_str = x[1] +' - '+ x[2]

        delay_avg = (  airport_delay_dict.get(x[1]) + airport_delay_dict.get(x[2])  ) / 2

        result_dict[my_str] = delay_avg









    #print(str(airport_delay_dict))
    print(str(result_dict))
if __name__ == "__main__":
    main()       