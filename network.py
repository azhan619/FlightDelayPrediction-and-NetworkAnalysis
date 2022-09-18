import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
import matplotlib.lines as graph_lines




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
    


    #df1_new = df1_new[['Date',	'AIRLINE','FLIGHT_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','SCHEDULED_DEPARTURE'
    #	,'DEPARTURE_TIME','DEPARTURE_DELAY']]   



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

    graph_delay = nx.from_pandas_edgelist(routes_mean, source = 'ORIGIN_AIRPORT', target = 'DESTINATION_AIRPORT',edge_attr = 'ARR_DELAY',create_using = nx.DiGraph())
    # without weights
    graph_un = nx.from_pandas_edgelist(routes_total, source = 'ORIGIN_AIRPORT', target = 'DESTINATION_AIRPORT')
    print(str(  graph_delay['ATL']['ABE']  ))
    print(str(  graph_delay['ABE']['ATL']  ))
    #graph1 = nx.from_pandas_edgelist(routes_total1, source = 'ORIGIN_AIRPORT', target = 'DESTINATION_AIRPORT',edge_attr = 'flight_cnt',create_using = nx.DiGraph())

    #print(str(nx.average_clustering(graph1)))
    
    def draw_network(graph,flight_cnt,pos_data,fileName):  
      
      plt.figure(figsize=(5,8))
    
    
      map = Basemap(projection='merc', llcrnrlon=-180,llcrnrlat=10,urcrnrlon=-50,urcrnrlat=70,lat_ts=0,resolution='l',suppress_ticks=True)
      map_long, map_lat = map(pos_data['LONGITUDE'].values, pos_data['LATITUDE'].values)
    
      nodes_position = {}
      
      for count, elem in enumerate (pos_data['IATA_CODE']):
          
          nodes_position[elem] = (map_long[count], map_lat[count])

      #print(str(nodes_position))        

      # creating nodes and edges on basemap
      nx.draw_networkx_nodes(G = graph, pos = nodes_position, nodelist = [x for x in graph.nodes() if flight_cnt['flight-count'][x] >= 100],node_color = 'r', alpha = 0.7,node_size = [flight_cnt['flight-count'][x] * 0.5 for x in graph.nodes() if flight_cnt['flight-count'][x] >= 100])

      nx.draw_networkx_labels(G = graph, pos = nodes_position, font_size=3,font_color="white",labels = {x:x for x in graph.nodes() if flight_cnt['flight-count'][x] >= 100})

      nx.draw_networkx_nodes(G = graph, pos = nodes_position, nodelist = [x for x in graph.nodes() if flight_cnt['flight-count'][x] < 100],node_color = 'yellow', alpha = 0.4,node_size = [flight_cnt['flight-count'][x]*0.5  for x in graph.nodes() if flight_cnt['flight-count'][x] < 100])

      nx.draw_networkx_edges(G = graph, pos = nodes_position, edge_color='black', alpha=0.1, arrows = False)
      plt.title("Domestic Flight Network of USA", fontsize = 8)
      map.drawcountries(linewidth = 3)
      map.drawstates(linewidth = 0.3)
      map.drawcoastlines(linewidth=1.5)
      map.fillcontinents(alpha = 0.3)
      legend_elem1 = graph_lines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red")
      legend_elem2 = graph_lines.Line2D(range(1), range(1), color="white", marker='o',markerfacecolor="yellow")
      legend_elem3 = graph_lines.Line2D(range(1), range(1), color="black", marker='',markerfacecolor="black")
      #plt.legend((legend_elem1, legend_elem2, legend_elem3), ('Airports with > 100 routes', 'Airports with < 100 routes', 'Direct route'),
                #loc=4, fontsize = 'xx-large')
    
      plt.tight_layout()
      plt.savefig(fileName+".png", format = "png", dpi = 150)

    
    
    def bet_centrality(graph):

          result_dict = nx.betweenness_centrality(graph,normalized=False)

          bet_centrality_df = pd.DataFrame.from_dict({
          'node': list(result_dict.keys()),
          'bet_centrality': list(result_dict.values())
            })

          bet_centrality_df = bet_centrality_df.sort_values('bet_centrality', ascending=False)

          bet_centrality_df.to_csv("bet_centrality-2018jan.csv")
    
    def degree_centrality(graph):

          result_dict = nx.degree_centrality(graph)

          deg_centrality_df = pd.DataFrame.from_dict({
          'node': list(result_dict.keys()),
          'deg_centrality': list(result_dict.values())
            })

          deg_centrality_df = deg_centrality_df.sort_values('deg_centrality', ascending=False)

          deg_centrality_df.to_csv("deg_centrality-2018jan.csv")

    def pagerank_measure(graph):
          
          result_dict = nx.pagerank(graph)

          pagerank_df = pd.DataFrame.from_dict({
          'node': list(result_dict.keys()),
          'pagerank': list(result_dict.values())
            })

          pagerank_df = pagerank_df.sort_values('pagerank', ascending=False)

          pagerank_df.to_csv("pagerank-2018jan.csv")

          
    def harmonic_centrality(graph):

          result_dict = nx.harmonic_centrality(graph)

          har_centrality_df = pd.DataFrame.from_dict({
              'node': list(result_dict.keys()),
              'har_centrality': list(result_dict.values())
                 })

          har_centrality_df = har_centrality_df.sort_values('har_centrality', ascending=False)

          har_centrality_df.to_csv("har_centrality-2018jan.csv")
    
    
   
   
   
   
   
   
   
    draw_network(graph,flight_cnt,pos_data,fileName="2018_data-NETWORK")
    #draw_network(graph=graph1,flight_cnt=flight_cnt1,pos_data=pos_data1,fileName="2017_data")
    bet_centrality(graph)
    degree_centrality(graph)
    pagerank_measure(graph)
    harmonic_centrality(graph)


if __name__ == "__main__":
    main()  