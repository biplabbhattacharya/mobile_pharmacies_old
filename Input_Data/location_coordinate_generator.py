# https://gis.stackexchange.com/questions/25877/generating-random-locations-nearby/68275#68275
# https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points


from __future__ import division
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from haversine import haversine
from copy import deepcopy
from math import radians, cos, sin, asin, sqrt

# def haversine(lon1, lat1, lon2, lat2):
#     """
#     Calculate the great circle distance between two points 
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians 
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

#     # haversine formula 
#     dlon = lon2 - lon1 
#     dlat = lat2 - lat1 
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a)) 
#     #r = 3956
#     r = 6371 # Radius of earth in kilometers. Use 3956 for miles
#     return c * r

def make_dict():
    return defaultdict(make_dict)

def create_random_point(x0,y0,distance):
    """
            Utility method for simulation of the points
             there are about 111,300 meters in a degree.
    """   
    r = distance/ 111300
    u = np.random.uniform(0,1)
    v = np.random.uniform(0,1)
    w = r * np.sqrt(u)
    t = 2 * np.pi * v
    x = w * np.cos(t)
    x1 = x / np.cos(y0)
    y = w * np.sin(t)
    return (x0+x1, y0 +y)


def locations_gen(number_of_locations, depot_lat,depot_long, approx_range):

    latitude1,longitude1 = depot_lat,depot_long
    fig = plt.figure()
    ax = host_subplot(111, axes_class=AA.Axes)
    #ax.set_ylim(76,78)
    #ax.set_xlim(13,13.1)
    ax.set_autoscale_on(True)
    ax.plot(latitude1,longitude1,'ro')
   
    location_id = [0]
    gps_coord = [[latitude1,longitude1]]
    #long_gps = []

    for i in range(1,number_of_locations+1):
        x,y = create_random_point(latitude1,longitude1 ,approx_range)
        gps_coord.append([x,y])
        location_id.append(i)
        ax.plot(x,y,'bo')
        dist = haversine((x,y),(latitude1,longitude1))
        #print "Distance between points 0 and %d are: %f" %(i, dist)    # a value approxiamtely less than 500 meters   )

    #print gps_coord

    #print "this is gps",gps_coord[0]

    distance_df = []

    for i in range (0,number_of_locations+1):
        temp_array = []
        for j in range (0,number_of_locations+1):
            dist = haversine(gps_coord[i],gps_coord[j])
            temp_array.append(dist)

        distance_df.append(temp_array)
        
    distance_df = pd.DataFrame(distance_df)

    #print distance_df
    
    #print "location farthest from Depot", max(distance_df[0])

    #return(max(distance_df[0]), gps_coord, distance_df)
    return(gps_coord)

number_of_locations = 5
depot_lat = 0.40
depot_long = 32.48
approx_range = 15000 #(in meters)

locations_gen(number_of_locations, depot_lat, depot_long, approx_range)



#plt.show()