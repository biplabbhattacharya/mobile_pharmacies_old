## Importing functions for using MapQuest

import urllib2
from numpy import mean, absolute
import json
import pandas as pd
from pandas.io.json import json_normalize

mapquestKey = 'pRLuCGM4quOm1bIHqOQ0ToCNQkfU4n7W'

## B. Define function to return distances using MapQuest

def genTravelMatrix(coordList, locIDlist, all2allStr, one2manyStr, many2oneStr):
	# We'll use MapQuest to calculate.
	transportMode = 'fastest'
	# Other option includes:  'pedestrian' (apparently 'bicycle' has been removed from the API)
	routeTypeStr = 'routeType:%s' % transportMode
	
	# Assemble query URL
	myUrl = 'http://www.mapquestapi.com/directions/v2/routematrix?'
	myUrl += 'key={}'.format(mapquestKey)
	myUrl += '&inFormat=json&json={locations:['
	
	# Insert coordinates into the query:
	n = len(coordList)
	for i in range(0,n):
		if i != n-1:
			myUrl += '{{latLng:{{lat:{},lng:{}}}}},'.format(coordList[i][0], coordList[i][1])
		elif i == n-1:
			myUrl += '{{latLng:{{lat:{},lng:{}}}}}'.format(coordList[i][0], coordList[i][1])
	myUrl += '],options:{{{},{},{},{},doReverseGeocode:false}}}}'.format(routeTypeStr, all2allStr,one2manyStr,many2oneStr)
	
	# print "\nThis is the URL we created.  You can copy/paste it into a Web browser to see the result:"
	# print myUrl
	
	
	# Now, we'll let Python go to mapquest and request the distance matrix data:
	request = urllib2.Request(myUrl)
	response = urllib2.urlopen(request)	
	data = json.loads(response.read())
	
	# print "\nHere's what MapQuest is giving us:"
	# print data

	# This info is hard to read.  Let's store it in a pandas dataframe.
	# We're goint to create one dataframe containing distance information:
	distance_df = json_normalize(data, "distance")
	# print "\nHere's our 'distance' dataframe:"
	# print distance_df	

	# print "\nHere's the distance between the first and second locations:"
	# print distance_df.iat[0,1]	
	
	# Our dataframe is a nice table, but we'd like the row names (indexes)and column names to match our location IDs.
	# This would be important if our locationIDs are [1, 2, 3, ...] instead of [0, 1, 2, 3, ...]
	distance_df.index = locIDlist
	distance_df.columns = locIDlist
		
	# Now, we can find the distance between location IDs 1 and 2 as:
	# print "\nHere's the distance between locationID 1 and locationID 2:"
	# print distance_df.loc[1,2]
	
	# We can create another dataframe containing the "time" information:
	time_df = json_normalize(data, "time")

	# print "\nHere's our 'time' dataframe:"
	# print time_df
	
	# Use our locationIDs as row/column names:
	time_df.index = locIDlist
	time_df.columns = locIDlist

	# We could also create a dataframe for the "locations" information (although we don't need this for our problem):
	#print "\nFinally, here's a dataframe for 'locations':"
	#df3 = json_normalize(data, "locations")
	#print df3
	#print time_df
	
	return(distance_df, time_df)