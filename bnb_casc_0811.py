##########################################

# Code by Biplab Bhattacharya, August 08, 2018
# run code using:  python bnb_casc_0808.py <row_number_from_input_file>
# This code will use Generalized greedy and BnB to solve the mobile pharmacies problem
#

# python bnb_casc_0808.py <row_number_from_input_file>
# python bnb_casc_0811.py 24
###########################################

import io
import os
import csv
import sys
from collections import defaultdict
from math import *
from parseCSVstring import *
from copy import deepcopy
import time
import random
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import urllib2
import json
import operator
import folium

from Gen_trav_mat import genTravelMatrix
from SSI_calc import SSI_calculate
from NN_py_script import nn_run

mapquestKey = 'pMWRXaXRFPbT7GMQlZ1NY2jVBT7ECmqt'

## A. Define function for data manipulations and storage
def make_dict():
	return defaultdict(make_dict)

class vehicles():
	def __init__(self):
		self.id = 1
		self.cap_M = 100
		self.cap_D = 100
		self.cap_P = 100

class Node():

	def __init__(self):
		self.id = None
		self.time = None
		self.path = []
		self.visits = []
		#self.current_run_visits = []
		self.max_visits = None
		self.MAD_val = None
		self.veh_cap_M_node = None
		self.veh_cap_D_node = None
		self.veh_cap_P_node = None
		self.LB = 0
		

	def show(self):
		print "this is the node id: ", self.id
		print "this is the node time: ", self.time 
		print "this is the node path: ", self.path
		print "this is the node visits so far: ",self.visits
		print "Maximum number of visits to any settlement are: ", self.max_visits
		print "this is the node MAD score: ", self.MAD_val
		print "this is the node LB: ", self.LB
		print "this is the vehicle capacity for M at node", self.veh_cap_M_node
		print "this is the vehicle capacity for D at node", self.veh_cap_D_node
		print "this is the vehicle capacity for P at node", self.veh_cap_P_node


## C. Define function to calculate SSI

def genShapepoints(startCoords, endCoords):
	# We'll use MapQuest to calculate.
	transportMode = 'fastest'		# Other option includes:  'pedestrian' (apparently 'bicycle' has been removed from the API)

	# assemble query URL 
	myUrl = 'http://www.mapquestapi.com/directions/v2/route?key={}&routeType={}&from={}&to={}'.format(mapquestKey, transportMode, startCoords, endCoords)
	myUrl += '&doReverseGeocode=false&fullShape=true'

	# print "\nThis is the URL we created.  You can copy/paste it into a Web browser to see the result:"
	# print myUrl
	
	# Now, we'll let Python go to mapquest and request the distance matrix data:
	request = urllib2.Request(myUrl)
	response = urllib2.urlopen(request)	
	data = json.loads(response.read())
	
	# print "\nHere's what MapQuest is giving us:"
	# print data
		
	# retrieve info for each leg: start location, length, and time duration
	lats = [data['route']['legs'][0]['maneuvers'][i]['startPoint']['lat'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	lngs = [data['route']['legs'][0]['maneuvers'][i]['startPoint']['lng'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	secs = [data['route']['legs'][0]['maneuvers'][i]['time'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	dist = [data['route']['legs'][0]['maneuvers'][i]['distance'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]

	# print "\nHere are all of the lat coordinates:"
	# print lats

	# create list of dictionaries (one dictionary per leg) with the following keys: "waypoint", "time", "distance"
	legs = [dict(waypoint = (lats[i],lngs[i]), time = secs[i], distance = dist[i]) for i in range(0,len(lats))]

	# create list of waypoints (waypoints define legs)
	wayPoints = [legs[i]['waypoint'] for i in range(0,len(legs))]
	# print wayPoints

	# get shape points (each leg has multiple shapepoints)
	shapePts = [tuple(data['route']['shape']['shapePoints'][i:i+2]) for i in range(0,len(data['route']['shape']['shapePoints']),2)]
	# print shapePts
					
	return shapePts

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = np.array(array)
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def split_tours(path):
	
	i = 1
	x = []
	while i<(len(path)):
		a = [0]
		for j in range(i,len(path)):
			
			a.append(path[j])
		#	print a
			i = i+1
			if i == len(path):
				a.append(0)
			if path[j]==0:
				break

		x.append(a)
	print "inside split tour function", x
	return x


def Number_of_Visits(path,N):
	#array_path = Counter(path)
	#print "this is array path", np.array([v for k, v in sorted(array_path.iteritems())])
	#return np.array([v for k, v in sorted(array_path.iteritems())])

	visits_array_temp = [0]*(N+1)

	array_path = np.array(path)
	max_freq = max(np.bincount(array_path))
	freq = np.bincount(array_path)
	for i in range (0,len(freq)):
		visits_array_temp[i] = freq[i]

	visits_array_temp[0] = 0
	return (visits_array_temp,max_freq)
	#return ((np.bincount(array_path)),max_freq)

def calc_LB(max_visits_allowed, path, visits, total_time, current_time, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P):

	#print path, visits, total_time, current_time

	#location_LB = 0
	temp_SSI_LB = 0

	planning_horizon = total_time - current_time
	#print planning_horizon

	LB_path = deepcopy(path)
	visits_LB = deepcopy(visits)

	NN_LB_so_M = deepcopy(village_sod_M)
	NN_LB_so_D = deepcopy(village_sod_D)
	NN_LB_so_P = deepcopy(village_sod_P)

	location_LB = LB_path[-1]

	visit_flag_SSI = {}

	while planning_horizon>=0:
		#calculate the temporary SSI of the villages at this state
		temp_SSI_LB = SSI_calculate(V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P, NN_LB_so_M, NN_LB_so_D, NN_LB_so_P)


		# Create a dictionary to store all villages and their corresponding SSI
		SSI_dict_LB = {}

		# Now store all temp SSI in the dictionary
		for i in V:
			SSI_dict_LB[i] = temp_SSI_LB[i-1]

		#If a village has been visited the amount of their max visits, remove them from the list
		for i in V:
			if (visits_LB[i] >= max_visits_allowed):
				del SSI_dict_LB[i]		

		# If the list is null, go to location 0
		if bool(SSI_dict_LB)==False:
			location_LB = 0
			LB_path.append(location_LB)
			#print "LB_path", LB_path
		
		else:
			#Next location is the location in the list with highest SSI
			next_loc_LB = max(SSI_dict_LB.iteritems(), key=operator.itemgetter(1))[0]
			LB_path.append(next_loc_LB)
			
			#print "LB_path", LB_path
			# add to the visits taht a the selected location has been visited
			visits_LB[next_loc_LB] = visits_LB[next_loc_LB] + 1
			
			location_LB = next_loc_LB

			NN_LB_so_M[next_loc_LB] = max(0,NN_LB_so_M[next_loc_LB] - visit_effect)
			NN_LB_so_D[next_loc_LB] = max(0,NN_LB_so_D[next_loc_LB] - visit_effect)
			NN_LB_so_P[next_loc_LB] = max(0,NN_LB_so_P[next_loc_LB] - visit_effect)



		
		
		planning_horizon = planning_horizon -1
	else:
		node_LB = 0
	SSI_LB = SSI_calculate(V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P,NN_LB_so_M,NN_LB_so_D,NN_LB_so_P)
	node_LB = mad(SSI_LB)
	#print SSI_LB
	# print NN_LB_so_M
	# print NN_LB_so_D
	# print NN_LB_so_P
	# print "Node_LB_score and path: ", node_LB, LB_path
	
	return(node_LB)

def solve_BNB(initial_node_input,upper_bound,T_horiz,max_dist_allowed,max_visits_allowed,visit_effect,distance_df,veh_1,V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P,village_cpd_M,village_cpd_D,village_cpd_P):

	print "upper bound", upper_bound
	Node_count = 1
	village_cpd_M[0]=0
	village_cpd_D[0]=0
	village_cpd_P[0]=0
	
	#initiate sod_count_M that you will use to store the value of the new stock out days at each node.
	sod_count_M = deepcopy(village_sod_M)
	sod_count_D = deepcopy(village_sod_D)
	sod_count_P = deepcopy(village_sod_P)

	# print "SOD input"

	# print sod_count_M
	# print sod_count_D
	# print sod_count_P
	initial_node = deepcopy(initial_node_input)

	#initial_node.show()

	#upper_bound = 1.0 ##0.00958072877179  ##0.008555
	current_time_at_bnb_init = len(initial_node.path)
	#print current_time_at_bnb_init
	projected_len_bnb_run = (run_number+1)*T_horiz
	#print projected_len_bnb_run
	#initial_node.LB = calc_LB(max_visits_allowed, initial_node_input.path, initial_node.visits, projected_len_bnb_run, current_time_at_bnb_init, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P)

	#print "best_prev",best_node_prev_run

	T = T_horiz #total time periods after T0 
	N = len(V) #excluding depot
	max_dist = max_dist_allowed
	Total_leaf_nodes = (N+1)**T

	Z_best_node_t = 1
	Z_best_node_overall = 1
	
	Nodes_list = [initial_node]

	Nodes_list[0].show()

	prev_visits = deepcopy(initial_node.visits)


	#print "LB and UB - at start ", Nodes_list[0].LB,  upper_bound

	Current_time_Nodes = deepcopy(Nodes_list)
	#print "hey",Current_time_Nodes
	temp_nodes_created = []

	Initial_MAD_score = mad(SSI_calculate(V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P))
	#Z_best_cur = copy.deepcopy(Initial_MAD_score)
	
	Z_best_t = 100.0
	Z_best_overall = 100.0
	new_visits = []

	for t in range(1,T+1):
		print t
		#print Current_time_Nodes
		if t!=1:
			#print "d"
			#print "copy begins"
			#Current_time_Nodes = copy.deepcopy(temp_nodes_created)
			#del Current_time_Nodes
			Current_time_Nodes = deepcopy(temp_nodes_created)
			#print "copy ends"
		
		temp_nodes_created = []
		#print Current_time_Nodes
	    #print Current_time_Nodes
		
		for c in Current_time_Nodes:
			
			any_feasible = 0
			
			for n in range(1,N+1):
				#print n
				if (len(c.path)>0):
					if (c.veh_cap_M_node>=village_cpd_M[n]*visit_effect and c.veh_cap_D_node>=village_cpd_D[n]*visit_effect and c.veh_cap_P_node>=village_cpd_P[n]*visit_effect and c.LB<=upper_bound and distance_df[n][c.path[-1]]<=max_dist and c.visits[n] < max_visits_allowed):
						any_feasible = any_feasible + 1
						#print "Feasible node", n
				if (len(c.path)==0):
					if (c.veh_cap_M_node>=village_cpd_M[n]*visit_effect and c.veh_cap_D_node>=village_cpd_D[n]*visit_effect and c.veh_cap_P_node>=village_cpd_P[n]*visit_effect and c.LB<=upper_bound and c.visits[n] < max_visits_allowed):
						any_feasible = any_feasible + 1


			#if (any_feasible>0):
			if (any_feasible>2 or c.path[-1]==0 or(len(c.path)>1) and c.path[-2]==0):	
				next_lowest_node = 1
					#print "if "
			else:
				next_lowest_node = 0
					#print "else"
					


			# if c.path[-1] == 0 or ((len(c.path)>1) and c.path[-2]==0):
			#   	next_lowest_node = 1
			# else:
			#   	next_lowest_node = 0	
			
			#create all children from thelist of Current_time_Nodes
			# if c.path[-1] == 0:
			# 	print "yes, path -1 == 0"
			# 	c.veh_cap_M_node = veh_1.cap_M
			# 	c.veh_cap_D_node = veh_1.cap_D
			# 	c.veh_cap_P_node = veh_1.cap_P

			if c.path[-1] == 0:
				
				c.veh_cap_M_node = 100
				c.veh_cap_D_node = 100
				c.veh_cap_P_node = 100



			#here next_lowest_node could be 0 which is depot or 1, which is the first village
			for n in range(next_lowest_node,N+1):

				if n == 0:
					village_cpd_M[0] = 0
					village_cpd_D[0] = 0
					village_cpd_P[0] = 0

				#if (c.LB<=upper_bound and distance_df[n][c.path[-1]]<=max_dist) and (c.visits[n] < max_visits_allowed) and (c.veh_cap_M_node >=village_cpd_M[n]*visit_effect) and (c.veh_cap_D_node >=village_cpd_D[n]*visit_effect) and (c.veh_cap_P_node >=village_cpd_P[n]*visit_effect):
				
				#print distance_df[n][c.path[-1]]
				#print max_dist

				#print c.visits[n]
				#print max_visits_allowed

				#print "veh cap vs demand for Malaria was", c.veh_cap_M_node, village_cpd_M[n]
				#print "veh cap vs demand for Diarrhea was", c.veh_cap_D_node, village_cpd_D[n]
				#print "veh cap vs demand for Pneumonia was", c.veh_cap_P_node, village_cpd_P[n]

				if (c.LB<=upper_bound and distance_df[n][c.path[-1]]<=max_dist) and (c.visits[n] < max_visits_allowed) and (c.veh_cap_M_node >=village_cpd_M[n]*visit_effect) and (c.veh_cap_D_node >=village_cpd_D[n]*visit_effect) and (c.veh_cap_P_node >=village_cpd_P[n]*visit_effect):
					
					#print "passed test - ",n
					#if (distance_df[n][c.path[-1]]<=max_dist) :	
					Nodes_list.append(Node())
					Nodes_list[Node_count].id = Node_count
					Nodes_list[Node_count].time = t
					Nodes_list[Node_count].path = c.path+[n]
					#print Nodes_list[Node_count].path
					
					# after appending the path, update the stock-out values

					visits,max_visits_temp = Number_of_Visits(Nodes_list[Node_count].path,N)
					Nodes_list[Node_count].max_visits = max_visits_temp
					Nodes_list[Node_count].visits = visits
					
					new_visits = map(operator.sub, visits, prev_visits)


					#print visits, "List of visits\n"

					for j in range(1 ,len(new_visits)):
						sod_count_M[j] = max(0,(sod_count_M[j] - (new_visits[j]*visit_effect)))
						sod_count_D[j] = max(0,(sod_count_D[j] - (new_visits[j]*visit_effect)))
						sod_count_P[j] = max(0,(sod_count_P[j] - (new_visits[j]*visit_effect)))	

					Nodes_list[Node_count].MAD_val = mad(SSI_calculate(V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,sod_count_M,sod_count_D,sod_count_P))
					#print "this is the SSI MAD_score", Nodes_list[Node_count].MAD_val
					
					# Now that all parameters are updated and this child node was created because the LB of parent was < Some UB:
					# We create the LB of this child node.

					t_left = T-t
					t_half = int(0.5*T)

					lowest_LB = []

					if t_left >=t_half:
						for i in range(0,t_half+1):
							lowest_LB.append(calc_LB(max_visits_allowed, Nodes_list[Node_count].path, Nodes_list[Node_count].visits, T_horiz-i, t, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P, sod_count_M, sod_count_D, sod_count_P))
					else:
						for i in range(0,t_left+1):
							lowest_LB.append(calc_LB(max_visits_allowed, Nodes_list[Node_count].path, Nodes_list[Node_count].visits, T_horiz-i, t, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P, sod_count_M, sod_count_D, sod_count_P))
					
					#print "the LB array is", lowest_LB
					Nodes_list[Node_count].LB = min(lowest_LB)
					#print "the LB_selected is ", Nodes_list[Node_count].LB

					#Nodes_list[Node_count].LB = calc_LB(max_visits_allowed, Nodes_list[Node_count].path, Nodes_list[Node_count].visits, T_horiz, t, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P, sod_count_M, sod_count_D, sod_count_P)
					#print "LB ",Nodes_list[Node_count].LB, n
					#print "UB ",upper_bound
					#Nodes_list[Node_count].LB = calc_LB(max_visits_allowed, c.path+[n], visits, T_horiz, t, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P, sod_count_M, sod_count_D, sod_count_P)
					#For some reason using visits works better than the node visits. Apparently the LB calc changes it.

					## Here we update the Vehicle capacity
					#print "AAANEWWSS", Nodes_list[Node_count].path,n
					if n == 0:
						#print "Reload",Nodes_list[Node_count].path
						Nodes_list[Node_count].veh_cap_M_node = 100
						Nodes_list[Node_count].veh_cap_D_node = 100
						Nodes_list[Node_count].veh_cap_P_node = 100

					else:
						Nodes_list[Node_count].veh_cap_M_node = max(0,c.veh_cap_M_node - (village_cpd_M[n]*visit_effect))
						Nodes_list[Node_count].veh_cap_D_node = max(0,c.veh_cap_D_node - (village_cpd_D[n]*visit_effect))
						Nodes_list[Node_count].veh_cap_P_node = max(0,c.veh_cap_P_node - (village_cpd_P[n]*visit_effect))
						#print "cap M, D, P are: ", Nodes_list[Node_count].veh_cap_M_node,Nodes_list[Node_count].veh_cap_D_node,Nodes_list[Node_count].veh_cap_P_node

					if Nodes_list[Node_count].MAD_val<Z_best_overall:
							Z_best_overall = Nodes_list[Node_count].MAD_val
							Z_best_node_overall = Nodes_list[Node_count].id
							Z_best_overall_SSI = SSI_calculate(V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,sod_count_M,sod_count_D,sod_count_P)

					if t == T:
						if Nodes_list[Node_count].MAD_val<Z_best_t:
							Z_best_t = Nodes_list[Node_count].MAD_val
							upper_bound = deepcopy(Z_best_t)
							Z_best_node_t = Nodes_list[Node_count].id
							Z_best_t_SSI = SSI_calculate(V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,sod_count_M,sod_count_D,sod_count_P)
							#optimality_gap = Z_best_t - c.LB
							#optimality_gap_percent = (optimality_gap/c.LB)*100
							sod_M_best_t = deepcopy (sod_count_M)
							sod_D_best_t = deepcopy (sod_count_D)
							sod_P_best_t = deepcopy (sod_count_P)


					# now that all the values into the node are updated, append this node to the temp, nodes created
					temp_nodes_created.append(Nodes_list[Node_count])

					Node_count = Node_count + 1

					#reset all the SODs for the next iteration
					sod_count_M = deepcopy(village_sod_M)
					sod_count_D = deepcopy(village_sod_D)
					sod_count_P = deepcopy(village_sod_P)
					#print "this is", Node_count, t
				
						
				'''
				else:
					# if the feasibility conditions are not met, that child node is not created"
					print "\n pruned due to Infeasibility", c.id, c.path,n

					if c.LB>upper_bound:
						print "pruned due to bound where LB = %f and UB = %f" % (c.LB, upper_bound)

					if c.visits[n] >= max_visits_allowed:
						print "Infeasible due to Max visits exceeded for node: ", n, c.visits[n]
					if distance_df[n][c.path[-1]]>max_dist:
						print "infeasible due to distance constraint for node %d and parent node %d" % (n, c.path[-1])
					if (c.veh_cap_M_node <village_cpd_M[n]) or (c.veh_cap_D_node <village_cpd_D[n]) or (c.veh_cap_P_node <village_cpd_P[n]):
						print "infeasible due to capacity"
						print "veh cap vs demand for Malaria was", c.veh_cap_M_node, village_cpd_M[n]
					 	print "veh cap vs demand for Diarrhea was", c.veh_cap_D_node, village_cpd_D[n]
					 	print "veh cap vs demand for Pneumonia was", c.veh_cap_P_node, village_cpd_P[n]
				'''
				
	# for i in range(0,len(Nodes_list)):
	#  	Nodes_list[i].show()
	
	print "length of nodes list:",len(Nodes_list)

	Initial_MAD_score = mad(SSI_calculate(V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P))
	#print "\n Initial MAD_score is ", Initial_MAD_score
	# print "\n overall optimal is Node %d with a MAD score of %f at time %d" % (Z_best_node_overall,Z_best_overall,Nodes_list[Z_best_node_overall].time)
	print "\n At time t = %d, optimal is Node %d with a MAD score of %f" % (Nodes_list[Z_best_node_t].time,Z_best_node_t,Z_best_t)
	#print Nodes_list[Z_best_node_t].show()
	#print "\n Optimal overall", Z_best_overall
	#print Nodes_list[Z_best_node_overall].show()
	#print V
	#print "Optimality gap", optimality_gap
	#print "optimality gap percent", optimality_gap_percent
	#print "cpd", village_cpd_M,village_cpd_D,village_cpd_P
	return (Nodes_list[Z_best_node_t],Z_best_t,Z_best_t_SSI,sod_M_best_t,sod_D_best_t,sod_P_best_t,)

### Start of code

start_time = time.time()

## 2. Import datasets

datContent = [i.strip().split(',') for i in open("input_args.txt").readlines()]

#print datContent

row = int(sys.argv[1])

input_file 			= str(datContent[row][0])
upper_bound_input 	= float(datContent[row][1])
visit_effect 		= int(datContent[row][2])
T_horiz 			= int(datContent[row][3])
max_visits_allowed 	= int(datContent[row][4])
max_dist_allowed 	= float(datContent[row][5])

# print "\n start of code"

# print "Input file", input_file 			
# print "upper bound", upper_bound_input		
# print "visit effect", visit_effect 	
# print "T_horiz", T_horiz 			
# print "max_visits_allowed", max_visits_allowed 	
# print "max_distance_allowed", max_dist_allowed 	

# ### print termial output to a file
# f = str("output_%d.txt" % row)
# out = open("Output/%s" % f, 'w')
# sys.stdout = out


print "\n start of code"

print "Input file:           ", input_file 			
print "upper bound:          ", upper_bound_input		
print "visit effect:         ", visit_effect 	
print "T_horiz:              ", T_horiz 			
print "max_visits_allowed:   ", max_visits_allowed 	
print "max_distance_allowed: ", max_dist_allowed 	
#print input_file
input_village_data = 'Input_Data/Input/%s' % input_file

#max_visits_allowed = 2

no_bnb_runs = T_horiz/



length_of_bnb_run = 7

print "number of runs needed",no_bnb_runs

V=[]

village_id = defaultdict(make_dict)
village_lat_deg= defaultdict(make_dict)
village_long_deg = defaultdict(make_dict)
village_D1 = defaultdict(make_dict)
village_D2 = defaultdict(make_dict)
village_D3 = defaultdict(make_dict)
village_E = defaultdict(make_dict)
village_ap_M= defaultdict(make_dict)
village_ap_D= defaultdict(make_dict)
village_ap_P= defaultdict(make_dict)
village_sod_M= defaultdict(make_dict)
village_sod_D= defaultdict(make_dict)
village_sod_P= defaultdict(make_dict)
village_cpd_M= defaultdict(make_dict)
village_cpd_D= defaultdict(make_dict)
village_cpd_P= defaultdict(make_dict)
village_visited = defaultdict(make_dict)
#sod_count_M = defaultdict(make_dict)


depot_lat_deg = 0.40
depot_long_deg = 32.48	

village_SSI=defaultdict(make_dict)


village_data = parseCSVstring(input_village_data, returnJagged=False, fillerValue=-1, delimiter=',')
for i in range(1,len(village_data)):
	if (len(village_data[i]) == 16):
		tmp_ID = int(village_data[i][0])		
		tmp_long_deg = float(village_data[i][1])
		tmp_lat_deg = float(village_data[i][2])
		tmp_D1 = int(village_data[i][3])
		tmp_D2 = int(village_data[i][4])
		tmp_D3 = int(village_data[i][5])
		tmp_E = int(village_data[i][6])
		tmp_ap_M = float(village_data[i][7])
		tmp_ap_D = float(village_data[i][8])
		tmp_ap_P = float(village_data[i][9])
		tmp_sod_M = float(village_data[i][10])
		tmp_sod_D = float(village_data[i][11])
		tmp_sod_P = float(village_data[i][12])
		tmp_cpd_M = float(village_data[i][13])
		tmp_cpd_D = float(village_data[i][14])
		tmp_cpd_P = float(village_data[i][15])


		V.append(tmp_ID)
		village_id[tmp_ID]= tmp_ID
		village_lat_deg[tmp_ID] = tmp_lat_deg
		village_long_deg[tmp_ID] = tmp_long_deg
		village_D1 [tmp_ID] = tmp_D1
		village_D2 [tmp_ID] = tmp_D2
		village_D3 [tmp_ID] = tmp_D3
		village_E [tmp_ID] = tmp_E
		village_ap_M[tmp_ID] = tmp_ap_M
		village_ap_D[tmp_ID]= tmp_ap_D
		village_ap_P[tmp_ID]= tmp_ap_P
		village_sod_M[tmp_ID]= tmp_sod_M
		village_sod_D[tmp_ID]= tmp_sod_D
		village_sod_P[tmp_ID]= tmp_sod_P
		village_cpd_M[tmp_ID]= tmp_cpd_M
		village_cpd_D[tmp_ID]= tmp_cpd_D
		village_cpd_P[tmp_ID]= tmp_cpd_P
		#print loc_village_lat_rad
		#print loc_village_long_rad
		#print (village_id, village_name, loc_village_lat_rad, loc_village_long_rad)

	else:
		print 'ERROR: Row %d of customers_with_GPS.csv has %d columns of data.  I expected 6 columns (CUSTOMER_ID, CITY, STATE, DEMAND, LONG (degrees), LAT (degrees)).  Sorry things did not work out.  Bye.' % (i, len(customer_data[i]))
		exit()

original_SSI = SSI_calculate(V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P)

#print original_SSI

## 3. Run the MapQuest function to get distance matrix

# Use MapQuest to generate two pandas dataframes.
# One dataframe will contain a matrix of travel distances, 
# the other will contain a matrix of travel times.
coordList = []
locIDlist = []

coordList.append([depot_lat_deg,depot_long_deg])
locIDlist.append(0)

for i in V:
	coordList.append([village_lat_deg[i], village_long_deg[i]])
	locIDlist.append(i)

all2allStr	= 'allToAll:true' 
one2manyStr	= 'oneToMany:false'
many2oneStr	= 'manyToOne:false'

[distance_df, time_df] = genTravelMatrix(coordList, locIDlist, all2allStr, one2manyStr, many2oneStr)
max_dist_allowed = max(distance_df[0])

#initialize first vehicle
veh_1 = vehicles()

selected_node_i = []
best_node_prev_run = 0

Nodes_list = [Node()]
Nodes_list[0].id = 0
Nodes_list[0].time = 0
Nodes_list[0].path.append(0)
Nodes_list[0].MAD_val = 1000
Nodes_list[0].visits = [0]*(len(V)+1)
Nodes_list[0].current_run_visits = [0]*(len(V)+1)
Nodes_list[0].visits[0] = 1
Nodes_list[0].veh_cap_M_node = veh_1.cap_M
Nodes_list[0].veh_cap_D_node = veh_1.cap_D
Nodes_list[0].veh_cap_P_node = veh_1.cap_P


#calculate node Lower bound


o_t_left_node_0 = length_of_bnb_run
o_t_half_node_0 = int(0.5*length_of_bnb_run)

o_lowest_LB_node_0= []

for i in range(0,o_t_half_node_0):
 	o_lowest_LB_node_0.append(calc_LB(max_visits_allowed, Nodes_list[0].path, Nodes_list[0].visits, length_of_bnb_run-1, i, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P, village_sod_M, village_sod_D, village_sod_P))
 	#print o_lowest_LB
# #print "the LB array is", lowest_LB
Nodes_list[0].LB = min(o_lowest_LB_node_0)

#Nodes_list[0].LB = calc_LB(max_visits_allowed, Nodes_list[0].path, Nodes_list[0].visits, length_of_bnb_run, 0, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P)


o_t_left = T_horiz
o_t_half = int(0.5*T_horiz)

o_lowest_LB = []

# for i in range(0,o_t_half+1):
for i in range(0,o_t_half):
 	o_lowest_LB.append(calc_LB(max_visits_allowed, Nodes_list[0].path, Nodes_list[0].visits, T_horiz-1, i, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P, village_sod_M, village_sod_D, village_sod_P))
 	#print o_lowest_LB
# #print "the LB array is", lowest_LB
overall_LB = min(o_lowest_LB)



#overall_LB = calc_LB(max_visits_allowed, Nodes_list[0].path, Nodes_list[0].visits, T_horiz, 0, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P)
#overall_LB = calc_LB(max_visits_allowed, Nodes_list[0].path, Nodes_list[0].visits, 19, 0, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P)

print "overall LB", overall_LB
print "nodelist 0 LB,", Nodes_list[0].LB
#best_node = Nodes_list[0]
best_node_prev_run = Nodes_list[0]

upper_bound = []

print "this is NN solution"
nn_solution = nn_run(max_visits_allowed,best_node_prev_run.visits,max_dist_allowed,0,T_horiz,best_node_prev_run.path,distance_df,best_node_prev_run.veh_cap_M_node,best_node_prev_run.veh_cap_D_node,best_node_prev_run.veh_cap_P_node,V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P,village_cpd_M,village_cpd_D,village_cpd_P,visit_effect)

# if no_bnb_runs == max_visits_allowed:
# 	max_visits_allowed_per_run = 1
# elif no_bnb_runs<max_visits_allowed:
# 	max_visits_allowed_per_run = 

max_visits_added_per_run = [1,2,1]  
max_visits_allowed = 0
print "max distance", max(distance_df)

for i in range (0,no_bnb_runs):

	run_number = i
	start_time_run = time.time()
	print "This is run number:", i

	print"best node prev run's visits", best_node_prev_run.visits

	max_visits_allowed = max_visits_added_per_run[i]+ max_visits_allowed
	print "max_visits_allowed = ", max_visits_allowed

	#upper_bound.append(nn_run(max_visits_allowed,best_node_prev_run.visits,max_dist_allowed,run_number,length_of_bnb_run,best_node_prev_run.path,distance_df,best_node_prev_run.veh_cap_M_node,best_node_prev_run.veh_cap_D_node,best_node_prev_run.veh_cap_P_node,V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P,village_cpd_M,village_cpd_D,village_cpd_P,visit_effect))
	upper_bound.append(nn_run(max_visits_allowed,best_node_prev_run.visits,max_dist_allowed,run_number,length_of_bnb_run,best_node_prev_run.path,distance_df,best_node_prev_run.veh_cap_M_node,best_node_prev_run.veh_cap_D_node,best_node_prev_run.veh_cap_P_node,V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P,village_cpd_M,village_cpd_D,village_cpd_P,visit_effect))

	print "\n this is the upper bound", upper_bound[i]
	#best_node_prev_run.show()
	#print "\n this is the veh_cap being passed"
	
	#(best_node, node_score_bnb,best_node_SSI,village_sod_M_best_node,village_sod_D_best_node,village_sod_P_best_node)=solve_BNB(best_node_prev_run,upper_bound[i],length_of_bnb_run,max_dist_allowed,max_visits_allowed,visit_effect, distance_df,veh_1,V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P,village_cpd_M,village_cpd_D,village_cpd_P)
	(best_node, node_score_bnb,best_node_SSI,village_sod_M_best_node,village_sod_D_best_node,village_sod_P_best_node)=solve_BNB(best_node_prev_run,upper_bound[i],length_of_bnb_run,max_dist_allowed,max_visits_allowed,visit_effect, distance_df,veh_1,V,village_E,village_D1,village_D2,village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P,village_cpd_M,village_cpd_D,village_cpd_P)
	
	print("\n--- %s seconds --- is the time for this run" % (time.time() - start_time_run))
	village_sod_M = deepcopy(village_sod_M_best_node)
	village_sod_D = deepcopy(village_sod_D_best_node)
	village_sod_P = deepcopy(village_sod_P_best_node)
	selected_node_i.append(best_node)

	best_node.show()

	#print village_sod_M
	#print village_sod_D
	#print village_sod_P	
	best_node_prev_run = deepcopy(best_node)
	

	#print "CCC"
	o_t_left_node_a = length_of_bnb_run*(i+1)
	o_t_half_node_a = int(0.5*(length_of_bnb_run*(i+1)))

	o_lowest_LB_node_a= []
	#print "AAAA"
	#print (length_of_bnb_run)*(i+1)
	#print o_t_half_node_a+1
	for j in range((length_of_bnb_run)*i,(o_t_half_node_a+1+(length_of_bnb_run*i))):
		#print "This is inside the LB feeding loop", j
	 	o_lowest_LB_node_a.append(calc_LB(max_visits_allowed, Nodes_list[0].path, Nodes_list[0].visits, length_of_bnb_run*(i+1),j, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P, village_sod_M, village_sod_D, village_sod_P))
	
	print "\n best node SSI", best_node_SSI

	# #print "the LB array is", lowest_LB
	#print o_lowest_LB_node_a
	best_node_prev_run.LB = min(o_lowest_LB_node_a)
	print best_node_prev_run.LB
	#print "BBBB"



	#a = calc_LB(max_visits_allowed, best_node.path, best_node.visits, length_of_bnb_run*(i+1), length_of_bnb_run*i, V, village_E, village_D1, village_D2, village_D3,village_ap_M,village_ap_D,village_ap_P,village_sod_M,village_sod_D,village_sod_P)
	#best_node_prev_run.LB = a
	#print "best_node_new_LB",a


#print upper_bound

print best_node.show()



#print node_score_bnb


print "Problem SSI ", mad(original_SSI)
print "Z_best_t_MAD_SSI", best_node.MAD_val
print "Initial gini coefficient", gini(original_SSI)
print "final gini coefficient", gini(best_node_SSI)

print("\n--- %s seconds ---" % (time.time() - start_time))

optimality_gap = best_node.MAD_val - overall_LB
optimality_gap_percent = (optimality_gap/overall_LB)*100
print "Optimality gap and percentage", optimality_gap,optimality_gap_percent

percent_improvement_MAD = (((mad(original_SSI)-node_score_bnb)/mad(original_SSI))*100)

print ("\n percent impovement MAD is %s %%" % (((mad(original_SSI)-node_score_bnb)/mad(original_SSI))*100))
print ("\n percent impovement gini is %s %%\n" % (((gini(original_SSI)-gini(best_node_SSI))/gini(original_SSI))*100))

#print distance_df

#out.close()


############# Create Maps ##############################

map = folium.Map(location=[0.4, 32.48], zoom_start=10)

locations_visited = np.unique(best_node.path)
#locations_visited = np.unique([18, 18, 14, 7, 10, 19, 2, 16, 19,  2, 16, 19, 0, 14, 7, 10, 13])
#locations_visited = [7, 7, 3, 2, 2, 5]
village_lat_deg[0] = 0.4
village_long_deg[0] = 32.48

folium.Marker(location = [village_lat_deg[0], village_long_deg[0]], popup = 'Depot', icon=folium.Icon(color='black')).add_to(map)

for i in V:
	if i!=0:
		folium.Circle(location = [village_lat_deg[i], village_long_deg[i]], popup = str(i), radius = 50, color = 'red', fill = True, fill_color = 'red').add_to(map)

#print original_SSI
# for i in V:
# 	#print village_lat_deg[i], village_long_deg[i]
# 	#print original_SSI[i]
# 	if i not in locations_visited and (i!=0):
# 		folium.Circle(location = [village_lat_deg[i], village_long_deg[i]], popup = str(i), radius = 50, color = 'red', fill = True, fill_color = 'red').add_to(map)

for i in locations_visited:
		if i!=0:
		#print village_lat_deg[i], village_long_deg[i]
		#folium.Circle(location = [village_lat_deg[i], village_long_deg[i]], popup = str(i), radius = 200, color = 'green', fill = True, fill_color = 'green').add_to(map)
			folium.Marker(location = [village_lat_deg[i], village_long_deg[i]],popup = str(i),  icon=folium.Icon(color='green')).add_to(map)


tour_list = split_tours(best_node.path)

#tour_list = split_tours([0, 18, 18, 0, 14, 7, 10, 19, 0, 2, 16, 19, 0, 2, 16, 19, 0, 14, 7, 10, 13, 0])
#tour_list = split_tours([0, 7, 7, 3, 2, 2, 0, 5])
colors = ["red", "blue", 'green', 'yellow', 'black','orange', 'pink', 'gray', 'purple','cadetblue']

print "this is tour list", tour_list
col_index = 0
for tour in tour_list:
	#print tour
	i = tour[0]
	for j in tour[1:]:
		startCoords = '%f,%f' % (village_lat_deg[i],village_long_deg[i])
		endCoords = '%f,%f' % (village_lat_deg[j],village_long_deg[j])
		myShapepoints = genShapepoints(startCoords, endCoords)
		#print colors[col_index]	       
		folium.PolyLine(myShapepoints, color=colors[col_index], weight=8.5, opacity=0.5).add_to(map)	
		i = j
	
	
 	col_index = col_index + 1
	


# i = best_node.path[0]
# for j in best_node.path[1:]:
#     startCoords = '%f,%f' % (village_lat_deg[i], village_long_deg[i])
#     endCoords = '%f,%f' % (village_lat_deg[j], village_long_deg[j])
#     myShapepoints = genShapepoints(startCoords, endCoords)	       
#     folium.PolyLine(myShapepoints, color="red", weight=8.5, opacity=0.5).add_to(map)	
#     i = j


for i in V:
	if i != 0:
		folium.Circle(location=[village_lat_deg[i], village_long_deg[i]], radius=original_SSI[i-1]*15000, color='blue', fill=True, fill_color='blue').add_to(map)

#map.save('Toy_Impl_enum_Rand.html')

#print locations_visited

