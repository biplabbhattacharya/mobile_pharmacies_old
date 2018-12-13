#####################

# Code by Biplab Bhattacharya, Sept 26 2017
# 

#####################

## This code is written to create test cases for testing the SSI formula
## in this part of the code, we specify the following parameters:
# Number of test cases (User input)
# Number of settlemenments (user input)
# number of disease types
# d_min_m and d_max_m (demographic factor for settlement and disease type)
# e_min and e_max (demographic factor for settlement)
# pop_min and pop_max (population values for min and max)
# prev_min_m and prev_max_m (min and max prevalance rates for disease type)
# so_min_m and so_max_m (min and max observed stock out days for each disease type)


# to run the code: python SSI_testcase_builder.py <#testcases> <#settlements>


# 0. Import required functions

import sys			# Allows us to capture command line arguments
import csv
import math
import random
import pandas as pd
from collections import defaultdict
from location_coordinate_generator import locations_gen


# 1. Define command line inputs
if (len(sys.argv) == 3):
	testcases_num		= int(sys.argv[1])		# Ex:  20
	settlements_num		= int(sys.argv[2])		# Ex: 25
else:
	print 'ERROR: You passed', len(sys.argv)-1, 'input parameters.'
	quit()

# 2. Set all parameters

d_min_1 = 5
d_max_1 = 13
d_min_2 = 6
d_max_2 = 36
d_min_3 = 50
d_max_3 = 70
e_min = 2.6
e_max = 29
pop_min = 250 #0.2 because 20% of the population are under 5
pop_max = 1000 #0.2 because of the 20% of the population are under 5
prev_min_1 = 0.1 # percentage value eg. 0.1
prev_max_1 = 0.3
prev_min_2 = 0.15
prev_max_2 = 0.34
prev_min_3 = 0.23
prev_max_3 = 0.28
so_min_1 = 10
so_max_1 = 23
so_min_2 = 5
so_max_2 = 19
so_min_3 = 7
so_max_3 = 28


'''
eps = 
alpha = 
gamma = 
delta = 

param_sum = eps+alpha+gamma+delta

if (param_sum!=1):
	print 'Error, your parameters do not add up to 1', param_sum
	quit()
'''

## define function for node

def make_dict():
	return defaultdict(make_dict)


class make_node:
	def __init__(self, testcase, settlement_num, D_1,D_2,D_3,E, pop, prev_1, prev_2, prev_3, a_1, a_2, a_3, so_1,so_2,so_3):
		# Set node[nodeID]
		self.testcase	= testcase
		self.settlement_num	= settlement_num
		self.D_1	= D_1
		self.D_2	= D_2
		self.D_3	= D_3
		self.E 		= E
		self.pop 	= pop
		self.prev_1 = prev_1
		self.prev_2 = prev_2
		self.prev_3 = prev_3
		self.a_1 	= a_1
		self.a_2	= a_2
		self.a_3	= a_3
		self.so_1 	= so_1
		self.so_2 	= so_2
		self.so_3 	= so_3




number_of_locations = settlements_num
depot_lat = 0.40
depot_long = 32.48
approx_range = 15000 #(in meters)
gps_loc = locations_gen(number_of_locations, depot_lat, depot_long, approx_range)

#print "settlements_num", settlements_num

#print gps_loc

# for j in range (0,settlements_num):
# 	print gps_loc[j+1][0]
# 	print gps_loc[j+1][1]

## initialize dictionary
## create csv output file
str = 'settlement_num,longitude,latitude,D_1,D_2,D_3,E,a_1,a_2,a_3,so_1,so_2,so_3,demand_1,demand_2,demand_3 \n'

for i in range(0, testcases_num):
	for j in range (0,settlements_num):
		
		percentage_children = random.uniform(0.15,0.25) #Since average population is 20%

		#testcase = i
		settlement_num = j+1
		latitude = gps_loc[j+1][0]
		longitude = gps_loc[j+1][1]
		D_1 = random.randint(d_min_1,d_max_1)
		D_2 = random.randint(d_min_2,d_max_2)
		D_3 = random.randint(d_min_3,d_max_3)
		E = random.uniform(e_min,e_max)
		pop = random.randint(pop_min,pop_max)*percentage_children
		prev_1 = random.uniform(prev_min_1,prev_max_1)
		prev_2 = random.uniform(prev_min_1,prev_max_2)
		prev_3 = random.uniform(prev_min_1,prev_max_3)
		a_1 = math.ceil(pop*prev_1)
		a_2 = math.ceil(pop*prev_2)
		a_3 = math.ceil(pop*prev_3)
		so_1 = random.randint(so_min_1,so_max_1)
		so_2 = random.randint(so_min_2,so_max_2)
		so_3 = random.randint(so_min_3,so_max_3)
		dem_1 = random.randint(1,a_1)
		dem_2 = random.randint(1,a_2)
		dem_3 = random.randint(1,a_3)

		#print settlement_num,latitude ,longitude,D_1,D_2,D_3,E,pop,prev_1,prev_2, prev_3, a_1, a_2, a_3, so_1,so_2,so_3

		
		#str += '%d, %d,%f, %f %d, %d, %d, %d,%d,  %f, %f,%f, %d,%d,%d,%d,%d,%d \n' % (testcase, settlement_num,latitude, longitude,D_1,D_2,D_3,E,pop,prev_1,prev_2, prev_3, a_1, a_2, a_3, so_1,so_2,so_3)
		
		#ignoring testcase number
		str += ' %d,%f, %f,%d, %d, %d, %d,%d,%d,%d,%d, %d,%d,%d,%d,%d \n' % (settlement_num,longitude,latitude,D_1,D_2,D_3,E,a_1,a_2,a_3,so_1,so_2,so_3,dem_1,dem_2,dem_3)
myFile = open('38_dataset_15.csv','a')
myFile.write(str)
myFile.close()		
		#node.append(testcase, settlement_num,D_1,D_2,D_3,pop,prev_1,prev_2, prev_3, a_1, a_2, a_3, so_1,so_2,so_3)
		#node[i][j] = make_node(i, j,D_1,D_2,D_3, E, pop, prev_1, prev_2, prev_3, a_1, a_2, a_3, so_1,so_2,so_3)
		#print node[i][j]

