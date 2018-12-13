## written by: Biplab Bhattacharya
## Last modified: September 27, 2017
## 
## The purpose of this code is to read in inputs from a different file and calculate the SSI for all test cases

#from parseCSVstring import *
import sys			# Allows us to capture command line arguments
import csv
import math
import pandas as pd
import numpy as np
from collections import defaultdict

# define a function that will calculate the SSI for us
def SSI_calculate(settlements,e,d_1,d_2,d_3,ap_1,ap_2,ap_3,so_1,so_2,so_3):

	sum_e 		= 0.0
	sum_d_1 	= 0.0
	sum_d_2 	= 0.0
	sum_d_3 	= 0.0
	sum_ap_1 	= 0.0
	sum_ap_2 	= 0.0
	sum_ap_3 	= 0.0
	sum_so_1 	= 0.0
	sum_so_2 	= 0.0
	sum_so_3 	= 0.0

	# eps = 0.25
	# alpha = 0.25
	# beta = 0.25
	# delta = 0.25
	eps = 0.2
	alpha = 0.2
	beta = 0.4
	delta = 0.2
	
	#print "This is the length of settlements",len(settlements)

	for i in range(1,len(settlements)+1):
		
		sum_e 		= sum_e + e[i]
		sum_d_1		= sum_d_1 + d_1[i]
		sum_d_2		= sum_d_2 + d_2[i]
		sum_d_3		= sum_d_3 + d_3[i]
		sum_ap_1 	= sum_ap_1 + ap_1[i]
		sum_ap_2 	= sum_ap_2 + ap_2[i]
		sum_ap_3 	= sum_ap_3 + ap_3[i]
		sum_so_1 	= sum_so_1 + so_1[i]
		sum_so_2 	= sum_so_1 + so_1[i]
		sum_so_3 	= sum_so_1 + so_1[i]
	

	# print sum_e
	# print sum_d_1
	# print sum_d_1
	# print sum_d_1
	# print sum_ap_1
	# print sum_ap_2
	# print sum_ap_3
	# print sum_so_1
	# print sum_so_2
	# print sum_so_3



	SSI = []


	for i in range(1,len(settlements)+1):
		
		val= (eps*(float(e[i])/sum_e))+(\
			(((alpha*(float(ap_1[i])/sum_ap_1))+(beta*(float(so_1[i])/sum_so_1))+(delta*(float(d_1[i])/sum_d_1)))/3)+\
			(((alpha*(float(ap_2[i])/sum_ap_2))+(beta*(float(so_2[i])/sum_so_2))+(delta*(float(d_2[i])/sum_d_2)))/3)+\
			(((alpha*(float(ap_3[i])/sum_ap_3))+(beta*(float(so_3[i])/sum_so_3))+(delta*(float(d_3[i])/sum_d_3)))/3))
		
		SSI.append(val)

	#print len(SSI)
	#print SSI
		#print val
	return(SSI)



'''

# Read the values from the input file:

#SSI_input_file = 'SSI_input.csv'

settlement_data = pd.read_csv('SSI_testing_py.csv')


SSI = {}
tempkey= {}
tempval = {}
str = ''
str1 = ''

for i in range(0,max(settlement_data['TC_num'])+1):

#for i in range(0,1):
	TC = settlement_data[settlement_data['TC_num'] == i]
	TC = TC.reset_index(drop=True)
	settlements = TC['settlement']
	e = TC['E']
	d_1 = TC['D_1']
	d_2 = TC['D_2']
	d_3 = TC['D_3']
	ap_1 = TC['a_1']
	ap_2 = TC['a_2']
	ap_3 = TC['a_3']
	so_1 = TC['so_1']
	so_2 = TC['so_2']
	so_3 = TC['so_3']

	#print TC
	total_settlements = max(settlements)
	

	SSI[i] = SSI_calculate(settlements,e,d_1,d_2,d_3,ap_1,ap_2,ap_3,so_1,so_2,so_3)

	tempkey[i] =np.argsort(SSI[i])
	#print tempkey[i]
	
	tempval[i] =np.sort(SSI[i])

	#print tempval[i]
	#numpy.savetxt("foo.csv",tempkey, delimiter =",")
	#numpy.savetxt("foo.csv",tempval, delimiter =",")
	#str += '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f \n' % (SSI[i][0],SSI[i][1],SSI[i][2],SSI[i][3],SSI[i][4],SSI[i][5],SSI[i][6],SSI[i][7],SSI[i][8],SSI[i][9],SSI[i][10],SSI[i][11],SSI[i][12],SSI[i][13],SSI[i][14],SSI[i][15],SSI[i][16],SSI[i][17],SSI[i][18],SSI[i][19])
	
	#This worked

	#str1 += '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d \n' % (tempkey[i][0],tempkey[i][1],tempkey[i][2],tempkey[i][3],tempkey[i][4],tempkey[i][5],tempkey[i][6],tempkey[i][7],tempkey[i][8],tempkey[i][9],tempkey[i][10],tempkey[i][11],tempkey[i][12],tempkey[i][13],tempkey[i][14],tempkey[i][15],tempkey[i][16],tempkey[i][17],tempkey[i][18],tempkey[i][59])
	#str1 += '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f \n' % (tempval[i][0],tempval[i][1],tempval[i][2],tempval[i][3],tempval[i][4],tempval[i][5],tempval[i][6],tempval[i][7],tempval[i][8],tempval[i][9],tempval[i][10],tempval[i][11],tempval[i][12],tempval[i][13],tempval[i][14],tempval[i][15],tempval[i][16],tempval[i][17],tempval[i][18],tempval[i][59])

	# till here


	#loop that prints all ranked SSI indexes in one line
### USE SECTION A  FOR INDEXES AND SECTION B FOR VALUES

#SECTION A

	# for j in range(0, len(settlements)-1):
	#   	str1 += '%d,' % (tempkey[i][j])
	 	
	# str1 += '%d'% (tempkey[i][len(settlements)-1])
	# str1 += '\n'

#SECTION B
	
	#loop that prints the ssi values in one line
	for j in range(0, len(settlements)-1):
	 	str1 += '%f,' % (tempval[i][j])
	 	
	str1 += '%f'% (tempval[i][len(settlements)-1])
	str1 += '\n'



print SSI

#print np.argsort(SSI[1])
#print np.sort(SSI[1])
#print SSI[1]
#print SSI[2]
#print SSI[3]
#myFile = open('SSI_OP.csv','a')
#myFile.write(str)
#myFile.close()	


#print str1



## Uncomment this section - we use this to print output into an excel file

myFile = open('SSI_Output_Apr27.csv','a')
myFile.write(str1)
myFile.close()	

##

#print SSI

#print settlement_data['TC_num'][0]

#for i in range(0,len())


'''
'''
	for i in settlements:
		SSI[i] = (eps*(e[i]/sum_e))+((\
			(alpha*(ap_1[i]/sum_ap_1))+(beta*(so_1[i]/sum_so_1))+(delta*(d_1[i]/sum_d_1))\
			(alpha*(ap_2[i]/sum_ap_2))+(beta*(so_2[i]/sum_so_2))+(delta*(d_2[i]/sum_d_2))\
			(alpha*(ap_3[i]/sum_ap_3))+(beta*(so_3[i]/sum_so_3))+(delta*(d_3[i]/sum_d_3)))/3)

		# new value that worked
		val= (eps*(float(e[i])/sum_e))+((\
			(alpha*(float(ap_1[i])/sum_ap_1))+(beta*(float(so_1[i])/sum_so_1))+(delta*(float(d_1[i])/sum_d_1))+\
			(alpha*(float(ap_2[i])/sum_ap_2))+(beta*(float(so_2[i])/sum_so_2))+(delta*(float(d_2[i])/sum_d_2))+\
			(alpha*(float(ap_3[i])/sum_ap_3))+(beta*(float(so_3[i])/sum_so_3))+(delta*(float(d_3[i])/sum_d_3)))/3)

'''


'''
MAYbe this will work
for i in range (0,100):
	with open('mycsvfile.csv', 'wb') as f:  
		w = csv.DictWriter(f, np.argsort(SSI[i]))
		w.writeheader()
		w.writerow(np.sort(SSI[i]))
'''
