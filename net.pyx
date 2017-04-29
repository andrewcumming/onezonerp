cimport cython

cdef extern from "math.h":
	double exp(double m)
	double log(double m)
 
import numpy as np
import itertools
#from math import exp,log

elements = { 'h':1, 'he':2, 'c':6, 'n':7, 'o':8, 'f':9, 'ne':10, 'na':11, 'mg':12, 'al':13,
	'si':14, 'p':15, 's':16, 'cl':17, 'ar':18, 'k':19, 'ca':20, 'sc':21, 'ti':22, 'v':23, 
	'cr':24, 'mn':25, 'fe':26, 'co':27, 'ni':28, 'cu':29, 'zn':30, 'ga':31, 'ge':32, 
	'as':33, 'se':34, 'br':35, 'kr':36, 'rb':37, 'sr':38, 'y':39, 'zr':40, 'nb':41, 'mo':42,
	'tc':43, 'ru':44, 'rh':45, 'pd':46, 'ag':47, 'cd':48, 'in':49, 'sn':50, 'sb':51,}

def make_species_list(input_list):
	output_list = []
	for spec in input_list.split(' '):
		f = [ ''.join(x) for _,x in itertools.groupby(spec,key=str.isdigit)]
		if len(f) == 2:
			output_list.append(spec)
		if len(f) == 4:
			a1, a2 = int(f[1]), int(f[3])
			for i in range(a2-a1+1):
				output_list.append(f[0]+str(a1+i))
	return output_list
	
def get_AZ(species):
	AA = np.array([])
	ZZ = np.array([])
	for spec in species:
		f = [ ''.join(x) for _,x in itertools.groupby(spec,key=str.isdigit)]
		AA = np.append(AA, int(f[1]))
		ZZ = np.append(ZZ, elements[f[0]])
	return AA,ZZ

@cython.boundscheck(False)
@cython.cdivision(True)
def calculate_rate(double T9,double a0,double a1,double a2,double a3,double a4,double a5,double a6):
	cdef double T913, T953, rate
	T913 = T9**(1.0/3.0)
	T953 = T9**(5.0/3.0)
	rate = exp(a0 + a1/T9 + a2/T913 + a3*T913 + a4*T9 + a5*T953 + a6*log(T9))
	return rate

def read_rates(species):
	# Reads the Reaclib library to get rates involving the species in 
	# our network
	rates = []
	# open REACLIB file
	fp = open('20170427default','r')	
	count = 0
	include_rate = False
	chapter = 0
	for line in fp:
		if count==0:
			if line[0]!=' ':
				# if the first character is an integer, this is a header
				# so extract the chapter number
				chapter = int(line[0:2])
			else:
				# get the nuclides involved
				nuclides = [line[5*(i+1):5*(i+2)].strip() for i in range(5)]
				nuclides = list(filter(None,nuclides))   # remove empty strings
				try:
					nuclides[nuclides.index('p')] = 'h1'
				except:
					pass
				# make sure they are all in our network
				if all(nuc in species for nuc in nuclides):
					# if so, include the rate and get the Q value
					include_rate = True
					Q = float(line[52:65])*9.64e17
		# extract the reaction coefficients from the next two lines
		if include_rate:
			if count==1:
				a0,a1,a2,a3 = [float(line[13*i:13*(i+1)]) for i in range(4)]
			if count==2:
				a4,a5,a6 = [float(line[13*i:13*(i+1)]) for i in range(3)]
				ind = [species.index(spec) for spec in nuclides]
				n_reac_vec = (1,1,1,2,2,2,2,3)	
				n_prod_vec = (1,2,3,1,2,3,4,1)
				rates.append((n_reac_vec[chapter-1],n_prod_vec[chapter-1],nuclides,ind,(a0,a1,a2,a3,a4,a5,a6),Q))
				include_rate=False
		count = (count+1) % 3
	fp.close()
	return rates

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_dYdt(double rho,double T,species,double [:] Y,rates):
	cdef int n_reac, n_prod, count, i
	cdef double ydot, eps, rateval
	cdef double[:] dYdt=np.zeros(len(species))
	eps = 0.0
	for rate in rates:   
		n_reac = rate[0]
		n_prod = rate[1]
		rateval = calculate_rate(T*1e-9,*rate[4])
		ind = rate[3]   # the indices of the species involved in this reaction
		
		ydot = rateval*Y[ind[0]]
		if n_reac>1:
			ydot *= rho*Y[ind[1]]    # note: currently no check for identical particles
			if n_reac>2:
				ydot *= rho*Y[ind[2]] / 6.0    # assume this is triple alpha

		count = 0
		for i in ind:
			if count<n_reac:
				dYdt[i] = dYdt[i] - ydot
			else:
				dYdt[i] = dYdt[i] + ydot
			count+=1
			
		eps += rate[5]*ydot    # rate[4] is the Qvalue in MeV/mu
	return dYdt, eps
