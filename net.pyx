cimport cython

cdef extern from "math.h":
	double exp(double m)
	double log(double m)
	double sqrt(double m)
 
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
@cython.cdivision(True)
def calculate_dYdt(double rho,double T,double Ye,species,double [:] Y,double [:] AA,double [:] ZZ,rates):
	cdef int n_reac, n_prod, count, i, i0, i1, i2
	cdef double ydot, eps, rateval, Q_val
	cdef double a0, a1, a2, a3, a4, a5, a6
	cdef double[:] dYdt=np.zeros(len(species))
	eps = 0.0
	for rate in rates:   
		n_reac = rate[0]
		n_prod = rate[1]
		Q_val = rate[5]
		a0, a1, a2, a3, a4, a5, a6 = rate[4]
		rateval = calculate_rate(T*1e-9, a0, a1, a2, a3, a4, a5, a6)

		ind = rate[3]   # the indices of the species involved in this reaction
		i0 = ind[0]
		if n_reac>1:
			i1 = ind[1]
			if n_reac>2:
				i2 = ind[2]
		
		ydot = rateval*Y[i0]
		if n_reac>1:
			ydot *= rho*Y[i1]*exp(screening(T, rho, Ye, ZZ[i0], ZZ[i1], AA[i0], AA[i1]))
						# note: currently no check for identical particles
			if n_reac>2:
				#ydot *= rho*Y[ind[2]] / 6.0    # assume this is triple alpha
				ydot = eps_triple_alpha(rho, T, Ye) * Y[i0]**3 / Q_val

		count = 0
		for i in ind:
			if count<n_reac:
				dYdt[i] = dYdt[i] - ydot
			else:
				dYdt[i] = dYdt[i] + ydot
			count+=1
			
		eps += Q_val*ydot    # rate[4] is the Qvalue in MeV/mu
	return dYdt, eps

@cython.cdivision(True)
cdef double calculate_rate(double T9,double a0,double a1,double a2,double a3,double a4,double a5,double a6):
	# numerically evaluates the rate
	cdef double T913, T953, rate
	T913 = T9**(1.0/3.0)
	T953 = T9**(5.0/3.0)
	rate = exp(a0 + a1/T9 + a2/T913 + a3*T913 + a4*T9 + a5*T953 + a6*log(T9))
	return rate

@cython.cdivision(True)
cdef double screening(double T, double rho, double Ye, double Z1, double Z2, double A1, double A2):
	# screening factor from Ogata et al 1993
	# eqs. (19)--(21)
	cdef double T8, gam, r6, r613, f, lgam, hf, A, tau, Z13,Z23, QQ, DD,BB
	
	A = A1*A2/(A1+A2)
	T8 = 1e-8*T
	r6 = Ye*rho/1e6
	r613 = r6**(1.0/3.0)
	Z13 = Z1**(1.0/3.0)
	Z23 = Z2**(1.0/3.0)
	
	hf = 0.25*(Z13+Z23)**3/(Z1+Z2)
	gam = 0.23 * r613/T8 * 2.0*Z1*Z2/(Z13+Z23)
	lgam=log(gam)
	
	tau = 9.18*(Z1*Z2)**(2.0/3.0)*(A/T8)**(1.0/3.0)
	f = 3.0*gam/tau
	
	QQ = (1.148-0.00944*lgam-0.000168*lgam*lgam)*gam-0.15625*gam*f*f*hf+(-0.18528+0.03863*lgam+0.01095*f)*gam*f*f*f
	
	return QQ

@cython.cdivision(True)
cdef double eps_triple_alpha(double rho, double T, double Ye):
	# Fushiki and Lamb's fit to the triple alpha rate
	# includes screening and is good for pycnonuclear regime
	cdef double r6, T6, r613, r616, T613, T623, T653, T632, T612, u, G1, G2, f1, f2, f3, f4, u32
	r6 = 1e-6*rho*Ye*2.0
	T6 = 1e-6*T
	
	r613 = r6**(1.0/3.0)
	r616 = sqrt(r613)
	T613 = T6**(1.0/3.0)
	T623 = T613*T613
	T653 = T6*T623
	T632 = T6**1.5
	T612 = sqrt(T6)
	
	u = 1.35*r613/T623
	u32 = u**1.5
	
	f1 = exp(60.492*r613/T6)
	f2 = exp(106.35*r613/T6)
	if r6 < 5.458e3:
		f3 = exp(-1065.1/T6)/T632
	else:
		f3 = 0.0
	if r6 < 1.836e4:
		f4 = exp(-3336.4/T6)/T632
	else:
		f4 = 0.0
  		
	if (u < 1):
		G1 = f1*(f3+16.16*exp(-134.92/T613)/(T623*((1-4.222e-2*T623)**2+2.643e-5*T653)))
		G2=f2*(f4+244.6*(1+3.528e-3*T623)**5*exp(-235.72/T613)/(T623*((1-2.807e-2*T623)**2+2.704e-6*T653)))
	else:
		G1=f1*f3+1.178*(1+(1.0/u32))*exp(-77.554/r616)/(T612*((1-5.680e-2*r613)**2+8.815e-7*T6*T6))
		G2=f2*f4+13.48*(1+(1.0/u32))*(1+5.070e-3*r613)**5*exp(-135.08/r616)/(T612*((1-3.791e-2*r613)**2+5.162e-8*T6*T6))
  		
	r6=r6/(2.0*Ye)
	# multiply the following by Y_He^3 to get eps
	return 5.12e29*64*r6*r6*G1*G2
