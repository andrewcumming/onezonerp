#cython: profile=False

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

# some networks
nucs = {}
# small network while developing
nucs['48'] = 'h1 he4 o14-16 c12 n13-15 f17-18 ne18-19 na20-21 mg21-22 al22-25 si23-27 p27-30 s29-31 cl32-34 ar33-35 k36-38 ca37-39 sc40-42 ti41-43 v44-46 cr45-47 mn48'
# the smaller rp net from MESA  (~140 isotopes)
nucs['56'] = 'h1 he4 o14-18 c12-13 n13-15 f17-19 ne18-21 na20-23 mg21-25 al22-27 si24-30 p26-31 s27-34 cl30-35 ar31-38 k35-39 ca36-44 sc39-45 ti40-47 v43-49 cr44-52 mn47-53 fe48-56 co51-56 ni52-56'
# bigger (~200 isotopes)
nucs['78'] = 'h1 he4 o14-18 c12-13 n13-15 f17-19 ne18-21 na20-23 mg21-25 al22-27 si24-30 p26-31 s27-34 cl30-35 ar31-38 k35-39 ca36-44 sc39-45 ti40-47 v43-49 cr44-52 mn47-53 fe48-56 co51-56 ni52-57 cu54-63 zn55-66 ga59-67 ge60-68 as64-69 se65-72 br68-73 kr69-74 rb73-77 sr74-78'
# biggest (~300 isotopes)
nucs['106'] = 'h1 he4 o14-18 c12-13 n13-15 f17-19 ne18-21 na20-23 mg21-25 al22-27 si24-30 p26-31 s27-34 cl30-35 ar31-38 k35-39 ca36-44 sc39-45 ti40-47 v43-49 cr44-52 mn47-53 fe48-56 co51-56 ni52-57 cu54-63 zn55-66 ga59-67 ge60-68 as64-69 se65-72 br68-73 kr69-74 rb73-77 sr74-78 y77-82 zr78-83 nb81-85 mo82-86 tc85-88 ru86-91 rh89-93 pd90-94 ag94-98 cd95-99 in98-104 sn99-105 sb106'

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
	# number of reactants and products for each REACLIB chapter
	n_reac_vec = (1,1,1,2,2,2,2,3)	
	n_prod_vec = (1,2,3,1,2,3,4,1)
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
				if len(ind)>4:
					print("more than 4 reactants and products!!")
				else:
					ind = ind + [0]*(4-len(ind))
				rates.append((n_reac_vec[chapter-1],n_prod_vec[chapter-1],ind[0],ind[1],ind[2],ind[3],a0,a1,a2,a3,a4,a5,a6,Q))
				include_rate=False
		count = (count+1) % 3
	fp.close()
	return rates

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculate_dYdt(double rho,double T,double Ye,double [:] Y,double [:] AA,double [:] ZZ,rates):
	cdef int n_reac, n_prod, count, i, i0, i1, i2, i3
	cdef double ydot, eps, rateval, Q_val
	cdef double a0, a1, a2, a3, a4, a5, a6
	cdef double[:] dYdt=np.zeros(len(Y))
	cdef double T9, T913, T953, T9log, r613
	cdef tuple rate
	eps = 0.0
	T9 = 1e-9*T
	T913 = T9**(1.0/3.0)
	T953 = T9**(5.0/3.0)
	T9log = log(T9)
	r613 = (Ye*rho/1e6)**(1.0/3.0)
	done_triple_alpha = False
	for i in range(len(rates)): 
		n_reac, n_prod, i0, i1, i2, i3, a0, a1, a2, a3, a4, a5, a6, Q_val = rates[i]

		rateval = calculate_rate(T9,T913,T953,T9log, a0, a1, a2, a3, a4, a5, a6)

		if n_reac == 1:
			ydot = rateval*Y[i0]
			dYdt[i0] += -ydot
			dYdt[i1] += ydot
			if n_prod >= 2: 
				dYdt[i2] += ydot
			if n_prod == 3:
				dYdt[i3] += ydot
		if n_reac == 2:
			ydot = rateval*Y[i0]*rho*Y[i1]*exp(screening(T9, r613, Ye, ZZ[i0], ZZ[i1], AA[i0], AA[i1]))
			dYdt[i0] += -ydot
			dYdt[i1] += -ydot
			dYdt[i2] += ydot
			if n_prod == 2: 
				dYdt[i3] += ydot
		if n_reac == 3:
			# because we are hard coding triple alpha, we need to make sure we only add this once 
			# (the REACLIB library has >1 entry for triple alpha)
			if not done_triple_alpha:
				ydot = eps_triple_alpha(rho, T, Ye) * Y[i0]**3 / Q_val
				dYdt[i0] += -ydot
				dYdt[i1] += -ydot
				dYdt[i2] += -ydot
				dYdt[i3] += ydot
				done_triple_alpha = True
			
		eps += Q_val*ydot
	return dYdt, eps

@cython.cdivision(True)
cdef double calculate_rate(double T9,double T913,double T953,double T9log,double a0,double a1,double a2,double a3,double a4,double a5,double a6):
	# numerically evaluates the rate
	#cdef double T913, T953, rate
	#T913 = T9**(1.0/3.0)
	#T953 = T9**(5.0/3.0)
	rate = exp(a0 + a1/T9 + a2/T913 + a3*T913 + a4*T9 + a5*T953 + a6*log(T9))
	return rate

@cython.cdivision(True)
cdef double screening(double T9, double r613, double Ye, double Z1, double Z2, double A1, double A2):
	# screening factor from Ogata et al 1993
	# eqs. (19)--(21)
	cdef double gam, f, lgam, hf, A, tau, Z13,Z23, QQ, DD,BB
	
	A = A1*A2/(A1+A2)
	Z13 = Z1**(1.0/3.0)
	Z23 = Z2**(1.0/3.0)
	
	hf = 0.25*(Z13+Z23)**3/(Z1+Z2)
	gam = 0.023 * r613/T9 * 2.0*Z1*Z2/(Z13+Z23)
	lgam=log(gam)
	
	tau = 9.18*(Z1*Z2)**(2.0/3.0)*(0.1*A/T9)**(1.0/3.0)
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
