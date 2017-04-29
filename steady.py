#----------------------------------------------------------------------------
# Steady-state rp-process burning on an accreting neutron star
# as in Schatz et al. (1999)
#
# Nuclear rates are from REACLIB (see net.py)
# Not included:
# * neutrino losses
# * screening
# * density-dependent weak rates
# * conduction in the opacity
# * free-free Gaunt factor has been set to 1
#
#----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import optimize
import itertools
import eos
import net
import time

def derivs(Y,t):
	T = Y[-2]
	F = Y[-1]
	P = grav * mdot * t
	Ye = sum(Y[:-2]*ZZ)
	Yi = sum(Y[:-2])
	rho = eos.find_rho(P,T,Ye,Yi)

	# abundance derivatives
	dYdt, eps = net.calculate_dYdt(rho,T,species,Y[:-2],rates)

	# temperature gradient
	dTdt = mdot * 3.0*eos.kappa(rho,T,Ye)*F / (4.0*arad*clight*T**3)

	# dF/dt
	dFdt = -mdot * eps

	# return all the derivatives together
	dYdt = np.append(dYdt,dTdt)
	dYdt = np.append(dYdt,dFdt)
	return dYdt

# parameters
grav = 2e14
mdot_Edd = 8.8e4
arad = 7.5657e-15
clight = 3e10

mdot = 0.5 * mdot_Edd

# ----- set up network -----
# small network while developing
#species = net.make_species_list('h1 he4 o14-16 c12 n13-15 f17-18 ne18-19 na20-21 mg21-22 al22-25 si23-27 p27-30 s29-31 cl32-34 ar33-35 k36-38 ca37-39 sc40-42 ti41-43 v44-46 cr45-47 mn48')
# the smaller rp net from MESA  (~140 isotopes)
#species = net.make_species_list('h1 he4 o14-18 c12-13 n13-15 f17-19 ne18-21 na20-23 mg21-25 al22-27 si24-30 p26-31 s27-34 cl30-35 ar31-38 k35-39 ca36-44 sc39-45 ti40-47 v43-49 cr44-52 mn47-53 fe48-56 co51-56 ni52-56')
# bigger (~200 isotopes)
species = net.make_species_list('h1 he4 o14-18 c12-13 n13-15 f17-19 ne18-21 na20-23 mg21-25 al22-27 si24-30 p26-31 s27-34 cl30-35 ar31-38 k35-39 ca36-44 sc39-45 ti40-47 v43-49 cr44-52 mn47-53 fe48-56 co51-56 ni52-56 cu54-63 zn55-66 ga59-67 ge60-68 as64-69 se 65-72 br68-73 kr69-74 rb73-77 sr74-78')
#
print("Number of species=",len(species))

rates = net.read_rates(species)
print("Number of rates = ",len(rates))

# initial mass fractions
XX = np.append(np.array([0.7,0.28,0.01,0.01]),np.zeros(len(species)-4))

# convert to number fraction for the network evolution
AA, ZZ = net.get_AZ(species)
YY = np.array([X/A for X,A in zip(XX,AA)])

# ----- integrate ----- 
T0 = 2e8
F0 = 6.5*9.64e17*mdot
time_to_run = 8000.0
nsteps = 1000
t = np.arange(nsteps+1)*time_to_run/nsteps  + 10.0
t0 = time.time()
result = odeint(derivs,np.append(YY,np.array([T0,F0])),t)
print('Integration took ',time.time()-t0,' seconds')

# output final mass fractions
Xfinal = AA*result[-1,:-2]
Xfinal[Xfinal<1e-12] = 1e-12 
ind = np.argsort(Xfinal)
fp = open('finalabun.dat','w')
for i in ind[::-1]:
	print(species[i],Xfinal[i],AA[i],ZZ[i],file=fp)
print("Final mass fractions sum to: 1+", sum(Xfinal)-1.0)
fp.close()

# plot abundances vs. A
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Avec = np.arange(101)
Yvec = np.zeros(101)
for i in range(len(species)):
	if result[-1,i]>1e-12:
		Yvec[int(AA[i])] = Yvec[int(AA[i])] + result[-1,i]
Yvec[Yvec==0.0] = 1e-12
plt.plot(Avec,np.log10(Yvec),'ko')
plt.plot(Avec,np.log10(Yvec),'k')
plt.ylim((-10,0))
plt.xlim((0,100))
ax.set_yscale('linear')
plt.savefig('finalabun.pdf')

# plot T
fig = plt.figure( )
ax = fig.add_subplot(3,1,1)
plt.plot(mdot*t,result[:,-2])
ax.set_yscale('log')
ax.set_xscale('log')
ax = fig.add_subplot(3,1,2)
plt.plot(mdot*t,result[:,-1])
ax.set_yscale('log')
ax.set_xscale('log')
ax = fig.add_subplot(3,1,3)
plt.plot(mdot*t,result[:,-1]/(mdot*9.64e17))
print("Flux at the base = ", result[-1,-1], result[-1,-1]/(mdot*9.64e17))
ax.set_yscale('linear')
ax.set_xscale('log')
plt.savefig('TP.pdf')

# plot abundances over time
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
for i in range(len(species)):
	if max(AA[i]*result[:,i]) > 1e-12:
		plt.plot(mdot*t,AA[i]*result[:,i])#label=species[i])

plt.xlabel(r'$\mathrm{Column\ depth}\ (g\ cm^{-2})$')
plt.ylabel(r'$\mathrm{Mass\ fraction}\ X_i$')
ax.set_xscale('log')
ax.set_yscale('log')
plt.ylim((1e-5,1.0))
#plt.legend()
plt.savefig('abun.pdf')
