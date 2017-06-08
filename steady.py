#----------------------------------------------------------------------------
# Steady-state rp-process burning on an accreting neutron star
# as in Schatz et al. (1999)
#
# Nuclear rates are from REACLIB (see net.py)
# Not included:
# * neutrino losses
# * density-dependent weak rates
#
#----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import optimize
import itertools
import eos
import kappa
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
	dYdt, eps = net.calculate_dYdt(rho,T,Ye,Y[:-2],AA,ZZ,rates)

	# temperature gradient  (eq. 6 of Schatz et al. 1999)
	dTdt = mdot * 3.0*kappa.kappa(rho,T,Y[:-2],AA,ZZ)*F / (4.0*arad*clight*T**3)

	# dF/dt, ignoring compressional heating (eq. 5 of Schatz et al 1999)
	dFdt = -mdot * eps

	# return all the derivatives together
	dYdt = np.append(dYdt,dTdt)
	dYdt = np.append(dYdt,dFdt)
	return dYdt

# -------- parameters --------
grav = 2e14
mdot_Edd = 8.8e4
arad = 7.5657e-15
clight = 3e10

mdot = 1.0
Ftop = 6.3
print('Accretion rate ',mdot, ' Eddington; flux at the top = ',Ftop, ' MeV/mu')
mdot = mdot*mdot_Edd
time_to_run = 3e4
nsteps = 10000

# ----- set up network -----
species = net.make_species_list( net.nucs['106'] )
AA, ZZ = net.get_AZ(species)
print("net: Number of species=",len(species))
rates = net.read_rates(species)
print("net: Number of rates = ",len(rates))

# initial mass fractions
XX = np.append(np.array([0.7,0.28,0.01,0.01]),np.zeros(len(species)-4))

# convert to number fraction for the network evolution
YY = np.array([X/A for X,A in zip(XX,AA)])

# -------- initialize opacities ---------
kappa.init()

# ----- integrate ----- 
T0 = 2e8
F0 = Ftop*9.64e17*mdot
t = np.arange(nsteps+1)*time_to_run/nsteps  + 1e5/mdot  # start at column depth 1e5
t0 = time.time()
result = odeint(derivs,np.append(YY,np.array([T0,F0])),t)
print('Integration took ',time.time()-t0,' seconds')

# -------- output final mass fractions --------
Xfinal = AA*result[-1,:-2]
Xfinal[Xfinal<1e-12] = 1e-12 
ind = np.argsort(Xfinal)
fp = open('finalabun.dat','w')
for i in ind[::-1]:
	print(species[i],Xfinal[i],AA[i],ZZ[i],file=fp)
print("Final mass fractions sum to: 1+", sum(Xfinal)-1.0)
fp.close()

# -------- extract opacities --------
print("Extracting opacities")
kap_vec = np.array([])
for i in range(len(t)):
	T = result[i,-2]
	Ye = sum(result[i,:-2]*ZZ)
	Yi = sum(result[i,:-2])
	rho = eos.find_rho(grav * mdot * t[i],T,Ye,Yi)
	kap = kappa.kappa(rho, T, result[i,:-2],AA,ZZ)
	kap_vec = np.append(kap_vec, kap)

#------- plots ----------
print("Making plots")

# plot abundances vs. A
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Avec = np.arange(int(max(AA))+1)
Yvec = np.zeros(len(Avec))
for i in range(len(species)):
	if result[-1,i]>1e-12:
		Yvec[int(AA[i])] = Yvec[int(AA[i])] + result[-1,i]
Yvec[Yvec==0.0] = 1e-12
plt.plot(Avec,np.log10(Yvec),'ko')
plt.plot(Avec,np.log10(Yvec),'k')
plt.ylim((-12,0))
plt.xlim((0,100))
ax.set_yscale('linear')
plt.savefig('steady_finalabun.pdf')

# plot T and flux profile
fig = plt.figure(figsize=(6,8))

ax = fig.add_subplot(4,1,1)
plt.plot(mdot*t,result[:,-2])
plt.ylabel(r'$T (\mathrm{K})$')
ax.set_yscale('log')
ax.set_xscale('log')

ax = fig.add_subplot(4,1,2)
plt.plot(mdot*t,result[:,-1])
plt.ylabel(r'$\mathrm{Flux}\ (\mathrm{erg\ cm^{-2}\ s^{-1}})$')
ax.set_yscale('log')
ax.set_xscale('log')

ax = fig.add_subplot(4,1,3)
plt.plot(mdot*t,result[:,-1]/(mdot*9.64e17))
plt.ylabel(r'$F/\dot m\ (\mathrm{MeV/nuc})$')
print("y,T at the base = ", mdot*t[-1],result[-1,-2])
print("Flux at the base = ", result[-1,-1], ' cgs ,', result[-1,-1]/(mdot*9.64e17), ' MeV/mu')
print("Nuclear energy release ", Ftop-result[-1,-1]/(mdot*9.64e17), " MeV/mu")
ax.set_yscale('linear')
ax.set_xscale('log')

ax = fig.add_subplot(4,1,4)
plt.plot(mdot*t,kap_vec)
plt.ylabel(r'$\kappa\ (\mathrm{cm^2\ g^{-1}})$')
ax.set_yscale('linear')
ax.set_xscale('log')

plt.xlabel(r'$\mathrm{Column\ depth}\ (\mathrm{g\ cm^{-2}})$')
plt.tight_layout()
plt.savefig('steady_TP.pdf')

# plot abundances over time
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
maxX = np.array([max(AA[i]*result[:,i]) for i in range(len(species))])
ind = np.argsort(maxX)
ind = ind[::-1]
for i in range(len(species)):
	if i<10:
		plt.plot(mdot*t,AA[ind[i]]*result[:,ind[i]],label=species[ind[i]])
	else:
		plt.plot(mdot*t,AA[ind[i]]*result[:,ind[i]])

plt.xlabel(r'$\mathrm{Column\ depth\ (g\ cm^{-2})}$')
plt.ylabel(r'$\mathrm{Mass\ fraction}\ X_i$')
ax.set_xscale('log')
ax.set_yscale('log')
plt.ylim((1e-5,1.0))
plt.legend(ncol=1,prop={'size':6})
plt.savefig('steady_abun.pdf')
