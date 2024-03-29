#----------------------------------------------------------------------------
# One zone Type I X-ray burst
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
	T = Y[-1]
	Ye = sum(Y[:-1]*ZZ)
	Yi = sum(Y[:-1])
	P = grav*ycolumn
	rho = eos.find_rho(P,T,Ye,Yi)
	kap, _ = kappa.kappa(rho,T,Y[:-1],AA,ZZ)
	#kap = 0.2

	# abundance derivatives
	dYdt, eps = net.calculate_dYdt(rho,T,Ye,Y[:-1],AA,ZZ,rates)

	# temperature derivatives
	F = arad*clight*T**4 / (3*kap*ycolumn)
	dTdt = (eps - F/ycolumn)/eos.CP(rho,T,Ye,Yi)

	# return all the derivatives together
	dYdt = np.append(dYdt,dTdt)
	return dYdt

# ------ parameters ------
grav = 2.45e14
mdot_Edd = 8.8e4
arad = 7.5657e-15
clight = 3e10
kB = 1.38e-16
mp = 1.67e-24

X0 = 0.7
T0 = 2.5e8
ycolumn = 2e8
time_to_run =  1000.0
nsteps = 1e5

# ------ initialize opacities ------
kappa.init()

# ----- set up network -----
species = net.make_species_list( net.nucs['106'] )
print("Number of species=",len(species))
AA, ZZ = net.get_AZ(species)
rates = net.read_rates(species)
print("Number of rates = ",len(rates))

# initial mass fractions
XX = np.append(np.array([X0,1.0-X0,0.00]),np.zeros(len(species)-3))

# convert to number fraction for the network evolution
YY = np.array([X/A for X,A in zip(XX,AA)])

# ----- integrate ----- 
t = np.arange(nsteps+1)*time_to_run/nsteps
t0 = time.time()
result = odeint(derivs,np.append(YY,T0),t,mxstep=100000)
print('Integration took ',time.time()-t0,' seconds')

# output final mass fractions
Xfinal = AA*result[-1,:-1]
Xfinal[Xfinal<1e-12] = 1e-12 
ind = np.argsort(Xfinal)
fp = open('burst_finalabun.dat','w')
for i in ind[::-1]:
	print(species[i],Xfinal[i],AA[i],ZZ[i],file=fp)
print("Final mass fractions sum to: 1+", sum(Xfinal)-1.0)
fp.close()

# plot lightcurve and output
print("Plotting lightcurve")
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
F = np.array([])
Ye_vec = np.array([])
Yi_vec = np.array([])
rho_vec = np.array([])
T_vec = np.array([])
for i, T in enumerate(result[:,-1]):
	Ye = sum(result[i,:-1]*ZZ)
	P = grav*ycolumn
	Yi = sum(result[i,:-1])
	rho = eos.find_rho(P,T,Ye,Yi)
	kap, _ = kappa.kappa(rho,T,result[i,:-1],AA,ZZ)
	#kap = 0.2
	flux = arad*clight*T**4/(3*kap*ycolumn)
	F = np.append(F,flux)
	Ye_vec = np.append(Ye_vec,Ye)
	Yi_vec = np.append(Yi_vec,Yi)
	rho_vec = np.append(rho_vec,rho)
	T_vec = np.append(T_vec,T)
	
ind = F>0.001*max(F)
tburst = t[ind]
tstart = tburst[0]
tend = tburst[-1]
print("Burst start time = ",tstart)
Fburst = F[ind]
plt.plot(tburst-tstart,Fburst)
ax.set_yscale('linear')
ax.set_xscale('linear')
plt.savefig('burst_prof.pdf')
#plt.show()
np.savetxt('burst_prof.dat', np.c_[tburst,Fburst,rho_vec[ind],T_vec[ind],Ye_vec[ind],Yi_vec[ind]])

# plot final abundances vs. A
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
Avec = np.arange(int(max(AA))+1)
Yvec = np.zeros(len(Avec))
for i in range(len(species)):
	if result[t==tend,i]>1e-12:
		Yvec[int(AA[i])] = Yvec[int(AA[i])] + AA[i]*result[t==tend,i]
Yvec[Yvec==0.0] = 1e-12
plt.plot(Avec,np.log10(Yvec),'ko')
plt.plot(Avec,np.log10(Yvec),'k')
plt.ylim((-5,0))
plt.xlim((0,100))
ax.set_yscale('linear')
plt.savefig('burst_finalabun.pdf')

# plot abundances over time
print("Plotting abundances")
fig = plt.figure( )
ax = fig.add_subplot(1,1,1)
maxX = np.array([max(AA[i]*result[:,i]) for i in range(len(species))])
sortind = np.argsort(maxX)
sortind = sortind[::-1]
for i in range(len(species)):
	X = AA[sortind[i]]*result[:,sortind[i]]
	tt = t[ind]-tstart
	XX = X[ind]
	if i<10:
		plt.plot(tt[1:],XX[1:],label=species[sortind[i]])
	elif i<20:
		plt.plot(tt[1:],XX[1:],':',label=species[sortind[i]])
	elif i<30:
		plt.plot(tt[1:],XX[1:],'--',label=species[sortind[i]])

plt.legend(ncol=1,prop={'size':6},loc=5)
plt.xlabel(r'$\mathrm{Time (s)}$')
plt.ylabel(r'$\mathrm{Mass\ fraction}\ X_i$')
ax.set_xscale('linear')
ax.set_yscale('log')
plt.ylim((1e-5,1.0))
plt.savefig('burst_abun.pdf')
