import numpy as np
from scipy import optimize

def pressure(rho,T,Ye,Yi):
	Ped=9.91e12*(Ye*rho)**(5.0/3.0)   # non-relativistic degenerate electron pressure
	Pend=Ye*rho*1.38e-16*T/1.67e-24   # non-degenerate electron pressure 
	Pe=(Ped**2+Pend**2)**0.5   # fitting formula for electron pressure (Pacynski 1983)
	Pion=Yi*rho*1.38e-16*T/1.67e-24	# ion pressure
	arad = 7.5657e-15
	Prad=arad*T**4/3.0	# radiation pressure
	return Pion+Pe+Prad

def find_rho_eqn(rho,P,Ye,Yi,T):
	return pressure(rho,T,Ye,Yi)-P
		
def find_rho(P,T,Ye,Yi):
	# inverts the equation of state to find the density
	rho = optimize.brentq(find_rho_eqn,1.0,1e8,xtol=1e-6,args=(P,Ye,Yi,T))
	return rho
		
def kappa(rho,T,Ye,YZ2):     # Need to add conduction !!
	rho5=rho*1e-5
	T8=T*1e-8
	# radiative opacity
	kff = 0.753*Ye*rho5*YZ2/T8**3.5    # free-free Gaunt factor set to 1
	kes = 0.4*Ye/((1+2.7*rho5/T8**2)*(1.0+(T8/4.5)**0.86))
	# Thompson scattering with corrections from Paczynski 1983
	return kes + kff   # note that strictly Rosseland mean opacities don't add (<30% error)
	