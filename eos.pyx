#cython: profile=False

cimport cython

cdef extern from "math.h":
	double exp(double m)
	double log(double m)
	double sqrt(double m)
	double MPI "M_PI"

import numpy as np
from scipy import optimize

# ------------------ Pressure and density -------------------------------
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


# -------------------------- Fermi energy -------------------------------

# Inverse Fermi integral  X_{1/2}
# given by equation 6 of Antia (1993)
def X_n(double F):
	cdef double Xn, F23
	if F < 4:
		Xn = log(F*Rmk_1(F))
	else:
		F23 = F**(2.0/3.0)
		Xn = F23 * Rmk_2(1.0/F23)
	return Xn

#Equation (4) in Antia (1993) (Constants are from Table 8)
cdef double Rmk_1(double x):
	cdef double a0 = 4.4593646E+01, a1 = 1.1288764E+01, a2 = 1.0
	cdef double b0 = 3.9519346E+01, b1 = -5.7517464E+00, b2 = 2.6594291E-01
	return (a0 + a1*x + a2*x**2)/(b0 + b1*x + b2*x**2)

cdef double Rmk_2(double x):
	cdef double c0 = 34.87, c1 = -26.92, c2 = 1.0
	cdef double d0 = 26.61, d1 = -20.45, d2 = 11.81
	return (c0 + c1*x + c2*x**2)/(d0 + d1*x + d2*x**2)

# Fermi energy from Chabrier and Potekhin 1998
# calculate chi = E_F/kT   where E_F does *not* include the rest mass
def chi(double T_keV, double rho, double Ye):

	rY = rho * Ye
	x = 1.007e-2*rY**(1.0/3.0)
	tau = 0.0019545 * T_keV     # tau is kT/(m_e c^2)
	theta = tau/(sqrt(1.0+x*x)-1.0)   # theta is T/T_F
	
	# non-relativistic guess for E_F/kT
	F = 2.0*theta**(-1.5)/3.0
	EFnr = X_n(F)
	
	# functions defined in CP1998 eq. 24
	if theta > 69.0:
		et = 1e30
		etu = 1e-30
	else:
		et = exp(theta)
		etu = 1.0/et
	q1 = 1.5/(et-1.0)
	q2 = 12.0 + 8.0/theta**1.5
	q3 = 1.366 - (etu + 1.612*et) / (6.192*theta**0.0944*etu + 5.535*theta**0.698*et)
	
	# This is the correction to the non-relativistic EF
	corr = (1.0 + q1*sqrt(tau) + q2*q3*tau)/(1.0 + q2*tau);
	corr = corr * tau/(1.0 + 0.5*tau/theta)
	corr = 1.5 * log(1.0 + corr)

	return EFnr-corr
