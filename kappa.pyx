#cython: profile=False, boundscheck=False, wraparound=False, cdivision=True

cimport cython

cdef extern from "math.h":
	double exp(double m)
	double log(double m)
	double sqrt(double m)
	double MPI "M_PI"
	
import numpy as np
from scipy import interpolate
import eos

# ----------------------- Free-free opacity -------------------------------

# total absorption opacity
def kappa_abs(double T_keV, double rho, double Ye, XA, AA, ZZ, double chi):

	cdef double kff, gaunt, Z, Y, A

	kff = 0.753e-5*rho*Ye/(0.1160452*T_keV)**3.5
	# last term is the sum of Z^2/A with gaunt factor
	gaunt = 0.0
	for Z, Y, A in zip(ZZ,XA,AA):
		gaunt += Y * Z * Z * gff(T_keV, rho, Ye, Z, chi)
	kff = kff * gaunt

	return kff

# free-free Gaunt factor
cdef double gff(double T_keV, double rho, double Ye, double Z, double chi):
	cdef double u = 10.0
	cdef double lchi, PP, gam, f1, f2, f3, rho5, T8
	
	rho5 = rho*1e-5
	T8 = T_keV/8.6173324
	
	lchi = log(1.0 + exp(chi))
	
	PP = (1.0 + lchi)**(2.0/3.0)
	gam = sqrt(1.58e-3 * Z**2 / T8)
	
	f1 = 0.08 * T8**1.5 / (rho5*Ye) * lchi
	f2 = (1.0 - exp(-2.0*MPI*gam / sqrt(PP + u))) / (1.0 - exp(-2.0*MPI*gam / sqrt(PP)))
	f3 = 1.0 + (T8/7.7)**1.5

	return 1.16 * f1 * f2 * f3

# initialize
def init():
	# nothing to do here yet
	pass


# ----------------------- Thermal conductivity ------------------------------------

cdef double K_cond(double T_keV, double rho, double Ye, double Yi, double YZ2, double chi):

	cdef double m_star, gamma, lam, x, nu

	# effective electron mass (in keV)
	m_star = 510.999 + T_keV * chi

	# Coulomb log from Yakovlev & Urpin
	gamma = 0.0196*(YZ2/Yi)*(rho*Yi)**(1.0/3.0)/T_keV
	lam = log( 1.2794*(Ye/Yi)**(1.0/3.0)*sqrt(1.5 + 3.0/gamma) )
	x = (Ye*1e-6*rho)**(1.0/3.0)
	lam = lam - 0.5 * x**2 / (1.0 + x**2)
	
	# electron-ion collision frequency (s^-1)
	nu = 3.47e13 * m_star * (YZ2/Ye) * lam
	
	# return the conductivity in cgs
	return 2.45e29 * T_keV * Ye * rho / (m_star * nu)
	
# ----------------------- Opacity ------------------------------------

def kappa(double rho, double T, XA, AA, ZZ):
	cdef double Yi, Ye, YZ2, Z_avg, rho5, T8, T_keV, kff, kes, f, TRy, Afac, krad, kcond, chi

	# composition
	Yi = np.sum(XA)
	Ye = np.sum(XA*ZZ)
	YZ2 = np.sum(XA*ZZ*ZZ)
	Z_avg = np.sum(XA*ZZ*AA)

	# density and temperature
	rho5 = rho*1e-5
	T8 = T*1e-8
	T_keV = 1e-8*T*8.6173324

	# chi = EF/kT, used in ff and cond
	chi = eos.chi( T_keV, rho, Ye)

	# free-free opacity
	kff = kappa_abs(T_keV, rho, Ye, XA, AA, ZZ, chi)
	
	# electron scattering fitting formula from Paczynski (1983)
	kes = 0.4006 * Ye / ((1.0 + 2.7 * rho5 / T8**2) * (1.0 + (T8/4.5)**0.86))

	# 'non-additivity' factor for radiative opacity
	# from eqs. 19 and 20 of Potekhin & Yakovlev 2001
	f = kff / (kff+kes)
	TRy = 73.5*T_keV/Z_avg**2
	Afac = (1.097 + 0.777*TRy) / (1.0 + 0.536*TRy)
	Afac = 1.0 + Afac * f**0.617 * (1.0-f)**0.77
	krad = (kff + kes) * Afac

	# thermal conductivity
	kcond = 4.72854e17 * T_keV**3 / (rho * K_cond(T_keV, rho, Ye, Yi, YZ2, chi))
	
	# return the total opacity
	return 1.0 / (1.0/krad + 1.0/kcond), (krad,kff,kes,kcond)
