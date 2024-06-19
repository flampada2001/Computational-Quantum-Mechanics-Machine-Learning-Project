# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:43:11 2024

@author: Foivos Lampadaridis Kokkinakis

This module holds all the functions that were desinged for the Computational Quantum
Mechanics Project
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

r=sp.symbols("r")
k=sp.symbols("k")

# -----------Position space functions------------------
def Sr1s(z,r):
	return 2*z**(3/2)*sp.exp(-z*r)

def Sr2s(z,r):
	return 2/sp.sqrt(3)*z**(5/2)*r*sp.exp(-z*r)

def Sr3s(z,r):
	return 2**(3/2)/(3*sp.sqrt(5))*z**(7/2)*r**2*sp.exp(-z*r)

def Sr2p(z,r):
	return 2/sp.sqrt(3)*z**(5/2)*r*sp.exp(-z*r)

# This function, given the apropiate coefficiants, will return the RHF wavefunctions
# in the position space
def RHF_wavefunctions_R_one_3s(coeficiants_1s,coeficiants_2s,coeficiants_2p,z_values_s,z_values_p):
    R1s=0
    R2s=0
    R2p=0
    
    for i in range(7):
        R2p=R2p+coeficiants_2p[i]*Sr2p(z_values_p[i],r)
    
        if i==0 or i==1:
            R1s=R1s+coeficiants_1s[i]*Sr1s(z_values_s[i],r)
            R2s=R2s+coeficiants_2s[i]*Sr1s(z_values_s[i],r)
        elif i==2:
            R1s=R1s+coeficiants_1s[i]*Sr3s(z_values_s[i],r)
            R2s=R2s+coeficiants_2s[i]*Sr3s(z_values_s[i],r)
        else:
            R1s=R1s+coeficiants_1s[i]*Sr2s(z_values_s[i],r)
            R2s=R2s+coeficiants_2s[i]*Sr2s(z_values_s[i],r)
            
    return R1s,R2s,R2p

# This function, given the apropiate coefficiants, will return the RHF wavefunctions
# in the position space when two 3s terms are needed (used in Be, B)
def RHF_wavefunctions_R_two_3s(coeficiants_1s,coeficiants_2s,coeficiants_2p,z_values_s,z_values_p):
    R1s=0
    R2s=0
    R2p=0
    
    for i in range(7):
        R2p=R2p+coeficiants_2p[i]*Sr2p(z_values_p[i],r)
    
        if i==0 or i==1:
            R1s=R1s+coeficiants_1s[i]*Sr1s(z_values_s[i],r)
            R2s=R2s+coeficiants_2s[i]*Sr1s(z_values_s[i],r)
        elif i==2 or i==3:
            R1s=R1s+coeficiants_1s[i]*Sr3s(z_values_s[i],r)
            R2s=R2s+coeficiants_2s[i]*Sr3s(z_values_s[i],r)
        else:
            R1s=R1s+coeficiants_1s[i]*Sr2s(z_values_s[i],r)
            R2s=R2s+coeficiants_2s[i]*Sr2s(z_values_s[i],r)
            
    return R1s,R2s,R2p

# This function, given the apropiate coefficiants, will return the RHF wavefunctions
# in the position space when only one 1s term is needed (used in He)
def RHF_wavefunctions_R_one_1s(coeficiants_1s,coeficiants_2s,coeficiants_2p,z_values_s,z_values_p):
    R1s=0
    R2s=0
    R2p=0
    
    for i in range(7):
        R2p=R2p+coeficiants_2p[i]*Sr2p(z_values_p[i],r)
    
        if i==0:
            R1s=R1s+coeficiants_1s[i]*Sr1s(z_values_s[i],r)
            R2s=R2s+coeficiants_2s[i]*Sr1s(z_values_s[i],r)
        elif i==1:
            R1s=R1s+coeficiants_1s[i]*Sr3s(z_values_s[i],r)
            R2s=R2s+coeficiants_2s[i]*Sr3s(z_values_s[i],r)
        else:
            R1s=R1s+coeficiants_1s[i]*Sr2s(z_values_s[i],r)
            R2s=R2s+coeficiants_2s[i]*Sr2s(z_values_s[i],r)
            
    return R1s,R2s,R2p



# This function will be used to integrate symbolic expresions of r from 0 to infinity
def Integrate_symbolic_0_inf_R(func):
    numerical_func=sp.lambdify(r,func,modules=["numpy"])
    def integrand(r):
        return numerical_func(r)
     #Perform the numerical integration from 0 to infinity
    Int_result, error_result=quad(integrand,0, np.inf)
    return Int_result

# This function is used to plot a symbolic function from r 0 up to rmax
# we can also give the title and the labels in the x and y axis
# The element_name is used to create the proper .pdf figure file
def Plot_sumbolic_R(func,rmax,title,xlabel,ylabel,element_name):
    numerical_func=sp.lambdify(r,func,modules=["numpy"])
    r_space=np.linspace(0, rmax,10000)
    
    func_space=numerical_func(r_space)
    
    
    plt.plot(r_space,func_space)
    plt.title(str(title))
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    plt.grid()
    plt.savefig('Dos_r_'+element_name+'.pdf', format='pdf', dpi=300)
    plt.show()
    
# -----------k-space functions------------------  

def Sk1s(z,k):
	return 1/(2*sp.pi)**(3/2)*16*sp.pi*z**(5/2)/(z**2+k**2)**2

def Sk2s(z,k):
	return 1/(2*sp.pi)**(3/2)*16*sp.pi*z**(5/2)*(3*z**2-k**2)/(sp.sqrt(3)*(z**2+k**2)**3)

def Sk3s(z,k):
	return 1/(2*sp.pi)**(3/2)*64*sp.sqrt(10)*sp.pi*z**(9/2)*(z**2-k**2)/(5*(z**2+k**2)**4)

def Sk2p(z,k):
	return 1/(2*sp.pi)**(3/2)*64*sp.pi*k*z**(7/2)/(sp.sqrt(3)*(z**2+k**2)**3)

# This function, given the apropiate coefficiants, will return the RHF wavefunctions
# in the k-space
def RHF_wavefunctions_k_one_3s(coeficiants_1s,coeficiants_2s,coeficiants_2p,z_values_s,z_values_p):
    K1s=0
    K2s=0
    K2p=0
    
    for i in range(7):
        K2p=K2p+coeficiants_2p[i]*Sk2p(z_values_p[i],k)
    
        if i==0 or i==1:
            K1s=K1s+coeficiants_1s[i]*Sk1s(z_values_s[i],k)
            K2s=K2s+coeficiants_2s[i]*Sk1s(z_values_s[i],k)
        elif i==2:
            K1s=K1s+coeficiants_1s[i]*Sk3s(z_values_s[i],k)
            K2s=K2s+coeficiants_2s[i]*Sk3s(z_values_s[i],k)
        else:
            K1s=K1s+coeficiants_1s[i]*Sk2s(z_values_s[i],k)
            K2s=K2s+coeficiants_2s[i]*Sk2s(z_values_s[i],k)
            
    return K1s,K2s,K2p

# This function, given the apropiate coefficiants, will return the RHF wavefunctions
# in the k-space when two 3s terms are needed (used in Be, B)
def RHF_wavefunctions_k_two_3s(coeficiants_1s,coeficiants_2s,coeficiants_2p,z_values_s,z_values_p):
    K1s=0
    K2s=0
    K2p=0
    
    for i in range(7):
        K2p=K2p+coeficiants_2p[i]*Sk2p(z_values_p[i],k)
    
        if i==0 or i==1:
            K1s=K1s+coeficiants_1s[i]*Sk1s(z_values_s[i],k)
            K2s=K2s+coeficiants_2s[i]*Sk1s(z_values_s[i],k)
        elif i==2 or i==3:
            K1s=K1s+coeficiants_1s[i]*Sk3s(z_values_s[i],k)
            K2s=K2s+coeficiants_2s[i]*Sk3s(z_values_s[i],k)
        else:
            K1s=K1s+coeficiants_1s[i]*Sk2s(z_values_s[i],k)
            K2s=K2s+coeficiants_2s[i]*Sk2s(z_values_s[i],k)
            
    return K1s,K2s,K2p

# This function, given the apropiate coefficiants, will return the RHF wavefunctions
# in the k-space when only one 1s term is needed (used in He)
def RHF_wavefunctions_k_one_1s(coeficiants_1s,coeficiants_2s,coeficiants_2p,z_values_s,z_values_p):
    K1s=0
    K2s=0
    K2p=0
    
    for i in range(7):
        K2p=K2p+coeficiants_2p[i]*Sk2p(z_values_p[i],k)
    
        if i==0:
            K1s=K1s+coeficiants_1s[i]*Sk1s(z_values_s[i],k)
            K2s=K2s+coeficiants_2s[i]*Sk1s(z_values_s[i],k)
        elif i==1:
            K1s=K1s+coeficiants_1s[i]*Sk3s(z_values_s[i],k)
            K2s=K2s+coeficiants_2s[i]*Sk3s(z_values_s[i],k)
        else:
            K1s=K1s+coeficiants_1s[i]*Sk2s(z_values_s[i],k)
            K2s=K2s+coeficiants_2s[i]*Sk2s(z_values_s[i],k)
            
    return K1s,K2s,K2p


# This function will be used to integrate symbolic expresions of k from 0 to infinity
def Integrate_symbolic_0_inf_k(func):
    numerical_func=sp.lambdify(k,func,modules=["numpy"])
    def integrand(k):
        return numerical_func(k)
     #Perform the numerical integration from 0 to infinity
    Int_result, error_result=quad(integrand,0, np.inf)
    return Int_result

# This function is used to plot a symbolic function from k 0 up to rmax
# we can also give the title and the labels in the x and y axis
# The element_name is used to create the proper .pdf figure file
def Plot_sumbolic_k(func,kmax,title,xlabel,ylabel,element_name):
    numerical_func=sp.lambdify(k,func,modules=["numpy"])
    k_space=np.linspace(0, kmax,10000)
    
    func_space=numerical_func(k_space)
    
    
    plt.plot(k_space,func_space)
    plt.title(str(title))
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    plt.grid()
    plt.savefig('Dos_k_'+element_name+'.pdf', format='pdf', dpi=300)
    plt.show()    