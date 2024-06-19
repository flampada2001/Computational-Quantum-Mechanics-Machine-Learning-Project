"""
Created on Sat May 25 11:06:53 2024

@author: Foivos Lampadaridis Kokkinakis

This is the main .py that calculates all the needed quantities for all 9 of our elements.
The DOS in both the position and k-space is calculated, ploted and used for the
calculation of each atoms entropy. The above is done by using the user created
module RHF_functions.
"""
import RHF_functions as rhf

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from prettytable import PrettyTable

s_R_values=[]
s_K_values=[]
s_ol_values=[]
z_values=range(2,11)


r=sp.symbols("r")
k=sp.symbols("k")

#Element 1, Z=2: He
print("Studying He:")
coeficiants_1s_He=[1.347900 ,-0.001613 ,-0.100506 ,-0.270779 ,0.0 ,0.0 ,0.0]
coeficiants_2s_He=[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]
coeficiants_2p_He=[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]


z_values_s_He=[1.4595 ,5.3244 ,2.6298 ,1.7504 ,1,1,1]
z_values_p_He=[1,1,1,1,1,1,1]

R1sHe,R2sHe,R2pHe=rhf.RHF_wavefunctions_R_one_1s(coeficiants_1s_He, coeficiants_2s_He, coeficiants_2p_He, z_values_s_He, z_values_p_He)

t_He=t = PrettyTable(["Quantity", "Result"])

Int_R1s_He=rhf.Integrate_symbolic_0_inf_R(R1sHe**2*r**2)
t_He.add_row(['Int_R1s',Int_R1s_He])


Int_R2s_He=rhf.Integrate_symbolic_0_inf_R(R2sHe**2*r**2)
t_He.add_row(['Int_R2s',Int_R2s_He])


Int_R2p_He=rhf.Integrate_symbolic_0_inf_R(R2pHe**2*r**2)
t_He.add_row(['Int_R2p',Int_R2p_He])


#pr=pollaplasiazw me 1/(4*pi*10) giati einai 1/4pi kai epi 10 gia ta hlektronia
p_R_He=1/(4*np.pi*2)*(2*R1sHe**2+0.0*R2sHe**2+0.0*R2pHe**2)#DOS


Int_pr_He=rhf.Integrate_symbolic_0_inf_R(4*sp.pi*r**2*p_R_He)
t_He.add_row(['Int_pr',Int_pr_He])

rhf.Plot_sumbolic_R(p_R_He, 1.3, "The DOS of He in the position space", "r", "rho(r)","He")

# epsilon is used so there we avoid calculating log0
epsilon=10**(-16)
S_r_He=rhf.Integrate_symbolic_0_inf_R(-4*sp.pi*p_R_He*sp.log(p_R_He+epsilon)*r**2)
s_R_values.append(S_r_He)


# b) k space
K1sHe,K2sHe,K2pHe=rhf.RHF_wavefunctions_k_one_1s(coeficiants_1s_He, coeficiants_2s_He, coeficiants_2p_He, z_values_s_He, z_values_p_He)

Int_K1s_He=rhf.Integrate_symbolic_0_inf_k(K1sHe**2*k**2)
t_He.add_row(['Int_K1s',Int_K1s_He])

Int_K2s_He=rhf.Integrate_symbolic_0_inf_k(K2sHe**2*k**2)
t_He.add_row(['Int_K2s',Int_K2s_He])

Int_K2p_He=rhf.Integrate_symbolic_0_inf_k(K2pHe**2*k**2)
t_He.add_row(['Int_K2p',Int_K2p_He])

#pr=pollaplasiazw me 1/(4*pi*10) giati einai 1/4pi kai epi 10 gia ta hlektronia
n_k_He=1/(4*np.pi*2)*(2*K1sHe**2+0.0*K2sHe**2+0.0*K2pHe**2)#DOS

Int_nk_He=rhf.Integrate_symbolic_0_inf_k(4*sp.pi*k**2*n_k_He)
t_He.add_row(['Int_nk',Int_R1s_He])

t_He.add_row(['S_r',S_r_He])

rhf.Plot_sumbolic_k(n_k_He, 3, "The DOS of He in the k space", "k", "n(k)","He")

S_k_He=rhf.Integrate_symbolic_0_inf_k(-4*sp.pi*n_k_He*sp.log(n_k_He+epsilon)*k**2)
s_K_values.append(S_k_He)
t_He.add_row(['S_k',S_k_He])

S_ol_He=S_r_He+S_k_He
s_ol_values.append(S_ol_He)
t_He.add_row(['S_ol',S_ol_He])

print(t_He)
print("\n")



#Element 2, Z=3: Li
print("Studying Li:")
coeficiants_1s_Li=[0.141279 ,0.874231 ,-0.005201 ,-0.002307 ,0.006985 ,-0.000305 ,0.000760]
coeficiants_2s_Li=[-0.022416,-0.135791,0.000389,-0.000068,-0.076544,0.340542,0.715708]
coeficiants_2p_Li=[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]


z_values_s_Li=[4.3069 ,2.4573 ,6.7850 ,7.4527 ,1.8504,0.7667,0.6364]
z_values_p_Li=[1,1,1,1,1,1,1]

R1sLi,R2sLi,R2pLi=rhf.RHF_wavefunctions_R_one_3s(coeficiants_1s_Li, coeficiants_2s_Li, coeficiants_2p_Li, z_values_s_Li, z_values_p_Li)

t_Li=t = PrettyTable(["Quantity", "Result"])

Int_R1s_Li=rhf.Integrate_symbolic_0_inf_R(R1sLi**2*r**2)
t_Li.add_row(['Int_R1s',Int_R1s_Li])


Int_R2s_Li=rhf.Integrate_symbolic_0_inf_R(R2sLi**2*r**2)
t_Li.add_row(['Int_R2s',Int_R2s_Li])


Int_R2p_Li=rhf.Integrate_symbolic_0_inf_R(R2pLi**2*r**2)
t_Li.add_row(['Int_R2p',Int_R2p_Li])


#pr=pollaplasiazw me 1/(4*pi*3) giati einai 1/4pi kai epi 3 gia ta hlektronia
p_R_Li=1/(4*np.pi*3)*(2*R1sLi**2+1*R2sLi**2+0.0*R2pLi**2)#DOS


Int_pr_Li=rhf.Integrate_symbolic_0_inf_R(4*sp.pi*r**2*p_R_Li)
t_Li.add_row(['Int_pr',Int_pr_Li])

rhf.Plot_sumbolic_R(p_R_Li, 1.2, "The DOS of Li in the position space", "r", "rho(r)","Li")

# epsilon is used so there we avoid calculating log0
epsilon=10**(-16)
S_r_Li=rhf.Integrate_symbolic_0_inf_R(-4*sp.pi*p_R_Li*sp.log(p_R_Li+epsilon)*r**2)
s_R_values.append(S_r_Li)


# b) k space
K1sLi,K2sLi,K2pLi=rhf.RHF_wavefunctions_k_one_3s(coeficiants_1s_Li, coeficiants_2s_Li, coeficiants_2p_Li, z_values_s_Li, z_values_p_Li)

Int_K1s_Li=rhf.Integrate_symbolic_0_inf_k(K1sLi**2*k**2)
t_Li.add_row(['Int_K1s',Int_K1s_Li])

Int_K2s_Li=rhf.Integrate_symbolic_0_inf_k(K2sLi**2*k**2)
t_Li.add_row(['Int_K2s',Int_K2s_Li])

Int_K2p_Li=rhf.Integrate_symbolic_0_inf_k(K2pLi**2*k**2)
t_Li.add_row(['Int_K2p',Int_K2p_Li])

#pr=pollaplasiazw me 1/(4*pi*3) giati einai 1/4pi kai epi 3 gia ta hlektronia
n_k_Li=1/(4*np.pi*3)*(2*K1sLi**2+1*K2sLi**2+0.0*K2pLi**2)#DOS

Int_nk_Li=rhf.Integrate_symbolic_0_inf_k(4*sp.pi*k**2*n_k_Li)
t_Li.add_row(['Int_nk',Int_R1s_Li])

t_Li.add_row(['S_r',S_r_Li])

rhf.Plot_sumbolic_k(n_k_Li, 1, "The DOS of Li in the k space", "k", "n(k)","Li")

S_k_Li=rhf.Integrate_symbolic_0_inf_k(-4*sp.pi*n_k_Li*sp.log(n_k_Li+epsilon)*k**2)
s_K_values.append(S_k_Li)
t_Li.add_row(['S_k',S_k_Li])

S_ol_Li=S_r_Li+S_k_Li
s_ol_values.append(S_ol_Li)
t_Li.add_row(['S_ol',S_ol_Li])

print(t_Li)
print("\n")



#Element 3, Z=4: Be
print("Studying Be:")
coeficiants_1s_Be=[0.285107 ,0.474813 ,-0.001620 ,0.052852,0.243499 ,0.000106 ,-0.000032]
coeficiants_2s_Be=[-0.016378,-0.155066,0.000426,-0.059234,-0.031925,0.387968,0.685674]
coeficiants_2p_Be=[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]


z_values_s_Be=[5.7531 ,3.7156 ,9.9670 ,3.7128 ,4.4661,1.2919,0.8555]
z_values_p_Be=[1,1,1,1,1,1,1]

R1sBe,R2sBe,R2pBe=rhf.RHF_wavefunctions_R_two_3s(coeficiants_1s_Be, coeficiants_2s_Be, coeficiants_2p_Be, z_values_s_Be, z_values_p_Be)

t_Be=t = PrettyTable(["Quantity", "Result"])

Int_R1s_Be=rhf.Integrate_symbolic_0_inf_R(R1sBe**2*r**2)
t_Be.add_row(['Int_R1s',Int_R1s_Be])


Int_R2s_Be=rhf.Integrate_symbolic_0_inf_R(R2sBe**2*r**2)
t_Be.add_row(['Int_R2s',Int_R2s_Be])


Int_R2p_Be=rhf.Integrate_symbolic_0_inf_R(R2pBe**2*r**2)
t_Be.add_row(['Int_R2p',Int_R2p_Be])


#pr=pollaplasiazw me 1/(4*pi*4) giati einai 1/4pi kai epi 4 gia ta hlektronia
p_R_Be=1/(4*np.pi*4)*(2*R1sBe**2+2*R2sBe**2+0.0*R2pBe**2)#DOS


Int_pr_Be=rhf.Integrate_symbolic_0_inf_R(4*sp.pi*r**2*p_R_Be)
t_Be.add_row(['Int_pr',Int_pr_Be])

rhf.Plot_sumbolic_R(p_R_Be, 0.9, "The DOS of Be in the position space", "r", "rho(r)","Be")

# epsilon is used so there we avoid calculating log0
epsilon=10**(-16)
S_r_Be=rhf.Integrate_symbolic_0_inf_R(-4*sp.pi*p_R_Be*sp.log(p_R_Be+epsilon)*r**2)
s_R_values.append(S_r_Be)


# b) k space
K1sBe,K2sBe,K2pBe=rhf.RHF_wavefunctions_k_two_3s(coeficiants_1s_Be, coeficiants_2s_Be, coeficiants_2p_Be, z_values_s_Be, z_values_p_Be)

Int_K1s_Be=rhf.Integrate_symbolic_0_inf_k(K1sBe**2*k**2)
t_Be.add_row(['Int_K1s',Int_K1s_Be])

Int_K2s_Be=rhf.Integrate_symbolic_0_inf_k(K2sBe**2*k**2)
t_Be.add_row(['Int_K2s',Int_K2s_Be])

Int_K2p_Be=rhf.Integrate_symbolic_0_inf_k(K2pBe**2*k**2)
t_Be.add_row(['Int_K2p',Int_K2p_Be])

#pr=pollaplasiazw me 1/(4*pi*4) giati einai 1/4pi kai epi 4 gia ta hlektronia
n_k_Be=1/(4*np.pi*4)*(2*K1sBe**2+2*K2sBe**2+0.0*K2pBe**2)#DOS

Int_nk_Be=rhf.Integrate_symbolic_0_inf_k(4*sp.pi*k**2*n_k_Be)
t_Be.add_row(['Int_nk',Int_R1s_Be])

t_Be.add_row(['S_r',S_r_Be])

rhf.Plot_sumbolic_k(n_k_Be, 1.5, "The DOS of Be in the k space", "k", "n(k)","Be")

S_k_Be=rhf.Integrate_symbolic_0_inf_k(-4*sp.pi*n_k_Be*sp.log(n_k_Be+epsilon)*k**2)
s_K_values.append(S_k_Be)
t_Be.add_row(['S_k',S_k_Be])

S_ol_Be=S_r_Be+S_k_Be
s_ol_values.append(S_ol_Be)
t_Be.add_row(['S_ol',S_ol_Be])

print(t_Be)
print("\n")




#Element 4, Z=5: B
print("Studying B:")
coeficiants_1s_B=[0.381607 ,0.423958 ,-0.001316 ,-0.000822,0.237016 ,0.001062 ,-0.000137]
coeficiants_2s_B=[-0.022549,0.321716,-0.000452,-0.072032,-0.050313,-0.484281,-0.518986]
coeficiants_2p_B=[0.007600 ,0.045137 ,0.184206 ,0.394754 ,0.432795 ,0.0 ,0.0]


z_values_s_B=[7.0178 ,3.9468 ,12.7297 ,2.7646 ,5.7420,1.5436,1.0802]
z_values_p_B=[5.7416,2.6341,1.8340,1.1919,0.8494,1,1]

R1sB,R2sB,R2pB=rhf.RHF_wavefunctions_R_two_3s(coeficiants_1s_B, coeficiants_2s_B, coeficiants_2p_B, z_values_s_B, z_values_p_B)

t_B=t = PrettyTable(["Quantity", "Result"])

Int_R1s_B=rhf.Integrate_symbolic_0_inf_R(R1sB**2*r**2)
t_B.add_row(['Int_R1s',Int_R1s_B])


Int_R2s_B=rhf.Integrate_symbolic_0_inf_R(R2sB**2*r**2)
t_B.add_row(['Int_R2s',Int_R2s_B])


Int_R2p_B=rhf.Integrate_symbolic_0_inf_R(R2pB**2*r**2)
t_B.add_row(['Int_R2p',Int_R2p_B])


#pr=pollaplasiazw me 1/(4*pi*5) giati einai 1/4pi kai epi 5 gia ta hlektronia
p_R_B=1/(4*np.pi*5)*(2*R1sB**2+2*R2sB**2+1*R2pB**2)#DOS


Int_pr_B=rhf.Integrate_symbolic_0_inf_R(4*sp.pi*r**2*p_R_B)
t_B.add_row(['Int_pr',Int_pr_B])

rhf.Plot_sumbolic_R(p_R_B, 0.65, "The DOS of B in the position space", "r", "rho(r)","B")

# epsilon is used so there we avoid calculating log0
epsilon=10**(-16)
S_r_B=rhf.Integrate_symbolic_0_inf_R(-4*sp.pi*p_R_B*sp.log(p_R_B+epsilon)*r**2)
s_R_values.append(S_r_B)


# b) k space
K1sB,K2sB,K2pB=rhf.RHF_wavefunctions_k_two_3s(coeficiants_1s_B, coeficiants_2s_B, coeficiants_2p_B, z_values_s_B, z_values_p_B)

Int_K1s_B=rhf.Integrate_symbolic_0_inf_k(K1sB**2*k**2)
t_B.add_row(['Int_K1s',Int_K1s_B])

Int_K2s_B=rhf.Integrate_symbolic_0_inf_k(K2sB**2*k**2)
t_B.add_row(['Int_K2s',Int_K2s_B])

Int_K2p_B=rhf.Integrate_symbolic_0_inf_k(K2pB**2*k**2)
t_B.add_row(['Int_K2p',Int_K2p_B])

#pr=pollaplasiazw me 1/(4*pi*5) giati einai 1/4pi kai epi 5 gia ta hlektronia
n_k_B=1/(4*np.pi*5)*(2*K1sB**2+2*K2sB**2+1*K2pB**2)#DOS

Int_nk_B=rhf.Integrate_symbolic_0_inf_k(4*sp.pi*k**2*n_k_B)
t_B.add_row(['Int_nk',Int_R1s_B])

t_B.add_row(['S_r',S_r_B])

rhf.Plot_sumbolic_k(n_k_B, 2, "The DOS of B in the k space", "k", "n(k)","B")

S_k_B=rhf.Integrate_symbolic_0_inf_k(-4*sp.pi*n_k_B*sp.log(n_k_B+epsilon)*k**2)
s_K_values.append(S_k_B)
t_B.add_row(['S_k',S_k_B])

S_ol_B=S_r_B+S_k_B
s_ol_values.append(S_ol_B)
t_B.add_row(['S_ol',S_ol_B])

print(t_B)
print("\n")





#Element 5, Z=6: C
print("Studying C:")
coeficiants_1s_C=[0.352872 ,0.473621 ,-0.001199 ,0.210887,0.000886 ,0.000465 ,-0.000119]
coeficiants_2s_C=[-0.071727,0.438307,-0.000383,-0.091194,-0.393105,-0.579121,-0.126067]
coeficiants_2p_C=[0.006977 ,0.070877 ,0.230802 ,0.411931 ,0.350701 ,0.0 ,0.0]


z_values_s_C=[8.4936 ,4.8788 ,15.4660 ,7.0500 ,2.2640,1.4747,1.1639]
z_values_p_C=[7.0500,3.2275,2.1908,1.4413,1.0242,1,1]

R1sC,R2sC,R2pC=rhf.RHF_wavefunctions_R_one_3s(coeficiants_1s_C, coeficiants_2s_C, coeficiants_2p_C, z_values_s_C, z_values_p_C)

t_C=t = PrettyTable(["Quantity", "Result"])

Int_R1s_C=rhf.Integrate_symbolic_0_inf_R(R1sC**2*r**2)
t_C.add_row(['Int_R1s',Int_R1s_C])


Int_R2s_C=rhf.Integrate_symbolic_0_inf_R(R2sC**2*r**2)
t_C.add_row(['Int_R2s',Int_R2s_C])


Int_R2p_C=rhf.Integrate_symbolic_0_inf_R(R2pC**2*r**2)
t_C.add_row(['Int_R2p',Int_R2p_C])


#pr=pollaplasiazw me 1/(4*pi*6) giati einai 1/4pi kai epi 6 gia ta hlektronia
p_R_C=1/(4*np.pi*6)*(2*R1sC**2+2*R2sC**2+2*R2pC**2)#DOS


Int_pr_C=rhf.Integrate_symbolic_0_inf_R(4*sp.pi*r**2*p_R_C)
t_C.add_row(['Int_pr',Int_pr_C])

rhf.Plot_sumbolic_R(p_R_C, 0.6, "The DOS of C in the position space", "r", "rho(r)","C")

# epsilon is used so there we avoid calculating log0
epsilon=10**(-16)
S_r_C=rhf.Integrate_symbolic_0_inf_R(-4*sp.pi*p_R_C*sp.log(p_R_C+epsilon)*r**2)
s_R_values.append(S_r_C)


# b) k space
K1sC,K2sC,K2pC=rhf.RHF_wavefunctions_k_one_3s(coeficiants_1s_C, coeficiants_2s_C, coeficiants_2p_C, z_values_s_C, z_values_p_C)

Int_K1s_C=rhf.Integrate_symbolic_0_inf_k(K1sC**2*k**2)
t_C.add_row(['Int_K1s',Int_K1s_C])

Int_K2s_C=rhf.Integrate_symbolic_0_inf_k(K2sC**2*k**2)
t_C.add_row(['Int_K2s',Int_K2s_C])

Int_K2p_C=rhf.Integrate_symbolic_0_inf_k(K2pC**2*k**2)
t_C.add_row(['Int_K2p',Int_K2p_C])

#pr=pollaplasiazw me 1/(4*pi*5) giati einai 1/4pi kai epi 5 gia ta hlektronia
n_k_C=1/(4*np.pi*6)*(2*K1sC**2+2*K2sC**2+2*K2pC**2)#DOS

Int_nk_C=rhf.Integrate_symbolic_0_inf_k(4*sp.pi*k**2*n_k_C)
t_C.add_row(['Int_nk',Int_R1s_C])

t_C.add_row(['S_r',S_r_C])

rhf.Plot_sumbolic_k(n_k_C, 3, "The DOS of C in the k space", "k", "n(k)","C")

S_k_C=rhf.Integrate_symbolic_0_inf_k(-4*sp.pi*n_k_C*sp.log(n_k_C+epsilon)*k**2)
s_K_values.append(S_k_C)
t_C.add_row(['S_k',S_k_C])

S_ol_C=S_r_C+S_k_C
s_ol_values.append(S_ol_C)
t_C.add_row(['S_ol',S_ol_C])

print(t_C)
print("\n")


#Element 6, Z=7: N
print("Studying N:")
coeficiants_1s_N=[0.354839 ,0.472579 ,-0.001038 ,0.208492,0.001687 ,0.000206 ,0.000064]
coeficiants_2s_N=[-0.067498,0.434142,-0.000315,-0.080331,-0.374128,-0.522775,-0.207735]
coeficiants_2p_N=[0.006323 ,0.082938 ,0.260147 ,0.418361 ,0.308272 ,0.0 ,0.0]


z_values_s_N=[9.9051 ,5.7429 ,17.9816 ,8.3087 ,2.7611,1.8223,1.4191]
z_values_p_N=[8.3490,3.8827,2.5920,1.6946,1.1914,1,1]

R1sN,R2sN,R2pN=rhf.RHF_wavefunctions_R_one_3s(coeficiants_1s_N, coeficiants_2s_N, coeficiants_2p_N, z_values_s_N, z_values_p_N)

t_N=t = PrettyTable(["Quantity", "Result"])

Int_R1s_N=rhf.Integrate_symbolic_0_inf_R(R1sN**2*r**2)
t_N.add_row(['Int_R1s',Int_R1s_N])


Int_R2s_N=rhf.Integrate_symbolic_0_inf_R(R2sN**2*r**2)
t_N.add_row(['Int_R2s',Int_R2s_N])


Int_R2p_N=rhf.Integrate_symbolic_0_inf_R(R2pN**2*r**2)
t_N.add_row(['Int_R2p',Int_R2p_N])


#pr=pollaplasiazw me 1/(4*pi*7) giati einai 1/4pi kai epi 7 gia ta hlektronia
p_R_N=1/(4*np.pi*7)*(2*R1sN**2+2*R2sN**2+3*R2pN**2)#DOS


Int_pr_N=rhf.Integrate_symbolic_0_inf_R(4*sp.pi*r**2*p_R_N)
t_N.add_row(['Int_pr',Int_pr_N])

rhf.Plot_sumbolic_R(p_R_N, 0.5, "The DOS of N in the position space", "r", "rho(r)","N")

# epsilon is used so there we avoid calculating log0
epsilon=10**(-16)
S_r_N=rhf.Integrate_symbolic_0_inf_R(-4*sp.pi*p_R_N*sp.log(p_R_N+epsilon)*r**2)
s_R_values.append(S_r_N)


# b) k space
K1sN,K2sN,K2pN=rhf.RHF_wavefunctions_k_one_3s(coeficiants_1s_N, coeficiants_2s_N, coeficiants_2p_N, z_values_s_N, z_values_p_N)

Int_K1s_N=rhf.Integrate_symbolic_0_inf_k(K1sN**2*k**2)
t_N.add_row(['Int_K1s',Int_K1s_N])

Int_K2s_N=rhf.Integrate_symbolic_0_inf_k(K2sN**2*k**2)
t_N.add_row(['Int_K2s',Int_K2s_N])

Int_K2p_N=rhf.Integrate_symbolic_0_inf_k(K2pN**2*k**2)
t_N.add_row(['Int_K2p',Int_K2p_N])

#pr=pollaplasiazw me 1/(4*pi*7) giati einai 1/4pi kai epi 7 gia ta hlektronia
n_k_N=1/(4*np.pi*7)*(2*K1sN**2+2*K2sN**2+3*K2pN**2)#DOS

Int_nk_N=rhf.Integrate_symbolic_0_inf_k(4*sp.pi*k**2*n_k_N)
t_N.add_row(['Int_nk',Int_R1s_N])

t_N.add_row(['S_r',S_r_N])

rhf.Plot_sumbolic_k(n_k_N, 3.2, "The DOS of N in the k space", "k", "n(k)","N")

S_k_N=rhf.Integrate_symbolic_0_inf_k(-4*sp.pi*n_k_N*sp.log(n_k_N+epsilon)*k**2)
s_K_values.append(S_k_N)
t_N.add_row(['S_k',S_k_N])

S_ol_N=S_r_N+S_k_N
s_ol_values.append(S_ol_N)
t_N.add_row(['S_ol',S_ol_N])

print(t_N)
print("\n")




#Element 7, Z=8: O
print("Studying O:")
coeficiants_1s_O=[0.360063 ,0.466625 ,-0.000918 ,0.208441,0.002018 ,0.000216 ,0.000133]
coeficiants_2s_O=[-0.064363,0.433186,-0.000275,-0.072497,-0.369900,-0.512627,-0.227421]
coeficiants_2p_O=[0.005626 ,0.126618 ,0.328966 ,0.395422 ,0.231788 ,0.0 ,0.0]


z_values_s_O=[11.2970 ,6.5966 ,20.5019 ,9.5546 ,3.2482,2.1608,1.6411]
z_values_p_O=[9.6471,4.3323,2.7502,1.7525,1.2473,1,1]

R1sO,R2sO,R2pO=rhf.RHF_wavefunctions_R_one_3s(coeficiants_1s_O, coeficiants_2s_O, coeficiants_2p_O, z_values_s_O, z_values_p_O)

t_O=t = PrettyTable(["Quantity", "Result"])

Int_R1s_O=rhf.Integrate_symbolic_0_inf_R(R1sO**2*r**2)
t_O.add_row(['Int_R1s',Int_R1s_O])


Int_R2s_O=rhf.Integrate_symbolic_0_inf_R(R2sO**2*r**2)
t_O.add_row(['Int_R2s',Int_R2s_O])


Int_R2p_O=rhf.Integrate_symbolic_0_inf_R(R2pO**2*r**2)
t_O.add_row(['Int_R2p',Int_R2p_O])


#pr=pollaplasiazw me 1/(4*pi*8) giati einai 1/4pi kai epi 8 gia ta hlektronia
p_R_O=1/(4*np.pi*8)*(2*R1sO**2+2*R2sO**2+4*R2pO**2)#DOS


Int_pr_O=rhf.Integrate_symbolic_0_inf_R(4*sp.pi*r**2*p_R_O)
t_O.add_row(['Int_pr',Int_pr_O])

rhf.Plot_sumbolic_R(p_R_O, 0.45, "The DOS of O in the position space", "r", "rho(r)","O")

# epsilon is used so there we avoid calculating log0
epsilon=10**(-16)
S_r_O=rhf.Integrate_symbolic_0_inf_R(-4*sp.pi*p_R_O*sp.log(p_R_O+epsilon)*r**2)
s_R_values.append(S_r_O)


# b) k space
K1sO,K2sO,K2pO=rhf.RHF_wavefunctions_k_one_3s(coeficiants_1s_O, coeficiants_2s_O, coeficiants_2p_O, z_values_s_O, z_values_p_O)

Int_K1s_O=rhf.Integrate_symbolic_0_inf_k(K1sO**2*k**2)
t_O.add_row(['Int_K1s',Int_K1s_O])

Int_K2s_O=rhf.Integrate_symbolic_0_inf_k(K2sO**2*k**2)
t_O.add_row(['Int_K2s',Int_K2s_O])

Int_K2p_O=rhf.Integrate_symbolic_0_inf_k(K2pO**2*k**2)
t_O.add_row(['Int_K2p',Int_K2p_O])

#pr=pollaplasiazw me 1/(4*pi*8) giati einai 1/4pi kai epi 8 gia ta hlektronia
n_k_O=1/(4*np.pi*8)*(2*K1sO**2+2*K2sO**2+4*K2pO**2)#DOS

Int_nk_O=rhf.Integrate_symbolic_0_inf_k(4*sp.pi*k**2*n_k_O)
t_O.add_row(['Int_nk',Int_R1s_O])

t_O.add_row(['S_r',S_r_O])

rhf.Plot_sumbolic_k(n_k_O, 3.5, "The DOS of O in the k space", "k", "n(k)","O")

S_k_O=rhf.Integrate_symbolic_0_inf_k(-4*sp.pi*n_k_O*sp.log(n_k_O+epsilon)*k**2)
s_K_values.append(S_k_O)
t_O.add_row(['S_k',S_k_O])

S_ol_O=S_r_O+S_k_O
s_ol_values.append(S_ol_O)
t_O.add_row(['S_ol',S_ol_O])

print(t_O)
print("\n")




#Element 8, Z=9: F
print("Studying F:")
coeficiants_1s_F=[0.377498 ,0.443947 ,-0.000797 ,0.213846,0.002183 ,0.000335 ,0.000147]
coeficiants_2s_F=[-0.058489,0.426450,-0.000274,-0.063457,-0.358939,-0.516660,-0.239143]
coeficiants_2p_F=[0.004879 ,0.130794 ,.337876 ,0.396122 ,0.225374 ,0.0 ,0.0]


z_values_s_F=[12.6074 ,7.4101 ,23.2475 ,10.7416 ,3.7543,2.5009,1.8577]
z_values_p_F=[11.0134,4.9962,3.1540,1.9722,1.3632,1,1]

R1sF,R2sF,R2pF=rhf.RHF_wavefunctions_R_one_3s(coeficiants_1s_F, coeficiants_2s_F, coeficiants_2p_F, z_values_s_F, z_values_p_F)

t_F=t = PrettyTable(["Quantity", "Result"])

Int_R1s_F=rhf.Integrate_symbolic_0_inf_R(R1sF**2*r**2)
t_F.add_row(['Int_R1s',Int_R1s_F])


Int_R2s_F=rhf.Integrate_symbolic_0_inf_R(R2sF**2*r**2)
t_F.add_row(['Int_R2s',Int_R2s_F])


Int_R2p_F=rhf.Integrate_symbolic_0_inf_R(R2pF**2*r**2)
t_F.add_row(['Int_R2p',Int_R2p_F])


#pr=pollaplasiazw me 1/(4*pi*9) giati einai 1/4pi kai epi 9 gia ta hlektronia
p_R_F=1/(4*np.pi*9)*(2*R1sF**2+2*R2sF**2+5*R2pF**2)#DFS


Int_pr_F=rhf.Integrate_symbolic_0_inf_R(4*sp.pi*r**2*p_R_F)
t_F.add_row(['Int_pr',Int_pr_F])

rhf.Plot_sumbolic_R(p_R_F, 0.45, "The DOS of F in the position space", "r", "rho(r)","F")

# epsilon is used so there we avoid calculating log0
epsilon=10**(-16)
S_r_F=rhf.Integrate_symbolic_0_inf_R(-4*sp.pi*p_R_F*sp.log(p_R_F+epsilon)*r**2)
s_R_values.append(S_r_F)


# b) k space
K1sF,K2sF,K2pF=rhf.RHF_wavefunctions_k_one_3s(coeficiants_1s_F, coeficiants_2s_F, coeficiants_2p_F, z_values_s_F, z_values_p_F)

Int_K1s_F=rhf.Integrate_symbolic_0_inf_k(K1sF**2*k**2)
t_F.add_row(['Int_K1s',Int_K1s_F])

Int_K2s_F=rhf.Integrate_symbolic_0_inf_k(K2sF**2*k**2)
t_F.add_row(['Int_K2s',Int_K2s_F])

Int_K2p_F=rhf.Integrate_symbolic_0_inf_k(K2pF**2*k**2)
t_F.add_row(['Int_K2p',Int_K2p_F])

#pr=pollaplasiazw me 1/(4*pi*9) giati einai 1/4pi kai epi 9 gia ta hlektronia
n_k_F=1/(4*np.pi*9)*(2*K1sF**2+2*K2sF**2+5*K2pF**2)#DFS

Int_nk_F=rhf.Integrate_symbolic_0_inf_k(4*sp.pi*k**2*n_k_F)
t_F.add_row(['Int_nk',Int_R1s_F])

t_F.add_row(['S_r',S_r_F])

rhf.Plot_sumbolic_k(n_k_F, 3.5, "The DOS of F in the k space", "k", "n(k)","F")

S_k_F=rhf.Integrate_symbolic_0_inf_k(-4*sp.pi*n_k_F*sp.log(n_k_F+epsilon)*k**2)
s_K_values.append(S_k_F)
t_F.add_row(['S_k',S_k_F])

S_ol_F=S_r_F+S_k_F
s_ol_values.append(S_ol_F)
t_F.add_row(['S_ol',S_ol_F])

print(t_F)
print("\n")




#Element 9, Z=10: Ne
# a) Position Space
print("Studying Ne:")
coeficiants_1s_Ne=[0.392290,0.425817,-0.000702,0.217206,+0.002300,0.000463,0.000147]
coeficiants_2s_Ne=[-0.053023,0.419502,-0.000263,-0.055723,-0.349457,-0.523070,-0.246038]
coeficiants_2p_Ne=[0.004391,0.133955,0.342978,0.395742,0.221831,0.0,0.0]


z_values_s_Ne=[13.9074,8.2187,26.0325,11.9249,4.2635,2.8357,2.0715]
z_values_p_Ne=[12.3239,5.6525,3.5570,2.2056,1.4948,1,1]

R1sNe,R2sNe,R2pNe=rhf.RHF_wavefunctions_R_one_3s(coeficiants_1s_Ne, coeficiants_2s_Ne, coeficiants_2p_Ne, z_values_s_Ne, z_values_p_Ne)

t_Ne=t = PrettyTable(["Quantity", "Result"])

Int_R1s_Ne=rhf.Integrate_symbolic_0_inf_R(R1sNe**2*r**2)
t_Ne.add_row(['Int_R1s',Int_R1s_Ne])


Int_R2s_Ne=rhf.Integrate_symbolic_0_inf_R(R2sNe**2*r**2)
t_Ne.add_row(['Int_R2s',Int_R2s_Ne])


Int_R2p_Ne=rhf.Integrate_symbolic_0_inf_R(R2pNe**2*r**2)
t_Ne.add_row(['Int_R2p',Int_R2p_Ne])


#pr=pollaplasiazw me 1/(4*pi*10) giati einai 1/4pi kai epi 10 gia ta hlektronia
p_R_Ne=1/(4*np.pi*10)*(2*R1sNe**2+2*R2sNe**2+6*R2pNe**2)#DOS


Int_pr_Ne=rhf.Integrate_symbolic_0_inf_R(4*sp.pi*r**2*p_R_Ne)
t_Ne.add_row(['Int_pr',Int_pr_Ne])

rhf.Plot_sumbolic_R(p_R_Ne, 0.4, "The DOS of Ne in the position space", "r", "rho(r)","Ne")

# epsilon is used so there we avoid calculating log0
epsilon=10**(-16)
S_r_Ne=rhf.Integrate_symbolic_0_inf_R(-4*sp.pi*p_R_Ne*sp.log(p_R_Ne+epsilon)*r**2)
s_R_values.append(S_r_Ne)


# b) k space
K1sNe,K2sNe,K2pNe=rhf.RHF_wavefunctions_k_one_3s(coeficiants_1s_Ne, coeficiants_2s_Ne, coeficiants_2p_Ne, z_values_s_Ne, z_values_p_Ne)

Int_K1s_Ne=rhf.Integrate_symbolic_0_inf_k(K1sNe**2*k**2)
t_Ne.add_row(['Int_K1s',Int_K1s_Ne])

Int_K2s_Ne=rhf.Integrate_symbolic_0_inf_k(K2sNe**2*k**2)
t_Ne.add_row(['Int_K2s',Int_K2s_Ne])

Int_K2p_Ne=rhf.Integrate_symbolic_0_inf_k(K2pNe**2*k**2)
t_Ne.add_row(['Int_K2p',Int_K2p_Ne])

#pr=pollaplasiazw me 1/(4*pi*10) giati einai 1/4pi kai epi 10 gia ta hlektronia
n_k_Ne=1/(4*np.pi*10)*(2*K1sNe**2+2*K2sNe**2+6*K2pNe**2)#DOS

Int_nk_Ne=rhf.Integrate_symbolic_0_inf_k(4*sp.pi*k**2*n_k_Ne)
t_Ne.add_row(['Int_nk',Int_R1s_Ne])

t_Ne.add_row(['S_r',S_r_Ne])

rhf.Plot_sumbolic_k(n_k_Ne, 4, "The DOS of Ne in the k space", "k", "n(k)","Ne")

S_k_Ne=rhf.Integrate_symbolic_0_inf_k(-4*sp.pi*n_k_Ne*sp.log(n_k_Ne+epsilon)*k**2)
s_K_values.append(S_k_Ne)
t_Ne.add_row(['S_k',S_k_Ne])

S_ol_Ne=S_r_Ne+S_k_Ne
s_ol_values.append(S_ol_Ne)
t_Ne.add_row(['S_ol',S_ol_Ne])

print(t_Ne)
print("\n")


plt.scatter(z_values,s_R_values)
plt.xlabel("Z")
plt.ylabel("Sr(Z)")
plt.grid()
plt.title("The entropy in the position space Sr as a function of Z")
#plt.savefig('S_r.pdf', format='pdf', dpi=300)
plt.show()


plt.scatter(z_values,s_K_values)
plt.xlabel("Z")
plt.ylabel("Sk(Z)")
plt.grid()
plt.title("The entropy in the k-space Sk as a function of Z")
#plt.savefig('S_k.pdf', format='pdf', dpi=300)
plt.show()

plt.scatter(z_values,s_ol_values)
plt.xlabel("Z")
plt.ylabel("S(Z)")
plt.grid()
plt.title("The whole entropy S as a function of Z")
#plt.savefig('S.pdf', format='pdf', dpi=300)
plt.show()