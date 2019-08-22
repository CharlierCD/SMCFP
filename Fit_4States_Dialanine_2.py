#!/usr/bin/env python
from __future__ import division
from scipy.optimize import minimize
from make_plot_grid import make_plot_grid
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
import mpmath
mpmath.mp.dps = 25
from pylab import *
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from matplotlib import pylab
from mpl_toolkits.mplot3d.axes3d import Axes3D
from random import randint
import pandas as pd
import numpy as np
import os
from scipy.linalg import expm, sinm, cosm
import sys
import pandas as pd
import nmrglue as ng
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm
import mpmath
from scipy.optimize import minimize

plt.rcParams["figure.figsize"] = (6.5,5)
plt.rcParams.update({'font.size': 8})

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
np.set_printoptions(precision = 100, linewidth=90, suppress=False)


 
def B1_Inhomogeneity(B1,cw_span_frac,cw_span_std_num,n_sims):
    Sat_Strengh     = B1*2*np.pi
    #cw_span_frac    = 0.10
    #cw_span_std_num = 3
    #n_sims          = 10
    wy_min      =   Sat_Strengh-Sat_Strengh*cw_span_std_num*cw_span_frac
    wy_max      =   Sat_Strengh+Sat_Strengh*cw_span_std_num*cw_span_frac
    x           =   np.linspace(wy_min,wy_max,n_sims)
    sigma       =   cw_span_frac*cw_span_std_num*Sat_Strengh
    gauss_pro   =   norm.pdf(x,Sat_Strengh,sigma)
    scale       =   sigma*np.sqrt(2*np.pi)
    weight_ar   =   gauss_pro*scale
    sum_f       =   np.sum(weight_ar)
    return x, weight_ar, sum_f

def populations_4states(ExchRates):
    kab,    kba,    kbc,    kcb, kcd, kdc   = ExchRates
    pa = 1/(1+(kab/kba)*(1+(kbc/kcb)*(1+(kcd/kdc))))
    pb = pa*(kab/kba)
    pc = pa*(kab/kba)*(kbc/kcb)
    pd = pa*(kab/kba)*(kbc/kcb)*(kcd/kdc)
    #print(pa,pb,pc,pd)
    return pa, pb, pc, pd

def Four_State_Matrix(B0, T1Rates, T2Rates, ExchRates, Shifts, tsat, B1, freq):

    da,     db,     dc,     dd              = Shifts
    T1a,    T1b,    T1c,    T1d             = T1Rates
    T2a,    T2b,    T2c,    T2d             = T2Rates
    kab,    kba,    kbc,    kcb, kcd, kdc   = ExchRates

    pa , pb, pc, pd = populations_4states(ExchRates)
    R1a =   1/T1a
    R1b =   1/T1b
    R1c =   1/T1c
    R1d =   1/T1d

    R2a =   1/T2a
    R2b =   1/T2b
    R2c =   1/T2c
    R2d =   1/T2d

    w1  =   B1*2*np.pi
    wa  =   B0*2*np.pi*(da-freq);
    wb  =   B0*2*np.pi*(db-freq);
    wc  =   B0*2*np.pi*(dc-freq);
    wd  =   B0*2*np.pi*(dd-freq);

    H=np.asarray([
        [0,     0,0,0,0,0,0,0,0,0,0,0,0],#E
        [0,    -R2a-kab,-wa,w1,kba,0,0,  0,0,0,  0,0,0],#Max
        [0,      wa,-R2a-kab,0,0,kba,0,  0,0,0,  0,0,0],#May
        [2*R1a*pa,   -w1,0,-R1a-kab,0,0,kba,  0,0,0,  0,0,0],#Maz
        [0,    kab,0,0,    -R2b-kba-kbc,-wb,w1,   kcb,0,0,  0,0,0],#Mbx
        [0,    0,kab,0,      wb,-R2b-kba-kbc,0,   0,kcb,0,  0,0,0],#Mby
        [2*R1b*pb,  0,0,kab,     -w1,0,-R1b-kba-kbc,   0,0,kcb,  0,0,0],#Mbz
        [0,    0,0,0,    kbc,0,0,    -R2c-kcd-kcb,-wc,w1,   kdc,0,0],#Mcx
        [0,    0,0,0,    0,kbc,0,      wc,-R2c-kcd-kcb,0,   0,kdc,0],#Mcy
        [2*R1c*pc,  0,0,0,    0,0,kbc,     -w1,0,-R1c-kcd-kcb,   0,0,kdc],#Mcz
        [0,    0,0,0,    0,0,0,    kcd,0,0,    -R2d-kdc,-wd,w1],#Mdx
        [0,    0,0,0,    0,0,0,    0,kcd,0,      wd,-R2d-kdc,0],#Mdy
        [2*R1d*pd,  0,0,0,    0,0,0,    0,0,kcd,      -w1,0,-R1d-kdc],#Mdz
        ])

    M0 = np.array([[0.5, 0.0, 0.0, pa, 0.0, 0.0, pb, 0.0, 0.0, pc, 0.0, 0.0, pd ]]).T
    blochmat = H*tsat
    M = np.dot(expm(blochmat),M0)
    Mz = M[6]
    return Mz

def unpack_fit_parameters(x,temp):
    temperatures = [temp]
    fit_par = {}
    for i, temp in enumerate(temperatures):
        pb = Populations_2States[temp]
        pa = 1 - pb
        kex =  kex_2states[temp]

        pd_pb = 0.04
        kab, kba, kbc, kcb, kcd = pb*kex, pa*kex, x[0] , x[1]*300, x[2]
        kdc = (kcd * kbc) / (pd_pb * kcb)
        fit_par[temp] = {
            'rates': (kab, kba, kbc, kcb, kcd, kdc),
            #'shift': (x[3],x[4])
        }
    return fit_par

def unpack_fit_parameters_Export(x,temp):
    pb = Populations_2States[temp]
    pa = 1 - pb
    kex =  kex_2states[temp]

    pd_pb = 0.04
    kab, kba, kbc, kcb, kcd = pb*kex, pa*kex, x[0], x[1]*300, x[2]
    kdc = (kcd * kbc) / (pd_pb * kcb)
    fit_par = np.round([kab, kba, kbc, kcb, kcd, kdc],8)

    return fit_par

def simulate_data(x,df,temp):
    fit_par = unpack_fit_parameters(x,temp)
    sim_intensity = np.zeros(len(df.index))
    
    for i, row in df.iterrows():
        k = fit_par[row['temp']]['rates']
        sim, sim_0 = 0.0, 0.0
        x_s, weight_ar, sum_f = B1_Inhomogeneity(row['b1'],0.1,2,n_sim)
        for b in range(len(x_s)):
            sim += Four_State_Matrix(
                B0          =   row['b0'],
                T1Rates     =   (T1, T1, T1, T1),
                T2Rates     =   (T2, T2, T2, T2),
                ExchRates   =   k,
                Shifts      =   chemical_shifts[row['temp']],
                tsat        =   row['t_sat'],
                B1          =   x_s[b]/(2*np.pi),
                freq        =   row['offset']
            ) * weight_ar[b] / sum_f

        tsat = row['t_sat']
        freq = row['offset']
        if row['reference'] in ['freq','b1']:
            freq = 0
            x_s, weight_ar, sum_f = B1_Inhomogeneity(0.1, 0.1, 2, n_sim)
        else:
            tsat = 0
        for b in range(len(x_s)):
            sim_0 += Four_State_Matrix(
                B0              =   row['b0'],
                T1Rates         =   (T1, T1, T1, T1),
                T2Rates         =   (T2, T2, T2, T2),
                ExchRates       =   k,
                Shifts          =   chemical_shifts[row['temp']],
                tsat            =   tsat,
                B1              =   x_s[b]/(2*np.pi),
                freq            =   freq
            ) * weight_ar[b] / sum_f
        sim_intensity[i] = (sim / (sim_0 + 1e-10))[0]
    return sim_intensity

def plot_data(x, temp, tsat, b1, b0, freq, reference):

    l = max(np.array(temp).size, np.array(tsat).size, np.array(b1).size, np.array(b0).size, np.array(freq).size)
    data = np.zeros((l, 7))
    data[:, 0] = temp
    data[:, 1] = freq
    data[:, 2] = tsat
    data[:, 3] = b1
    data[:, 5] = b0

    df = pd.DataFrame(data,
        columns = ['temp', 'offset', 't_sat', 'b1', 'intensity', 'b0', 'reference']
    )
    df['reference'] = reference
    return simulate_data(x, df,temp)

def fit_objective_BootStrap(x,df,temp):
    sim_intensity = simulate_data(x,df,temp)
    rmsd = np.sqrt(np.mean((sim_intensity - df['intensity'].values)**2))
    #print(rmsd)
    return rmsd

def fit_objective(x,df,temp):
    sim_intensity = simulate_data(x,df,temp)
    diff = (sim_intensity - df['intensity'].values)**2
    n = sim_intensity.size
    noise = df['rmsd'].values

    log_lik = np.sum(
        -diff / (2 * noise**2)
    )
    # print(-log_lik)
    return -log_lik

def One_BootStrap_Run(s,ImputData,T_Fit):
    data = ImputData.copy(deep=True)
    data = data[
        (data['temp'] == T_Fit )]
    data.index = np.arange(0, len(data))

    #BootStrapping
    n_data = len(data)

    p = np.ones(n_data)
    p[:int(round(0.1*n_data,0))] = 0
    np.random.shuffle(p)
    p /= np.sum(p)

    new_idx = np.random.choice(
        np.arange(n_data),
        size=n_data,
        replace=True,
        p=p
    )


    data = data.loc[new_idx,:]
    data.index = np.arange(0, len(data))

    res = minimize(
        fit_objective_BootStrap,
        x0=[
             Initial_Conditions[T_Fit][0],     # kex
             Initial_Conditions[T_Fit][1],     # kex
             Initial_Conditions[T_Fit][2],     # kex
             # 4.3,
             # 7.7
        ],
        bounds=[
             (1e-6, np.inf),
             (1e-6, np.inf),
             (1e-6, np.inf),
             # (1e-6, np.inf),
             # (1e-6, np.inf)
        ],
        method='SLSQP',
        options={'ftol': 1e-6},
        args=(data,T_Fit),
    )
    rates = unpack_fit_parameters([res.x[0], res.x[1], res.x[2]],T_Fit)[T_Fit]['rates']
    pop   = populations_4states(rates)

    return T_Fit, res.x[0], res.x[1], res.x[2], pop[0], pop[1], pop[2], pop[3]

def Final_Run_Error(ImputData,T_Fit):

    data = ImputData.copy(deep=True)
    data = data[
        (data['temp'] == T_Fit )]
    data.index = np.arange(0, len(data))

    res = minimize(
        lambda *args: fit_objective(*args) * 0.01,
        x0=[
             Initial_Conditions[T_Fit][0],     # kex
             Initial_Conditions[T_Fit][1],     # kex
             Initial_Conditions[T_Fit][2],     # kex
             # 4.3,
             # 7.7
        ],
        bounds=[
             (1e-6, np.inf),
             (1e-6, np.inf),
             (1e-6, np.inf),
             # (1e-6, np.inf),
             # (1e-6, np.inf)
        ],
        method='SLSQP',
        options={'ftol': 1e-6},
        args=(data,T_Fit),
    )
    err = error_from_hessian(fit_objective, res.x, args=(data, T_Fit))
    return res.x, err

def rmsd_calculation_Error(ImputData,ResultsFit,T_Fit):
    data = ImputData.copy(deep=True)
    data['rmsd']= 0

    for temperature in T_Fit:
        Data = data[
            (data['temp']       ==  temperature) 
        ]
        Data.index = np.arange(0, len(Data))

        Res = ResultsFit[
            (ResultsFit['Temp'] == temperature)]
        Res.drop(['Temp'],axis=1,inplace=True)

        kbc_fit = np.mean(Res['kbc'])
        kcb_fit = np.mean(Res['kcb'])
        kcd_fit = np.mean(Res['kcd'])
        k_vect = [kbc_fit,kcb_fit,kcd_fit]
        sim_intensity = simulate_data(k_vect,Data,temperature)

        for i in range(len(sim_intensity)):
            rmsd = np.sqrt(np.mean((sim_intensity[i] - Data['intensity'].loc[i])**2))

            state = Data.loc[i,'state']
            ref = Data.loc[i,'reference']
            temp = Data.loc[i,'temp']
            intensity = Data.loc[i,'intensity']

            _data = data[
            (data['state']      ==  state) &
            (data['intensity']      ==  intensity) &
            (data['temp']       ==  temp) &
            (data['reference']  ==  ref)
            ]
            idx = _data.index
            data['rmsd'].iloc[idx] = rmsd
    return data

def error_from_hessian(f, x_0, args, eps = 1e-6):
    x_0 = np.asarray(x_0, dtype=np.float64)
    eps = np.finfo(x_0.dtype).eps**0.25 * np.ones(x_0.shape)
    hessian = np.zeros((x_0.shape[0], x_0.shape[0]), dtype=np.float64)
    for i in range(x_0.shape[0]):
        for j in range(x_0.shape[0]):
            if i == j:
                x = np.copy(x_0)
                x[i] += 2*eps[i]
                f_0 = f(x, *args)
 
                x = np.copy(x_0)
                x[i] += eps[i]
           
                f_1 = f(x, *args)
 
                x = np.copy(x_0)
                x[i] -= eps[i]
           
                f_2 = f(x, *args)
 
                x = np.copy(x_0)
                x[i] -= 2*eps[i]
           
                f_3 = f(x, *args)
 
                x = np.copy(x_0)
 
                hessian[i,i] = (-f_0 + 16*f_1 -30*f(x_0, *args) + 16*f_2 - f_3) / (12*eps[i]**2 + 1e-10)
            else:           
                x = np.copy(x_0)
                x[i] += eps[i]
                x[j] += eps[j]
                f_0 = f(x, *args)
 
                x = np.copy(x_0)
                x[i] += eps[i]
                x[j] -= eps[j]
                f_1 = f(x, *args)
 
                x = np.copy(x_0)
                x[i] -= eps[i]
                x[j] += eps[j]
                f_2 = f(x, *args)
 
                x = np.copy(x_0)
                x[i] -= eps[i]
                x[j] -= eps[j]
                f_3 = f(x, *args)
 
                x = np.copy(x_0)
 
                hessian[i,j] = (f_0 - f_1 - f_2 + f_3) / (4*eps[i]*eps[j] + 1e-10)
 
    err = np.sqrt(np.abs(np.diag(np.linalg.pinv(hessian))))
    return err

def Rates_Pop_error(Res,Temp):
    kbc     = Res.loc[Temp,'kbc']
    kbc_err = Res.loc[Temp,'kbc_err']
    kcb     = Res.loc[Temp,'kcb']*300
    kcb_err = Res.loc[Temp,'kcb_err']
    kcd     = Res.loc[Temp,'kcd']
    kcd_err = Res.loc[Temp,'kcd_err']
    pd_pb = 0.04
    pd_pb_err = 0.0001    
    kdc = (kcd * kbc) / (pd_pb * kcb)
    
    kdc_err =  kdc*sqrt(
        (kcd_err*2+kbc_err**2)/(kcd+kbc)**2 +
        (pd_pb_err*2+kcb_err**2)/(pd_pb_err+kcb)**2 
    )
    rates   = [kbc,kcb,kcd,kdc]
    err     = [kbc_err,kcb_err,kcd_err,kdc_err]
    return rates, err

T1  =   25
T2  =   1
#
chemical_shifts = {
        293: (5.043,    9.552,      4.3438,      7.7371+0.0005),
        298: (5.099,    9.609,      4.3969,      7.7934+0.0005),
        303: (5.154,    9.664,      4.4509,      7.8479+0.0005),
        308: (5.208,    9.718,      4.5196,      7.9012+0.0005),
        313: (5.261,    9.771,      4.5704,      7.9539+0.0005),
        318: (5.313,    9.823,      4.6195,      8.0049+0.0005),
                    }
kex_2states =   {
        293: (0.018),
        298: (0.027),
        303: (0.038),
        308: (0.056),
        313: (0.08),
        318: (0.12),
}
Populations_2States = {
        293: (0.21),
        298: (0.24),
        303: (0.28),
        308: (0.32),
        313: (0.35),
        318: (0.40),
}

Initial_Conditions = {
        293: (2.0, 0.1, 0.2),
        298: (2.00, 0.4, 0.4),
        303: (3.00, 1.8, 1.8),
        308: (4.00, 1.2, 1.2),
        313: (5.00, 1.6, 1.6),
        318: (6.00, 2.0, 2.0)
}

T_Fit = [293,298,303,308,313,318]
dopal_diala_data = pd.read_csv('./dopal_Dialanine_consolidated.csv')

n_sim       = 10
n_threads   = 20
n_mc        = 1000


if __name__ == '__main__':
    # seeds = np.random.randint(0, 2**32, size=(n_mc))
    # result = Parallel(n_jobs=n_threads)(delayed(One_BootStrap_Run)(s,dopal_diala_data,t) for s in tqdm(seeds) for t in tqdm(T_Fit))
    # Res = pd.DataFrame(result)

    # Col = ['Temp','kbc','kcb','kcd','pa','pb','pc','pd']
    # Res.columns =  Col
    # Res.round(4).to_csv('Results_Dopal_dialanine_BootStrap_1000',sep='\t')
    # # import Figure_CEST_Dialanine_4States_2D
    # # import Results_CEST_Dialanine_4States
    # #import Figure_CEST_Dialanine_4States_3D

    try:
        Res
    except NameError:
        Res = pd.read_csv('Results_Dopal_dialanine_BootStrap_1000',sep='\t',usecols=['Temp','kbc','kcb','kcd','pa','pb','pc','pd'])
    
    SaveData = 'N'
    if SaveData == 'Y':
        data_rmsd = rmsd_calculation_Error(dopal_diala_data,Res,[293,298,303,308,313,318])
        data_rmsd.to_csv('./dopal_Dialanine_consolidated_Error.csv',sep='\t')
    else:
        data_rmsd = pd.read_csv('./dopal_Dialanine_consolidated_Error.csv',sep='\t')
    Res_FinalFit = pd.DataFrame(index=T_Fit,columns=['kbc','kbc_err','kcb','kcb_err','kcd','kcd_err'])

    for temp in tqdm(T_Fit):
        rates, err = Final_Run_Error(data_rmsd,temp)
        Res_FinalFit.loc[temp,'kbc'] = rates[0]
        Res_FinalFit.loc[temp,'kcb'] = rates[1]
        Res_FinalFit.loc[temp,'kcd'] = rates[2]

        Res_FinalFit.loc[temp,'kbc_err'] = err[0]
        Res_FinalFit.loc[temp,'kcb_err'] = err[1]*300
        Res_FinalFit.loc[temp,'kcd_err'] = err[2]

    Res_FinalFit.to_csv('FinalFit_Dopal_Dialanine.csv',sep='\t')
