# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:02:35 2025

@author: cdrg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from nelder_mead import nelder_mead, log_likelihood

#==================================================================
noise_dir = "../data/"
file_list = os.listdir(noise_dir)

output_dir = "../output/"

#==================================================================
def import_data(_dir, filename):
    data = []
    with open(_dir + filename) as file:
        for line in file:
            data.append(line.strip().split())
            
    return data

#==================================================================
for st in file_list:
    name, ext = st.split('.')
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(noise_dir + st):
        [time, color, white] = np.array(import_data(noise_dir, st)[2::], 
                                        dtype=float).T
        
        miss_ind = np.isnan(white)
        
        time = time[~miss_ind]
        color = color[~miss_ind]
        
        dt = np.array([1/365.25]+[time[i+1]-time[i] 
                                  for i in range(len(time)-1)],dtype=float)
        
        result = nelder_mead(log_likelihood, np.array([2, -1.2]), color, dt)
        
        amp = round(result[0][0], 2)
        kappa = round(result[0][1], 2)
        
        #==========================================
        fig0 = plt.figure(figsize=(18, 9))
        ax0 = plt.subplot2grid((1, 1), (0, 0))
            
        ax0.plot(time, color, c='k', lw=0.8)
        ax0.plot(time, color, 'o', c='k', ms=6.5)
        
        ax0.yaxis.set_minor_locator(AutoMinorLocator(5))  
        ax0.xaxis.set_minor_locator(AutoMinorLocator(10))
        
        ax0.tick_params(axis='both', which='major',direction='in',
                                bottom=True,top=True,right=True,left=True,
                                length=15, width=3.1, labelsize=22,pad=12)
        
        ax0.tick_params(axis='both', which='minor',direction='in',
                        bottom=True,top=True,right=True,left=True,
                        length=9, width=2.5, labelsize=22,pad=12)
        
        [ax0.spines[b].set_linewidth(2.6) for b in ['top', 'bottom','left','right']]
        [ax0.spines[b].set_color("black") for b in ['top', 'bottom','left','right']]
        
        ax0.set_title("spectral index($\kappa$): {0}\n \
                       colored noise amplitude: {1}".format(kappa, amp), 
                       fontsize=18, pad=20, ha='right', x=0.985, linespacing=1.6) 
        
        ax0.set_xlabel("time in year", fontsize=28, labelpad=15)
        ax0.set_ylabel("colored noise (mm)", fontsize=28, labelpad=15)
        
        fig0.suptitle("{0} Colored Noise".format(name), fontsize=45, y=0.91)
        plt.tight_layout()
        plt.savefig(output_dir + name + ".png", dpi=400)
