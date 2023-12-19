# BREWSTER_FFT
# Alastair McLean
#
#
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from scipy.integrate import odeint
import numpy as np
from scipy import fftpack
from scipy.integrate import odeint
from scipy import signal as sig
from scipy.signal import butter, lfilter, freqz
import scipy.optimize
import scipy.constants as cst
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # for unequal plot boxes
import sys
#from scipy.optimize import curve_fit
#import aifc
#%matplotlib inline

# one lorentzian 
def lorenztian(x, mu1, a1, sigma1):
    return (a1/np.pi)*(sigma1/(pow(x-mu1,2)+pow(sigma1,2)))

# two lorentzian functions
def two_lorenztians(x, mu1, mu2, a1, a2, sigma1, sigma2):
    return (a1/np.pi)*(sigma1/(pow(x-mu1,2)+pow(sigma1,2))) + \
    (a2/np.pi)*(sigma2/(pow(x-mu2,2)+pow(sigma2,2)))

    
# one gaussian 
def gaussian(x, mu1, a1, sigma1):
    return a1*np.exp(-pow(x-mu1,2)/(2*pow(sigma1,2)))

# two gaussian functions
def two_gaussians(x, mu1, mu2, a1, a2, sigma1, sigma2):
    return a1*np.exp(-pow(x-mu1,2)/(2*pow(sigma1,2))) + \
    a2*np.exp(-pow(x-mu2,2)/(2*pow(sigma2,2)))

# response function for the single oscillator
def single(x, A, tau, w_d, phi, y_offset):
    return y_offset+A*np.exp(-x/tau)*np.cos(w_d*x+phi) 
  
# response function for the double oscillator
def fitfunction(x, A1, A2, tau1, tau2, w_d1, w_d2, phi1, phi2, y_offset1, y_offset2):
    return y_offset1+A1*np.exp(-x/tau1)*np.cos(w_d1*x+phi1)+y_offset2+A2*np.exp(-x/tau2)*np.cos(w_d2*x+phi2)
   
def frequency(w):
    return w/(2*np.pi)
   
def period(w):
    return 1/frequency(w)
    
def qualityfactor(zeta):
    return 1/(2.0*zeta)

def zeta(tau,w_d):
    return 1/(tau*w_d)
   
def powerspectrum(real, imag):
    return np.sqrt(real**2+imag**2)
      
def calculatepowerspectrum(wave, dt):
    wavefft = fftpack.fft(wave)                      # FFT of the wave
    f = fftpack.fftfreq(wave.size, dt)               # frequencies 
    f = fftpack.fftshift(f)                          # shift frequencies from min to max
    wavefftshift = fftpack.fftshift(wavefft)         # shift wavefft order to correspond to f
    power = powerspectrum(np.real(wavefftshift), np.imag(wavefftshift))  # calculate the power spectrum
    power = power/np.max(power)
    return power, f
    
def plotwave(xsize, ysize, x, y, xlabel, ylabel, title, output):
    fignum = 1
    plt.figure(fignum, figsize=(xsize, ysize))
    plt.plot(x, y)
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)
    plt.title(title, fontsize = 16)
    if output == True:
        plt.savefig(title +'.pdf')
    plt.show()
    
def plot_power_model(xsize, ysize, x, y_data, y_model, xlabel, ylabel, title, output):
    fignum = 1
    plt.figure(fignum, figsize=(xsize, ysize))
    plt.plot(x, y_data,'r',)
    plt.plot(x, y_model,'b',)
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)
    plt.title(title, fontsize = 16)
    if output == True:
        plt.savefig(title +'.pdf')
    plt.show()
     
def plotwaveandtwopoints(xsize, ysize, x, y, xlabel, ylabel, title, output, n1, n2):
    fignum = 1
    plt.figure(fignum, figsize=(xsize, ysize))
    plt.plot(x, y)
    plt.plot(x[n1],y[n1],'ro')
    plt.plot(x[n2],y[n2],'ro')
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)
    plt.title(title, fontsize = 16)
    if output == True:
        plt.savefig(title +'.pdf')
    plt.show()
    
def plotfitfunction(x_fit, y_fit, xt, yt, resids, A1, dA1):
    fig = plt.figure(1, figsize=(14,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[6, 2])

    # Top plot: data and fit
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x_fit, y_fit,'b-')
    #ax1.plot(x_fit, y_fit)
    #ax1.errorbar(xt, yt, yerr=dy, fmt='or', ecolor='black')
    ax1.plot(xt, yt, 'ro')
    ax1.set_ylabel('Response (V)', fontsize = 16)
    #plt.axis([387.9,390.5,0,70])

    # Bottom plot: residuals
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(xt, resids, 'g-')
    #ax2.errorbar(xt, resids, yerr = dy, ecolor="black", fmt="ro")
    ax2.axhline(color="gray", zorder=-1)
    ax2.set_xlabel('Time (s)', fontsize = 16)
    ax2.set_ylabel('Residuals', fontsize = 16)
    #ax2.set_ylim(-1, 0.1)
    #ax2.set_yticks((-5, 0, 5))
    #plt.savefig('figure.pdf')
    
    #print('A1 = {0:0.1e}+/-{1:0.1e}'.format(np.absolute(A1), dA1))
    #print('tau1 = {0:0.1e}+/-{1:0.1e}'.format(tau1, dtau1))
    #print('w_d1 = {0:0.1e}+/-{1:0.1e}'.format(w_d1, dw_d1))
    #print('phi1 = {0:0.1e}+/-{1:0.1e}'.format(phi1, dphi1))
    #print('y_offset1 = {0:0.1e}+/-{1:0.1e}'.format(y_offset1, dy_offset1))
    #print('A2 = {0:0.1e}+/-{1:0.1e}'.format(np.absolute(A2), dA2))
    #print('tau2 = {0:0.1e}+/-{1:0.1e}'.format(tau2, dtau2))
    #print('w_d2 = {0:0.1e}+/-{1:0.1e}'.format(w_d2, dw_d2))
    #print('phi2 = {0:0.1e}+/-{1:0.1e}'.format(phi2, dphi2))
    #print('y_offset2 = {0:0.1e}+/-{1:0.1e}'.format(y_offset2, dy_offset2))
    plt.show()
    
    
def powerfitreport(mu1,dmu1,mu2,dmu2,a1,da1,a2,da2,sigma1,dsigma1,sigma2, dsigma2):
    print("mu1 = {0:0.2e}+/-{1:0.1e}".format(np.absolute(mu1), dmu1))
    print("mu2 = {0:0.2e}+/-{1:0.1e}".format(np.absolute(mu2), dmu2))
    print("sigma1 = {0:0.2e}+/-{1:0.1e}".format(np.absolute(sigma1), dsigma1))
    print("sigma2 = {0:0.2e}+/-{1:0.1e}".format(np.absolute(sigma2), dsigma2))  
    
      
def singlefitreport(A,dA,tau,dtau,w_d,dw_d,phi,dphi,y_offset,dy_offset):
    #print("A =%10.3e"% A)
    print("A = {0:0.1e}+/-{1:0.1e}".format(np.absolute(A), dA))
    #print("tau =%10.3e"% tau)
    print("tau = {0:0.1e}+/-{1:0.1e}".format(np.absolute(tau), dtau))
    #print("w_d =%10.3e"% w_d)
    print("w_d = {0:0.1e}+/-{1:0.1e}".format(np.absolute(w_d), dw_d))
    print("f_d =%10.3e"% frequency(w_d))
    #print("f_d = {0:0.1e}+/-{1:0.1e}".format(np.absolute(f_d), df_d))
    #print("phi =%10.3e rad"% phi)
    print("phi = {0:0.1e}+/-{1:0.1e}".format(np.absolute(phi), dphi))
    #print("phi =%10.3e deg"% np.rad2deg(phi))
    #print("y_offset =%10.3e"% y_offset)  
    print("y_offset = {0:0.1e}+/-{1:0.1e}".format(np.absolute(y_offset), dy_offset))
    print("zeta =%10.3e"% zeta(tau,w_d))
    z = zeta(tau,w_d)
    print("Q = %.1f"% qualityfactor(z))
    
    
def coupledfitreport(A1,dA1,A2,dA2,tau1,dtau1,tau2,dtau2,w_d1,dw_d1,w_d2,dw_d2,phi1,dphi1,phi2,dphi2,y_offset1,dy_offset1,y_offset2,dy_offset2):
	print("A1 = {0:0.1e}+/-{1:0.1e}".format(np.absolute(A1), dA1))
	print("A2 = {0:0.1e}+/-{1:0.1e}".format(np.absolute(A2), dA2))
	
	print("tau1 = {0:0.1e}+/-{1:0.1e}".format(np.absolute(tau1), dtau1))
	print("tau2 = {0:0.1e}+/-{1:0.1e}".format(np.absolute(tau2), dtau2))	
	#print("tau1 =%10.3e"% tau1)
	#print("tau2 =%10.3e"% tau2)
	
	print("w_d1 = {0:0.1e}+/-{1:0.1e}".format(np.absolute(w_d1), dw_d1))
	print("w_d2 = {0:0.1e}+/-{1:0.1e}".format(np.absolute(w_d2), dw_d2))
	#print("w_d1 =%10.3e"% w_d1)
	#print("w_d2 =%10.3e"% w_d2)
	
	#print("f_d1 = {0:0.1e}+/-{1:0.1e}".format(np.absolute(f_d1), df_d1))
	#print("f_d2 = {0:0.1e}+/-{1:0.1e}".format(np.absolute(f_d2), df_d2))
	print("f_d1 =%10.3e"% frequency(w_d1))
	print("f_d2 =%10.3e"% frequency(w_d2))

	print("phi1 = {0:0.1e}+/-{1:0.1e} rad".format(np.absolute(phi1), dphi1))
	print("phi2 = {0:0.1e}+/-{1:0.1e} rad".format(np.absolute(phi2), dphi2))		
	#print("phi1 =%10.3e rad"% phi1)
	#print("phi2 =%10.3e rad"% phi2)
	
	#print("phi1 = {0:0.1e}+/-{1:0.1e} deg".format(np.absolute(phi1), dphi1))
	#print("phi2 = {0:0.1e}+/-{1:0.1e} deg".format(np.absolute(phi2), dphi2))
	print("phi1 =%10.3e deg"% np.rad2deg(phi1))
	print("phi2 =%10.3e deg"% np.rad2deg(phi2))
			
	print("y_offset1 =%10.3e"% y_offset1)  
	print("y_offset2 =%10.3e"% y_offset2)  
	
	print("zeta1 =%10.3e"% zeta(tau1,w_d1))
	print("zeta2 =%10.3e"% zeta(tau2,w_d2))
	z1 = zeta(tau1,w_d1)
	z2 = zeta(tau2,w_d2)
	print("Q1 = %.1f"% qualityfactor(z1))
	print("Q2 = %.1f"% qualityfactor(z2))	


def plotpowerspectrum(t, power, startfrequency, stopfrequency, save, name, ymax):
    fig = plt.figure(2, figsize=(14,6))
    plt.plot(t, power, label='power spectrum')
    plt.legend(loc='upper right')
    plt.xlabel('frequency / Hz', fontsize = 16)
    plt.ylabel('power', fontsize = 16)
    plt.grid(True)
    plt.xlim(startfrequency, stopfrequency)
    plt.ylim(0, ymax)
    if save == True:
        plt.savefig(name) 
    plt.show()
    
    
    