import hydropt.hydropt as hd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from hydropt.bio_optics import H2O_IOP_DEFAULT, a_phyto_base_HSI
import lmfit
import csv

# for i in range (1,63):
#     with open('hydropt/data/PACE_polynom_04_h2o.csv','a',newline='',encoding='utf-8') as f:
#         writer=csv.writer(f)
#         c=710+i*5
#         c=str(c)
#         writer.writerow([c,'-2.9647619351540513','-1.2361097525010263','0.9104449363817371',
#                           '12.840776041695658','1.264877309731138','0.06692509149478654',
#                           '69.09924920855089','8.443800353333097','0.49886792963342286',
#                           '0.008579899967093102','706.9733216481112','59.20770873339014',
#                           '1.4976681723040033','-0.010053516793850963','-0.0003469438780323575'])


wavebands=np.arange(400,711,5)
# wavebands=np.array([400,410,440,490,510,560,620,665,675,
#                     680,710,755,760,765,770,
#                     780,865,885,900,940,1020])


#creation of an optical model
def clear_nat_water(*args):
    return H2O_IOP_DEFAULT.T.values

#Absorption and backscattering of the phytoplankton
def phytoplankton(*args):
    chl=args[0]
    a=a_phyto_base_HSI.absorption.values #basis vector
    bb=np.repeat(.014*0.18,len(a)) #constant spectral backscatter with backscatter ratio of 1.4%
    
    return chl*np.array([a, bb])

#Colored dissolved organic matter
def cdom(*args):
    a_440=args[0] #absorption at 440 nm
    a=np.array(np.exp(-0.017*(wavebands-440))) #spectral absorption
    bb=np.zeros(len(a)) #no backscatter
    
    return a_440*np.array([a, bb])

#adding of all optical components created to an instance of th BioOpticalModel class
bio_opt=hd.BioOpticalModel()
#set optical models
bio_opt.set_iop(
    wavebands=wavebands,
    water=clear_nat_water,
    phyto=phytoplankton,
    cdom=cdom)

#checking
#bio_opt.plot(water=None, phyto=1, cdom=1)

#initializing the HYDROPT forward model with the bio-optical model, calculate
#Rrs when the phytoplankton concentration is 0.15mg/m^3 and CDOM absorption
#is 0.02 m^-1

#HYDROPT polynomial forward model
fwd_model=hd.PolynomialForward(bio_opt)

#atmospheric correction
j=146
d = 1 + 0.016*np.cos(2*np.pi*((j-3)/365))
teta_v = 16.10000
teta_s = 62.06009
# day=223
# teta_v=12.6
# teta_s=29.4

fid=open('hydropt/petite_image_HICO_var_26_mai_2013.img','rb')
C=400
L=400
B=63


data=np.fromfile(fid,dtype=np.uint8,count=C*L*B)

fid.close()
CNt=np.reshape(data,(C,L,B))
CN=np.zeros((L,C,B))

for i in range (B):
    CN[:,:,i]=np.transpose(CNt[:,:,i])

L_TOA=CN/50   #top of the atmosphere luminance

#Earth sun distance
#d=1+0.016*np.cos(2*np.pi*(day-3)/365)
with open ('hydropt/F0.txt') as data:
    lambda_,F0=np.loadtxt(data,dtype=float, unpack=True)
    
rho_toa=np.zeros((L,C,B))
for i in range (B):
    rho_toa[:,:,i]=np.pi*L_TOA[:,:,i]/(d*F0[i]*np.cos(teta_s*np.pi/180))

#water mask
# plt.figure()
# plt.hist(rho_toa[:,:,3])
# plt.show()

# threshold=0.008
# mask=rho_toa[:,:,3]<threshold
# mask=mask.astype('uint8')

# plt.figure
# plt.imshow(mask)
    
tau_a_440=0.059
tau_a_1020=0.023
n=np.log(tau_a_1020/tau_a_440)/np.log(1020/440)
tau_a=tau_a_440*(lambda_/440)**n

with open ('hydropt/to_g.txt') as f: #gases optical thickness
    lambda_,to_g=np.loadtxt(f,dtype=float,unpack=True)
    
with open('hydropt/to_r.txt') as f: #air optical thickness
    lambda_,to_r=np.loadtxt(f,dtype=float,unpack=True)
    
ts=np.exp(-(0.5*to_r+tau_a)/np.cos(teta_s))
tv=np.exp(-(0.5*to_r+tau_a)/np.cos(teta_v))
t=ts*tv   #transmittance

tgs=np.exp(-to_g/np.cos(teta_s))
tgv=np.exp(-to_g/np.cos(teta_v))
tg=tgs*tgv

#Rayleigh corrections
with open('hydropt/rho_r.txt') as f:
    lambda_,rho_r=np.loadtxt(f,dtype=float,unpack=True)

rho_c=np.zeros((C,L,B))
for i in range (B):
    rho_c[:,:,i]=rho_toa[:,:,i]-tg[i]*rho_r[i]

#Aerosols correction
ind_776=0
ind_862=0
for i in range (B):
    if 776<=lambda_[i]<777:
        ind_776=i
    if 862<=lambda_[i]<863:
        ind_862=i
        
rho_a_776=rho_c[:,:,ind_776]
rho_a_862=rho_c[:,:,ind_862]
eps=rho_a_776/rho_a_862
c=np.log(eps)/(862-776)

rho_a=np.zeros((L,C,B))
for i in range (B):
    rho_a[:,:,i]=rho_c[:,:,ind_862]*np.exp((862-lambda_[i])*c) #aerosols reflectance
  
rho_s=np.zeros((L,C,B))
for i in range(B):
    rho_s[:,:,i]=1/t[i]*(rho_c[:,:,i]-tg[i]*rho_a[:,:,i])
    


#Calculate Rrs
rrs=fwd_model.forward(phyto=.15,cdom=.02)

#Inversion of the Rrs spectrum using the Levenberg-Marquart routine
#set initial guess parameters for LM
x0=lmfit.Parameters()
#some initial guess
x0.add('phyto',value=5,min=1E-9)
x0.add('cdom',value=.01, min=1E-9)
#initialize an inversion model
inv_model=hd.InversionModel(
    fwd_model=fwd_model,
    minimizer=lmfit.minimize)
#estimate parameters
phyto=[]
cdom=[]

#estimate parameters
for i in range (L):
    print(i)
    for j in range (C):
        Rrs_mesure=np.squeeze(rho_s[i,j,:])
        xhat=inv_model.invert(y=Rrs_mesure,x=x0)
        phyto.append(xhat.last_internal_values[0])
        cdom.append(xhat.last_internal_values[1])