#LIBRERIE
import numpy as np 
import sympy as sym
from filterpy.kalman import KalmanFilter 
import scipy
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
import matplotlib.pyplot as plt
from matplotlib import style
import pymap3d as pym
import pandas as pd
import astropy
from poliastro.bodies import Moon 
from astropy import units as u
from poliastro.twobody.propagation import cowell as cowell
from poliastro.core.perturbations import J3_perturbation, J2_perturbation
from poliastro.core.propagation import func_twobody
import glob
import math
from sklearn import linear_model, datasets
import matplotlib.style as style
import time
from tkinter import SW
from numpy.linalg import inv, det
import pickle
from utilitys.utils import *
from utilitys.MieFunzioni import *
from poliastro.twobody.orbit import Orbit
from poliastro.plotting import OrbitPlotter3D
from poliastro.constants import J2000
import cv2
from poliastro.twobody.orbit import Orbit

import seaborn as sns
sns.set()
###########################
#COSTANTI
dt = 10
mi = 4.9048695e3 # km^3/s^2 
S = 0.006 # m (focal length 600 mm per la wide angle camera)
FOV=61.4 #° WIDE ANGLE CAMERA
SFT = 0

#############################
#DATI REALI
#Lat, Long, Alt
df = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\definitivo\nuova2\lla2.csv") 
real_Latitudes, real_Longitudes, real_Altitudes = df['Lat (deg)'].to_numpy().astype(float), df['Lon (deg)'].to_numpy().astype(float), df['Alt (km)'].to_numpy().astype(float)

#posizione
dpf = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\definitivo\nuova2\pos2.csv") 
real_X, real_Y, real_Z  = dpf['x (km)'].to_numpy().astype(float), dpf['y (km)'].to_numpy().astype(float),dpf['z (km)'].to_numpy().astype(float)

#real_Vxs, real_Vys, real_Vzs = [(float(real_X[i+1])-float(real_X[i]))/10 for i in range(len(dpf)-2)],[(float(real_Y[i+1])-float(real_Y[i]))/10 for i in range(len(dpf)-2)],[(float(real_Z[i+1])-float(real_Z[i]))/10 for i in range(len(dpf)-2)]
#real_Vxs, real_Vys, real_Vzs = np.array(real_Vxs), np.array(real_Vys), np.array(real_Vzs)
#angoli di Eulero
dq = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\definitivo\nuova2\att2.csv")  
real_t1 = dq['yaw (deg)'].to_numpy().astype(float)
real_t2 = dq['pitch (deg)'].to_numpy().astype(float)  
real_t3 = dq['roll (deg)'].to_numpy().astype(float)

# #velocità angolari
real_om1, real_om2, real_om3 = [(float(real_t1[i+1])-float(real_t1[i]))/10 for i in range(len(dq)-2)],[(float(real_t2[i+1])-float(real_t2[i]))/10 for i in range(len(dq)-2)],[(float(real_t3[i+1])-float(real_t3[i]))/10 for i in range(len(dq)-2)]
real_om1, real_om2, real_om3 = np.array(real_om1), np.array(real_om2), np.array(real_om3)

#velocita' 
dpv = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\definitivo\nuova2\fixed_pos_vel2.csv") 
real_Vxs, real_Vys, real_Vzs  = dpv['vx (km/sec)'].to_numpy().astype(float), dpv['vy (km/sec)'].to_numpy().astype(float),dpv['vz (km/sec)'].to_numpy().astype(float)
 

real_Latitudes, real_Longitudes, real_Altitudes = real_Latitudes[SFT:], real_Longitudes[SFT:], real_Altitudes[SFT:]
real_X, real_Y, real_Z  = real_X[SFT:], real_Y[SFT:], real_Z[SFT:]
real_Vxs, real_Vys, real_Vzs  = real_Vxs[SFT:]*(-1), real_Vys[SFT:]*(-1), real_Vzs[SFT:]*(-1)
real_t1, real_t2, real_t3 = real_t1[SFT:], real_t2[SFT:], real_t3[SFT:]
real_om1, real_om2, real_om3 = real_om1[SFT:], real_om2[SFT:], real_om3[SFT:]


#Condizioni iniziali
initx, inity, initz = real_X[0], real_Y[0], real_Z[0]
initvx, initvy, initvz = real_Vxs[0], real_Vys[0], real_Vzs[0]
initteta1, initteta2, initteta3 = real_t1[0], real_t2[0], real_t3[0]
initom1, initom2, initom3 = real_om1[0], real_om2[0], real_om3[0]


########################################
#FILTRO DI KALMAN

#State transition matrix
F=find_F_matrix(initx, inity, initz,initom1, initom2, initom3) 
phii= np.eye(12)+F*dt 

#Process noise matrix
q = Q_discrete_white_noise(dim=3, dt=dt, var=0.001)
Q = block_diag(q, q, q, q)   


X = np.array([[initx, inity, initz, initvx, initvy, initvz, initteta1, initteta2, initteta3,initom1, initom2, initom3]]).T
X_in = np.array([[initx, inity, initz, initvx, initvy, initvz, initteta1, initteta2, initteta3,initom1, initom2, initom3]]).T

#Process covariance matrix

P=np.diag([ 36.77411499,33.63011419,40.90912484,10.62378216 ,12.12832917,16.23225206,2.81504593,0.04198674,0.61019117,3.35623915,4.84792314,4.60937118]) 

#Measurements noise matrix 
R=np.eye(6)*0.001


#Measurements matrix
H_matrix=np.zeros((6,12))
H_matrix[0,0]=1
H_matrix[1,1]=1
H_matrix[2,2]=1
H_matrix[3,6]=1
H_matrix[4,7]=1
H_matrix[5,8]=1

#################
#MISURE

with open('test_1_30.pkl', 'rb') as handle:
    b = pickle.load(handle)



#############################
kvx=[]
mus = []
mus.append(X)
teta1,teta2,teta3 = X[6,0],X[7,0],X[8,0]
omm1,omm2,omm3=X[9,0],X[10,0],X[11,0]

pos,vel=[],[]
for i in range(1000):
    foo,foo2=cowell_propagation(X_in)
    pos.append(foo)
    vel.append(foo2)
    X_in=np.array([[foo[0],foo[1],foo[2], foo2[0],foo2[1],foo2[2],0,0,0,0,0,0]]).T

pos=np.array(pos)
vel=np.array(vel)

soluz=attitude_propagator(teta1,teta2,teta3, omm1,omm2,omm3, dt)
#for i in range(len(df)):    
for i in range(998):
    step=1 
    #print("Iterazione ", i)
    
    t1=soluz[0,i]
    t2=soluz[1,i]
    t3=soluz[2,i]
    om1=soluz[3,i]
    om2=soluz[4,i]
    om3=soluz[5,i]
    F = find_F_matrix(pos[i,0],pos[i,1],pos[i,2],om1,om2,om3)
    phii = np.eye(12)+F*dt 
    X=np.array([[pos[i,0],pos[i,1],pos[i,2], vel[i,0],vel[i,1],vel[i,2], t1,t2,t3,om1, om2, om3]]).T  
    # Predict:
    P = kf_predict(P, phii, Q)

    
    pnp_x=b['x'][i+step]
    pnp_y=b['y'][i+step]
    pnp_z=b['z'][i+step]
    pnp_te1=b['teta1'][i+step]
    pnp_te2=b['teta2'][i+step]
    pnp_te3=b['teta3'][i+step]


    if pnp_x is not None and pnp_y is not None and pnp_z is not None and pnp_te1 is not None and pnp_te2 is not None and pnp_te3 is not None:
        mis=np.array([[pnp_x,pnp_y,pnp_z,pnp_te1,pnp_te2,pnp_te3]]).T
        (X, P, K, IM, IS) = kf_update(X, P, mis, H_matrix, R) 

    mus.append(X)
#print(K)
mus=np.array(mus)

lw=1


x_pred = []
y_pred = []
z_pred = []
vx_pred = []
vy_pred = []
vz_pred = []
t1_pred = []
t2_pred = []
t3_pred = []
om1_pred = []
om2_pred = []
om3_pred = []

for mu in mus:
    x = mu[0]
    x_pred.append(x)

    y = mu[1]
    y_pred.append(y)
       
    z = mu[2]
    z_pred.append(z)
    
    vx = mu[3]
    vx_pred.append(vx)
    
    vy = mu[4]
    vy_pred.append(vy)
    
    vz = mu[5]
    vz_pred.append(vz)
    
    t1 = mu[6]
    t1_pred.append(t1)
    
    t2 = mu[7]
    t2_pred.append(t2)
    
    t3 = mu[8]
    t3_pred.append(t3)
    
    om1 = mu[9]
    om1_pred.append(om1)
    
    om2 = mu[10]
    om2_pred.append(om2)
    
    om3 = mu[11]
    om3_pred.append(om3)

x_true = real_X[:len(x_pred)]
y_true = real_Y[:len(y_pred)] 
z_true = real_Z[:len(z_pred)]
vx_true = real_Vxs[:len(vx_pred)]
vy_true = real_Vys[:len(vy_pred)]
vz_true = real_Vzs[:len(vz_pred)]
t1_true = real_t1[:len(t1_pred)]
t2_true = real_t2[:len(t2_pred)]
t3_true = real_t3[:len(t3_pred)]
om1_true = real_om1[:len(om1_pred)]
om2_true = real_om2[:len(om2_pred)]
om3_true = real_om3[:len(om3_pred)]



lw=1

plt.figure(dpi=200, tight_layout=True)
#plt.figure()
plt.subplot(3,1,1) 
plt.plot(x_pred, '-k', linewidth=lw)
plt.plot(x_true, 'r', linewidth=lw)
plt.legend(("Predizione","Reale"))
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('X [Km]')

plt.subplot(3,1,2) 
plt.plot(y_pred, '-k', linewidth=lw)
plt.plot(y_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Y [Km]')

plt.subplot(3,1,3) 
plt.plot(z_pred, '-k', linewidth=lw)
plt.plot(z_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Z [Km]')
plt.show(block=False)

plt.figure(dpi=200, tight_layout=True)
plt.subplot(3,1,1) 
plt.plot(vx_pred, '-k', linewidth=lw)
plt.plot(vx_true, 'r', linewidth=lw)
plt.legend(("Predizione","Reale"))
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Vx [km/sec]')

plt.subplot(3,1,2) 
plt.plot(vy_pred, '-k', linewidth=lw)
plt.plot(vy_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Vy [km/sec]')

plt.subplot(3,1,3) 
plt.plot(vz_pred, '-k', linewidth=lw)
plt.plot(vz_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Vz [km/sec]')
plt.show(block=False)

plt.figure(dpi=200, tight_layout=True)
plt.subplot(3,1,1) 
plt.plot(t1_pred, '-k', linewidth=lw)
plt.plot(t1_true, 'r', linewidth=lw)
plt.legend(("Predizione","Reale"))
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Teta1')

plt.subplot(3,1,2) 
plt.plot(t2_pred, '-k', linewidth=lw)
plt.plot(t2_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Teta2')

plt.subplot(3,1,3) 
plt.plot(t3_pred, '-k', linewidth=lw)
plt.plot(t3_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Teta3')
plt.show(block=False)

plt.figure(dpi=200, tight_layout=True)
plt.subplot(3,1,1) 
plt.plot(om1_pred, '-k', linewidth=lw)
plt.plot(om1_true, 'r', linewidth=lw)
plt.legend(("Predizione","Reale"))
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Omega 1 [°/sec]')

plt.subplot(3,1,2) 
plt.plot(om2_pred, '-k', linewidth=lw)
plt.plot(om2_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Omega 2 [°/sec]')

plt.subplot(3,1,3) 
plt.plot(om3_pred, '-k', linewidth=lw)
plt.plot(om3_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Omega 3 [°/sec]')
plt.show(block=False)

plt.figure(dpi=200, tight_layout=True)
plt.subplot(311)
x_pred = np.array(x_pred)
x_true = np.array(x_true)
diff_x = []
for x,y in zip(x_pred,x_true):
    d = (x - y)
    diff_x.append(d)
plt.title('Error along X (LCLF)')
plt.plot(diff_x, '-k', linewidth=lw)
plt.ylabel('Km')
plt.ylim([-1,1])

plt.subplot(312)
y_pred = np.array(y_pred)
y_true = np.array(y_true)
diff_y = []
for x,y in zip(y_pred,y_true):
    d = (x - y)
    diff_y.append(d)
plt.title('Error along Y (LCLF)')
plt.plot(diff_y, '-k', linewidth=lw)
plt.ylabel('Km')
plt.ylim([-1,1])

plt.subplot(313)
z_pred = np.array(z_pred)
z_true = np.array(z_true)
diff_z = []
for x,y in zip(z_pred,z_true):
    d = (x - y)
    diff_z.append(d)
plt.title('Error along Z (LCLF)')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Km')
plt.ylim([-1,1])
plt.show(block=False)


plt.figure(dpi=200, tight_layout=True)
plt.subplot(311)
vx_pred = np.array(vx_pred)
vx_true = np.array(vx_true)
diff_x = []
for x,y in zip(vx_pred,vx_true):
    d = (x - y)
    diff_x.append(d)
plt.title('Error along Vx (LCLF)')
plt.plot(diff_x, '-k', linewidth=lw)
plt.ylabel('Km/sec')
plt.ylim([-1,1])

plt.subplot(312)
vy_pred = np.array(vy_pred)
vy_true = np.array(vy_true)
diff_y = []
for x,y in zip(vy_pred,vy_true):
    d = (x - y)
    diff_y.append(d)
plt.title('Error along Vy (LCLF)')
plt.plot(diff_y, '-k', linewidth=lw)
plt.ylabel('Km/sec')
plt.ylim([-1,1])

plt.subplot(313)
vz_pred = np.array(vz_pred)
vz_true = np.array(vz_true)
diff_z = []
for x,y in zip(vz_pred,vz_true):
    d = (x - y)
    diff_z.append(d)
plt.title('Error along Vz (LCLF)')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Km/sec')
plt.ylim([-1,1])
plt.show(block=False)


plt.figure(dpi=200, tight_layout=True)
plt.subplot(311)
t1_pred = np.array(t1_pred)
t1_true = np.array(t1_true)
diff_x = []
for x,y in zip(t1_pred,t1_true):
    d = (x - y)
    diff_x.append(d)
plt.title('Error in $\Theta$1')
plt.plot(diff_x, '-k', linewidth=lw)
plt.ylabel('deg')
plt.ylim([-1,1])

plt.subplot(312)
t2_pred = np.array(t2_pred)
t2_true = np.array(t2_true)
diff_y = []
for x,y in zip(t2_pred,t2_true):
    d = (x - y)
    diff_y.append(d)
plt.title('Error in $\Theta$2')
plt.plot(diff_y, '-k', linewidth=lw)
plt.ylabel('deg')
plt.ylim([-1,1])

plt.subplot(313)
t3_pred = np.array(t3_pred)
t3_true = np.array(t3_true)
diff_z = []
for x,y in zip(t3_pred,t3_true):
    d = (x - y)
    diff_z.append(d)
plt.title('Error in $\Theta$3')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('deg')
plt.ylim([-1,1])
plt.show(block=False)

plt.figure(dpi=200, tight_layout=True)
plt.subplot(311)
om1_pred = np.array(om1_pred)
om1_true = np.array(om1_true)
diff_x = []
for x,y in zip(om1_pred,om1_true):
    d = (x - y)
    diff_x.append(d)
plt.title('Error along $\omega$1')
plt.plot(diff_x, '-k', linewidth=lw)
plt.ylabel('deg/sec')
plt.ylim([-1,1])

plt.subplot(312)
om2_pred = np.array(om2_pred)
om2_true = np.array(om2_true)
diff_y = []
for x,y in zip(om2_pred,om2_true):
    d = (x - y)
    diff_y.append(d)
plt.title('Error along $\omega$2')
plt.plot(diff_y, '-k', linewidth=lw)
plt.ylabel('deg/sec')
plt.ylim([-1,1])

plt.subplot(313)
om3_pred = np.array(om3_pred)
om3_true = np.array(om3_true)
diff_z = []
for x,y in zip(om3_pred,om3_true):
    d = (x - y)
    diff_z.append(d)
plt.title('Error along $\omega$3')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('deg/sec')
plt.ylim([-1,1])
plt.show()


