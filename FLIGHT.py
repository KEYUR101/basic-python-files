### FLIGHT.py -- Generic 6-DOF Trim, Linear Model, and Flight Path Simulation

import math
import numpy as np
from scipy.optimize import fmin
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


print('** ======================= **')
print('** 6-DOF FLIGHT Simulation **')
print('** ======================= **\n')

#### Main File. It contain ...
#       Define initial conditions
#       Contain aerodynamic data tables(if required)
#       Calculates longitudinal trim condition
#       Calculate stability & control matrices for linear model
#       Simulate flight path using nonlinear equation of motion

#### Function used by FLIGHT:
#       AeroModelAlpha    High-Alpha, Low-Mach aerodynamic coefficients of the aircraft,
#                         thrust model, and geometric and inertial properties
#       AeroModelMach     Low-Alpha, High-Mach aerodynamic coefficients of the aircraft,
#                         thrust model, and geometric and inertial properties
#       AeroMoedelUser    User-defined aerodynamic corfficietns of the aircraft,
#                         thrust model, and geometric and inertial properties
#       Atoms             Air density, sound speed
#       DCM               Direction-cosine (Rotation) matrix from Euler Angles
#       RMQ               Direction-cosine (Rotation) matrix from Quaternions
#       EoM               Equations of motion for integration (Euler Angles)
#       EoMQ              Equations of motion for integration (Quaternions)    ##### WILL DO THIS LATER ######
#       Linmodel          Equations of motion for defining linear-model
#                         (F & G) matrices via central differences
#       TrimCost          Cost function for trim solution
#       WindField         Wind velocity components

### DEFINITION OF THE STATE VECTOR
##   With Euler Angle DCM option (QUAT=0):
#       x[0]    =   Body-axis x in inertial velocity, ub, m/s
#       x[1]    =   Body-axis y in inertial velocity, vb, m/s
#       x[2]    =   Body-axis z in inertial velocity, wb, m/s
#       x[3]    =   North position of centre of mass w.r.t Earth, xe, m
#       x[4]    =   East position of centre of mass w.r.t Earth, ye, m
#       x[5]    =   Negative position of centre of mass w.r.t Earth, ze = -h, m
#       x[6]    =   Body-axis roll rate, pr, rad/s
#       x[7]    =   Body-axis pitch rate, qr, rad/s
#       x[8]    =   Body-axis yaw rate, rr, rad/s
#       x[9]    =   Roll angle of body w.r.t Earth, phir, rad
#       x[10]   =   Pitch angle of body w.r.t Earth, qhir, rad
#       x[11]   =   Yaw angle of body w.r.t Earth, psir, rad 
##  With Quaternions DCM option (QUAT=1):
#       ##### WILL DO THIS LATER ######


### DEFINITION OF THE CONTROL VECTOR
#       u[0]    =   Elevator, dEr, rad, positive: trailing edge down
#       u[1]    =   Aileron, dAr, rad, positive: left trailing edge down
#       u[2]    =   Rudder, dRr, rad, positive: trailing edge left
#       u[3]    =   Throttle, dT
#       u[4]    =   Asymmetric Spoiler, dASr, rad
#       u[5]    =   Flap, dFr, rad
#       u[6]    =   Stabilator, dSr, rad


### ================================================================================
##  USER INPUT
##  ============
##  --  Flag define which analysis or Input conditions(I.C.)/Inputs will be engaged
##  ===============================================================
##  FLIGHT Flags (1 = ON, 0 = OFF)

MODEL   =   2       # Aerodynamic model selection
                    # 0: Incompressible flow, high angle of attack  ## WILL DO THIS LATER ##
                    # 1: Compressible flow, low angle of attack  ## WILL DO THIS LATER ##
                    # 2: User-Defined model
QUAT    =   0       # 0: Rotation matrix (DCM) from Euler Angles
                    # 1: Rotation matrix (DCM) from Quaternions  ## WILL DO THIS LATER ##
TRIM    =   1       # Trim Flag (=1 to calculate trim at Intial Condition)
LINEAR  =   1       # Linear model flag (=1 to calculate and store F and G)
SIMUL   =   1       # Flight path flag (=1 for nonlinear simulation)
CONHIS  =   1       # Control History ON(=1) or OFF(=0)
RUNNING =   0       # internal flag
GEAR    =   0       # Landing gear DOWN(=1) or UP(=0)
SPOIL   =   0       # Symmetric Spoiler DEPLOYED(=1) or CLOSED(=0)
dF      =   0       # Flap setting, deg


##########################################################################################################################################
#####   Functions             ############################################################################################################
##########################################################################################################################################

def Atoms(geomAlt):
    '''
    Note:   Function does not extrapolate outside altitude range
    Input:  Geometric Altitude, m (positive up)
    Output: Air Density, kg/m^3
            Air Pressure, N/m^2
            Air Temprature, K
            Speed of Sound, m/s
    '''

    # Values Tabualated by Geometric Altitude
    Z       = np.array([-1000,0,2500,5000,10000,11100,15000,20000,47400,51000])  # geometric altitude, m
    H       = np.array([-1000,0,2499,4996,9984,11081,14965,19937,47049,50594])   # geopotential altitude, m
    ppo     = np.array([1,1,0.737,0.533,0.262,0.221,0.12,0.055,0.0011,0.0007])   # p/p0*  #p=standard pressure  #p0*=1.013*10^5 N/m^2
    rro     = np.array([1,1,0.781,0.601,0.338,0.293,0.159,0.073,0.0011,0.0007])  # rho/rho0*  #r=standard density  #r0*=1.225 kg/m^3
    T	    = np.array([288.15,288.15,271.906,255.676,223.252,216.65,216.65,216.65,270.65,270.65])  # Tempraure, K
    a	    = np.array([340.294,340.294,330.563,320.545,299.532,295.069,295.069,295.069,329.799,329.799])  # speed of sound, m/s
    R		= 6367435	# Mean radius of the earth, m
    Dens	= 1.225	    # Air density at sea level, Kg/m^3
    Pres	= 101300	# Air pressure at sea level, N/m^2    

    # Geopotential Altitude, m
    geopAlt = R * geomAlt / (R + geomAlt)

    # Linear Interpolation in Geopotential Altitude for Temprature and Speed of Sound
    temp        = np.interp(geopAlt, Z,T)
    soundSpeed  = np.interp(geopAlt, Z,a)

    # Exponential Interpolation in Geometric Altitude for Air Density and Pressure
    for k in range(2, 10+1):
        if geomAlt <= Z[k]:
            betap	= np.log(ppo[k] / ppo[k-1]) / (Z[k] - Z[k-1])
            betar   = np.log(rro[k] / rro[k-1]) / (Z[k] - Z[k-1])
            airPres = Pres * ppo[k-1] * np.exp(betap * (geomAlt - Z[k-1]))
            airDens = Dens * rro[k-1] * np.exp(betar * (geomAlt - Z[k-1]))
            break
    
    return (airDens, airPres, temp, soundSpeed)

################################################################################
def WindField(height,phir,thetar,psir):
    '''
        Flight Wind Field Interpolation for 3-D wind as a Function of Altitude
    '''

    windh = np.array([-10, 0, 100, 200, 500, 1000, 2000, 4000, 8000, 16000])	# Height, m
    windx = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])	# Northerly wind, m/s
    windy =	np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])	# Easterly wind, m/s
    windz =	np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])	# Vertical wind. m/s
    
    winde = [np.interp(height, windh,windx), np.interp(height, windh,windy), np.interp(height, windh,windz)];   # Earth-relative frame
    winde = np.reshape(winde, (3,1))
    HEB     =   DCM(phir,thetar,psir)
    windb   =   np.matmul(HEB, winde).ravel()                # Body-axis frame  (3x1) matrix

    return windb

################################################################################
def DCM(Phi,Theta,Psi):
        '''
                FLIGHT Earth-to-Body-Axis Direction-Cosine Matrix

                # Euler Angles:
                        # phi,    Roll Angle,  rad
                        # Theta,  Pitch Angle, rad
                        # Psi,    Yaw Angle,   rad
        '''
        sinR = math.sin(Phi)
        cosR = math.cos(Phi)
        sinP = math.sin(Theta)
        cosP = math.cos(Theta)
        sinY = math.sin(Psi)
        cosY = math.cos(Psi)

        H = np.zeros((3,3))

        H[0,0] = cosP*cosY
        H[0,1] = cosP*sinY
        H[0,2] = -sinP
        H[1,0] = sinR*sinP*cosY - cosR*sinY
        H[1,1] = sinR*sinP*sinY + cosR*cosY
        H[1,2] = sinR*cosP
        H[2,0] = cosR*sinP*cosY + sinR*sinY
        H[2,1] = cosR*sinP*sinY - sinR*cosY
        H[2,2] = cosR*cosP

        return H

################################################################################
def RMQ(q1,q2,q3,q4):
        '''
        FLIGHT Earth-to-Body-Axis Rotation (Direction-Cosine) Matrix from Quaternion
        #   Quaternion:
                q1, x Component of quaternion
                q2, y Component of quaternion
                q3, z Component of quaternion
                q4, cos(Euler) Component of quaternion   
        '''

        H = np.zeros((3,3))

        H[0,0] = q1**2 - q2**2 - q3**2 + q4**2
        H[0,1] = 2*(q1*q2 + q3*q4)
        H[0,2] = 2*(q1*q3 - q2*q4)
        H[1,0] = 2*(q1*q2 - q3*q4)
        H[1,1] = -q1**2 + q2**2 - q3**2 + q4**2
        H[1,2] = 2*(q2*q3 + q1*q4)
        H[2,0] = 2*(q1*q3 + q2*q4)
        H[2,1] = 2*(q2*q3 - q1*q4)
        H[2,2] = -q1**2 - q2**2 + q3**2 + q4**2    

        return H

################################################################################
def AeroModelUser(x,u,Mach,alphar,betar,V):
    '''
        T-X Trainer Aerodynamic Coefficients, Thrust Model,
        and Geometric and Inertial Properties for FLIGHT.py
        Low-Angle-of-Attack, Mach-Dependent Model

        Called by:
                EoM
                EoMQ
    '''

    # Mass, Inertial, and Reference Properties
    m               =   4800
    Ixx             =   20950
    Iyy             =   49675
    Izz             =   62525
    Ixz             =   -1710
    cBar            =   3.03
    b               =   10
    S               =   27.77
    lHT             =   5.2
    lVT             =   3.9
    StaticThrust    =   49000

    (airDens,airPres,temp,soundSpeed) = Atoms(-x[5])
    Thrust   =   u[3]*StaticThrust*(airDens/1.225)*(1 - math.exp((-x[5] - 18000)/2000))
    
    # Mach Number Effect on All Incompressible Coefficients
    Prandtl  = 1 / math.sqrt(1 - Mach**2)	 # Prandtl Fac
    
    # Current Longitudinal Characteristics
    # ====================================
    # Lift Coefficient
    CLo     =   0
    CLa     =   4.92
    CLqhat  =   2.49
    CLq     =	CLqhat*cBar/(2*V)
    CLdE    =   0.72
    CLdS    =   CLdE

    # Total Lift Coefficient, w/Mach Correction
    CL      =	(CLo + CLa*alphar + CLq*x[7] + CLdS*u[6] + CLdE*u[0])*Prandtl

    # Drag Coefficient
    CDo     =   0.019
    Epsilon =   0.093*Prandtl
        
    # Total Drag Coefficient, w/Mach Correction
    CD      =	CDo*Prandtl + (Epsilon * CL**2)
                
    # Pitching Moment Coefficient
    StaticMargin    =   0.2
    Cmo     =   0
    Cma 	=	-CLa*StaticMargin
    Cmqhat  =   -4.3
    Cmq     =	Cmqhat*cBar/(2*V)
    CmV     =   0
    CmdE	=	-1.25
    CmdS 	=	CmdE
        
    # Total Pitching Moment Coefficient, w/Mach Correction
    Cm      =	(Cmo + Cma*alphar + Cmq*x[7] + CmdS*u[6] + CmdE*u[0])*Prandtl

    # Current Lateral-Directional Characteristics
    # ===========================================
    # Side-Force Coefficient
    CYo     =   0
    CYb     =   -0.5
    CYphat  =   0
    CYp     =   CYphat*(b/(2*V))
    CYrhat  =   0
    CYr     =   CYrhat*(b/(2*V))
    CYdA	=	0
    CYdR	=	0.04
        
    # Total Side-Force Coefficient, w/Mach Correction
    CYo     =   0
    CY      =	(CYo + CYb*betar + CYdR*u[2] + CYdA*u[1] + CYp*x[6] + CYr*x[8])*Prandtl

    # Rolling Moment Coefficient
    Clo     =   0
    Clb  	=	-0.066
    Clphat  =   -0.5
    Clp 	=	Clphat*(b/(2*V))				
    Clrhat  =   -0.5
    Clr 	=	Clrhat* (b/(2*V))
    CldA 	=	0.12
    CldR 	=	0.03
        
    # Total Rolling-Moment Coefficient, w/Mach Correction
    Cl      =	(Clo + Clb*betar + Clp * x[6] + Clr * x[8] + CldA*u[1] + CldR*u[2])* Prandtl

    # Yawing Moment Coefficient
    Cno     =   0
    CnBeta  =	0.37
    Cnphat  =   -0.06
    Cnp 	=	Cnphat*(b/(2*V))				
    Cnrhat  =   -0.5
    Cnr 	=	Cnrhat*(b/(2*V))				
    CndA 	=	0
    CndR 	=	-0.2
        
    # Total Yawing-Moment Coefficient, w/Mach Correction
    Cn      =	(Cno + CnBeta*betar + Cnp*x[6] + Cnr*x[8] + CndA*u[1] + CndR*u[2])*Prandtl


    return (CD,CL,CY,Cl,Cm,Cn,Thrust,m,Ixx,Iyy,Izz,Ixz,cBar,b,S)

################################################################################
def EoM(t,x):
    '''
		FLIGHT Equations of Motion
	'''
    global CONHIS
    global SPOIL
    global u
    global V
    global uInc
    global tuHis
    global deluHis
    global MODEL
    global RUNNING

    # if MODEL == 0:
    #     from AeroModelAlpha import AeroModelAlpha as AeroModel
    # elif MODEL == 1:
    #     from AeroModelMach import AeroModelMach as AeroModel
    # else:
    #     from AeroModelUser import AeroModelUser as AeroModel

	## Event Function
    (value, terminal, direction) = event(t,x)

	# Earth-to-Body-Axis Transformation Matrix
    HEB = DCM(x[9],x[10],x[11])
	# Atmospheric States
    x[5] = min(x[5], 0)     #Limit x[5] to <=0
    (airDens,airPres,temp,soundSpeed) = Atoms(-x[5])
	# Body-Axis Wind Field
    windb	=	WindField(-x[5],x[9],x[10],x[11])
	# Body-Axis Gravity Components
    gb = np.matmul(HEB, np.array([0,0,9.80665]).reshape((3,1))).ravel()

	# Air-Relative Velocity Vector
    x[0]    =   max(x[0],0)     # Limit axial velocity to >= 0 m/s
    Va		=   np.array([[x[0],x[1],x[2]]]).reshape(3,1).ravel() + windb
    V = math.sqrt(np.matmul(np.matrix.transpose(Va), Va))
    
    alphar = math.atan(Va[2]/abs(Va[0]))
	#alphar  =   min(alphar, (pi/2 - 1e-6))     # Limit angle of attack to <= 90 deg
	#alpha	=	57.2957795 * float(alphar)
    alpha = 57.2957795 * alphar
    betar	= 	math.asin(Va[1] / V)
    beta	= 	57.2957795 * betar
    Mach	= 	V / soundSpeed
    qbar	=	0.5 * airDens * V**2

	# Incremental Flight Control Effects
    if CONHIS >=1 and RUNNING ==1:
		## uInc = array([])
        uInc    =   np.interp(t, tuHis.ravel(),deluHis[:, 0])
        uInc    =   np.matrix.transpose(np.array(uInc))   # Transpose
        uTotal  =   u + uInc
    else:
        uTotal  =   u
	
	# Force and Moment Coefficients; Thrust
    if MODEL == 0:
        (CD,CL,CY,Cl,Cm,Cn,Thrust) = AeroModelAlpha(x,uTotal,Mach,alphar,betar,V)
    elif MODEL == 1:
        (CD,CL,CY,Cl,Cm,Cn,Thrust) = AeroModelMach(x,uTotal,Mach,alphar,betar,V)
    else:
        (CD,CL,CY,Cl,Cm,Cn,Thrust,m,Ixx,Iyy,Izz,Ixz,cBar,b,S) = AeroModelUser(x,uTotal,Mach,alphar,betar,V)
    
    
    #(CD,CL,CY,Cl,Cm,Cn,Thrust) = AeroModelUser(x,uTotal,Mach,alphar,betar,V)
    
    qbarS   =   qbar * S
    
    CX	=	-CD * math.cos(alphar) + CL * math.sin(alphar)	# Body-axis X coefficient
    CZ	= 	-CD * math.sin(alphar) - CL * math.cos(alphar)	# Body-axis Z coefficient 

	# State Accelerations
    Xb =	(CX * qbarS + Thrust) / m
    Yb =	CY * qbarS / m
    Zb =	CZ * qbarS / m
    Lb =	Cl * qbarS * b
    Mb =	Cm * qbarS * cBar
    Nb =	Cn * qbarS * b
    nz	=	-Zb / 9.80665               # Normal load factor

	# Dynamic Equations
    xd1 = Xb + gb[0] + x[8] * x[1] - x[7] * x[2]
    xd2 = Yb + gb[1] - x[8] * x[0] + x[6] * x[2]
    xd3 = Zb + gb[2] + x[7] * x[0] - x[6] * x[1]

    HEB_T = np.matrix.transpose(HEB)
    y = np.matmul(HEB_T, np.array([x[0], x[1], x[2]]))
     
    xd4 = y[0]
    xd5 = y[1]
    xd6 = y[2]
    
    xd7	= 	((Izz * Lb + Ixz * Nb - (Ixz * (Iyy - Ixx - Izz) * x[6] + (Ixz**2 + Izz * (Izz - Iyy)) * x[8]) * x[7]) / (Ixx * Izz - Ixz**2))
    xd8 = 	((Mb - (Ixx - Izz) * x[6] * x[8] - Ixz * (x[6]**2 - x[8]**2)) / Iyy)
    xd9 =	((Ixz * Lb + Ixx * Nb + (Ixz * (Iyy - Ixx - Izz) * x[8] + (Ixz**2 + Ixx * (Ixx - Iyy)) * x[6]) * x[7]) / (Ixx * Izz - Ixz**2))
    
    cosPitch	=	math.cos(x[10])
    if abs(cosPitch)	<=	0.00001:
        cosPitch	=	0.00001 * (abs(cosPitch)/cosPitch)      # sign(cosPitch) == (abs(cosPitch)/cosPitch) for python
    tanPitch	=	math.sin(x[10]) / cosPitch
    
    xd10	=	x[6] + (math.sin(x[9]) * x[7] + math.cos(x[9]) * x[8]) * tanPitch
    xd11	=	math.cos(x[9]) * x[7] - math.sin(x[9]) * x[8]
    xd12	=	(math.sin(x[9]) * x[7] + math.cos(x[9]) * x[8]) / cosPitch
    
    xdot	=	np.array([xd1,xd2,xd3,xd4,xd5,xd6,xd7,xd8,xd9,xd10,xd11,xd12])

	# for i in range(1,13):
	# 	print(f'xd{str(i)} = {xdot[i-1]} ') 
    
    return xdot

################################################################################
def TrimCost(OptParam):
    '''
        FLIGHT Cost Function for Longitudinal Trim in Steady Level Flight
    '''
    global u
    global x
    global V
    global TrimHist

    R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = R.reshape((3,3))

    # Optimizing Vector:
    #   1 = Stabilator, rad
    #   2 = Throttle, #
    #   3 = Pitch Angle, rad
    OptParam = OptParam.reshape((3,1))

    u = np.array([u[0],
        u[1],
        u[2],
        OptParam[1],
        u[4],
        u[5],
        OptParam[0]]).reshape((7,1)).ravel()

    x	=	np.array([V * math.cos(OptParam[2]),
            x[1],
            V * math.sin(OptParam[2]),
            x[3],
            x[4],
            x[5],
            x[6],
            x[7],
            x[8],
            x[9],
            OptParam[2],
            x[11]]).reshape((12,1)).ravel()
    
    xdot    = EoM(1,x)
    xCost   = np.array([xdot[0], xdot[2], xdot[7]]).reshape((3,1))
    # J = quadretic cost function    # otehr choices avilable to compute cost function(J)
    J       =   (np.matmul(np.matmul(xCost.T, R), xCost)).ravel().reshape((1,1))    ## xCost' * R * xCost
    ParamCost   =   (np.vstack((OptParam, J))).reshape((4,1))   ##[OptParam;J];   # column vector
    TrimHist    =   np.concatenate((TrimHist, ParamCost), axis=1)   # 4 rows showing history of trim computation over "INDEX" times

    return J.ravel()[0]

################################################################################
def LinModel(tj,xj):
    '''
        Flight Equations of Motion for Linear Model (Jacobian) Evaluation,
        with dummy state elements added for controls
    '''

    global u
    global x

    x   =   xj[0:11+1]
    u   =   xj[12:18+1]

    xdot    =   EoM(tj,x)
    xdotj   =   np.concatenate((xdot, np.array([0,0,0,0,0,0,0])))

    return xdotj

###############################################################################
def event(t,x):
    '''
        Time when (-height) passes through zero in an increasing direction.
    '''
    value       =   x[5]    # detect positive x[5] = 0
    terminal    =   True    # stop the integration
    direction   =   1       # positive direction

    return (value, terminal, direction)

###############################################################################
import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################



### Initial Altitude(ft), Indicated Airspeed(kt)
hft     =   10000   # Altitude above Sea Level, ft
VKIAS   =   150     # Indicated Airspeed, kt

hm      =   hft * 0.3048    # Altitude above Sea Level, m
VmsIAS  =   VKIAS * 0.5154  # Indicated Airspeed, m/s

print('Initial Conditions')
print('==================')
print(f'Altitude           = {str(hft)} ft,   = {str(hm)} m')
print(f'Indicated Airspeed = {str(VKIAS)} kt,   = {str(VmsIAS)} m/s')

### US Standard Atmosphere, 1976, Table lookup for I.C.
(airDens, airPres, temp, soundSpeed) = Atoms(hm) 
print(f'Air Density     = {str(airDens)} kg/m**3, Air Pressure = {str(airPres)} N/m**2')
print(f'Air Temperature = {str(temp)} K,         Sound Speed  = {str(soundSpeed)} m/s')
        
# Dynamic Pressure (N/m**2), and True Airspeed (m/s)
qBarSL  =   0.5*1.225 * VmsIAS**2      # Dynamic Pressure at sea level, N/m**2
V       =   np.sqrt(2*qBarSL/airDens)	# True Airspeed, TAS, m/s
TASms   =   V
print(f'Dynamic Pressure = {str(qBarSL)} N/m**2, True Airspeed = {str(V)} m/s')

### Alphabetical List of Initial Conditions

alpha   =	0      # Angle of attack, deg (relative to air mass)
beta    =	0      # Sideslip angle, deg (relative to air mass)
dA      =	0      # Aileron angle, deg
dAS     =	0      # Asymmetric spoiler angle, deg
dE      =	0      # Elevator angle, deg
dR      =	0      # Rudder angle, deg
dS      = 	0      # Stabilator setting, deg
dT      = 	0      # Throttle setting,   # / 100
hdot    =	0      # Altitude rate, m/s
p       =	0      # Body-axis roll rate, deg/s
phi     =	0      # Body roll angle wrt earth, deg
psi     =	0      # Body yaw angle wrt earth, deg
q       =	0      # Body-axis pitch rate, deg/sec
r       =	0      # Body-axis yaw rate, deg/s
SMI     =	0      # Static margin increment due to center-of-mass variation from reference, #/100
tf      =	100    # Final time for simulation, sec
ti      = 	0      # Initial time for simulation, sec
theta   =	alpha  # Body pitch angle wrt earth, deg [theta = alpha if hdot = 0]
xe      =	0      # Initial longitudinal position, m
ye      = 	0      # Initial lateral position, m
ze      = 	-hm    # Initial vertical position, m [h: + up, z: + down]

## Initial Conditions Depending on Prior Initial Conditions
gamma   =   57.2957795 * np.arctan(hdot / np.sqrt(V**2 - hdot**2))    #      ## Inertial Vertical Flight Path Angle, deg  # deg = 57.2957795 * rad
qbar	= 	0.5 * airDens * V**2	    # Dynamic Pressure, N/m**2
IAS		=	np.sqrt(2 * qbar / 1.225)      # Indicated Air Speed, m/s
Mach	= 	V / soundSpeed              # Mach Number
print(f'Mach number = {str(Mach)}, Flight Path Angle = {str(gamma)} deg\n')

uInc    =   [];      #      #
if MODEL == 0:
    print('<<Low-Mach, High-Alpha Model>>')
elif MODEL == 1:
    print('<<High-Mach, Low-Alpha Aerodynamic Model>>')
else:
    print('<<USer Defined AeroModel>>')
print('  ======================================')

## Initial Control Perturbation (Test Inputs: rad or 100#)			
delu	=	np.array([0,0,0,0,0,0,0])  # column vector  # delta u   in u = u0 + delta u
## Initial State Perturbation (Test Inputs: m, m/s, rad, or rad/s)
delx	=	np.array([0,0,0,0,0,0,0,0,0,0,0,0])  ##.reshape((4,3)) ###### ????? SHAPE ????   # delta x   in x = x0 + delta x

print('Initial Perturbations to Trim for Step Response')
print('===============================================')

print('Control Vector')
print('--------------')
print(f'Elevator   = {str(delu[0])} rad, Aileron = {str(delu[1])} rad, Rudder = {str(delu[2])} rad')
print(f'Throttle   = {str(delu[3])} x 100%, Asymm Spoiler = {str(delu[4])} rad, Flap = {str(delu[5])} rad')
print(f'Stabilator = {str(delu[6])} rad\n')

print('State Vector')
print('------------')
print(f'u   = {str(delx[0])} m/s, v = {str(delx[1])} m/s, w = {str(delx[2])} m/s')
print(f'x   = {str(delx[3])} m, y = {str(delx[4])} m, z = {str(delx[5])} m')
print(f'p   = {str(delx[6])} rad/s, q = {str(delx[7])} rad/s, r = {str(delx[8])} rad/s')
print(f'Phi = {str(delx[9])} rad, Theta = {str(delx[10])} rad, Psi = {str(delx[11])} rad')

### Control Perturbation History (Test Inputs: rad or 100%)
#   =======================================================
# Each control effector represented by a column
# Each row contains control increment delta-u(t) at time t:
print('\nControl Vector Time History Table')
print('=================================')
print('Time, sec: t0, t1, t2, ...')
tuHis	=	np.array([[0, 33, 67, 100]]).T    #  size = (4,1)
print(f'tuHis = {tuHis}')

print('Columns:  Elements of the Control Vector')
print('Rows:     Value at time, t0, t1, ...')
deluHis	=	np.zeros((4,7))    # size = (4,7)  # 7 = controls  # 4 = time increment
print(f'deluHis = {deluHis}')

# State Vector and Control Initialization, rad
phir	=	phi * 0.01745329       #rad = 0.01745329 * deg
thetar	=	theta * 0.01745329
psir	=	psi * 0.01745329

windb	=	WindField(-ze,phir,thetar,psir)
alphar	=	alpha * 0.01745329    # alphar == alpha in rad
betar	=	beta * 0.01745329

print(V)

x	=	np.array([V*math.cos(alphar)*math.cos(betar) - float(windb[0]),
        V*math.sin(betar) - float(windb[1]),
        V*math.sin(alphar)*math.cos(betar) - float(windb[2]),
        xe,
        ye,
        ze,
        p * 0.01745329,
        q * 0.01745329,
        r * 0.01745329,
        phir,
        thetar,
        psir]).reshape(12,1).ravel()
print(f'x = {x}')
print(type(x))
print(x.shape)

u	=	np.array([dE * 0.01745329,
        dA * 0.01745329,
        dR * 0.01745329,
        dT,
        dAS * 0.01745329,
        dF * 0.01745329,
        dS * 0.01745329]).reshape((7,1)).ravel()

print(f'u = {u}')
print(type(u))
print(u.shape)

### Trim Calculation (for Steady Level Flight at Initial V and h)
#   =============================================================
#   Always use Euler Angles for trim calculation
#   Trim Parameter Vector (OptParam):
#		1 = Stabilator, rad
#		2 = Throttle, %
#		3 = Pitch Angle, rad

if TRIM >= 1:
    print('\nTRIM Stabilator, Thrust, and Pitch Angle')
    print('========================================')
    OptParam        =   np.zeros(3).reshape((3,1))
    TrimHist        =    np.zeros(4).reshape((4,1))
    InitParam		=	np.array([0.0369,0.1892,0.0986]).reshape((3,1)).ravel()

    #(OptParam,J,ExitFlag,Output) = fmin(TrimCost,InitParam)
    InitParam = [0.0369, 0.1992, 0.0986]
    (OptParam) = fmin(TrimCost,InitParam, xtol=1e-10)
    print(f'OptParam = {OptParam}')
    J = TrimCost(OptParam)
    print(f'J = {J}')
    print(f'Trim Cost = {str(J)}')

    ## Optimizing Trim Error Cost with respect to dSr, dT, and Theta
    Index=  list(range(0,len(TrimHist[0])))   # row lenght of TrimHist
    TrimStabDeg     =   57.2957795*OptParam[0]
    TrimThrusPer    =   100*OptParam[1]
    TrimPitchDeg    =   57.2957795*OptParam[2]
    TrimAlphaDeg    =   TrimPitchDeg - gamma
    print(f'Stabilator  = {str(TrimStabDeg)} deg, Thrust = {str(TrimThrusPer)} x 100%')
    print(f'Pitch Angle = {str(TrimPitchDeg)} deg, Angle of Attack = {str(TrimAlphaDeg)} deg')
    
    ## Insert trim values in nominal control and state vectors
    print('\nTrimmed Initial Control and State Vectors')
    print('=========================================')
    u	=	[u[0],u[1],u[2],OptParam[1],u[4],u[5],OptParam[0]]

    x	=	[V * math.cos(OptParam[2]),
            x[1],
            V * math.sin(OptParam[2]),
            x[3],
            x[4],
            x[5],
            x[6],
            x[7],
            x[8],
            x[9],
            OptParam[2],
            x[11]]
    print('Control Vector')
    print('--------------')
    print([f'Elevator   = {str(u[0])} rad, Aileron = {str(u[1])} rad, Rudder = {str(u[2])} rad'])
    print([f'Throttle   = {str(u[3])} x 100%, Asymm Spoiler = {str(u[4])} rad, Flap = {str(u[5])} rad'])
    print([f'Stabilator = {str(u[6])} rad'])

    print('\nState Vector')
    print('------------')
    print(f'u   = {str(x[0])} m/s, v = {str(x[1])} m/s, w = {str(x[2])} m/s')
    print(f'x   = {str(x[3])} m, y = {str(x[4])} m, z = {str(x[5])} m')
    print(f'p   = {str(x[6])} rad/s, q = {str(x[7])} rad/s, r = {str(x[8])} rad/s')
    print(f'Phi = {str(x[9])} rad, Theta = {str(x[10])} rad, Psi = {str(x[11])} rad\n')
		

plt.figure('Convergence of trim parameters and trim-error cost function', figsize=(16,9))
plt.subplot(121)
plt.plot(Index,TrimHist[0,:],'b-', Index,TrimHist[1,:],'g-', Index,TrimHist[2,:],'r-')
plt.xlabel('Iterations')
plt.ylabel('Stabilator(blue), Thrust(green), Pitch Angle(red)')
plt.legend(('Stabilator', 'Thrust', 'Pitch Angle'))
plt.title('Trim Parameters')
plt.axis([0, 350, -0.15, 0.25])
plt.grid(True)
plt.subplot(122)
plt.plot(Index,TrimHist[3,:],'b-')
plt.xlabel('Iterations')
plt.ylabel('Trim Cost')
plt.title('Trim Cost')
plt.axis([0, 350, -1, 6])
plt.grid(True)
#plt.show()

xdot = EoM(1,x)

print(f'x = {x}')
print(f'xdot = {xdot}')

### Stability-and-Control Derivative Calculation           ##### WILL DO THIS LATER ######
#   ============================================
if LINEAR >= 1:
    print('\nGenerate and Save LINEAR MODEL')
    print('==============================')
    #thresh	=	np.array([.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1])
    xj		=	np.array(x+u)
    uTemp   =   u           # 'numjac' modifies 'u'; reset 'u' after the call
    xdotj		=	LinModel(ti,xj)
    print(f'xdotj = {xdotj}')
    #[dFdX,fac]	=	numjac('LinModel',ti,xj,xdotj,thresh,[],0)
    #u           =   uTemp
    #Fmodel		=	dFdX(1:12,1:12)
    #Gmodel		=	dFdX(1:12,13:19)
    #save ('Fmodel','Fmodel','TASms','hm')
    #save ('Gmodel','Gmodel')


### Flight Path Simulation, with Quaternion Option
#   ==============================================
if SIMUL >= 1:
    RUNNING =   1  
    tspan	=	(ti, tf)
    xo      =   x + delx
    u		=	u + delu
    
    print('Trimmed Initial Control and State Vectors PLUS Test Inputs')
    print('==========================================================')
    print('Control Vector')
    print('--------------')
    print(f'Elevator   = {str(u[0])} rad, Aileron = {str(u[1])} rad, Rudder = {str(u[2])} rad')
    print(f'Throttle   = {str(u[3])} x 100%, Asymm Spoiler = {str(u[4])} rad, Flap = {str(u[5])} rad')
    print(f'Stabilator = {str(u[5])} rad')

    print('\nState Vector')
    print('------------')
    print(f'u   = {str(xo[0])} m/s, v = {str(xo[1])} m/s, w = {str(xo[2])} m/s')
    print(f'x   = {str(xo[3])} m, y = {str(xo[4])} m, z = {str(xo[5])} m')
    print(f'p   = {str(xo[6])} rad/s, q = {str(xo[7])} rad/s, r = {str(xo[8])} rad/s')
    print(f'Phi = {str(xo[9])} rad, Theta = {str(xo[10])} rad, Psi = {str(xo[11])} rad\n')
    
# switch QUAT  :  0: Euler Angles    1: Quaternions
if QUAT == 0:
    print('Use Euler Angles to form Rotation Matrix')
    print('========================================')
    
    tic()
    sol = solve_ivp(EoM,tspan,xo,method='BDF',events=event,rtol=1e-13,atol=1e-13)     ### change rtol & atol and observe effects on plot
    toc()
    t = sol.t
    x = sol.y
    kHis	=	len(t)
    print(f'# of Time Steps = {str(kHis)}\n')
    #print(f't = {sol.t}')
    #print(f'x = {sol.y}')
        
elif QUAT == 1:
    pass          ##### WILL DO THIS LATER ######


### Plot Control History
plt.figure('Test Inputs(Controls) Vs Time', figsize=(16,9))
plt.subplot(221)
plt.plot(tuHis, 57.29578*deluHis[:,0] + 57.29578*delu[0],'b-' , tuHis, 57.29578*deluHis[:,6] + 57.29578*delu[6],'g-')
plt.xlabel('Time, s')
plt.ylabel('Elevator (blue), Stabilator (green), deg')
plt.grid(True)
plt.title('Pitch Test Inputs')
plt.legend(('Elevator, dE', 'Stabilator, dS'))
plt.subplot(222)
plt.plot(tuHis, 57.29578*deluHis[:,1] + 57.29578*delu[1],'b-' , tuHis, 57.29578*deluHis[:,2] + 57.29578*delu[2],'g-' , tuHis, 57.29578*deluHis[:,4] + 57.29578*delu[4],'r-')    
plt.xlabel('Time, s')
plt.ylabel('Aileron (blue), Rudder (green), Asymmetric Spoiler (red), deg')
plt.grid(True)
plt.title('Lateral-Directional Test Inputs')
plt.legend(('Aileron, dA', 'Rudder, dR', 'Asymmetric Spoiler, dAS'))
plt.subplot(223)
plt.plot(tuHis, deluHis[:,3] + delu[3])    
plt.xlabel('Time, s')
plt.ylabel('Throttle Setting')
plt.grid(True)
plt.title('Throttle Test Inputs')
plt.subplot(224)
plt.plot(tuHis, 57.29578*deluHis[:,5] + 57.29578*delu[5])    
plt.xlabel('Time, s')
plt.ylabel('Flap, deg')
plt.grid(True)
plt.title('Flap Test Inputs')
#plt.show()


### Plot State History
plt.figure('Velocities Vs Time', figsize=(16,9))
plt.subplot(221)
plt.plot(t,x[0,:])
plt.xlabel('Time, s')
plt.ylabel('Axial Velocity (u), m/s')
plt.grid(True)
plt.title('Forward Body-Axis Component of Inertial Velocity, u')
plt.subplot(222)
plt.plot(t,x[1,:])
plt.xlabel('Time, s')
plt.ylabel('Side Velocity (v), m/s')
plt.grid(True)
plt.title('Side Body-Axis Component of Inertial Velocity, v')
plt.subplot(223)
plt.plot(t,x[2,:])
plt.xlabel('Time, s')
plt.ylabel('Normal Velocity (w), m/s')
plt.grid(True)
plt.title('Normal Body-Axis Component of Inertial Velocity, z')
plt.subplot(224)
plt.plot(t,x[0,:],'b-',t,x[1,:],'g-',t,x[2,:],'r-')
plt.xlabel('Time, s')
plt.ylabel('u (blue), v (green), w (red), m/s')
plt.grid(True)
plt.title('Body-Axis Component of Inertial Velocity') 
plt.legend(('Axial velocity, u', 'Side velocity, v', 'Normal velocity, w'))

plt.figure('Location Vs Time', figsize=(16,9))
plt.subplot(321)
plt.plot(t,x[3,:])
plt.xlabel('Time, s')
plt.ylabel('North (x), m')
plt.grid(True)
plt.title('North Location, x')
plt.subplot(322)
plt.plot(t,x[4,:])
plt.xlabel('Time, s')
plt.ylabel('East (y), m')
plt.grid(True)
plt.title('East Location, y')
plt.subplot(323)
plt.plot(t,-x[5,:])
plt.xlabel('Time, s')
plt.ylabel('Altitude (-z), m')
plt.grid(True)
plt.title('Altitude, -z')
plt.subplot(324)
plt.plot((np.sqrt(x[3,:]*x[3,:] + x[4,:]*x[5,:])),-x[5,:])       #### element wise
plt.xlabel('Ground Range, m')
plt.ylabel('Altitude, m')
plt.grid(True)
plt.title('Altitude vs. Ground Range')
plt.subplot(325)
plt.plot(x[3,:],x[4,:])
plt.xlabel('North, m')
plt.ylabel('East, m')
plt.grid(True)
plt.title('Ground Track, North vs. East')
fig = plt.figure('3D Flight Path', figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[3,:],x[4,:],-x[5,:])
ax.set_xlabel('North, m')
ax.set_ylabel('East, m')
ax.set_zlabel('Altitude, m')
#plt.show()

plt.figure('Pitch Roll Yaw rates Vs Time', figsize=(16,9))
plt.subplot(221)
plt.plot(t,x[6,:] * 57.29578)
plt.xlabel('Time, s')
plt.ylabel('Roll Rate (p), deg/s')
plt.grid(True)
plt.title('Body-Axis Roll Component of Inertial Rate, p')
plt.subplot(222)
plt.plot(t,x[7,:] * 57.29578)
plt.xlabel('Time, s')
plt.ylabel('Pitch Rate (q), deg/s')
plt.grid(True)
plt.title('Body-Axis Pitch Component of Inertial Rate, q')
plt.subplot(223)
plt.plot(t,x[8,:] * 57.29578)
plt.xlabel('Time, s')
plt.ylabel('Yaw Rate (r), deg/s')
plt.grid(True)
plt.title('Body-Axis Yaw Component of Inertial Rate, r')
plt.subplot(224)
plt.plot(t,x[6,:] * 57.29578,'b-',t,x[7,:] * 57.29578,'g-',t,x[8,:] * 57.29578,'r-')
plt.xlabel('Time, s')
plt.ylabel('p (blue), q (green), r (red), deg/s')
plt.grid(True)
plt.title('Body-Axis Inertial Rate Vector Components')
plt.legend(('Roll rate, p', 'Pitch rate, q', 'Yaw rate, r'))

plt.figure('Pitch Roll Yaw Attitude(Angles) Vs Time', figsize=(16,9))
plt.subplot(221)
plt.plot(t,x[9,:] * 57.29578)
plt.xlabel('Time, s')
plt.ylabel('Roll Angle (phi), deg')
plt.grid(True)
plt.title('Earth-Relative Roll Attitude')
plt.subplot(222)
plt.plot(t,x[10,:] * 57.29578)
plt.xlabel('Time, s')
plt.ylabel('Pitch Angle (theta), deg')
plt.grid(True)
plt.title('Earth-Relative Pitch Attitude')
plt.subplot(223)
plt.plot(t,x[11,:] * 57.29578)
plt.xlabel('Time, s')
plt.ylabel('Yaw Angle (psi, deg')
plt.grid(True)
plt.title('Earth-Relative Yaw Attitude')
plt.subplot(224)
plt.plot(t,x[9,:] * 57.29578,'b-',t,x[10,:] * 57.29578,'g-',t,x[11,:] * 57.29578,'r-')
plt.xlabel('Time, s')
plt.ylabel('phi (blue), theta (green), psi (red), deg')
plt.grid(True)
plt.title('Euler Angles')
plt.legend(('Roll angle, phi', 'Pitch angle, theta', 'Yaw angle, psi'))
#plt.show()

VAirRel         =   np.zeros(1)
vEarth          =   np.zeros(3)
AlphaAR         =   np.zeros(1)
BetaAR          =   np.zeros(1)
windBody        =   np.zeros(3).reshape((3,1))
airDensHis      =   np.zeros(1)
soundSpeedHis   =   np.zeros(1)
qbarHis         =   np.zeros(1)
GammaHis        =   np.zeros(1)
XiHis           =   np.zeros(1)

for i in range(1,kHis):
    windb           =	WindField(-x[5,i],x[8,i],x[10,i],x[11,i]).reshape((3,1))
    windBody        =   np.concatenate((windBody, windb), axis =1)
    (airDens,airPres,temp,soundSpeed) = Atoms(-x[5,i])
    airDensHis      =   np.concatenate((airDensHis, np.array([airDens])))
    soundSpeedHis   =   np.concatenate((soundSpeedHis, np.array([soundSpeed])))
#end

vBody           =   np.array([x[0,:], x[1,:], x[2,:]]).T      # shape :  (95,3)
vBodyAir        =   vBody[1:,:] + windBody[:,1:].T

for i in range(1,kHis-1):
    vE          =   np.matmul(DCM(x[9,i],x[10,i],x[11,i]).T, np.array([vBody[i,0],vBody[i,1],vBody[i,2]]))
    VER         =   math.sqrt(vE[0]**2 + vE[1]**2 + vE[2]**2)
    VAR         =   math.sqrt(vBodyAir[i,0]**2 + vBodyAir[i,1]**2 + vBodyAir[i,2]**2)
    VARB        =   math.sqrt(vBodyAir[i,0]**2 + vBodyAir[i,2]**2)

    if vBodyAir[i,0] >= 0:
        Alphar      =	math.asin(vBodyAir[i,2] / VARB)
    else:
        Alphar      =	math.pi - math.asin(vBodyAir[i,2] / VARB)

    Alphar = np.array([Alphar])
    AlphaAR     =   np.concatenate((AlphaAR, Alphar))
    Betar       = 	math.asin(vBodyAir[i,1] / VAR)
    Betar = np.array([Betar])
    BetaAR      =   np.concatenate((BetaAR, Betar))
    vEarth      =   (np.vstack((vEarth, vE)))
    Xir         =   math.asin((vEarth[i,1]) / (math.sqrt((vEarth[i,0])**2 + (vEarth[i,1])**2)))
    
    if vEarth[i,0] <= 0 and vEarth[i,1] <= 0:
        Xir     = -math.pi - Xir

    if vEarth[i,0] <= 0 and vEarth[i,1] >= 0:
        Xir     = math.pi - Xir

    Gammar      =   math.asin(-vEarth[i,2] / VER)
    Gammar      =   np.array([Gammar])
    GammaHis    =   np.concatenate((GammaHis, Gammar)) 
    Xir         =   np.array([Xir])         
    XiHis       =   np.concatenate((XiHis, Xir))
    VAR         =   np.array([VAR])
    VAirRel     =   np.vstack((VAirRel, VAR))

MachHis         =   np.divide(VAirRel[:,0], soundSpeedHis[1:])
AlphaDegHis 	=	57.2957795 * AlphaAR
BetaDegHis      =	57.2957795 * BetaAR
qbarHis         =	0.5 * airDensHis[1:] * VAirRel[:,0] * VAirRel[:,0]
GammaDegHis     =   57.2957795 * GammaHis
XiDegHis        =   57.2957795 * XiHis


plt.figure('Vair Mach qbar vs Time', figsize=(16,9))
plt.subplot(311)
plt.plot(t[1:], VAirRel[:,0])       ## zome out very high to see minor oscilation
plt.xlabel('Time, s')
plt.ylabel('Air-relative Speed, m/s')
plt.grid(True)
plt.title('True AirSpeed, Vair')
plt.subplot(312)
plt.plot(t[1:], MachHis)       ## zome out very high to see minor oscilation  
plt.xlabel('Time, s')
plt.ylabel('M')
plt.grid(True)
plt.title('Mach Number, M')
plt.subplot(313)
plt.plot(t[1:], qbarHis)       ## zome out very high to see minor oscilation    
plt.xlabel('Time, s')
plt.ylabel('qbar, N/m^2')
plt.grid(True)
plt.title('Dynamic Pressure, qbar')

plt.figure('Aerodynamic and Flight Path Angles Vs Time', figsize=(16,9))
plt.subplot(211)
plt.plot(t[1:], AlphaDegHis, t[1:], BetaDegHis)    
plt.xlabel('Time, s')
plt.ylabel('Angle of Attack, deg (blue), Sideslip Angle, deg (green)')
plt.grid(True)
plt.title('Aerodynamic Angles')
plt.legend(('Angle of Attack, alpha', 'Sideslip Angle, beta'))
plt.subplot(212)
plt.plot(t[1:], GammaDegHis, t[1:], XiDegHis)    
plt.xlabel('Time, s')
plt.ylabel(('Vertical, deg (blue), Horizontal, deg (green)'))
plt.grid(True)
plt.title('Flight Path Angles')
plt.legend(('Flight Path Angle, gamma', 'Heading Angle, psi'))

plt.show()

print('\nEnd of FLIGHT Simulation')
############################################################################################