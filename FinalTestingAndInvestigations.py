# -*- coding: utf-8 -*-
import Raytracing as rt
from scipy.optimize import minimize #to be used in biconvex optimisation (see testing file)
from mpl_toolkits.mplot3d import Axes3D    #to be used in ploting (see testing file)   
import numpy as np
import matplotlib.pyplot as plt

def rmsfinder(b, point):
    '''rmsfinder - finds rms value at a given point
    Parameters: b-> bundle of rays to find rms of, point -> point at which to find rms of b
    Returns: total root mean square value'''
    c= findinterz(b.bundle[0], point)
    rms=[]
    lenb=len(b.bundle)
    for r in b.bundle:
        p=findinterz(r, point)
        dif=p-c
        
        dif=dif[:-1]
        
        sq=np.square(dif[0])+np.square(dif[1])
        
        rms.append(sq)
        
    return np.sqrt(sum(rms)/lenb)    
        
def rmsLoc(a):
    '''rmsLoc - Uses rmsfinder to iterate between z=0 to z=300 checking to find the lowest rms
    - it does this by taking samples at increasingly small intervals to narrow down the minimum
    rms's exact position
    Parameters: a-> bundle to be iterated through 
    Returns: location of minimum rms and minimum rms value '''
    rmsa=[]
    space=np.linspace(3,300, 300)
    for a1 in space:
        rmsa.append(rmsfinder(a, a1))
    out1=rmsa.index(min(rmsa))
    
    space=np.linspace(space[out1]-1, space[out1]+1, 300)
    rmsa=[]
    for a1 in space:
        rmsa.append(rmsfinder(a, a1))
   
    out1=rmsa.index(min(rmsa))
    return space[out1], min(rmsa)

def findinterz(ray, z):
    '''Findinterz- finds an intercept with a given z plane
    Parameters: ray for the intercept to be found with, z -> z value for the plane
    Returns: intercept
    '''
    p=ray.getp()
    k=ray.getk()
    val = z-p[2]/k[2]
    return val*k+p

def findinter(ray):
    '''Findinter- finds an intercept with a x=0 point
    Parameters: ray for the intercept to be found with
    Returns: intercept'''
    p=ray.getp()
    k=ray.getk()
    val = -p[0]/k[0]
    return val*k[2]+p[2]
     
def plotprep(rays):
    '''plotprep: Prepares rays in a form that can be easily manipulated and plotted along one another - parameters: rays-> bundle of arrays'''
    rf=[]
    for r in rays:
        x, y, z=[], [], []
        for p in r.vertices(): 
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        rf.append([x,y,z])
    return rf

def fullplot(ray):
    '''Individual Plot for a single ray - Accepts Parameters: Vertices of array
       Produces 3D plot and plot of z,x and z,y
    '''
    
    rf=plotprep([r])[0]
    
    plt.plot(rf[2],rf[0])
    plt.title('Path of Ray in x,z')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.figure()
    
def bundleplot(rays, rays2=[]):
    '''bundleplot: Plots multiple bundles - Accepts parameters: points -> a bundle of rays, points2 -> a seperate bundle of arrays
    Creates plots of these bundles - in different colours for each bundle 
    '''
    if len(rays2)==0:
        rays2=rays
        
    rf1=plotprep(rays) 
    rf2=plotprep(rays2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for r, r2 in zip(rf1, rf2):
        ax.plot(r[0], r[1], r[2], 'b') 
        ax.plot(r2[0], r2[1], r2[2], 'k') 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.figure()
    
    for r, r2 in zip(rf1,rf2):  
       plt.plot(r[2],r[0], color="blue")
       plt.plot(r2[2],r2[0], color="black") 
    plt.title('Path of Bundles in x and z')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.show()
    plt.figure()

          
   
#%%~ 2
#Example Set up for Task 9: Testing through a given surface
def Example(ray, c):
    s= rt.SphericalRefraction(100.0, c, 1.0, 1.5)#given spherical surface
    out=rt.OutputPlane(250)#Given output plane
    s.propagate_ray(ray)#Propagate through optical surface
    out.propagate_ray(ray)#Propagate to output plan
    fullplot(r) #Create 3d plot and a plot of x vs z and y vs z 

#%%~ 3
#Testing the case of no intercept
r=rt.ray([9,4,0.0], [1,20,30]) #- no intercept occurs
Example(r, 0.03)
#%%~ 4
r=rt.ray([-9,-4,5], [1,2,20])#- intercepts  and propogates out
Example(r, 0.03)
#%%~ 5
r=rt.ray([9,2,0.0], [-20,-1,100])#near no change in x
Example(r, 0.03)
#%%~ 6
r=rt.ray([-9,0,0.0], [1,1,100])#neg
Example(r, -0.03)
#%%~ 7
r=rt.ray([9,4,0.0], [0,0,30])#Parralel to z
Example(r, 0.03)
#%%~ 8   
#Task 9 Testing through a given surface
r=rt.ray([9,0,0.0], [0,0,30])#Parralel to z
Example(r, 0.03)
print(r.getp()) # Final ends up at 250, is therefore reaches output plane
print(r.getk()) #values coming out appear to be in the right direction - negative in x as it converges towards optical axis             

'''Checked actual point values to ensure rays were ending up where intended
Values:
[array([-1.83942692e-01,  0.00000000e+00,  2.50000000e+02]), 
array([-9.19713677e-02,  0.00000000e+00,  2.50000000e+02]), 
array([9.19713677e-02, 0.00000000e+00, 2.50000000e+02]), 
array([1.83942692e-01, 0.00000000e+00, 2.50000000e+02])]       
  '''              
#%%~  Aditional tests - 9
#Task 10: Testing Rays parrallel to Optical axis             
#putting ray straight down optical axis 
out=rt.OutputPlane(250)
s= rt.SphericalRefraction(100.0, 0.03, 1.0, 1.5, 3)      
e= [0.1,0.05,0,-0.05, -0.1] #list of starting points in x
#if starting at 0,0,0 and travelling through optical axis then it should continue in a straight line
l=[]
#varying x distance from optical axis
for a in e:
   r=rt.ray([a,0.0,0.0], [0,0,100])
   s.propagate_ray(r)
   out.propagate_ray(r) 
   l.append(r)     
bundleplot(l,l)#plots all points     
#now we have to find the paraxial focal point - this can be done by finding where each ray meets the optical axis   
#Using the inter method we can find an estimate of our paraxial focal point
inter=findinter(l[0])
inter1=findinter(l[1])# as you can see the difference from the axis changes the focal point
print(inter, inter1)
'''Values:
199.9997999996499
199.99994999997796
199.99994999997796
199.9997999996499   
    '''     
#%%~  Aditional tests - 10
# For positive curvatures- Test of  Focal length using f=nR/((n-1))	- Iterating through different 
cc=np.linspace(0.1, 0.01, 10)
for c in cc:
    s= rt.SphericalRefraction(100.0, c, 1.0, 1.5)      
    e= [0.1,0.01, -0.01,-0.1] #list of starting points in x
    #if starting at 0,0,0 and travelling through optical axis then it should continue in a straight line
    #varying x distance from optical axis
    for a in e:
        r=rt.ray([a,0.0,0.0], [0,0,100])
        s.propagate_ray(r)
        inter=findinter(r)
        inter=np.array(inter)
        CalculatedF= 1.5*(1/c)/(0.5)
        print("Modeled,          Calulated")
        print(inter-100, CalculatedF)
  
#as compared to calculated value of paraxial foci

 #%%~ additional Tests - 11 
# we have checked it travels straight through, equal and opposite rays meet at z plane etc
# an additional test is plotting the image formed at our output plane

s= rt.SphericalRefraction(100.0, 0.03, 1.0, 1.5)      
e= np.linspace(6, -6, 20) #list of starting points in x
#if starting at 0,0,0 and travelling through optical axis then it should continue in a straight line
out=rt.OutputPlane(250)
lx=[]
ly=[]
#varying x distance from optical axis
for a in e:
   r=rt.ray([a,0.0,0.0], [0,0,100])
   s.propagate_ray(r)
   out.propagate_ray(r)
   lx.append(r.getp()[0])
   ly.append(r.getp()[1])
plt.plot(lx,ly,'o', 'b')
#this should plot a linear set of equally spaced points that have been refracted similarily due to all being paraxial 
#plt.gca().set_aspect('equal', adjustable='box')   

        
 #%%~Additional tests - 12
#Checking the correct intercepts are being considered
sp= rt.SphericalRefraction(100.0, 0.03, 1.0, 1.5)     
sm= rt.SphericalRefraction(100.0,-0.03, 1.0, 1.5)  
e= np.linspace(6, -6, 20) #list of starting points in x
rays=[]
#varying x distance from optical axis
for a in e:
   rays.append(rt.ray([a,0.0,0.0], [0,0,100]))
rays2= rays.copy()
Success= True
for r, r2 in zip(rays, rays2):
    sm.propagate_ray(r2)
    sp.propagate_ray(r)
    if (r.getp()[2]>r2.getp()[2]):
        print("Something is wrong. The intercept with the lens of negative curvature has arrived first")
        Success=False
if Success==True:
    print("Success: The intercept with the lens of Positive curvature has arrived first in every case")
        
#%%~   13       
#Task 12 - Two Bundle generator ->  Test case to see regularity of shape 
#Generates two bundles parrallel to z axis - also prints out their image in the starting plane
s= rt.SphericalRefraction(100.0, 0.02,1,1.5168, 1/0.02)    
b= rt.Bundle(5, [0,0.0,0.0], [0,0,100], 5, "black")
a= rt.Bundle(5, [3,0,0.0], [0,0,100], 5, "blue")
out=rt.OutputPlane(300)

a.propagate(s)
a.propagate(out)
b.propagate(s)
b.propagate(out)

bundleplot(a.bundle,b.bundle) 

#%%~  14        
#Task 12 - Single Bundle  
#Generates two bundles parrallel to z axis - also prints out their image in the starting plane
s= rt.SphericalRefraction(100.0, 0.02,1,1.5168, 1/0.02)    
a= rt.Bundle(5, [0,0,0.0], [0,0,100], 5, 'k')
out=rt.OutputPlane(300)

a.propagate(s)
a.propagate(out)
    
bundleplot(a.bundle,a.bundle) 


 #%%~     15
#Task 13: Paraxial focal plane - plotting at focal plane for ray b
s= rt.SphericalRefraction(100.0, 0.02,1,1.5168, 1/0.02)    
a= rt.Bundle(5, [0,0,0.0], [0,0,100], 5, 'k')
a.propagate(s)
location1, rms1=rmsLoc(a)
outa=rt.OutputPlane(location1)
a.propagate(outa)
a.plot()
print(rms1, location1)

#%%~   16
#Task 15 creating a planar surface and propagating through tests - Testing against snells law
r=rt.ray([0.1,0.0,0], [12,0,30])
#r=ray([0.1,0.0,0], [7,-10,30]) #from above 
r=rt.ray([0.1,5.0,3], [50,30,30]) #straight line through
ps1= rt.planeSurface(95, 1, 1.5168) 
ps1.propagate_ray(r)
out=rt.OutputPlane(250)
out.propagate_ray(r)
fullplot(r)  
#%%~17
#Task 15 - Part 1:
s= rt.SphericalRefraction(100.0, -0.02,1.5168, 1, 1/0.02)   
ps1= rt.planeSurface(95, 1, 1.5168)  
a= rt.Bundle(5, [0,0,0.0], [0,0,100], 5, "blue")
out=rt.OutputPlane(300)
a.propagate(ps1)
a.propagate(s)
location1, rms1=rmsLoc(a)
print(location1, rms1)
out=rt.OutputPlane(location1)
for ar in a.bundle:
    out.propagate_ray(ar)
bundleplot(a.bundle, a.bundle)
a.plot()

#%%~     18
#Task 15 - Part 2:
s= rt.SphericalRefraction(100.0, 0.02,1,1.5168, 1/0.02)   
ps1= rt.planeSurface(105,1.5168, 1)  
a= rt.Bundle(5, [0,0,0.0], [0,0,100], 5, "blue")
out=rt.OutputPlane(300)
a.propagate(s)
a.propagate(ps1)
location1, rms1=rmsLoc(a)
out=rt.OutputPlane(location1)
print(location1, rms1)
a.propagate(out)
bundleplot(a.bundle, a.bundle)
a.plot()
 #%%~ 19  
#best form lens optimisation function print : 
def gensing(c1, c2):
    a= rt.Bundle(5, [0,0,0], [0,0,1], 5, "blue")
    s= rt.SphericalRefraction(100.0, c1,1,1.5168, 3)  
    s1= rt.SphericalRefraction(105.0, c2,1.5168,1, 3)  
    try:#catches intercepts that do not occur
       a.propagate(s)
       a.propagate(s1)
    except:
        return 1#returns large rms value so that this point will not be selected
    location1, rms1=rmsLoc(a)
    outa=rt.OutputPlane(location1)
    a.propagate(outa)  
    bundleplot(a.bundle,a.bundle)
    a.plot()
    print(location1)
    return rms1 
    
print(gensing(0.01959562,-0.02042272))


 #%%~    20  
#best form lens optimisation function print : 
def gensing(estimates):
    a= rt.Bundle(5, [0,0,0], [0,0,1], 5, "blue")
    s= rt.SphericalRefraction(100.0, estimates[0],1,1.5168, 3)  
    s1= rt.SphericalRefraction(105.0, estimates[1],1.5168,1, 3)  
    try:#catches intercepts that do not occur
       a.propagate(s)
       a.propagate(s1)
    except: #prevents points of no intercept from causing a problem
        return 1#returns large rms value so that this point will not be selected
    location1, rms1=rmsLoc(a)
    return rms1 
estimate1s=[0.02,-0.02] 

res=minimize(gensing,estimate1s) 
   # run this for optimization - It is very slow sorrrrry but better than the one below!!!
'''    
def recopt():
    rms=[]
    c=[]
    c1s=np.linspace(0.001,0.02, 20)
    c2s=np.linspace(-0.001,-0.02, 20) 
    for c1 in c1s:
        for c2 in c2s:
            rms.append(gensing(c1, c2))
            c.append([c1, c2])
            print(c1, c2)
    out1=rms.index(min(rms)) 
    return c[out1]  
#my own optimisation function takes ages to run    
val=recopt()            
'''