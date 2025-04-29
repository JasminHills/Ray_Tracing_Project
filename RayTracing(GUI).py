#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:50:53 2022

@author: hills
"""

 
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
global na
na =1.003 #air
global ng  
ng= 1.5168 # Glass lens
global nv
nv=1

class App(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent=parent
        self.imagpln=0
        self.lfr = tk.LabelFrame(parent)
        self.lfr.grid(row=1, column=0)
        self.default()
        self.pixelsize=0.001 #in mm
       
       # Important Variables
        '''
       L2inc = tk.IntVar()
       Flip L2= tk.IntVar()
        '''
       
       
       # Inputs and buttons set up
        self.check_z = tk.Button(self.lfr, text="Include L2", command=self.l2r)
        self.check_z.grid(column=1, row=1)

        self.entry = tk.Button(self.lfr, text="Edit Setup",  command=self.InputWindow)
        self.entry.grid(column=1, row=3)
        self.wg2=tk.Entry(self.lfr)
        self.wg2.grid(column=1, row=5)
        self.wg1=tk.Entry(self.lfr)
        self.wg1.grid(column=2, row=5)
        tk.Label(self.lfr, text="Width").grid(row=4, column=2)
        tk.Label(self.lfr, text="Refractive Index").grid(row=5, column=1)
        tk.Label(self.lfr, text="Location").grid(row=5, column=2)

       
        # Allows initial input        
   
        
        

        
    def OutputWindow(self): 
        values=["Magnification: ", "Output Plane: "]
        #def total_mag(self, f1, f2, l1, l2):
        self.varreset()
    
       
        
        mn=round(np.mean(self.imagpln), 1)
        mii=round(max(self.imagpln), 1)
        mi=round(min(self.imagpln), 1)
        print([self.l1 ,self.l2],[self.f1, self.f2], mn)
        mag= round(total_mag([self.l1 ,self.l2],[self.f1, self.f2], mn), 1)
        tk.Label(self.parent, text=values[0]+str(self.mag)).grid(row=5, column=1)
        tk.Label(self.parent, text= "Rms Minimum: "+str(round(self.min2*10**3, 6))+"\u03BCm at "+str(round(self.loc, 4))+"mm").grid(row=6, column=1)
        tk.Label(self.parent, text= "Rms Minimum: "+str(round(self.min2*10**3, 6))+"\u03BCm at "+str(round(self.loc, 4))+"mm").grid(row=7, column=1)
        #   tk.Label(self.parent, text=values[1]+" "+str(mn) +" min: "+str(mi)+" max: "+str(mii)+" Rms Minimum: "+str(self.min2)+" at "+str(self.loc)).grid(row=6, column=1)
        
      
        
    def image(self, a):  #finds image plane
        inter=[]
        for r in a.bundle:
            try:
                inter.append(findinter(r))

            except:
                print("Jesus")
        m1=inter[1:-1]
        self.imagpln=m1
        #self.loc, self.min2=rmsLoc(a, min(m1), 1500)
        self.loc, self.min2=rmsLoc(a, min(m1), 1500)
        self.loc, self.min2=rmsLoc(a, min(m1), 1500)
        
    def default(self): 
        self.l1= 500
        self.l2= 800
        self.o=1500
        self.f1=500
        self.f2=300
        self.ang=0.02
        self.thk= 20
        self.f=self.f1
        self.r1=self.curve()
        self.f=self.f2
        self.r2=self.curve()
        self.vars=[ self.l1, self.l2, self.r1, self.r2]
        self.l2r()

        
    def InputWindow(self):  
        nw=tk.Toplevel(self.parent)
        nw.attributes('-topmost',True)
        nw.title("Grid Edit")
        tk.Label(nw, text="Focal Length").grid(row=1, column=3)
        tk.Label(nw, text="Centre").grid(row=1, column=2)
        tk.Label(nw, text="L1").grid(row=2, column=1)
        tk.Label(nw, text="L2").grid(row=3, column=1)
        ei=[]
        #r=2, c=2 l1 f, r3, c=2 l2f
        #r=2, c=3 c1 f, r3, c=3 c2 f
        for i in range(4):
            ei.append(tk.Entry(nw))
            if i<2:
                ei[-1].grid(row=2+i, column=2)# c1, c2
            else:
                ei[-1].grid(row=i, column=3)# l1, l2
        self.ei=ei
        self.submit = tk.Button(nw, text="Submit",  command=self.readInputs)
        self.submit.grid(column=1, row=4)
        
            
    def l2c(self):
        self.r2=-self.r2    
        self.l2r()
        
    def readInputs(self):
        rr=0
        fi=[]
        for ii in self.ei:
           if ii.get():
               if rr>1:
                   fi.append(int(ii.get()))
                   self.f=fi[-1]
                   cc=self.curve()
                   self.vars[rr]=cc
               else:
                   self.vars[rr]=int(ii.get())
          
               rr+=1
        self.varreset()
        self.l2r()
 
        '''self.OutputWindow()'''
        
    def varreset(self):
        self.l1=self.vars[0]
        self.l2=self.vars[1]
        self.r1=self.vars[2]
        self.r2=self.vars[3]

        
    def curve(self):
          ng=1.5168
          c=ng/(self.f*((ng-1)))
          return c
    def focal(self, c):
           f=ng/(c*((ng-1)))
           return f

    def l2r(self):
         #Single conv
         self.r=[]
         plt.clf()
         a=Bundle('s', 'k', self.thk, self.ang, 0)
         self.out=OutputPlane(self.o)
         fig,(ax) = plt.subplots(figsize=(12, 10))
         self.sa= SphericalRefraction(self.l1, -self.r1, nv, ng, 1/0.7) 
         if self.r2<0:
              s2= SphericalRefraction(self.l2, self.r2,ng, nv,  1/0.7) 
         else:
              s2= SphericalRefraction(self.l2, self.r2, nv, ng, 1/0.7) 
         self.sa.compplot()
         self.r.append(ray((self.sa.height,0, 0), (0,0,100)))
         self.r.append(ray((-self.sa.height,0, 0), (0,0,100)))
         a.propagate(self.sa)
         a.propagate(s2)
         self.image(a)
         imagepln=OpticalElement(self.loc, 1,1)
         for ri in self.r:
             self.sa.propagate_ray(ri)
             s2.propagate_ray(ri)
             self.intt=imagepln.planarIntercept(ri)
             self.r
             self.out.propagate_ray(ri)
         print(self.intt)
         a.propagate(self.out)
         ax.axvline(np.mean(self.imagpln),ymin=-100,ymax= 100,linestyle= '--')
         bundleplot(a.bundle,a.bundle, ax) 
         bundleplot(self.r, self.r, ax)
         canvas = FigureCanvasTkAgg(fig,master=self.parent)
         canvas.draw()
         canvas.get_tk_widget().grid(row=4, column=1)
         self.mag=self.intt[0]/self.sa.height
         self.OutputWindow()
         '''self.OutputWindow()'''


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("800x600")
    root.eval('tk::PlaceWindow . center')
    root.attributes('-topmost',True)
    App(root)
    root.mainloop()
    
    
    
    
    
    #%%~  
    
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:00:50 2020

@author: hills
"""
import numpy as np
import scipy as si
import matplotlib.pyplot as plt
import matplotlib

class ray:
    
    """
    Ray: Initialises with starting point and direction ray(p,k)
         Data attributes: p -> list of points, k -> current direction
         Methods: append(p,k)-> adds current direction and new point,
         getk() -> returns hidden variable direction, 
         getp()-> returns hidden point p, 
         vertices()-> returns all points
    """
    
    def __init__(self, p, k):
        """initialisation of point list and direction vector
           param: p -> starting point, k -> starting direction
        """
        self.__Vertices=[] 
        self.__Vertices.append(np.array(p)) #adding current vertice to vertice list
        self.k=np.array(k) #Adding direction
        self.terminated= False #Confirming that this is an active ray
        
    def getp(self):
        """returns current point"""
        return self.__Vertices[-1] #returns most recently added point from vertice list
    
    def getk(self):
        """returns current direction"""
        return self.k
    
    def append(self,p,k):
        """updates current point and direction"""
        self.__Vertices.append(np.array(p)) #adds point to vertice list
        self.k=np.array(k) #updates direction
        
    def vertices(self):
        """returns all ray points"""
        return self.__Vertices
    
    def terminate(self):
        """Updates rays status to inactive"""
        self.k = None #Removes Direction
        self.terminated = True #marks ray as inactive
    
import matplotlib.pyplot as plt
import numpy as np


        
class OpticalElement:
    
    """
    Optical Element: Creates a generalised optical element, acts as a super class for sphericalsurface, planarsurface and outputplane 
         Data Atrributes: z0-> centre on optical axis, n1-> refractive index of initial medium, n2-> refractive index of final medium
         Methods: Propagate_ray -> generalised propagation function for a given ray, 
         snells-> enacts snells law for a given initial direction, surface normal and refractive indices
         PlanarIntercept-> finds any intercept with a planar surface/output surface on the optical axis
         Propagater-> generalised propagation method that takes the elements precalculated values and propagates a ray
         
    """
    
    def __init__(self, z0, n1, n2):
         self.z0=z0 #assigns point on optical axis
         self.n1=n1 #assigns refractive index of initial medium
         self.n2=n2 #assigns refractive index of medium to be passed into
    
    def planarIntercept(self,ray):
        '''finds place it comes into contact with any planar surface: 
        Takes parameters: ray -> ray to be brought to plane, 
        Returns: intercept with plane '''
        k=ray.getk() #Finds ray direction
        p=ray.getp() #finds ray position 
        if k[2]<=0 or ray.getp()[2]>self.z0: # checks that the ray is moving in the k towards the lens and that it is located before the element on the optical axis
            finali=None
            raise Exception("No Intercept Occurs")
        val=(self.z0-p[2])/(k[2]) #finds scalar by which the other position values will change by the time the plane is reached
        finali= p+k*val #defines final intercept
        return finali 
    
    def propagator(self,ray, intercept, norm): #python doesnt allow overloading :( - this was easier for mine
        '''Propagates a ray through any optical element, 
        given the intercept of the ray and the normal at that point'''
        if intercept is None: # checks if intercept exists
           raise Exception("This ray will not come into contact with this Optical Element") #raises intercept exception
        newdirection=self.snells(ray.getk(), norm, self.n1, self.n2)  #finds new direction
        if newdirection is None:
           ray.terminate() #terminates ray in case of total internal reflection
        else:
           ray.append(intercept, newdirection)
       
    def snells(self, indir, surfnorm, n1, n2):
        '''snells: Applies snells law to a given input: Take parameters: indir -> initial direction of ray, surfnorm -> normal to lens at that point, 
        n1 -> refractive index of initial medium, n2 -> refractive index of new medium'''
        surfnorm=np.array(surfnorm)
        idnorm= np.array(indir/np.linalg.norm(indir)) #normalises initial direction
        costheta1= -np.dot(surfnorm, idnorm) #Finds incident costheta
        if np.sqrt(1-(costheta1**2)) >(n2/n1): #Tests for total internal reflection
    
            raise Exception("Total Internal Reflection has occured and the ray has been terminated")
        else: 
            costheta2= np.sqrt(1-((n1/n2)**2)*(1-(costheta1**2))) #finds refractive costheta
            fdir= (n1/n2)*idnorm+((n1/n2)*costheta1-costheta2)*surfnorm #finds final direction
        return fdir # finds final outward direction
        
class planeSurface(OpticalElement):
    
    """
    planeSurface - Subclass inheriting from opticalElement 
        Creates a planar surface with at a normal to the z-axis
        Parameters: z0, n1, n2
        Data attributes: z0 -> location of plane, n1 -> original mediums refractive index, n2 -> new mediums refractive index, surfnorm -> surface normal to plane
        Methods: intercept-> finds rays contact with plane, propagate_ray -> propagates ray to the surface and finds its new direction from it
    """
    def __init__(self, z0, n1, n2):
        OpticalElement.__init__(self,z0,n1,n2) #initialises values
        self.surfnorm= [0,0,-1] #creates a simple surface norm to be used in later calculations
        
    def propagate_ray(self, ray):
        '''Moves ray to plane - propagates it through, giving it a new direction and location, 
        parameters: ray to be propagated
        Appends new location and direction to ray 
        '''
        try: 
           intercept= self.planarIntercept(ray) # finds intercept
           if intercept<100 and intercept>-100:
                self.propagator(ray, intercept, self.surfnorm) # propagates ray through optical element
        except:
            print("Sad")
            
class film(OpticalElement):
    def __init__(self, z0, n1, n2, w):
        OpticalElement.__init__(self,z0,n1,n2) #initialises values
        self.surfnorm= [0,0,-1] #creates a simple surface norm to be used in later calculations
        self.pl1=planeSurface(z0, n1, n2)
        self.pl2=planeSurface(z0+w, n2, n1)
        self.z0=z0
        self.w=w
        self.c=1
        
    def propagate_ray(self, ray):
        '''Moves ray to plane - propagates it through, giving it a new direction and location, 
        parameters: ray to be propagated
        Appends new location and direction to ray 
        '''
        self.pl1.propagate_ray(ray)
        self.pl2.propagate_ray(ray) 
   
    def plot(self, h, ax=plt.gca()):
        cube=matplotlib.patches.Rectangle((self.z0, -h/2), self.w, h, fill=True, color='b')
        print("plot")
        ax.add_patch(cube)
        
class SphericalRefraction(OpticalElement):
    
    '''Spherical Refracting surface - Subclass inheriting from OpticalElement 
       initialises with z0-> x-axis intercept, c -> curvature of lens, n1,n2 -> refractive indices either side of surface, ar - Apeture radius
       Data Attributes: z0, c, n1, n1, n2
       Methods: intercet(ray) -> finds point of interception
      
    '''
    def __init__(self, z0, c, n1, n2, ar=None):
        OpticalElement.__init__(self,z0,n1,n2)
        self.c=c
        self.ar=1
        if c>0:
            OpticalElement.__init__(self,z0,n1,n2)
            self.plane=self.z0+self.ar
            self.pln= planeSurface(self.plane,n2, n1)
        else: 
            OpticalElement.__init__(self,z0,n2,n1)
            self.plane=self.z0-self.ar
            self.pln= planeSurface(self.plane,n1, n2)
        points=self.compplot()
        self.height=0


        
    def compplot(self):
        a=Bundle('s', 'k', 1000, 15, 0)
        points=[]
        for r in a.bundle:
            try:
                newpoint = self.Sphereintercept(r) 
                points.append(newpoint)
            except:
                print("suck it")
      
        points=np.transpose(points)
        total=[points[0], points[2]]
        total=np.transpose(total)
        xyt=sorted(total, key=lambda x : x[0])
        total=np.transpose(xyt)  
        x=[]
        y=[]
        if self.c>0:
          for i in range(len(total[1])):
            if total[1][i]<self.plane:
                x.append(total[1][i])
                y.append(total[0][i])
        else:
          for i in range(len(total[1])):
            if total[1][i]>self.plane:
                x.append(total[1][i])
                y.append(total[0][i])
        self.height=max(y)
            
            
            
        '''
        x,y = generate_semicircle(self.plane,0,100, 0.1)
        plt.plot(x,y, color='k' )
        '''
        plt.plot(x,y, color='k' )
        plt.vlines(self.plane,min(y), max(y), color='k' )
        '''
        xx=[total[1][0],total[1][-1] ]
        yy=[total[0][0],total[0][-1] ]
        plt.plot(xx,yy, color='k' )'''

    
    def findnorm(self, intercept):
        '''findnorm - Finds normal to the plane
        Parameters: intercept between ray and surface'''
        n = intercept-[0.0,0.0, self.z0 + 1/self.c] #finds vector from center to point
        n=n/np.linalg.norm(n) #normalises vector
        if self.c<0: #if the curvature is negative then the norm must be directed inwards and therefore needs to be negative
            n=-n #changes direction of norm     
        return n
        
    def propagate_ray(self, ray):
       '''propagate a ray through the optical element
       Parameters:ray to be propagated to Spherical lens'''
       try:
          if self.c>0:
             newpoint = self.Sphereintercept(ray) #finds intercept
             self.propagator(ray, newpoint, self.findnorm(newpoint)) #propagates ray to optical element
             self.pln.propagate_ray(ray)
          else:
              self.pln.propagate_ray(ray)
              newpoint = self.Sphereintercept(ray) #finds intercept
              self.propagator(ray, newpoint, self.findnorm(newpoint)) #propagates ray to optical element
   
              
       except:
           print("sadnes")
           
     
    def Sphereintercept( self, ray):
        '''Finds intercept between optical element and ray.
        - A ray object is taken as a parameter'''
        k=ray.getk() #finds direction of ray
        p= ray.getp() #finds location of ray
        if self.c == 0: #if curvature is 0-> acts like a planar surface
            finali=self.planarIntercept(ray) #finds intercept using method from parent function
        else:
            k=k/np.linalg.norm(ray.getk()) #normalises direction
            r=p-np.array([0.0,0.0, self.z0+1/self.c]) #finds radius
            rnorm=np.linalg.norm(r) 
            d = np.dot(r, k)*np.dot(r, k)-((rnorm*rnorm)-((1/self.c)*(1/self.c))) #finds determinant
            if d<0: #checks if it only returns imaginary values 
                raise "Death is nigh"
                #return self.planarIntercept(ray) 
            else:
                 if self.c>0: #concave lens    
                       nri= -np.dot(r, k) - np.sqrt(d) #furthest intercept
                 elif self.c<0: #convex lens
                       nri= -np.dot(r, k) + np.sqrt(d) #closest intercept
            #nri is the distance travelled in that direction to reach the spherical surface
            finali= nri*k + p 
         #finding intercept point by multiplying distance by direction and adding it to starting point
        return finali       
                
        
class OutputPlane(OpticalElement):
    '''OutputPlane - Finds the arrays intersection with the output plane
    Parameters: takes location on z - axis 
    Propagates the ray to the output plane '''
    
    def __init__(self, z):
        OpticalElement.__init__(self,z,None,None) #initialises z value without refractive indices
    
    def propagate_ray(self, ray):
        '''finds place it comes into contact with opticalPlane
        Parameters: ray -> ray to be propagated
        Returns: intercept with the output plane
        '''
        try:
           ri= self.planarIntercept(ray) #finds planar intercept with function from parent class
           ray.append(ri, ray.getk()) #append new direction and intercept
        except:
            print("Sad")

     
class Bundle():
    '''Bundle - Generates Bundle of arrays - circular with a predefined weight function: currently only for circular bundles
    Parameters: radius-> radius of bundle, centre -> centre of circle to be generated, direction -> direction of rays generated, number-> changes density per ring generated, colour -> colour to be plotted with'''
   
    def __init__(self, t, colour,number, direction, centre, radius=0):
       self.number= number # density of points
       self.centre=centre
       self.colour=colour
       self.direction=direction #direction for every ray

       if t=='c':
            self.r=radius 
            initialPoints= self.circle() #generates points in a circle 
            self.bundle=self.generate(initialPoints) #generates rays at given points
       else:
            if t=='s': 
             initialPoints, direc= self.source(direction)

             self.bundle=self.generate(initialPoints, direc) #generates rays at given points
              


    def propagate(self, element):
        '''Propagates each ray in a bundle through a given optical element
           Parameter-> optical element
        '''
        for ray in self.bundle:
            element.propagate_ray(ray) #propagates individual ray through element
                
        
    def plot(self):
        '''Plots xy slice at the bundles current z location
        '''
        x, y=[], []
        for ray in self.bundle:
            x.append(ray.getp()[0])
            y.append(ray.getp()[1])
        plt.plot(x,y,'o', color=self.colour) # plots slice
        plt.title("xy Plot of Bundle Cross Section")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
         
    def generate(self, locations, direct=0):
        '''generate - Generates rays corresponding to the pattern decided on
        Parameters: Locations -> points at which rays should be generated from
        Returns: returns list of rays at corresponding points'''
        r=[]
        if direct==0:
            for l in locations:
                r.append(ray(l, self.direction)) #creates array for each location in the array
        else: 
            for l in range(len(locations)):
                r.append(ray(locations[l], direct[l])) #creates array for each location in the array
      
        return r
    
    def source(self, maxtheta):
        points=[]
        direct=[]
        direct.append((0, 0, 1))
        points.append((0, self.centre, 0))        
        ang= np.linspace(0, maxtheta, self.number) #creates a number of values between 0 and 2pi, it creates n angles at equal intervals to create equally spaced pooints along a circles circumference

        for a in ang: 
            if np.tan(a)>0:
                direct.append((np.tan(a), 0, 1))
                points.append((0, self.centre, 0))
                direct.append((-np.tan(a), 0, 1))
                points.append((0, self.centre, 0))
      
        return points, direct
        
    def circle(self):
        '''circle - Generates circle of rays - uses a weight function to vary the number of points in each ring at equal distances from 
        the centre point'''
        radii=np.linspace(0.1, self.r, 6) #preset 6 rings
        total=[]
        for r in radii:
            total.append(r*self.number+1) #determines number of points at each radius- the number input increases the density everywhere
        points=[]
        points.append(self.centre) # ensures there is a central point
        for d in range(len(total)):         
            ang= np.linspace(0, 2*si.pi, int(total[d])) #creates a number of values between 0 and 2pi, it creates n angles at equal intervals to create equally spaced pooints along a circles circumference
            for a in ang: #for each of these angles
                points.append([radii[d]*np.cos(a)+self.centre[0], radii[d]*np.sin(a)+self.centre[1], self.centre[2]]) #determining x and y values for each to make a circle in addition to the z location they are all placed at
        return points
# -*- coding: utf-8 -*-
#import Raytracing as rt
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
        
def rmsLoc(a, c, d):
    '''rmsLoc - Uses rmsfinder to iterate between z=0 to z=300 checking to find the lowest rms
    - it does this by taking samples at increasingly small intervals to narrow down the minimum
    rms's exact position
    Parameters: a-> bundle to be iterated through 
    Returns: location of minimum rms and minimum rms value '''
    rmsa=[]
    
    space=np.linspace(c,d, 900)
    for a1 in space:
        rmsa.append(rmsfinder(a, a1))
    out1=rmsa.index(min(rmsa))
    
    space=np.linspace(space[out1]-1, space[out1]+1, 900)
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
    
    rf=plotprep([ray])[0]
    
    plt.plot(rf[2],rf[0])
    plt.title('Path of Ray in x,z')
    plt.xlabel('z')
    plt.ylabel('x')

    
def bundleplot(rays, rays2=[], ax=plt.subplots()):
    '''bundleplot: Plots multiple bundles - Accepts parameters: points -> a bundle of rays, points2 -> a seperate bundle of arrays
    Creates plots of these bundles - in different colours for each bundle 
    '''
    if len(rays2)==0:
        rays2=rays
        
    rf1=plotprep(rays) 
    rf2=plotprep(rays2)
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for r, r2 in zip(rf1, rf2):
        ax.plot(r[0], r[1], r[2], 'b') 
        ax.plot(r2[0], r2[1], r2[2], 'k') 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')'''
    if len(rays)>2:
        col="blue"
        col1="black"
        ls='-'
    else:
        col="red"
        col1="red"
        ls='--'
        
    for r, r2 in zip(rf1,rf2):  
       ax.plot(r[2],r[0], color=col, linestyle=ls)
       ax.plot(r2[2],r2[0], color=col1, linestyle=ls) 
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
  
def doublelens(centre, radi, a):
    sa= SphericalRefraction(centre-1/r1, r1, nv, ng, 1/0.7) 
    sr= SphericalRefraction(centre+1/r1, -r1, ng,nv, 1/0.7)

    return a, sa, sr
    

def focal(c, n):
     f=n/(c*((n-1)))
     return f
 
def curve(f):
      ng=1.5168
      c=ng/(f*((ng-1)))
      return c      
   


def total_mag(d, f, imgpln):

   image=0
   M=1
   d[1]=d[1]-d[0]
   print(d)
   for i in range(2):
      u = d[i] - imgpln
      fi = f[i]
      v = u*fi/(u-fi)
      image = d[i] + v
      M = M*v/u

   return M



'''

def total_mag(d, f, imgpln):
    f1=f[0]
    f2=f[1]
    l1=d[0]
    l2=d[1]
    
    di1 = 1/((1/f1)-(1/l1))
    L=l2-l1
    do2 = L-di1
    di2 = 1/((1/f2)-(1/do2))

    m1 = di1/l1
    m2 = di2/l2

    M=m1*m2
    return M
'''