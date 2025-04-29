# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:00:50 2020

@author: hills
"""
import numpy as np
import scipy as si
import matplotlib.pyplot as plt

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
        intercept= self.planarIntercept(ray) # finds intercept
        self.propagator(ray, intercept, self.surfnorm) # propagates ray through optical element
        
class SphericalRefraction(OpticalElement):
    
    '''Spherical Refracting surface - Subclass inheriting from OpticalElement 
       initialises with z0-> x-axis intercept, c -> curvature of lens, n1,n2 -> refractive indices either side of surface, ar - Apeture radius
       Data Attributes: z0, c, n1, n1, n2
       Methods: intercet(ray) -> finds point of interception
    '''
    def __init__(self, z0, c, n1, n2, ar=None):
        OpticalElement.__init__(self,z0,n1,n2) #initialises z0, n1 and n2
        self.c=c
        if ar==None: #if no value for ar has been given assign default
            self.ar=1/c
                     
        
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
       newpoint = self.Sphereintercept(ray) #finds intercept
       self.propagator(ray, newpoint, self.findnorm(newpoint)) #propagates ray to optical element
           
     
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
                raise  Exception("Intercept values are imaginary - therefore these points do not exist -")
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
        ri= self.planarIntercept(ray) #finds planar intercept with function from parent class
        ray.append(ri, ray.getk()) #append new direction and intercept
    
class Bundle():
    '''Bundle - Generates Bundle of arrays - circular with a predefined weight function: currently only for circular bundles
    Parameters: radius-> radius of bundle, centre -> centre of circle to be generated, direction -> direction of rays generated, number-> changes density per ring generated, colour -> colour to be plotted with'''
    def __init__(self, radius, centre, direction, number, colour):
        self.r=radius 
        self.number= number # density of points
        self.centre=centre
        self.colour=colour
        self.direction=direction #direction for every ray
        initialPoints= self.circle() #generates points in a circle 
        self.bundle=self.generate(initialPoints) #generates rays at given points

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
         
    def generate(self, locations):
        '''generate - Generates rays corresponding to the pattern decided on
        Parameters: Locations -> points at which rays should be generated from
        Returns: returns list of rays at corresponding points'''
        r=[]
        for l in locations:
            r.append(ray(l, self.direction)) #creates array for each location in the array
        return r

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
    