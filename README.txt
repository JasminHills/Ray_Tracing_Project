__________________________________________________________________
To create your own optical system: 


Initialising Rays:

rt.ray(PositionVector, DirectionVector)

Initialising Optical Elements:

 spherical= rt.SphericalRefraction(z0, c, n1, n2, r=None) - can chose to define r (but a automatic value will be assigned if not)

 output=rt.OutputPlane(zPosition) 

 Plane= rt.planeSurface(z0, n1, n2) 


Propagating rays:

Individual:

OpticalElementName.propagate_ray(rayName)

In a Bundle:

BundleName.propagate(OpticalElementName)


Finding RMS Value of Bundle and its location:

location1, rms1=rmsLoc(BundleName)


Plotting:

RayTracing:

bundleplot(BundleName.bundle, BundleName.bundle) - repeat twice for just one, put different for multiple bundles (generates 3D and 2D)
fullplot(rayName)

XY plot at most recent z value for a bundle (used when you propagate to focal point):

BundleName.plot()

_____________________________________________________________

Summary of testing file: 

To Begin testing run the intitial cell- This will define functions required for later optimization and plotting
- It will also import the raytracing module. 

Cell 2:
Sets up simple system where a spherical surface and output plane for propagation of single rays

Cells 3-8:
Example simple rays being propagated through system defined in cell 2.
Checks they converge for this positive curvature - change curvature to see them diverge

Cell 8 has an output of the rays final direction and location to verify it ends up at z=250
and verify that the direction is coherent - as in this case the ray should be heading towards optical axis/converging

Cell 9:
Compares paraxial arrays with opposite positions and directions and compares+ prints the final focal points

Cell 10:
Compares and estimate of focal lengths for a set of rays to what they are meant to be for a perfect system with no aberations 
- Ensures they close to each other

Cell 11:
Prints rays xy that has been linearly generated to see that the shape is maintained and that
our plotting is working correctly

Cell 12:
Checks that the intercept with the lens of Positive curvature arrives first - i.e. that 
if the curvature is positive it intercepts with the spherical surface before z0 and the
opposite for negative curvatures

Cell 13:
Generatures two bundles through a positive lens

Cell 14:
Generates a propagates a bundle through a given spherical surface

Cell 15:
Plots the cell 14 bundle at focal plane - can be run without cell 14

Cell 16:
Tests the planar surface against snells law - with example ray going through planar surface

Cell 17:
Plots plano-convex lens in configuration 1 (planar surface first)
PLots trace of ray in addition to focal point image

Cell 18:
Plots plano-convex lens in configuration 2 (spherical surface first)
PLots trace of ray in addition to focal point image

Cell 19:
Models and plots bundle through biconvex lens of given c1=0.02, c2=-0.02 
Plots ray trace and focal plane image

Cell 20:
Simplified Gensing - with no plots
uses scipy minimize to find optical c1, c2
Also contains my own little method to find minimum - though that is v inefficient and takes ages to run so has been commented out




