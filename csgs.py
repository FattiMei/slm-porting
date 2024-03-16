import numpy as np
import time


get_time=time.perf_counter


#For all five functions, x,y,z are one-dimensional arrays of floats with coordinates in micrometers, f is a float with the
#equivalent focal of the system in mm, d is a float with the pixel size in micrometers, lam is the wavelength used in
#micrometers, res is the lateral resolution of the SLM.The software assumes the use of a circular area within a square slm:
#additional zero padding may be needed to operate rectangular SLMs


def get_performance_metrics(intensities, t):
    min = np.min(intensities)
    max = np.max(intensities)

    efficiency = np.sum(intensities)
    uniformity = 1 - (max - min) / (max + min)
    variance   = np.sqrt(np.var(intensities)) / np.mean(intensities)

    return [efficiency, uniformity, variance, t]


# Random superposition algorithm: Fastest available algorithm, produces low quality holograms

def rs(x, y, z, f: float, d: float, lam: float, res: int, seed: int):
    rng = np.random.default_rng(seed)
    t=get_time()

    #creation of a list of the SLM pixels contained in the pupil
    # @rank(slm_xcoord) = 2 , @shape(slm_xcoord) = (res, res)
    # @rank(slm_ycoord) = 2 , @shape(slm_ycoord) = (res, res)
    slm_xcoord,slm_ycoord=np.meshgrid(np.linspace(-1.0,1.0,res),np.linspace(-1.0,1.0,res))

    # @shape(pup_coords) = tuple((m), (m))
    # where m is the number of pixels in the pupil
    pup_coords=np.where(slm_xcoord**2+slm_ycoord**2<1.0)

    #array containing the phase of the field at each created spot
    # @shape(pists) = (n)
    # where n is the number of desired points
    pists=rng.random(x.shape[0])*2*np.pi

    #conversion of the coordinates arrays in microns
    slm_xcoord=slm_xcoord*d*float(res)/2.0
    slm_ycoord=slm_ycoord*d*float(res)/2.0
    
    #computation of the phase patterns generating each single spot independently
    # @shape(slm_p_phase) = (n, m)
    slm_p_phase=np.zeros((x.shape[0],pup_coords[0].shape[0]))

    for i in range(x.shape[0]):
        # @shape(slm_p_phase[i,:]) = (m)
        # @shape(slm_xcoord[pup_coords]) = (m)
        # @OPT: I think there is even a smarter way to make this operation, not so relevant right now
        slm_p_phase[i,:]=2.0*np.pi/(lam*(f*10.0**3))*(x[i]*slm_xcoord[pup_coords]+y[i]*slm_ycoord[pup_coords])+(np.pi*z[i])/(lam*(f*10.0**3)**2)*(slm_xcoord[pup_coords]**2+slm_ycoord[pup_coords]**2)


    #creation of the hologram, as superposition of all the phase patterns with random pistons
    # @OPT: pup_coords[0].shape[0] = m
    # @OPT: move the constant term out of the sum (requires accurate regression tests)
    # @shape(slm_total_field) = (m) (need confirm)
    # @shape(slm_total_phase) = (m)
    slm_total_field=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_p_phase+pists[:,None])),axis=0)
    slm_total_phase=np.angle(slm_total_field)

    t=get_time()-t

    #evaluation of the algorithm performance, calculating the expected intensities of all spots

    # @OPT: pup_coords[0].shape[0] = m
    # @shape(spot_fields) = (n)
    # @shape(ints) = (n)
    spot_fields=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_total_phase[None,:]-slm_p_phase)),axis=1)
    ints=np.abs(spot_fields)**2


    #reshaping of the hologram in a square array

    out=np.zeros((res,res))
    out[pup_coords]=slm_total_phase

    #the function returns the hologram, and a list with efficiency, uniformity and variance of the spots, and hologram computation time
    
    return out, get_performance_metrics(ints, t)




# Standard GS algorithm: Slow, high efficiency holograms, better uniformity than RS. The parameter "iters" is the number of GS iterations to
# perform

def gs(x, y, z, f: float, d: float, lam: float, res: int, iters: int, seed: int):
    rng = np.random.default_rng(seed)
    t=get_time()

    #creation of a list of the SLM pixels contained in the pupil
    slm_xcoord,slm_ycoord=np.meshgrid(np.linspace(-1.0,1.0,res),np.linspace(-1.0,1.0,res))
    pup_coords=np.where(slm_xcoord**2+slm_ycoord**2<1.0)
    
    
    #array containing the phase of the field at each created spot
    pists=rng.random(x.shape[0])*2*np.pi

    #conversion of the coordinates arrays in microns
    slm_xcoord=slm_xcoord*d*float(res)/2.0
    slm_ycoord=slm_ycoord*d*float(res)/2.0
    
    #computation of the phase patterns generating each single spot independently
    slm_p_phase=np.zeros((x.shape[0],pup_coords[0].shape[0]))
    for i in range(x.shape[0]):
        slm_p_phase[i,:]=2.0*np.pi/(lam*(f*10.0**3))*(x[i]*slm_xcoord[pup_coords]+y[i]*slm_ycoord[pup_coords])+(np.pi*z[i])/(lam*(f*10.0**3)**2)*(slm_xcoord[pup_coords]**2+slm_ycoord[pup_coords]**2)


    #main GS loop
    for n in range(iters):
        #creation of the hologram, as superposition of all the phase patterns with random pistons
        slm_total_field=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_p_phase+pists[:,None])),axis=0)
        slm_total_phase=np.angle(slm_total_field)


        #Update of the phases at all spots locations. The intensities are evaluated too for performance
        #estimation of the algorithm
        spot_fields=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_total_phase[None,:]-slm_p_phase)),axis=1)
        pists=np.angle(spot_fields)
        ints=np.abs(spot_fields)**2

    t=get_time()-t
            
    #the function returns the hologram, and a list with efficiency, uniformity and variance of the spots, and hologram computation time
    out=np.zeros((res,res))
    out[pup_coords]=slm_total_phase

    return out, get_performance_metrics(ints, t)



# Standard WGS algorithm: Slow, high uniformity holograms, better efficiency than RS. The parameter "iters" is the number of GS iterations to
# perform

def wgs(x, y, z, f: float, d: float, lam: float, res: int, iters: int, seed: int):
    rng = np.random.default_rng(seed)
    t=get_time()

    #creation of a list of the SLM pixels contained in the pupil
    slm_xcoord,slm_ycoord=np.meshgrid(np.linspace(-1.0,1.0,res),np.linspace(-1.0,1.0,res))
    pup_coords=np.where(slm_xcoord**2+slm_ycoord**2<1.0)
    
    #initialization of the weights, all with equal value
    weights=np.ones(x.shape[0])/float(x.shape[0])
    
    #array containing the phase of the field at each created spot
    pists=rng.random(x.shape[0])*2*np.pi

    #conversion of the coordinates arrays in microns
    slm_xcoord=slm_xcoord*d*float(res)/2.0
    slm_ycoord=slm_ycoord*d*float(res)/2.0
    
    #computation of the phase patterns generating each single spot independently
    slm_p_phase=np.zeros((x.shape[0],pup_coords[0].shape[0]))
    for i in range(x.shape[0]):
        slm_p_phase[i,:]=2.0*np.pi/(lam*(f*10.0**3))*(x[i]*slm_xcoord[pup_coords]+y[i]*slm_ycoord[pup_coords])+(np.pi*z[i])/(lam*(f*10.0**3)**2)*(slm_xcoord[pup_coords]**2+slm_ycoord[pup_coords]**2)


    #main GS loop
    for n in range(iters):
        #Creation of the hologram, as superposition of all the phase patterns with random pistons and weighted intensity
        slm_total_field=np.sum(weights[:,None]/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_p_phase+pists[:,None])),axis=0)
        slm_total_phase=np.angle(slm_total_field)


        #Update of the phases at all spots locations. The intensities are evaluated too for performance
        #estimation of the algorithm
        spot_fields=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_total_phase[None,:]-slm_p_phase)),axis=1)
        pists=np.angle(spot_fields)
        ints=np.abs(spot_fields)**2
        #update and renormalization of the weights. Renormalization is required to avoid computation precision errors in the final performance for
        #large numbers of iterations
        weights=weights*(np.mean(np.sqrt(ints))/np.sqrt(ints))
        weights=weights/np.sum(weights)


    t=get_time()-t
            
    #the function returns the hologram, and a list with efficiency, uniformity and variance of the spots, and hologram computation time
    out=np.zeros((res,res))
    out[pup_coords]=slm_total_phase

    return out, get_performance_metrics(ints, t)



# Compressive sensing GS:Fast, similar performances to GS. The parameter "sub" is a float greater than 0.0 and up to 1.0, indicating the
# Subsampling of the pupil. For sub=1.0, CSGS is equivalent to GS. Smaller values of sub increase the speed, with a maximum speed
# equal to the speed of RS. If sub is too small, performance may be affected (as a rule of thumb res^2*sub should be bigger than the
# number of spots)

def csgs(x, y, z, f: float, d: float, lam: float, res: int, iters: int, sub: float, seed: int):
    rng = np.random.default_rng(seed)
    t=get_time()
    #creation of a list of the SLM pixels contained in the pupil
    slm_xcoord,slm_ycoord=np.meshgrid(np.linspace(-1.0,1.0,res),np.linspace(-1.0,1.0,res))
    pup_coords=np.where(slm_xcoord**2+slm_ycoord**2<1.0)

    #creation of a list of the indexes of pup_coords, shuffled in random order
    coordslist=np.asarray(range(pup_coords[0].shape[0]))
    rng.shuffle(coordslist)
    
    #array containing the phase of the field at each created spot
    pists=rng.random(x.shape[0])*2*np.pi

    
    #conversion of the coordinates arrays in microns
    slm_xcoord=slm_xcoord*d*float(res)/2.0
    slm_ycoord=slm_ycoord*d*float(res)/2.0
    
    #computation of the phase patterns generating each single spot independently
    slm_p_phase=np.zeros((x.shape[0],pup_coords[0].shape[0]))
    for i in range(x.shape[0]):
        slm_p_phase[i,:]=2.0*np.pi/(lam*(f*10.0**3))*(x[i]*slm_xcoord[pup_coords]+y[i]*slm_ycoord[pup_coords])+(np.pi*z[i])/(lam*(f*10.0**3)**2)*(slm_xcoord[pup_coords]**2+slm_ycoord[pup_coords]**2)

    #main GS loop
    for n in range(iters):
        #a new set of random points is chosen on the SLM
        coordslist=np.roll(coordslist,int(coordslist.shape[0]*sub))
        coordslist_sparse=coordslist[:int(coordslist.shape[0]*sub)]

        #Creation of the hologram, as superposition of all the phase patterns with the estimated phases and weighted intensity
        slm_total_field=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_p_phase[:,coordslist_sparse]+pists[:,None])),axis=0)
        slm_total_phase=np.angle(slm_total_field)

        #Update of the phases at all spots locations. The intensities are evaluated too for performance
        #estimation of the algorithm
        spot_fields=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_total_phase[None,:]-slm_p_phase[:,coordslist_sparse])),axis=1)
        pists=np.angle(spot_fields)
        ints=np.abs(spot_fields)**2


    #The output holograms is estimated with the full SLM resolution
    slm_total_field=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_p_phase+pists[:,None])),axis=0)
    slm_total_phase=np.angle(slm_total_field)

    t=get_time()-t


    #evaluation of the algorithm performance, calculating the expected intensities of all spots
    spot_fields=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_total_phase[None,:]-slm_p_phase)),axis=1)
    ints=np.abs(spot_fields)**2
            
    #the function returns the hologram, and a list with efficiency, uniformity and variance of the spots, and hologram computation time
    out=np.zeros((res,res))
    out[pup_coords]=slm_total_phase

    return out, get_performance_metrics(ints, t)

# Weighted compressive sensing GS:Fast, efficiency and uniformity between GS and WGS.
# The parameter "sub" is a float greater than 0.0 and up to 1.0, indicating the Subsampling of the pupil. For sub=1.0, CSGS is equivalent to GS.
# Smaller values of sub increase the speed, with a maximum speed equal to half the speed of RS. If sub is too small, performance may be affected
# (as a rule of thumb (res^2)*sub should be at least twice the number of spots)

def wcsgs(x, y, z, f: float, d: float, lam: float, res: int, iters: int, sub: float, seed: int):
    rng = np.random.default_rng(seed)
    t=get_time()
    #creation of a list of the SLM pixels contained in the pupil
    slm_xcoord,slm_ycoord=np.meshgrid(np.linspace(-1.0,1.0,res),np.linspace(-1.0,1.0,res))
    pup_coords=np.where(slm_xcoord**2+slm_ycoord**2<1.0)

    #creation of a list of the indexes of pup_coords, shuffled in random order
    coordslist=np.asarray(range(pup_coords[0].shape[0]))
    rng.shuffle(coordslist)
    
    #array containing the phase of the field at each created spot
    pists=rng.random(x.shape[0])*2*np.pi

    
    #conversion of the coordinates arrays in microns
    slm_xcoord=slm_xcoord*d*float(res)/2.0
    slm_ycoord=slm_ycoord*d*float(res)/2.0
    
    #computation of the phase patterns generating each single spot independently
    slm_p_phase=np.zeros((x.shape[0],pup_coords[0].shape[0]))
    for i in range(x.shape[0]):
        slm_p_phase[i,:]=2.0*np.pi/(lam*(f*10.0**3))*(x[i]*slm_xcoord[pup_coords]+y[i]*slm_ycoord[pup_coords])+(np.pi*z[i])/(lam*(f*10.0**3)**2)*(slm_xcoord[pup_coords]**2+slm_ycoord[pup_coords]**2)

    #main GS loop
    for n in range(iters-1):
        #a new set of random points is chosen on the SLM
        # @POTENTIAL_BUG: here we are using a random function on the global rng
        coordslist=np.roll(coordslist,int(coordslist.shape[0]*sub))
        coordslist_sparse=coordslist[:int(coordslist.shape[0]*sub)]

        #Creation of the hologram, as superposition of all the phase patterns with the estimated phases
        slm_total_field=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_p_phase[:,coordslist_sparse]+pists[:,None])),axis=0)
        slm_total_phase=np.angle(slm_total_field)

        #Update of the phases at all spots locations. The intensities are evaluated too for performance
        #estimation of the algorithm
        spot_fields=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_total_phase[None,:]-slm_p_phase[:,coordslist_sparse])),axis=1)
        pists=np.angle(spot_fields)
        ints=np.abs(spot_fields)**2


    # an additional single loop of WGS without compression
    slm_total_field=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_p_phase+pists[:,None])),axis=0)
    slm_total_phase=np.angle(slm_total_field)

    spot_fields=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_total_phase[None,:]-slm_p_phase)),axis=1)
    pists=np.angle(spot_fields)
    ints=np.abs(spot_fields)**2

    weights=np.ones(x.shape[0])/float(x.shape[0])*(np.mean(np.sqrt(ints))/np.sqrt(ints))
    weights=weights/np.sum(weights)

    slm_total_field=np.sum(weights[:,None]/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_p_phase+pists[:,None])),axis=0)
    slm_total_phase=np.angle(slm_total_field)

    t=get_time()-t

    #evaluation of the algorithm performance, calculating the expected intensities of all spots
    spot_fields=np.sum(1.0/(float(pup_coords[0].shape[0]))*np.exp(1j*(slm_total_phase[None,:]-slm_p_phase)),axis=1)
    ints=np.abs(spot_fields)**2    
            
    #the function returns the hologram, and a list with efficiency, uniformity and variance of the spots, and hologram computation time
    out=np.zeros((res,res))
    out[pup_coords]=slm_total_phase

    return out, get_performance_metrics(ints, t)


if __name__ == "__main__":
    # usage example for a 20mm focal length system, a 512x512 pixels slm with 15 micron pitch, and 488nm wavelength.
    # x,y,z are arrays with the desired positions of 100 points, chosen at random in a 100x100x10 volume.
    # GS,CSGS,WGS and WCSGS algorithms are run for 30 iterations. CSGS and WCSGS are run with a compression factor
    # of 0.05 (only 1 out of 20 pixels of the SLM is considered in the loop)
    FOCAL_LENGTH = 20.0
    PIXELS       = 512
    PITCH        = 15.0
    WAVELENGTH   = 0.488
    NPOINTS      = 100
    ITERATIONS   = 30
    COMPRESSION  = 0.05


    # make reproducible results, but be careful when changing the global seed!
    SEED = 42
    np.random.seed(SEED)


    x=(np.random.random(NPOINTS)-0.5)*100.0
    y=(np.random.random(NPOINTS)-0.5)*100.0
    z=(np.random.random(NPOINTS)-0.5)*10.0


    performance_pars = [
            "Efficiency           : ",
            "Uniformity           : ",
            "Variance             : ",
            "Computation time (s) : "
    ]


    print("Computing random superposition hologram:")
    phase, performance=rs(x,y,z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,SEED)

    for i in range(3):
        print(performance_pars[i],performance[i])
    print()


    print("Computing Gerchberg-Saxton hologram:")
    phase, performance=gs(x,y,z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,SEED)
   
    for i in range(3):
        print(performance_pars[i],performance[i])
    print()


    print("Computing Weighted Gerchberg-Saxton hologram:")
    phase, performance=wgs(x,y,z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,SEED)

    for i in range(3):
        print(performance_pars[i],performance[i])
    print()


    print("Computing Compressive Sensing Gerchberg-Saxton hologram:")
    phase, performance=csgs(x,y,z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,SEED)

    for i in range(3):
        print(performance_pars[i],performance[i])
    print()


    print("Computing Weighted Compressive Sensing Gerchberg-Saxton hologram:")
    phase, performance=wcsgs(x,y,z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,SEED)

    for i in range(3):
        print(performance_pars[i],performance[i])
    print()
