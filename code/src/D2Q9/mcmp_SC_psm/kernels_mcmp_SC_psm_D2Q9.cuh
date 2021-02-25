
# ifndef KERNELS_MCMP_SC_PSM_D2Q9_H
# define KERNELS_MCMP_SC_PSM_D2Q9_H
# include <cuda.h>



__global__ void mcmp_initial_equilibrium_psm_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int);


__global__ void mcmp_update_particles_on_lattice_psm_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,											              
	int*,
	int*,
	int*,
	float,													      
	int,
	int);


__global__ void mcmp_set_boundary_velocity_psm_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int,
	int);


__global__ void mcmp_compute_density_psm_D2Q9(
	float*,
	float*,
	float*,
	float*,
	int);
												  
												  
__global__ void mcmp_compute_velocity_psm_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int);	
	
	
__global__ void mcmp_compute_SC_forces_psm_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int*,
	float,
	float,
	float,
	float,											    
	int);


__global__ void mcmp_collide_stream_psm_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*, 
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int*,
	float,
	float,
	float,
	int);	
	
	
										  
												  

# endif  // KERNELS_MCMP_SC_PSM_D2Q9_H