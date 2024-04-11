# ifndef KERNELS_SCSP_ACTIVE_D2Q9_H
# define KERNELS_SCSP_ACTIVE_D2Q9_H

# include "../iolets/boundary_condition_iolet.cuh"
# include "tensor2D.h" 
# include <cuda.h>



__global__ void scsp_active_zero_forces_D2Q9(
	float2*,
	int);


__global__ void scsp_active_initial_equilibrium_D2Q9(
	float*,
	float*,
	float2*,
	int);


__global__ void scsp_active_stream_collide_save_D2Q9(
	float*,
	float*,
	float*,
	float2*,
	int*,
	int*,
	float,
	int);


__global__ void scsp_active_stream_collide_save_forcing_D2Q9(
	float*,
    float*,
	float*,
	float2*,
	float2*,
	int*,
	int*,
	float,
	int);


__global__ void scsp_active_update_orientation_D2Q9(
	float2*,
	float2*,
	float2*,
	int*,
	float,
	float,
	int);


__global__ void scsp_active_update_orientation_diffusive_D2Q9(
	float2*,
	float2*,
	int*,
	float,
	int);
		

__global__ void scsp_active_fluid_stress_D2Q9(
	float2*,
	float2*,
	tensor2D*,
	int*,
	float,
	float,
	float,
	int);


__global__ void scsp_active_fluid_forces_D2Q9(
	float2*,
	tensor2D*,
	int*,
	int);


__global__ void scsp_active_fluid_molecular_field_D2Q9(
	float2*,
	float2*,
	tensor2D*,
	int*,
	float,
	float,
	int);


__global__ void scsp_active_fluid_molecular_field_with_phi_D2Q9(
	float*,
	float2*,
	float2*,
	tensor2D*,
	int*,
	float,
	float,
	float,
	int);
			
	
__global__ void scsp_active_fluid_chemical_potential_D2Q9(
	float*,
	float*,
	float2*,
	int*,
	float,
	float,
	float,
	float,
	int);
	

__global__ void scsp_active_fluid_chemical_potential_2phi_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float2*,
	int*,
	float,
	float,
	float,
	float,
	int);

	
__global__ void scsp_active_fluid_capillary_force_D2Q9(
	float*,
	float*,
	float2*,
	int*,
	int);
	

__global__ void scsp_active_fluid_capillary_force_2phi_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float2*,
	int*,
	int);
		
	
__global__ void scsp_active_fluid_update_phi_D2Q9(
	float*,
	float*,
	float2*,
	int*,
	float,
	int);


__global__ void scsp_active_fluid_update_phi_2phi_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float2*,
	int*,
	float,
	int);


__global__ void scsp_active_fluid_update_phi_diffusive_D2Q9(
	float*,
	float*,
	float2*,
	int*,
	float,
	int);
	
	
__global__ void scsp_active_fluid_set_velocity_field_D2Q9(
	float2*,
	float2*,
	float,
	int);


__device__ tensor2D grad_vector_field_D2Q9(
	int,
	float2*,
	int*);
	
	
__device__ float2 grad_scalar_field_D2Q9(
	int,
	float*,
	int*);
		
		
__device__ float2 laplacian_vector_field_D2Q9(
	int,
	float2*,
	int*);
	
	
__device__ float laplacian_scalar_field_D2Q9(
	int,
	float*,
	int*);
	
	
__device__ float divergence_vector_field_D2Q9(
	int,
	float2*,
	int*);
	
	
__device__ float divergence_vector_scalar_field_D2Q9(
	int,
	float2*,
	float*,
	int*);


__device__ float2 divergence_tensor_field_D2Q9(
	int,
	tensor2D*,
	int*);


# endif  // KERNELS_SCSP_ACTIVE_D2Q9_H