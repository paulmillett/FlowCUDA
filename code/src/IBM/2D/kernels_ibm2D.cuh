# ifndef KERNELS_IBM2D_H
# define KERNELS_IBM2D_H
# include <cuda.h>



__global__ void update_node_position_IBM2D(
	float*,
    float*,
    float*,
	float*,
    int);


__global__ void update_node_position_IBM2D(
	float*,
    float*,
    float*,
	float*,
	float*,
    float*,
    float*,
	float*,
	int,
	int,
    int);


__global__ void update_node_ref_position_IBM2D(
	float*,
    float*,
    int);


__global__ void update_node_ref_position_IBM2D(
	float*,
    float*,
	float*,
	float*,
    int);


__global__ void update_node_ref_position_IBM2D(
	float*,
    float*,
	float*,
	float*,
	float*,
	float*,
    int,
	int,
	int);


__global__ void set_reference_node_positions_IBM2D(
	float*,
    float*,
    float*,
	float*,
    int);
	
	
__global__ void compute_node_force_IBM2D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
	float,
    int);
	
	
__global__ void extrapolate_force_IBM2D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
	int,
    int);
	

__global__ void extrapolate_velocity_IBM2D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
	float*,
	int,
    int);


__global__ void interpolate_velocity_IBM2D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
	int,
    int);
	
		
# endif  // KERNELS_IBM2D_H