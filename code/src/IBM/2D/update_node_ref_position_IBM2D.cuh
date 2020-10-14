# ifndef UPDATE_NODE_REF_POSITION_IBM2D_H
# define UPDATE_NODE_REF_POSITION_IBM2D_H
# include <cuda.h>

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
		
# endif  // UPDATE_NODE_REF_POSITION_IBM2D_H