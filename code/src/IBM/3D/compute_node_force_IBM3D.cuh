# ifndef COMPUTE_NODE_FORCE_IBM3D_H
# define COMPUTE_NODE_FORCE_IBM3D_H
# include <cuda.h>

__global__ void compute_node_force_IBM3D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
	float*,
	float*,
	float*,
	float,
    int);

# endif  // COMPUTE_NODE_FORCE_IBM3D_H