# ifndef SET_REFERENCE_NODE_POSITIONS_IBM3D_H
# define SET_REFERENCE_NODE_POSITIONS_IBM3D_H
# include <cuda.h>

__global__ void set_reference_node_positions_IBM3D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
    int);

# endif  // SET_REFERENCE_NODE_POSITIONS_IBM3D_H