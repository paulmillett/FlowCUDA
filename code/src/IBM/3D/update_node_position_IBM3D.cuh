# ifndef UPDATE_NODE_POSITION_IBM3D_H
# define UPDATE_NODE_POSITION_IBM3D_H
# include <cuda.h>

__global__ void update_node_position_IBM3D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
    int);

__global__ void update_node_position_IBM3D(
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
	int,
	int,
    int);
		
# endif  // UPDATE_NODE_POSITION_IBM3D_H