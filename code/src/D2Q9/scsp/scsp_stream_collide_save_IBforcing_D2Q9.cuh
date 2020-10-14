# ifndef SCSP_STREAM_COLLIDE_SAVE_IBFORCING_D2Q9_H
# define SCSP_STREAM_COLLIDE_SAVE_IBFORCING_D2Q9_H
# include "../iolets/boundary_condition_iolet.cuh"
# include <cuda.h>

__global__ void scsp_stream_collide_save_IBforcing_D2Q9(float*,
float*, float*, float*, float*, float*, float*, float*, int*, int*, iolet2D*, 
float, int);

# endif  // SCSP_STREAM_COLLIDE_SAVE_IBFORCING_D2Q9_H 