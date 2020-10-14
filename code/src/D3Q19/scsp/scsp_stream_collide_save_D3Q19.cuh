# ifndef SCSP_STREAM_COLLIDE_SAVE_D3Q19_H
# define SCSP_STREAM_COLLIDE_SAVE_D3Q19_H
# include "../iolets/boundary_condition_iolet.cuh"
# include <cuda.h>

__global__ void scsp_stream_collide_save_D3Q19(float*, float*, float*, float*, float*,
                                               float*, int*, int*, iolet*, float, int, bool);

# endif  // SCSP_STREAM_COLLIDE_SAVE_D3Q19_H 