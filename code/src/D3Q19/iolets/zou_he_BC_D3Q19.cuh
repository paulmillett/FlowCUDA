# ifndef ZOU_HE_BC_D3Q19_H
# define ZOU_HE_BC_D3Q19_H
# include "boundary_condition_iolet.cuh"
# include <cuda.h>


__device__ void zou_he_BC_D3Q19(int, float*, iolet*);


# endif  // ZOU_HE_BC_D3Q19_H