# ifndef ZOU_HE_BC_D2Q9_H
# define ZOU_HE_BC_D2Q9_H
# include "boundary_condition_iolet.cuh"
# include <cuda.h>


__device__ void zou_he_BC_D2Q9(int, float*, iolet2D*);


/*
__device__ void zou_he_velo_west_D2Q9(int, float*, float*, float, float, float);

__device__ void zou_he_velo_east_D2Q9(int, float*, float*, float, float, float);

__device__ void zou_he_velo_south_D2Q9(int, float*, float*, float, float, float);

__device__ void zou_he_velo_north_D2Q9(int, float*, float*, float, float, float);

__device__ void zou_he_pres_west_D2Q9(int, float*, float*, float, float, float);

__device__ void zou_he_pres_east_D2Q9(int, float*, float*, float, float, float);

__device__ void zou_he_pres_south_D2Q9(int, float*, float*, float, float, float);

__device__ void zou_he_pres_north_D2Q9(int, float*, float*, float, float, float);
*/


# endif  // ZOU_HE_BC_D2Q9_H