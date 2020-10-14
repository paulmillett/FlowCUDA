# ifndef ZOU_HE_BC_D3Q19_H
# define ZOU_HE_BC_D3Q19_H
# include <cuda.h>

__device__ void zou_he_velo_west_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_velo_east_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_velo_south_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_velo_north_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_velo_bottom_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_velo_top_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_pres_west_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_pres_east_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_pres_south_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_pres_north_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_pres_bottom_D3Q19(int, float*, float*, float, float, float, float);

__device__ void zou_he_pres_top_D3Q19(int, float*, float*, float, float, float, float);

# endif  // ZOU_HE_BC_D3Q19_H