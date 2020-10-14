
# ifndef INSIDE_HEMISPHERE_D3Q19_H
# define INSIDE_HEMISPHERE_D3Q19_H

# include <cuda.h>
# include <string>

__global__ void inside_hemisphere_D3Q19(float*,int*,int,int,int,int);

# endif  // INSIDE_HEMISPHERE_D3Q19_H