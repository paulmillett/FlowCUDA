# ifndef ZERO_FORCES_2D
# define ZERO_FORCES_2D
# include <cuda.h>

__global__ void zero_forces_2D(float*,float*,int); 

# endif  // ZERO_FORCES_2D