# ifndef SCSP_INITIAL_EQUILIBRIUM_D3Q19
# define SCSP_INITIAL_EQUILIBRIUM_D3Q19
# include <cuda.h>

__global__ void scsp_initial_equilibrium_D3Q19(float*,
                                               float*,
                                               float*,
                                               float*,
                                               float*,										  
                                               int);

# endif  // SCSP_INITIAL_EQUILIBRIUM_D3Q19