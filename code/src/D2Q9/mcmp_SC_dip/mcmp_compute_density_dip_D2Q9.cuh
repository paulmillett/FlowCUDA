# ifndef MCMP_COMPUTE_DENSITY_DIP_D2Q9
# define MCMP_COMPUTE_DENSITY_DIP_D2Q9
# include <cuda.h>

__global__ void mcmp_compute_density_dip_D2Q9(float*,
                                              float*,
                                           	  float*,
                                           	  float*,
                                          	  int);

# endif  // MCMP_COMPUTE_DENSITY_DIP_D2Q9