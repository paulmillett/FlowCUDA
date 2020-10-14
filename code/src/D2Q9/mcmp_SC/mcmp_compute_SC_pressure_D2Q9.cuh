# ifndef MCMP_COMPUTE_SC_PRESSURE_D2Q9_H
# define MCMP_COMPUTE_SC_PRESSURE_D2Q9_H
# include <cuda.h>

__global__ void mcmp_compute_SC_pressure_D2Q9(float*,
                                              float*,
                                              float*,
                                              float,
                                              int);

# endif  // MCMP_COMPUTE_SC_PRESSURE_D2Q9_H