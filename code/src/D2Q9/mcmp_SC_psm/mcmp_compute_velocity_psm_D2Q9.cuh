# ifndef MCMP_COMPUTE_VELOCITY_PSM_D2Q9_H
# define MCMP_COMPUTE_VELOCITY_PSM_D2Q9_H
# include <cuda.h>

__global__ void mcmp_compute_velocity_psm_D2Q9(float*,
                                         	   float*,
                                           	   float*,
                                           	   float*,
										   	   float*,
										       float*,
										       float*,
										       float*,
										       float*,
										       float*,
											   float*,
											   float*,
											   float*,
											   float*,
											   float*,
											   int*,
                                               int);

# endif  // MCMP_COMPUTE_VELOCITY_PSM_D2Q9_H