# ifndef MCMP_COMPUTE_VELOCITY_DIP_D2Q9_H
# define MCMP_COMPUTE_VELOCITY_DIP_D2Q9_H
# include <cuda.h>
# include "particle2D_dip.cuh"

__global__ void mcmp_compute_velocity_dip_D2Q9(float*,
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
											   particle2D_dip*,
											   int*,
                                               int);

# endif  // MCMP_COMPUTE_VELOCITY_DIP_D2Q9_H