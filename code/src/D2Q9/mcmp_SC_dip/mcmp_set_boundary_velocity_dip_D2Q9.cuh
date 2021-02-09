# ifndef MCMP_SET_BOUNDARY_VELOCITY_DIP_D2Q9_H
# define MCMP_SET_BOUNDARY_VELOCITY_DIP_D2Q9_H
# include <cuda.h>

__global__ void mcmp_set_boundary_velocity_dip_D2Q9(float*,
                                         	        float*,
                                           	        float*,
                                           	        float*,
										   	        float*,
										            float*,
										            float*,
										            float*,
													int*,
										            int,
                                                    int);

# endif  // MCMP_SET_BOUNDARY_VELOCITY_DIP_D2Q9_H