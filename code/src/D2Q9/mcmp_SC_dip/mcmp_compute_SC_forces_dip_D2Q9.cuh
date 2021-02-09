# ifndef MCMP_COMPUTE_SC_FORCES_DIP_D2Q9_H
# define MCMP_COMPUTE_SC_FORCES_DIP_D2Q9_H
# include <cuda.h>
# include "particle2D_dip.cuh"

__global__ void mcmp_compute_SC_forces_dip_D2Q9(float*,
                                                float*,
												float*,
                                                float*,
                                                float*,
					 		 		            float*,
						 				        float*,
												particle2D_dip*,
												int*,
										        int*,
											    float,
												float,
												float,
												float,											    
                                                int);


# endif  // MCMP_COMPUTE_SC_FORCES_DIP_D2Q9_H