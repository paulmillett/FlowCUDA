# ifndef MCMP_COMPUTE_SC_FORCES_PSM_D2Q9_H
# define MCMP_COMPUTE_SC_FORCES_PSM_D2Q9_H
# include <cuda.h>

__global__ void mcmp_compute_SC_forces_psm_D2Q9(float*,
                                                float*,
												float*,
                                                float*,
                                                float*,
					 		 		            float*,
						 				        float*,
												float*,
												float*,
												int*,
										        int*,
											    float,
												float,
												float,
												float,											    
                                                int);


# endif  // MCMP_COMPUTE_SC_FORCES_PSM_D2Q9_H