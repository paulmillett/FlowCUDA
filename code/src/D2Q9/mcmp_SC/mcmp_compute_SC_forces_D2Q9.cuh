# ifndef MCMP_COMPUTE_SC_FORCES_D2Q9_H
# define MCMP_COMPUTE_SC_FORCES_D2Q9_H
# include <cuda.h>

__global__ void mcmp_compute_SC_forces_1_D2Q9(float*,
                                              float*,
                                              float*,
                                              float*,
					 		 			      float*,
						 				      float*,
										      int*,
											  float,
                                              int);
											
__global__ void mcmp_compute_SC_forces_2_D2Q9(float*,
                                              float*,
											  float*,
											  float*,
											  float*,
											  float*,
											  int*,
											  float,
											  int);

# endif  // MCMP_COMPUTE_SC_FORCES_D2Q9_H