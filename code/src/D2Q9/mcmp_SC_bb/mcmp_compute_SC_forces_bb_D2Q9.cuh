# ifndef MCMP_COMPUTE_SC_FORCES_BB_D2Q9_H
# define MCMP_COMPUTE_SC_FORCES_BB_D2Q9_H
# include <cuda.h>

__global__ void mcmp_compute_SC_forces_bb_D2Q9(float*,
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
                                               int);


# endif  // MCMP_COMPUTE_SC_FORCES_BB_D2Q9_H