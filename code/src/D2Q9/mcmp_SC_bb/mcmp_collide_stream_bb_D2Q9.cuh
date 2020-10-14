# ifndef MCMP_COLLIDE_STREAM_BB_D2Q9_H
# define MCMP_COLLIDE_STREAM_BB_D2Q9_H
# include <cuda.h>

__global__ void mcmp_collide_stream_bb_D2Q9(float*,
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
                                         	int*,
                                         	float,
                                         	int);

# endif  // MCMP_COLLIDE_STREAM_BB_D2Q9_H