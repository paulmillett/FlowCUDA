# ifndef GPU_PARALLEL_REDUCTION_H
# define GPU_PARALLEL_REDUCTION_H
# include <cuda.h>


__global__ void add_array_elements(
	const float*,
	int,
	float*);


# endif  // GPU_PARALLEL_REDUCTION_H