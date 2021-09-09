
# include "gpu_parallel_reduction.cuh"



// --------------------------------------------------------
// This kernel sums the elements of an array within a 
// block:
// --------------------------------------------------------

__global__ void add_array_elements(const float* gArr,
                                   int arraySize,
								   float* gOut)
{
	// dimensions:
	int blockSize = blockDim.x;
	int thIdx = threadIdx.x;
	int gthIdx = thIdx + blockIdx.x*blockSize;
	const int gridSize = blockSize*gridDim.x;
	
	// tree sum:
	float sum = 0;
	for (int i = gthIdx; i < arraySize; i += gridSize) {
		sum += gArr[i];
	}	    
	extern __shared__ float shArr[];
	shArr[thIdx] = sum;
	__syncthreads();
	
	for (int size = blockSize/2; size>0; size/=2) { //uniform
	    if (thIdx<size) shArr[thIdx] += shArr[thIdx+size];
	    __syncthreads();
	}
	
	// final result:
	if (thIdx == 0) gOut[blockIdx.x] = shArr[0];
}
