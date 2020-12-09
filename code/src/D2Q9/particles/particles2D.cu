
# include "particles2D.cuh"
# include "../../IO/GetPot"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

particles2D::particles2D()
{
	GetPot inputParams("input.dat");	
	nVoxels = inputParams("Lattice/nVoxels",0);
	nParts = inputParams("Particles/nParts",0);	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

particles2D::~particles2D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void particles2D::allocate()
{
	// allocate array memory (host):
    xH = (float*)malloc(nParts*sizeof(float));
	yH = (float*)malloc(nParts*sizeof(float));
	radH = (float*)malloc(nParts*sizeof(float));
				
	// allocate array memory (device):
	cudaMalloc((void **) &x, nParts*sizeof(float));
	cudaMalloc((void **) &y, nParts*sizeof(float));
	cudaMalloc((void **) &vx, nParts*sizeof(float));
	cudaMalloc((void **) &vy, nParts*sizeof(float));
	cudaMalloc((void **) &fx, nParts*sizeof(float));
	cudaMalloc((void **) &fy, nParts*sizeof(float));
	cudaMalloc((void **) &rad, nParts*sizeof(float));
	cudaMalloc((void **) &pIDgrid, nVoxels*sizeof(int));	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void particles2D::deallocate()
{
	// free array memory (host):
	free(xH);
	free(yH);
	free(radH);
				
	// free array memory (device):
	cudaFree(x);
	cudaFree(y);
	cudaFree(vx);
	cudaFree(vy);
	cudaFree(fx);
	cudaFree(fy);	
	cudaFree(rad);
	cudaFree(pIDgrid);
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void particles2D::memcopy_host_to_device()
{
    cudaMemcpy(x, xH, sizeof(float)*nParts, cudaMemcpyHostToDevice);
	cudaMemcpy(y, yH, sizeof(float)*nParts, cudaMemcpyHostToDevice);
	cudaMemcpy(rad, radH, sizeof(float)*nParts, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void particles2D::memcopy_device_to_host()
{
    cudaMemcpy(xH, x, sizeof(float)*nParts, cudaMemcpyDeviceToHost);
	cudaMemcpy(yH, y, sizeof(float)*nParts, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Wrtie output:
// --------------------------------------------------------

void particles2D::write_output(std::string tagname, int step)
{
	
}








