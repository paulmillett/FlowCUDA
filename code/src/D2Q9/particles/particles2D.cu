
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
	rInnerH = (float*)malloc(nParts*sizeof(float));
	rOuterH = (float*)malloc(nParts*sizeof(float));
				
	// allocate array memory (device):
	cudaMalloc((void **) &x, nParts*sizeof(float));
	cudaMalloc((void **) &y, nParts*sizeof(float));
	cudaMalloc((void **) &vx, nParts*sizeof(float));
	cudaMalloc((void **) &vy, nParts*sizeof(float));
	cudaMalloc((void **) &fx, nParts*sizeof(float));
	cudaMalloc((void **) &fy, nParts*sizeof(float));
	cudaMalloc((void **) &rad, nParts*sizeof(float));
	cudaMalloc((void **) &rInner, nParts*sizeof(float));
	cudaMalloc((void **) &rOuter, nParts*sizeof(float));
	cudaMalloc((void **) &B, nVoxels*sizeof(float));
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
	free(rInnerH);
	free(rOuterH);
				
	// free array memory (device):
	cudaFree(x);
	cudaFree(y);
	cudaFree(vx);
	cudaFree(vy);
	cudaFree(fx);
	cudaFree(fy);	
	cudaFree(rad);
	cudaFree(B);
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
	cudaMemcpy(rInner, rInnerH, sizeof(float)*nParts, cudaMemcpyHostToDevice);
	cudaMemcpy(rOuter, rOuterH, sizeof(float)*nParts, cudaMemcpyHostToDevice);
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
// Calls to kernels:
// --------------------------------------------------------

void particles2D::zero_forces(int nBlocks, int nThreads)
{
    zero_forces_2D
	<<<nBlocks,nThreads>>>(fx,fy,nParts);
}

void particles2D::move_particles(int nBlocks, int nThreads)
{
    move_particles_2D
	<<<nBlocks,nThreads>>>(x,y,vx,vy,fx,fy,nParts);
}



// --------------------------------------------------------
// Wrtie output:
// --------------------------------------------------------

void particles2D::write_output(std::string tagname, int step)
{
	
}








