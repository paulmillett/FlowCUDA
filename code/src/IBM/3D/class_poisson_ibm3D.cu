
# include "class_poisson_ibm3D.cuh"
# include <math.h>
# include <iostream>
using namespace std;



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_poisson_ibm3D::class_poisson_ibm3D()
{
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_poisson_ibm3D::~class_poisson_ibm3D()
{
		
}



// --------------------------------------------------------
// Initialize arrays:
// --------------------------------------------------------

void class_poisson_ibm3D::initialize(int Nxin, int Nyin, int Nzin)
{
	
	Nx = Nxin;
	Ny = Nyin;
	Nz = Nzin;
	nVoxels = Nx*Ny*Nz;
	
	// define cuFFT plan
	cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_C2C);
		
	// wave-vector arrays (host)
	float* kxH = new float[Nx];
	float* kyH = new float[Ny];
	float* kzH = new float[Nz];
	for (int i=0; i<=Nx/2; i++)   kxH[i] = i*2*M_PI;
	for (int i=Nx/2+1; i<Nx; i++) kxH[i] = (i-Nx)*2*M_PI;
	for (int j=0; j<=Ny/2; j++)   kyH[j] = j*2*M_PI;
	for (int j=Ny/2+1; j<Ny; j++) kyH[j] = (j-Ny)*2*M_PI;	
	for (int k=0; k<=Nz/2; k++)   kzH[k] = k*2*M_PI;
	for (int k=Nz/2+1; k<Nz; k++) kzH[k] = (k-Nz)*2*M_PI;
	
	// allocate host arrays
	indicatorH = (float*)malloc(nVoxels*sizeof(float));
	
	// allocate device arrays
	cudaMalloc((void**)&kx, sizeof(float)*Nx);
	cudaMalloc((void**)&ky, sizeof(float)*Ny);
	cudaMalloc((void**)&kz, sizeof(float)*Nz);
	cudaMalloc((void**)&rhs, sizeof(cufftComplex)*Nx*Ny*Nz);
	cudaMalloc((void**)&indicator, sizeof(float)*Nx*Ny*Nz);	
	cudaMalloc((void**)&G, sizeof(float3)*Nx*Ny*Nz);
	
	// memcopy wave-vector arrays to device
	cudaMemcpy(kx,kxH,sizeof(float)*Nx, cudaMemcpyHostToDevice);
	cudaMemcpy(ky,kyH,sizeof(float)*Ny, cudaMemcpyHostToDevice);
	cudaMemcpy(kz,kzH,sizeof(float)*Nz, cudaMemcpyHostToDevice);
	
}



// --------------------------------------------------------
// Solve poisson equation:
// --------------------------------------------------------

void class_poisson_ibm3D::solve_poisson(triangle* faces, float3* r, int nFaces, int nBlocks, int nThreads)
{
	// extrapolate IBM interface normal vectors to fluid grid:
	extrapolate_interface_normal_poisson_IBM3D
	<<<nBlocks,nThreads>>> (r,G,Nx,Ny,Nz,nFaces,faces);	
	
	// calculate RHS of poisson equation (div.G):
	calculate_rhs_poisson_IBM3D
	<<<nBlocks,nThreads>>> (G,rhs,nVoxels,Nx,Ny,Nz);
	
	// forward FFT (in-place):
	cufftExecC2C(plan, rhs, rhs, CUFFT_FORWARD);
	
	// solve poisson equation in Fourier space:
	solve_poisson_inplace
	<<<nBlocks,nThreads>>> (rhs,kx,ky,kz,Nx,Ny,Nz);
	
	// inverse FFT (in-place):
	cufftExecC2C(plan, rhs, rhs, CUFFT_INVERSE);
	
	// change solution from complex to real:
	complex2real
	<<<nBlocks,nThreads>>> (rhs,indicator,nVoxels);
}



// --------------------------------------------------------
// write output for the 'indicatorH' array:
// --------------------------------------------------------

void class_poisson_ibm3D::write_output(std::string tagname, int tagnum,
                                       int iskip, int jskip, int kskip, int precision)
{
	// first, do a memcopy from device to host:
	cudaMemcpy(indicatorH, indicator, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	for (int i=0; i<nVoxels; i++) cout << i << " " << indicatorH[i] << endl;
	// second, write the output:
	write_vtk_structured_grid(tagname,tagnum,Nx,Ny,Nz,indicatorH,iskip,jskip,kskip,precision);
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_poisson_ibm3D::deallocate()
{
	cudaFree(kx);
	cudaFree(ky);
	cudaFree(kz);
	cudaFree(rhs);
	cudaFree(G);
	cudaFree(indicator);
	free(indicatorH);
}



