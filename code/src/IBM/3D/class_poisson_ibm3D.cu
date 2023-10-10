
# include "class_poisson_ibm3D.cuh"
# include <math.h>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <string>
# include <sstream>
# include <stdlib.h>
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
	
	// define cuFFT plan (here, the first dimension must be the slowest changing one i.e. Nz)
	cufftPlan3d(&plan, Nz, Ny, Nx, CUFFT_C2C);	
	
	// wave-vector arrays (host)
	float* kxH = new float[Nx];
	float* kyH = new float[Ny];
	float* kzH = new float[Nz];
	for (int i=0; i<=Nx/2; i++)   kxH[i] = float(i)*2*M_PI/float(Nx);
	for (int i=Nx/2+1; i<Nx; i++) kxH[i] = float(i-Nx)*2*M_PI/float(Nx);
	for (int j=0; j<=Ny/2; j++)   kyH[j] = float(j)*2*M_PI/float(Ny);
	for (int j=Ny/2+1; j<Ny; j++) kyH[j] = float(j-Ny)*2*M_PI/float(Ny);	
	for (int k=0; k<=Nz/2; k++)   kzH[k] = float(k)*2*M_PI/float(Nz);
	for (int k=Nz/2+1; k<Nz; k++) kzH[k] = float(k-Nz)*2*M_PI/float(Nz);
	
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

void class_poisson_ibm3D::solve_poisson(triangle* faces, node* nodes, cell* cells, int nFaces, int cellType, int nBlocks, int nThreads)
{
	// zero the 'G' vector array:
	zero_G_poisson_IBM3D
	<<<nBlocks,nThreads>>> (G,nVoxels);	
		
	// extrapolate IBM interface normal vectors to fluid grid:
	extrapolate_interface_normal_poisson_IBM3D
	<<<nBlocks,nThreads>>> (nodes,G,Nx,Ny,Nz,nFaces,cellType,cells,faces);	
		
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
	
	// rescale indicator function:
	rescale_indicator_array
	<<<nBlocks,nThreads>>> (indicator,nVoxels);
	
}



// --------------------------------------------------------
// write output for the 'indicatorH' array:
// --------------------------------------------------------

void class_poisson_ibm3D::write_output(std::string tagname, int tagnum,
                                       int iskip, int jskip, int kskip, int precision)
{
	cudaMemcpy(indicatorH, indicator, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	write_vtk_structured_grid(tagname,tagnum,Nx,Ny,Nz,indicatorH,iskip,jskip,kskip,precision);
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_poisson_ibm3D::deallocate()
{
	cufftDestroy(plan);
	cudaFree(kx);
	cudaFree(ky);
	cudaFree(kz);
	cudaFree(rhs);
	cudaFree(G);
	cudaFree(indicator);
	free(indicatorH);
}



// --------------------------------------------------------
// analyze volume fraction data using 'indicatorH' array:
// --------------------------------------------------------

void class_poisson_ibm3D::volume_fraction_analysis(std::string tagname, float threshold)
{
	// device-to-host memcopy:
	cudaMemcpy(indicatorH, indicator, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	
	// define the file location and name (flowrate thru time):
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << ".dat";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// create array for the current phi average in yz-plane:
	float* phi = (float*)malloc(Ny*Nz*sizeof(float));
	for (int i=0; i<Ny*Nz; i++) phi[i] = 0.0;
	
	// loop over grid and calculate average phi for each voxel
	// on the yz-plane (for current time):
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			float volTot = 0.0;
			for (int i=0; i<Nx; i++) {	
				int ndx = k*Nx*Ny + j*Nx + i;
				// 0.4 is found to be a good cut-off for RBC's
				if (indicatorH[ndx]>threshold) volTot += indicatorH[ndx];
			}
			phi[k*Ny + j] = volTot/float(Nx);
		}
	}
	
	// print phi array:
	for (int i=0; i<Ny*Nz; i++) {
		outfile << fixed << setprecision(5) << phi[i] << endl; 
	}
	
}




