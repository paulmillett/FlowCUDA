
# include "class_scsp_active_D2Q9.cuh"
# include "kernels_scsp_active_D2Q9.cuh"
# include "../../IO/GetPot"
# include <math.h>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <sstream>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_scsp_active_D2Q9::class_scsp_active_D2Q9()
{
	Q = 9;
	GetPot inputParams("input.dat");	
	nVoxels = inputParams("Lattice/nVoxels",0);
	numIolets = inputParams("Lattice/numIolets",0);
	nu = inputParams("LBM/nu",0.1666666);
	sf = inputParams("LBM/sf",1.0);
	fricR = inputParams("LBM/fricR",1.0);
	activity = inputParams("LBM/activity",0.0);
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = inputParams("Lattice/Nz",1);
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_scsp_active_D2Q9::~class_scsp_active_D2Q9()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_scsp_active_D2Q9::allocate()
{
	// allocate array memory (host):
    rH = (float*)malloc(nVoxels*sizeof(float));
	uH = (float2*)malloc(nVoxels*sizeof(float2));
	pH = (float2*)malloc(nVoxels*sizeof(float2));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));	
			
	// allocate array memory (device):
	cudaMalloc((void **) &r, nVoxels*sizeof(float));
	cudaMalloc((void **) &u, nVoxels*sizeof(float2));
	cudaMalloc((void **) &F, nVoxels*sizeof(float2));
	cudaMalloc((void **) &p, nVoxels*sizeof(float2));
	cudaMalloc((void **) &f1, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &stress, nVoxels*sizeof(tensor2D));		
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &nList, nVoxels*Q*sizeof(int));	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_scsp_active_D2Q9::deallocate()
{
	// free array memory (host):
	free(uH);
	free(pH);
	free(rH);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);	
		
	// free array memory (device):
	cudaFree(F);
	cudaFree(u);
	cudaFree(p);
	cudaFree(r);
	cudaFree(f1);
	cudaFree(f2);
	cudaFree(stress);
	cudaFree(voxelType);
	cudaFree(streamIndex);
	cudaFree(nList);
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_scsp_active_D2Q9::memcopy_host_to_device()
{
    cudaMemcpy(r, rH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(u, uH, sizeof(float2)*nVoxels, cudaMemcpyHostToDevice);	
	cudaMemcpy(p, pH, sizeof(float2)*nVoxels, cudaMemcpyHostToDevice);	
	cudaMemcpy(nList, nListH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_scsp_active_D2Q9::memcopy_device_to_host()
{
    cudaMemcpy(rH, r, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(uH, u, sizeof(float2)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(pH, p, sizeof(float2)*nVoxels, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Initialize lattice as a "box" with periodic BC's:
// --------------------------------------------------------

void class_scsp_active_D2Q9::create_lattice_box_periodic()
{
	build_box_lattice_D2Q9(nVoxels,Nx,Ny,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PULL streaming:
// --------------------------------------------------------

void class_scsp_active_D2Q9::stream_index_pull()
{
	stream_index_pull_D2Q9(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Setters for host arrays:
// --------------------------------------------------------

void class_scsp_active_D2Q9::setU(int i, float val)
{
	uH[i].x = val;
}

void class_scsp_active_D2Q9::setV(int i, float val)
{
	uH[i].y = val;
}

void class_scsp_active_D2Q9::setPx(int i, float val)
{
	pH[i].x = val;
}

void class_scsp_active_D2Q9::setPy(int i, float val)
{
	pH[i].y = val;
}

void class_scsp_active_D2Q9::setR(int i, float val)
{
	rH[i] = val;
}

void class_scsp_active_D2Q9::setVoxelType(int i, int val)
{
	voxelTypeH[i] = val;
}



// --------------------------------------------------------
// Getters for host arrays:
// --------------------------------------------------------

float class_scsp_active_D2Q9::getU(int i)
{
	return uH[i].x;
}

float class_scsp_active_D2Q9::getV(int i)
{
	return uH[i].y;
}

float class_scsp_active_D2Q9::getR(int i)
{
	return rH[i];
}



// --------------------------------------------------------
// Call to "scsp_initial_equilibrium_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_D2Q9::initial_equilibrium(int nBlocks, int nThreads)
{
	scsp_active_initial_equilibrium_D2Q9 
	<<<nBlocks,nThreads>>> (f1,r,u,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_zero_forces_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_D2Q9::zero_forces(int nBlocks, int nThreads)
{
	scsp_active_zero_forces_D2Q9 
	<<<nBlocks,nThreads>>> (F,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_D2Q9::stream_collide_save(int nBlocks, int nThreads)
{
	scsp_active_stream_collide_save_D2Q9 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,streamIndex,voxelType,nu,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_forcing_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_D2Q9::stream_collide_save_forcing(int nBlocks, int nThreads)
{
	scsp_active_stream_collide_save_forcing_D2Q9 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,F,streamIndex,voxelType,nu,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "scsp_active_update_orientation_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_D2Q9::scsp_active_update_orientation(int nBlocks, int nThreads)
{
	scsp_active_update_orientation_D2Q9 
	<<<nBlocks,nThreads>>> (u,p,nList,sf,fricR,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_stress_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_D2Q9::scsp_active_fluid_stress(int nBlocks, int nThreads)
{
	scsp_active_fluid_stress_D2Q9 
	<<<nBlocks,nThreads>>> (p,stress,activity,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_forces_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_D2Q9::scsp_active_fluid_forces(int nBlocks, int nThreads)
{
	scsp_active_fluid_forces_D2Q9 
	<<<nBlocks,nThreads>>> (F,stress,nList,nVoxels);
}










// --------------------------------------------------------
// Wrtie output:
// --------------------------------------------------------

void class_scsp_active_D2Q9::write_output(std::string tagname, int step)
{
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << step << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << Nx << d << Ny << d << Nz << endl;
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
	outfile << " " << endl;
	outfile << "POINT_DATA " << Nx*Ny*Nz << endl;
	outfile << "SCALARS " << tagname << " float" << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	
	// -----------------------------------------------
	// Write the 'rho' data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				outfile << fixed << setprecision(5) << rH[ndx] << endl;
			}
		}
	}	
	
	// -----------------------------------------------				
	// Write the 'velocity' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Velocity float" << endl;		
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				outfile << fixed << setprecision(5) << uH[ndx].x << " "
					                                << uH[ndx].y << " " 
													<< 0.0 << endl;
			}
		}
	}
	
	// -----------------------------------------------				
	// Write the 'orientation' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Orientation float" << endl;		
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				outfile << fixed << setprecision(5) << pH[ndx].x << " "
					                                << pH[ndx].y << " " 
													<< 0.0 << endl;
			}
		}
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}








