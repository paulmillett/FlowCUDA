# include "class_scsp_psm_D3Q19.cuh"
# include "../../IO/GetPot"
# include <math.h>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <string>
# include <sstream>
# include <stdlib.h>
using namespace std;  









// **********************************************************************************************
// Constructor, destructor, and array allocations...
// **********************************************************************************************










// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_scsp_psm_D3Q19::class_scsp_psm_D3Q19()
{
	Q = 19;
	GetPot inputParams("input.dat");
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = inputParams("Lattice/Nz",1);
	nVoxels = inputParams("Lattice/nVoxels",0);
	nSpheres = inputParams("Particles/nSpheres",0);
	nu = inputParams("LBM/nu",0.1666666);
	dt = inputParams("Time/dt",1.0);
	forceFlag = false;
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_scsp_psm_D3Q19::~class_scsp_psm_D3Q19()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::allocate()
{
	// allocate array memory (host):
    uH = (float*)malloc(nVoxels*sizeof(float));
    vH = (float*)malloc(nVoxels*sizeof(float));
	wH = (float*)malloc(nVoxels*sizeof(float));
    rH = (float*)malloc(nVoxels*sizeof(float));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));
	spheresH = (sphere*)malloc(nSpheres*sizeof(sphere));
			
	// allocate array memory (device):
	cudaMalloc((void **) &u, nVoxels*sizeof(float));
	cudaMalloc((void **) &v, nVoxels*sizeof(float));
	cudaMalloc((void **) &w, nVoxels*sizeof(float));
	cudaMalloc((void **) &r, nVoxels*sizeof(float));
	cudaMalloc((void **) &f1, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2, nVoxels*Q*sizeof(float));		
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &spheres, nSpheres*sizeof(sphere));
	cudaMalloc((void **) &eps, nVoxels*sizeof(float));
	cudaMalloc((void **) &pID, nVoxels*sizeof(int));	
	
}



// --------------------------------------------------------
// Allocate force arrays:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::allocate_forces()
{
	// allocate force arrays (device):
	cudaMalloc((void **) &Fx, nVoxels*sizeof(float));
	cudaMalloc((void **) &Fy, nVoxels*sizeof(float));
	cudaMalloc((void **) &Fz, nVoxels*sizeof(float));
	forceFlag = true;
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::deallocate()
{
	// free array memory (host):
	free(uH);
	free(vH);
	free(wH);
	free(rH);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);	
	free(spheresH);
			
	// free array memory (device):
	cudaFree(u);
	cudaFree(v);
	cudaFree(w);
	cudaFree(r);
	cudaFree(f1);
	cudaFree(f2);	
	cudaFree(streamIndex);
	cudaFree(spheres);
	cudaFree(eps);
	cudaFree(pID);
	if (forceFlag) {
		cudaFree(Fx);
		cudaFree(Fy);
		cudaFree(Fz);
	}
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::memcopy_host_to_device()
{
    cudaMemcpy(u, uH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(v, vH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(w, wH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(r, rH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(spheres, spheresH, sizeof(sphere)*nSpheres, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::memcopy_device_to_host()
{
    cudaMemcpy(rH, r, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(wH, w, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(spheresH, spheres, sizeof(sphere)*nSpheres, cudaMemcpyDeviceToHost);
}











// **********************************************************************************************
// Initialization Stuff...
// **********************************************************************************************










// --------------------------------------------------------
// Initialize lattice as a "box":
// --------------------------------------------------------

void class_scsp_psm_D3Q19::create_lattice_box()
{
	GetPot inputParams("input.dat");		
	int flowDir = inputParams("Lattice/flowDir",0);
	int xLBC = inputParams("Lattice/xLBC",0);
	int xUBC = inputParams("Lattice/xUBC",0);
	int yLBC = inputParams("Lattice/yLBC",0);
	int yUBC = inputParams("Lattice/yUBC",0);
	int zLBC = inputParams("Lattice/zLBC",0);
	int zUBC = inputParams("Lattice/zUBC",0);		
	build_box_lattice_D3Q19(nVoxels,flowDir,Nx,Ny,Nz,
	                        xLBC,xUBC,yLBC,yUBC,zLBC,zUBC,
							voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice as a "box" with periodic BC's:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::create_lattice_box_periodic()
{
	build_box_lattice_D3Q19(nVoxels,Nx,Ny,Nz,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice as a "box" set up for shear flow
// in the x-direction:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::create_lattice_box_shear()
{
	build_box_lattice_shear_D3Q19(nVoxels,Nx,Ny,Nz,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice as a "box" set up for slit flow
// in the x-direction (plane poiseuille):
// --------------------------------------------------------

void class_scsp_psm_D3Q19::create_lattice_box_slit()
{
	build_box_lattice_slit_D3Q19(nVoxels,Nx,Ny,Nz,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice as a "box" set up for channel flow
// in the x-direction:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::create_lattice_box_channel()
{
	build_box_lattice_channel_D3Q19(nVoxels,Nx,Ny,Nz,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PULL streaming:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::stream_index_pull()
{
	stream_index_pull_D3Q19(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Setters for host arrays:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::setNu(float val)
{
	nu = val;
}

void class_scsp_psm_D3Q19::setU(int i, float val)
{
	uH[i] = val;
}

void class_scsp_psm_D3Q19::setV(int i, float val)
{
	vH[i] = val;
}

void class_scsp_psm_D3Q19::setW(int i, float val)
{
	wH[i] = val;
}

void class_scsp_psm_D3Q19::setR(int i, float val)
{
	rH[i] = val;
}



// --------------------------------------------------------
// Getters for host arrays:
// --------------------------------------------------------

float class_scsp_psm_D3Q19::getU(int i)
{
	return uH[i];
}

float class_scsp_psm_D3Q19::getV(int i)
{
	return vH[i];
}

float class_scsp_psm_D3Q19::getW(int i)
{
	return wH[i];
}

float class_scsp_psm_D3Q19::getR(int i)
{
	return rH[i];
}











// **********************************************************************************************
// Calls to CUDA kernels for main calculations
// **********************************************************************************************










// --------------------------------------------------------
// Call to "scsp_initial_equilibrium_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::initial_equilibrium(int nBlocks, int nThreads)
{
	scsp_psm_initial_equilibrium_D3Q19 
	<<<nBlocks,nThreads>>> (f1,r,u,v,w,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::stream_collide_save(int nBlocks, int nThreads)
{
	scsp_psm_stream_collide_save_D3Q19 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,v,w,eps,pID,streamIndex,spheres,nu,Nx,Ny,Nz,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_forcing_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::stream_collide_save_forcing(int nBlocks, int nThreads)
{
	/*
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	scsp_stream_collide_save_forcing_D3Q19 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,v,w,Fx,Fy,Fz,streamIndex,voxelType,iolets,nu,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
	*/
}



// --------------------------------------------------------
// Call to "set_boundary_shear_velocity_D3Q19" kernel:
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

void class_scsp_psm_D3Q19::set_boundary_shear_velocity(float uBot, float uTop, int nBlocks, int nThreads)
{
	scsp_psm_set_boundary_shear_velocity_D3Q19 
	<<<nBlocks,nThreads>>> (uBot,uTop,f1,u,v,w,r,Nx,Ny,Nz,nVoxels);
}



/*


// --------------------------------------------------------
// Call to "set_boundary_slit_velocity_D3Q19" kernel:
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

void class_scsp_psm_D3Q19::set_boundary_slit_velocity(float uWall, int nBlocks, int nThreads)
{
	scsp_set_boundary_slit_velocity_D3Q19 
	<<<nBlocks,nThreads>>> (uWall,f1,u,v,w,r,Nx,Ny,Nz,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_set_channel_wall_velocity_D3Q19" kernel:
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

void class_scsp_psm_D3Q19::set_channel_wall_velocity(float uWall, int nBlocks, int nThreads)
{
	scsp_set_channel_wall_velocity_D3Q19 
	<<<nBlocks,nThreads>>> (uWall,f1,u,v,w,r,Nx,Ny,Nz,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_set_boundary_slit_density_D3Q19" kernel:
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

void class_scsp_psm_D3Q19::set_boundary_slit_density(int nBlocks, int nThreads)
{
	scsp_set_boundary_slit_density_D3Q19 
	<<<nBlocks,nThreads>>> (f1,Nx,Ny,Nz,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_set_boundary_duct_density_D3Q19" kernel:
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

void class_scsp_psm_D3Q19::set_boundary_duct_density(int nBlocks, int nThreads)
{
	scsp_set_boundary_duct_density_D3Q19 
	<<<nBlocks,nThreads>>> (f1,Nx,Ny,Nz,nVoxels);
}


*/



// --------------------------------------------------------
// Call to "scsp_zero_forces_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::zero_forces(int nBlocks, int nThreads)
{
	/*
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	scsp_zero_forces_D3Q19 
	<<<nBlocks,nThreads>>> (Fx,Fy,Fz,nVoxels);
	*/
}



// --------------------------------------------------------
// Call to "scsp_add_body_forces_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::add_body_force(float bx, float by, float bz, int nBlocks, int nThreads)
{
	/*
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	scsp_add_body_force_D3Q19 
	<<<nBlocks,nThreads>>> (bx,by,bz,Fx,Fy,Fz,nVoxels);
	*/
}



// --------------------------------------------------------
// Call to "scsp_add_body_forces_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::add_body_force_kolmogorov(float Fo, int nBlocks, int nThreads)
{
	/*
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	scsp_add_body_force_kolmogorov_D3Q19 
	<<<nBlocks,nThreads>>> (Fo,Fx,Fy,Fz,nVoxels,Nx,Ny,Nz);
	*/
}











// **********************************************************************************************
// Input/output calls
// **********************************************************************************************













// --------------------------------------------------------
// Write VTK output: structured with u[], v[], w[], r[]:
// --------------------------------------------------------

void class_scsp_psm_D3Q19::vtk_structured_output_ruvw(std::string tagname, int tagnum,
                                            int iskip, int jskip, int kskip, int precision)
{
	write_vtk_structured_grid(tagname,tagnum,Nx,Ny,Nz,rH,uH,vH,wH,iskip,jskip,kskip,precision);
}





