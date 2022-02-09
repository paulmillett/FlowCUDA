# include "class_scsp_D3Q19.cuh"
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

class_scsp_D3Q19::class_scsp_D3Q19()
{
	Q = 19;
	GetPot inputParams("input.dat");
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = inputParams("Lattice/Nz",1);
	nVoxels = inputParams("Lattice/nVoxels",0);
	numIolets = inputParams("Lattice/numIolets",0);
	nu = inputParams("LBM/nu",0.1666666);
	dt = inputParams("Time/dt",1.0);
	forceFlag = false;
	velIBFlag = false;
	inoutFlag = false;
	solidFlag = false;
	xyzFlag = false;	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_scsp_D3Q19::~class_scsp_D3Q19()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_scsp_D3Q19::allocate()
{
	// allocate array memory (host):
    uH = (float*)malloc(nVoxels*sizeof(float));
    vH = (float*)malloc(nVoxels*sizeof(float));
	wH = (float*)malloc(nVoxels*sizeof(float));
    rH = (float*)malloc(nVoxels*sizeof(float));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));	
	ioletsH = (iolet*)malloc(numIolets*sizeof(iolet));
			
	// allocate array memory (device):
	cudaMalloc((void **) &u, nVoxels*sizeof(float));
	cudaMalloc((void **) &v, nVoxels*sizeof(float));
	cudaMalloc((void **) &w, nVoxels*sizeof(float));
	cudaMalloc((void **) &r, nVoxels*sizeof(float));
	cudaMalloc((void **) &f1, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2, nVoxels*Q*sizeof(float));		
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &iolets, numIolets*sizeof(iolet));	
	
}



// --------------------------------------------------------
// Allocate voxel position arrays:
// --------------------------------------------------------

void class_scsp_D3Q19::allocate_voxel_positions()
{
	// allocate voxel position arrays (host):
	xH = (int*)malloc(nVoxels*sizeof(int));
	yH = (int*)malloc(nVoxels*sizeof(int));
	zH = (int*)malloc(nVoxels*sizeof(int));
	xyzFlag = true;	
}



// --------------------------------------------------------
// Allocate force arrays:
// --------------------------------------------------------

void class_scsp_D3Q19::allocate_forces()
{
	// allocate force arrays (device):
	cudaMalloc((void **) &Fx, nVoxels*sizeof(float));
	cudaMalloc((void **) &Fy, nVoxels*sizeof(float));
	cudaMalloc((void **) &Fz, nVoxels*sizeof(float));
	forceFlag = true;
}



// --------------------------------------------------------
// Allocate IB velocity arrays.  These arrays store IB
// node velocities extrapolated to LB voxels.
// --------------------------------------------------------

void class_scsp_D3Q19::allocate_IB_velocities()
{
	// allocate IB velocity arrays (device):
	cudaMalloc((void **) &uIBvox, nVoxels*sizeof(float));
	cudaMalloc((void **) &vIBvox, nVoxels*sizeof(float));
	cudaMalloc((void **) &wIBvox, nVoxels*sizeof(float));
	cudaMalloc((void **) &weights, nVoxels*sizeof(float));
	velIBFlag = true;
}



// --------------------------------------------------------
// Allocate inoutH[] and inout[] arrays (to declare if
// voxel is inside or outside immersed boundary).
// --------------------------------------------------------

void class_scsp_D3Q19::allocate_inout()
{
	// allocate inout arrays (host & device):
	inoutH = (int*)malloc(nVoxels*sizeof(int));
	cudaMalloc((void **) &inout, nVoxels*sizeof(int));	
	inoutFlag = true;
}



// --------------------------------------------------------
// Allocate solidH[] and solid[] arrays (to declare if
// voxel is fluid or solid).
// --------------------------------------------------------

void class_scsp_D3Q19::allocate_solid()
{
	// allocate inout arrays (host & device):
	solidH = (int*)malloc(nVoxels*sizeof(int));
	cudaMalloc((void **) &solid, nVoxels*sizeof(int));	
	solidFlag = true;
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_scsp_D3Q19::deallocate()
{
	// free array memory (host):
	free(uH);
	free(vH);
	free(wH);
	free(rH);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);	
	free(ioletsH);	
	if (xyzFlag) {
		free(xH);
		free(yH);
		free(zH);
	}
	if (inoutFlag) {
		free(inoutH);
	}
	if (solidFlag) {
		free(solidH);
	}
			
	// free array memory (device):
	cudaFree(u);
	cudaFree(v);
	cudaFree(w);
	cudaFree(r);
	cudaFree(f1);
	cudaFree(f2);	
	cudaFree(voxelType);
	cudaFree(streamIndex);
	cudaFree(iolets);
	if (forceFlag) {
		cudaFree(Fx);
		cudaFree(Fy);
		cudaFree(Fz);
	}
	if (velIBFlag) {
		cudaFree(uIBvox);
		cudaFree(vIBvox);
		cudaFree(wIBvox);
		cudaFree(weights);
	}
	if (inoutFlag) {
		cudaFree(inout);
	}	
	if (solidFlag) {
		cudaFree(solid);
	}	
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_scsp_D3Q19::memcopy_host_to_device()
{
    cudaMemcpy(u, uH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(v, vH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(w, wH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(r, rH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(iolets, ioletsH, sizeof(iolet)*numIolets, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from host to device (just iolets):
// --------------------------------------------------------

void class_scsp_D3Q19::memcopy_host_to_device_iolets()
{
	cudaMemcpy(iolets, ioletsH, sizeof(iolet)*numIolets, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from host to device (just solid array):
// --------------------------------------------------------

void class_scsp_D3Q19::memcopy_host_to_device_solid()
{
    cudaMemcpy(solid, solidH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_scsp_D3Q19::memcopy_device_to_host()
{
    cudaMemcpy(rH, r, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(wH, w, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Copy arrays from device to host (just inout array):
// --------------------------------------------------------

void class_scsp_D3Q19::memcopy_device_to_host_inout()
{
    cudaMemcpy(inoutH, inout, sizeof(int)*nVoxels, cudaMemcpyDeviceToHost);
}










// **********************************************************************************************
// Initialization Stuff...
// **********************************************************************************************










// --------------------------------------------------------
// Initialize lattice as a "box":
// --------------------------------------------------------

void class_scsp_D3Q19::create_lattice_box()
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

void class_scsp_D3Q19::create_lattice_box_periodic()
{
	build_box_lattice_D3Q19(nVoxels,Nx,Ny,Nz,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice as a "box" set up for shear flow
// in the x-direction:
// --------------------------------------------------------

void class_scsp_D3Q19::create_lattice_box_shear()
{
	build_box_lattice_shear_D3Q19(nVoxels,Nx,Ny,Nz,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice as a "box" set up for channel flow
// in the x-direction:
// --------------------------------------------------------

void class_scsp_D3Q19::create_lattice_box_channel()
{
	build_box_lattice_channel_D3Q19(nVoxels,Nx,Ny,Nz,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice as a "box" with periodic BC's, and
// internal solid walls:
// --------------------------------------------------------

void class_scsp_D3Q19::create_lattice_box_periodic_solid_walls()
{
	build_box_lattice_solid_walls_D3Q19(nVoxels,Nx,Ny,Nz,voxelTypeH,solidH,nListH);
}



// --------------------------------------------------------
// Construct nList[] using bounding box:
// --------------------------------------------------------

void class_scsp_D3Q19::bounding_box_nList_construct()
{
	bounding_box_nList_construct_D3Q19(nVoxels,xH,yH,zH,nListH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PUSH streaming:
// --------------------------------------------------------

void class_scsp_D3Q19::stream_index_push()
{
	// still need to add this...
}



// --------------------------------------------------------
// Build the streamIndex[] array for PULL streaming:
// --------------------------------------------------------

void class_scsp_D3Q19::stream_index_pull()
{
	stream_index_pull_D3Q19(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Read information about iolet:
// --------------------------------------------------------

void class_scsp_D3Q19::read_iolet_info(int i, const char* name) 
{
	char namemod[20];
	GetPot inputParams("input.dat");
	if (i >= 0 and i < numIolets) {
		strcpy(namemod, name);
		strcat(namemod, "/type");
		ioletsH[i].type = inputParams(namemod,1);
		strcpy(namemod, name);
		strcat(namemod, "/uBC");
		ioletsH[i].uBC  = inputParams(namemod,0.0);
		strcpy(namemod, name);
		strcat(namemod, "/vBC");
		ioletsH[i].vBC  = inputParams(namemod,0.0);
		strcpy(namemod, name);
		strcat(namemod, "/wBC");
		ioletsH[i].wBC  = inputParams(namemod,0.0);
		strcpy(namemod, name);
		strcat(namemod, "/rBC");
		ioletsH[i].rBC  = inputParams(namemod,1.0);
		strcpy(namemod, name);
		strcat(namemod, "/pBC");
		ioletsH[i].pBC  = inputParams(namemod,0.0);
	}
	else {
		cout << "iolet index is not correct" << endl;
	}
}



// --------------------------------------------------------
// Setters for host arrays:
// --------------------------------------------------------

void class_scsp_D3Q19::setNu(float val)
{
	nu = val;
}

void class_scsp_D3Q19::setU(int i, float val)
{
	uH[i] = val;
}

void class_scsp_D3Q19::setV(int i, float val)
{
	vH[i] = val;
}

void class_scsp_D3Q19::setW(int i, float val)
{
	wH[i] = val;
}

void class_scsp_D3Q19::setR(int i, float val)
{
	rH[i] = val;
}

void class_scsp_D3Q19::setS(int i, int val)
{
	solidH[i] = val;
}

void class_scsp_D3Q19::setVoxelType(int i, int val)
{
	voxelTypeH[i] = val;
}

void class_scsp_D3Q19::setIoletU(int i, float val)
{
	ioletsH[i].uBC = val;
}

void class_scsp_D3Q19::setIoletV(int i, float val)
{
	ioletsH[i].vBC = val;
}

void class_scsp_D3Q19::setIoletW(int i, float val)
{
	ioletsH[i].wBC = val;
}

void class_scsp_D3Q19::setIoletR(int i, float val)
{
	ioletsH[i].rBC = val;
}

void class_scsp_D3Q19::setIoletType(int i, int val)
{
	ioletsH[i].type = val;
}



// --------------------------------------------------------
// Getters for host arrays:
// --------------------------------------------------------

float class_scsp_D3Q19::getU(int i)
{
	return uH[i];
}

float class_scsp_D3Q19::getV(int i)
{
	return vH[i];
}

float class_scsp_D3Q19::getW(int i)
{
	return wH[i];
}

float class_scsp_D3Q19::getR(int i)
{
	return rH[i];
}

int class_scsp_D3Q19::getX(int i)
{
	return xH[i];
}

int class_scsp_D3Q19::getY(int i)
{
	return yH[i];
}

int class_scsp_D3Q19::getZ(int i)
{
	return zH[i];
}

int class_scsp_D3Q19::getNList(int i)
{
	return nListH[i];
}










// **********************************************************************************************
// Calls to CUDA kernels for main calculations
// **********************************************************************************************










// --------------------------------------------------------
// Call to "scsp_initial_equilibrium_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_D3Q19::initial_equilibrium(int nBlocks, int nThreads)
{
	scsp_initial_equilibrium_D3Q19 
	<<<nBlocks,nThreads>>> (f1,r,u,v,w,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_D3Q19::stream_collide_save(int nBlocks, int nThreads, bool save)
{
	scsp_stream_collide_save_D3Q19 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,v,w,streamIndex,voxelType,iolets,nu,nVoxels,save);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_forcing_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_D3Q19::stream_collide_save_forcing(int nBlocks, int nThreads)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	scsp_stream_collide_save_forcing_D3Q19 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,v,w,Fx,Fy,Fz,streamIndex,voxelType,iolets,nu,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_forcing_dt_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_D3Q19::stream_collide_save_forcing_solid(int nBlocks, int nThreads)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	if (!solidFlag) cout << "Warning: LBM solid arrays have not been initialized" << endl;
	scsp_stream_collide_save_forcing_solid_D3Q19 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,v,w,Fx,Fy,Fz,streamIndex,voxelType,solid,iolets,nu,dt,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_IBforcing_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_D3Q19::stream_collide_save_IBforcing(int nBlocks, int nThreads)
{
	if (!velIBFlag) cout << "Warning: IB velocity arrays have not been initialized" << endl;
	scsp_stream_collide_save_IBforcing_D3Q19 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,v,w,uIBvox,vIBvox,wIBvox,weights,streamIndex,voxelType,iolets,nu,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;	
}



// --------------------------------------------------------
// Call to "set_boundary_shear_velocity_D3Q19" kernel:
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

void class_scsp_D3Q19::set_boundary_shear_velocity(float uBot, float uTop, int nBlocks, int nThreads)
{
	scsp_set_boundary_shear_velocity_D3Q19 
	<<<nBlocks,nThreads>>> (uBot,uTop,f1,u,v,w,r,Nx,Ny,Nz,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_set_channel_wall_velocity_D3Q19" kernel:
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

void class_scsp_D3Q19::set_channel_wall_velocity(float uWall, int nBlocks, int nThreads)
{
	scsp_set_channel_wall_velocity_D3Q19 
	<<<nBlocks,nThreads>>> (uWall,f1,u,v,w,r,Nx,Ny,Nz,nVoxels);
}



// --------------------------------------------------------
// Call to "extrapolate_velocity_IBM3D" kernel.  
// Note: this kernel is in the IBM/3D folder, and one
//       should use nBlocks as if calling an IBM kernel.
// --------------------------------------------------------

void class_scsp_D3Q19::extrapolate_velocity_from_IBM(int nBlocks, int nThreads,
	                                                 float3* rIB, float3* vIB, int nNodes)
{
	if (!velIBFlag) cout << "Warning: IB velocity arrays have not been initialized" << endl;
	extrapolate_velocity_IBM3D
	<<<nBlocks,nThreads>>> (rIB,vIB,uIBvox,vIBvox,wIBvox,weights,Nx,Ny,nNodes);
}



// --------------------------------------------------------
// Call to "interpolate_velocity_IBM3D" kernel.  
// Note: this kernel is in the IBM/3D folder, and one
//       should use nBlocks as if calling an IBM kernel.
// --------------------------------------------------------

void class_scsp_D3Q19::interpolate_velocity_to_IBM(int nBlocks, int nThreads,
	                                               float3* rIB, float3* vIB, int nNodes)
{
	interpolate_velocity_IBM3D
	<<<nBlocks,nThreads>>> (rIB,vIB,u,v,w,Nx,Ny,Nz,nNodes);
}



// --------------------------------------------------------
// Call to "extrapolate_force_IBM3D" kernel.  
// Note: this kernel is in the IBM/3D folder, and one
//       should use nBlocks as if calling an IBM kernel.
// --------------------------------------------------------

void class_scsp_D3Q19::extrapolate_forces_from_IBM(int nBlocks, int nThreads,
	                                               float3* rIB, float3* fIB, int nNodes)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	extrapolate_force_IBM3D
	<<<nBlocks,nThreads>>> (rIB,fIB,Fx,Fy,Fz,Nx,Ny,Nz,nNodes);	
}



// --------------------------------------------------------
// Call to "viscous_force_velocity_difference_IBM3D" kernel.  
// Note: this kernel is in the IBM/3D folder, and one
//       should use nBlocks as if calling an IBM kernel.
// --------------------------------------------------------

void class_scsp_D3Q19::viscous_force_IBM_LBM(int nBlocks, int nThreads, float gam,
	                                         float3* rIB, float3* vIB, float3* fIB, int nNodes)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	viscous_force_velocity_difference_IBM3D
	<<<nBlocks,nThreads>>> (rIB,vIB,fIB,Fx,Fy,Fz,u,v,w,gam,Nx,Ny,Nz,nNodes);	
}



// --------------------------------------------------------
// Call to "scsp_zero_forces_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_D3Q19::zero_forces(int nBlocks, int nThreads)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	scsp_zero_forces_D3Q19 
	<<<nBlocks,nThreads>>> (Fx,Fy,Fz,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_zero_forces_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_D3Q19::zero_forces_with_IBM(int nBlocks, int nThreads)
{
	if (!velIBFlag) cout << "Warning: IB velocity arrays have not been initialized" << endl;
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	scsp_zero_forces_D3Q19
	<<<nBlocks,nThreads>>> (Fx,Fy,Fz,uIBvox,vIBvox,wIBvox,weights,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_add_body_forces_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_D3Q19::add_body_force(float bx, float by, float bz, int nBlocks, int nThreads)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	scsp_add_body_force_D3Q19 
	<<<nBlocks,nThreads>>> (bx,by,bz,Fx,Fy,Fz,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_add_body_forces_D3Q19" kernel:
// --------------------------------------------------------

void class_scsp_D3Q19::add_body_force_with_solid(float bx, float by, float bz, int nBlocks, int nThreads)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	if (!solidFlag) cout << "Warning: LBM solid arrays have not been initialized" << endl;
	scsp_add_body_force_solid_D3Q19 
	<<<nBlocks,nThreads>>> (bx,by,bz,Fx,Fy,Fz,solid,nVoxels);
}



// --------------------------------------------------------
// Call to "inside_hemisphere" kernel:
// --------------------------------------------------------

void class_scsp_D3Q19::inside_hemisphere(int nBlocks, int nThreads)
{
	if (!velIBFlag) cout << "Warning: IB velocity arrays have not been initialized" << endl;
	inside_hemisphere_D3Q19
	<<<nBlocks,nThreads>>> (weights,inout,Nx,Ny,Nz,nVoxels);
}










// **********************************************************************************************
// Analysis of flow field done by the host (CPU)
// **********************************************************************************************










void class_scsp_D3Q19::calculate_flow_rate_xdir(std::string tagname, int tagnum)
{
	
	// define the file location and name:
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".dat";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// create variables for this calculation:
	float* velx = (float*)malloc(Ny*Nz*sizeof(float));
	for (int i=0; i<Ny*Nz; i++) velx[i] = 0.0;
	
	// loop over grid and calculate average x-vel for each voxel
	// on the yz-plane:
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			float vxsum = 0.0;
			for (int i=0; i<Nx; i++) {		
				vxsum += uH[k*Nx*Ny + j*Nx + i];
			}
			velx[k*Ny + j] = vxsum/float(Nx);
		}
	}
	
	// calculate overall x-dir. volumetric flow rate:
	float Qx = 0.0;
	for (int i=0; i<Ny*Nz; i++) Qx += velx[i];  // assume dx=dy=dz=1
	
	// print results:
	outfile << Ny << " " << Nz << endl;
	//outfile << fixed << setprecision(4) << Qx << endl;
	for (int i=0; i<Ny*Nz; i++) {
		outfile << fixed << setprecision(4) << velx[i] << endl;
	}
	
}










// **********************************************************************************************
// Input/output calls
// **********************************************************************************************










// --------------------------------------------------------
// Read lattice geometry:
// --------------------------------------------------------

void class_scsp_D3Q19::read_lattice_geometry(int type)
{
	// read voxel position, voxel type, and voxel nList:
	if (type == 1) {
		read_lattice_geometry_D3Q19(nVoxels,xH,yH,zH,voxelTypeH,nListH);
	}
	// read voxel position, and voxel type:
	else if (type == 2) {
		read_lattice_geometry_D3Q19(nVoxels,xH,yH,zH,voxelTypeH);
	}
	// read voxel position:
	else if (type == 3) {
		read_lattice_geometry_D3Q19(nVoxels,xH,yH,zH);
	}
}



// --------------------------------------------------------
// Write VTK output: structured with u[], v[], w[], r[]:
// --------------------------------------------------------

void class_scsp_D3Q19::vtk_structured_output_ruvw(std::string tagname, int tagnum,
                                            int iskip, int jskip, int kskip)
{
	write_vtk_structured_grid(tagname,tagnum,Nx,Ny,Nz,rH,uH,vH,wH,iskip,jskip,kskip);
}



// --------------------------------------------------------
// Write VTK output: structured with u[], v[], w[], int[]:
// --------------------------------------------------------

void class_scsp_D3Q19::vtk_structured_output_iuvw_inout(std::string tagname, int tagnum,
                                                  int iskip, int jskip, int kskip)
{
	if (!inoutFlag) cout << "Warning: inout arrays have not been initialized" << endl;
	write_vtk_structured_grid(tagname,tagnum,Nx,Ny,Nz,inoutH,uH,vH,wH,iskip,jskip,kskip);
}



// --------------------------------------------------------
// Write VTK output: structured with u[], v[], w[], int[]:
// --------------------------------------------------------

void class_scsp_D3Q19::vtk_structured_output_iuvw_vtype(std::string tagname, int tagnum,
                                                  int iskip, int jskip, int kskip)
{
	write_vtk_structured_grid(tagname,tagnum,Nx,Ny,Nz,voxelTypeH,uH,vH,wH,iskip,jskip,kskip);
}



// --------------------------------------------------------
// Write VTK output: polydata with u[], v[], w[], r[]:
// --------------------------------------------------------

void class_scsp_D3Q19::vtk_polydata_output_ruvw(std::string tagname, int tagnum)
{
	write_vtk_polydata(tagname,tagnum,nVoxels,xH,yH,zH,rH,uH,vH,wH);
}







