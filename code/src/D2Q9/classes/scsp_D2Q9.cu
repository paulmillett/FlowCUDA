
# include "scsp_D2Q9.cuh"
# include "scsp_D2Q9_includes.cuh"
# include "../../IO/GetPot"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_D2Q9::scsp_D2Q9()
{
	Q = 9;
	GetPot inputParams("input.dat");	
	nVoxels = inputParams("Lattice/nVoxels",0);
	numIolets = inputParams("Lattice/numIolets",0);
	nu = inputParams("LBM/nu",0.1666666);
	forceFlag = false;
	velIBFlag = false;
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_D2Q9::~scsp_D2Q9()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void scsp_D2Q9::allocate()
{
	// allocate array memory (host):
    uH = (float*)malloc(nVoxels*sizeof(float));
    vH = (float*)malloc(nVoxels*sizeof(float));
    rH = (float*)malloc(nVoxels*sizeof(float));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));	
	ioletsH = (iolet2D*)malloc(numIolets*sizeof(iolet2D));
			
	// allocate array memory (device):
	cudaMalloc((void **) &u, nVoxels*sizeof(float));
	cudaMalloc((void **) &v, nVoxels*sizeof(float));
	cudaMalloc((void **) &r, nVoxels*sizeof(float));
	cudaMalloc((void **) &f1, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2, nVoxels*Q*sizeof(float));		
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &iolets, numIolets*sizeof(iolet2D));	
}



// --------------------------------------------------------
// Allocate force arrays:
// --------------------------------------------------------

void scsp_D2Q9::allocate_forces()
{
	// allocate force arrays (device):
	cudaMalloc((void **) &Fx, nVoxels*sizeof(float));
	cudaMalloc((void **) &Fy, nVoxels*sizeof(float));
	forceFlag = true;
}



// --------------------------------------------------------
// Allocate IB velocity arrays.  These arrays store IB
// node velocities extrapolated to LB voxels.
// --------------------------------------------------------

void scsp_D2Q9::allocate_IB_velocities()
{
	// allocate force arrays (device):
	cudaMalloc((void **) &uIBvox, nVoxels*sizeof(float));
	cudaMalloc((void **) &vIBvox, nVoxels*sizeof(float));
	cudaMalloc((void **) &weights, nVoxels*sizeof(float));
	velIBFlag = true;
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void scsp_D2Q9::deallocate()
{
	// free array memory (host):
	free(uH);
	free(vH);
	free(rH);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);	
	free(ioletsH);	
		
	// free array memory (device):
	cudaFree(u);
	cudaFree(v);
	cudaFree(r);
	cudaFree(f1);
	cudaFree(f2);	
	cudaFree(voxelType);
	cudaFree(streamIndex);
	cudaFree(iolets);
	if (forceFlag) {
		cudaFree(Fx);
		cudaFree(Fy);
	}
	if (velIBFlag) {
		cudaFree(uIBvox);
		cudaFree(vIBvox);
		cudaFree(weights);
	}	
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void scsp_D2Q9::memcopy_host_to_device()
{
    cudaMemcpy(u, uH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(v, vH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(r, rH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(iolets, ioletsH, sizeof(iolet2D)*numIolets, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from host to device (just iolets):
// --------------------------------------------------------

void scsp_D2Q9::memcopy_host_to_device_iolets()
{
	cudaMemcpy(iolets, ioletsH, sizeof(iolet2D)*numIolets, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void scsp_D2Q9::memcopy_device_to_host()
{
    cudaMemcpy(rH, r, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Initialize lattice as a "box":
// --------------------------------------------------------

void scsp_D2Q9::create_lattice_box()
{
	GetPot inputParams("input.dat");
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = 1;
	int flowDir = inputParams("Lattice/flowDir",0);
	int xLBC = inputParams("Lattice/xLBC",0);
	int xUBC = inputParams("Lattice/xUBC",0);
	int yLBC = inputParams("Lattice/yLBC",0);
	int yUBC = inputParams("Lattice/yUBC",0);			
	build_box_lattice_D2Q9(nVoxels,flowDir,Nx,Ny,
	                       xLBC,xUBC,yLBC,yUBC,
	                       voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice as a "box" with periodic BC's:
// --------------------------------------------------------

void scsp_D2Q9::create_lattice_box_periodic()
{
	GetPot inputParams("input.dat");
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = inputParams("Lattice/Nz",1);
	build_box_lattice_D2Q9(nVoxels,Nx,Ny,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice from "file":
// --------------------------------------------------------

void scsp_D2Q9::create_lattice_file()
{
	
}



// --------------------------------------------------------
// Build the streamIndex[] array for PUSH streaming:
// --------------------------------------------------------

void scsp_D2Q9::stream_index_push()
{
	stream_index_push_D2Q9(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PULL streaming:
// --------------------------------------------------------

void scsp_D2Q9::stream_index_pull()
{
	stream_index_pull_D2Q9(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Read information about iolet:
// --------------------------------------------------------

void scsp_D2Q9::read_iolet_info(int i, const char* name) 
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

void scsp_D2Q9::setU(int i, float val)
{
	uH[i] = val;
}

void scsp_D2Q9::setV(int i, float val)
{
	vH[i] = val;
}

void scsp_D2Q9::setR(int i, float val)
{
	rH[i] = val;
}

void scsp_D2Q9::setVoxelType(int i, int val)
{
	voxelTypeH[i] = val;
}



// --------------------------------------------------------
// Getters for host arrays:
// --------------------------------------------------------

float scsp_D2Q9::getU(int i)
{
	return uH[i];
}

float scsp_D2Q9::getV(int i)
{
	return vH[i];
}

float scsp_D2Q9::getR(int i)
{
	return rH[i];
}



// --------------------------------------------------------
// Call to "scsp_initial_equilibrium_D2Q9" kernel:
// --------------------------------------------------------

void scsp_D2Q9::initial_equilibrium(int nBlocks, int nThreads)
{
	scsp_initial_equilibrium_D2Q9 
	<<<nBlocks,nThreads>>> (f1,r,u,v,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_D2Q9" kernel:
// --------------------------------------------------------

void scsp_D2Q9::stream_collide_save(int nBlocks, int nThreads, bool save)
{
	scsp_stream_collide_save_D2Q9 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,v,streamIndex,voxelType,iolets,nu,nVoxels,save);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_forcing_D2Q9" kernel:
// --------------------------------------------------------

void scsp_D2Q9::stream_collide_save_forcing(int nBlocks, int nThreads)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	scsp_stream_collide_save_forcing_D2Q9 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,v,Fx,Fy,streamIndex,voxelType,iolets,nu,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_forcing_D2Q9" kernel:
// --------------------------------------------------------

void scsp_D2Q9::stream_collide_save_IBforcing(int nBlocks, int nThreads)
{
	if (!velIBFlag) cout << "Warning: IB velocity arrays have not been initialized" << endl;
	scsp_stream_collide_save_IBforcing_D2Q9 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,v,uIBvox,vIBvox,weights,streamIndex,voxelType,iolets,nu,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "extrapolate_velocity_IBM2D" kernel.  
// Note: this kernel is in the IBM/2D folder, and one
//       should use nBlocks as if calling an IBM kernel.
// --------------------------------------------------------

void scsp_D2Q9::extrapolate_velocity_from_IBM(int nBlocks, int nThreads,
	                                          float* xIB, float* yIB, float* vxIB,
											  float* vyIB, int nNodes)
{
	if (!velIBFlag) cout << "Warning: IB velocity arrays have not been initialized" << endl;
	extrapolate_velocity_IBM2D
	<<<nBlocks,nThreads>>> (xIB,yIB,vxIB,vyIB,uIBvox,vIBvox,weights,Nx,nNodes);
}



// --------------------------------------------------------
// Call to "extrapolate_force_IBM2D" kernel.  
// Note: this kernel is in the IBM/2D folder, and one
//       should use nBlocks as if calling an IBM kernel.
// --------------------------------------------------------

void scsp_D2Q9::extrapolate_forces_from_IBM(int nBlocks, int nThreads,
	                                        float* xIB, float* yIB, float* fxIB,
											float* fyIB, int nNodes)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	extrapolate_force_IBM2D
	<<<nBlocks,nThreads>>> (xIB,yIB,fxIB,fyIB,Fx,Fy,Nx,nNodes);	
}



// --------------------------------------------------------
// Call to "interpolate_velocity_IBM2D" kernel.  
// Note: this kernel is in the IBM/2D folder, and one
//       should use nBlocks as if calling an IBM kernel.
// --------------------------------------------------------

void scsp_D2Q9::interpolate_velocity_to_IBM(int nBlocks, int nThreads,
	                                        float* xIB, float* yIB, float* vxIB,
											float* vyIB, int nNodes)
{
	interpolate_velocity_IBM2D
	<<<nBlocks,nThreads>>> (xIB,yIB,vxIB,vyIB,u,v,Nx,nNodes);
}



// --------------------------------------------------------
// Call to "scsp_zero_forces_D2Q9" kernel:
// --------------------------------------------------------

void scsp_D2Q9::zero_forces(int nBlocks, int nThreads)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	scsp_zero_forces_D2Q9 
	<<<nBlocks,nThreads>>> (Fx,Fy,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_zero_forces_D2Q9" kernel:
// --------------------------------------------------------

void scsp_D2Q9::zero_forces_with_IBM(int nBlocks, int nThreads)
{
	if (!forceFlag) cout << "Warning: LBM force arrays have not been initialized" << endl;
	if (!velIBFlag) cout << "Warning: IB velocity arrays have not been initialized" << endl;
	scsp_zero_forces_D2Q9
	<<<nBlocks,nThreads>>> (Fx,Fy,uIBvox,vIBvox,weights,nVoxels);
}



// --------------------------------------------------------
// Wrtie output:
// --------------------------------------------------------

void scsp_D2Q9::write_output(std::string tagname, int step)
{
	GetPot inputParams("input.dat");
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = inputParams("Lattice/Nz",1);
	write_vtk_structured_grid_2D(tagname,step,Nx,Ny,Nz,rH,uH,vH);
}








