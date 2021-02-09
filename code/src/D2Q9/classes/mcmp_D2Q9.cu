
# include "mcmp_D2Q9.cuh"
# include "mcmp_D2Q9_includes.cuh"
# include "../../IO/GetPot"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

mcmp_D2Q9::mcmp_D2Q9()
{
	Q = 9;
	GetPot inputParams("input.dat");	
	nVoxels = inputParams("Lattice/nVoxels",0);
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = 1;
	if (nVoxels != Nx*Ny*Nz) cout << "nVoxels does not match Nx, Ny, Nz!" << endl;
	numIolets = inputParams("Lattice/numIolets",0);
	nu = inputParams("LBM/nu",0.1666666);
	gAB = inputParams("LBM/gAB",6.0);
	gAS = inputParams("LBM/gAS",4.5);
	gBS = inputParams("LBM/gBS",4.5); 
	omega = inputParams("LBM/omega",0.0);
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

mcmp_D2Q9::~mcmp_D2Q9()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void mcmp_D2Q9::allocate()
{
	// allocate array memory (host):
    uH = (float*)malloc(nVoxels*sizeof(float));
    vH = (float*)malloc(nVoxels*sizeof(float));
    rAH = (float*)malloc(nVoxels*sizeof(float));
	rBH = (float*)malloc(nVoxels*sizeof(float));
	sH = (int*)malloc(nVoxels*sizeof(int));
	xH = (int*)malloc(nVoxels*sizeof(int));
	yH = (int*)malloc(nVoxels*sizeof(int));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));	
	ioletsH = (iolet2D*)malloc(numIolets*sizeof(iolet2D));
			
	// allocate array memory (device):
	cudaMalloc((void **) &u, nVoxels*sizeof(float));
	cudaMalloc((void **) &v, nVoxels*sizeof(float));
	cudaMalloc((void **) &rA, nVoxels*sizeof(float));
	cudaMalloc((void **) &rB, nVoxels*sizeof(float));
	cudaMalloc((void **) &f1A, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2A, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &f1B, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2B, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &FxA, nVoxels*sizeof(float));
	cudaMalloc((void **) &FxB, nVoxels*sizeof(float));
	cudaMalloc((void **) &FyA, nVoxels*sizeof(float));
	cudaMalloc((void **) &FyB, nVoxels*sizeof(float));
	cudaMalloc((void **) &s, nVoxels*sizeof(int));	
	cudaMalloc((void **) &x, nVoxels*sizeof(int));	
	cudaMalloc((void **) &y, nVoxels*sizeof(int));	
	cudaMalloc((void **) &nList, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &iolets, numIolets*sizeof(iolet2D));	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void mcmp_D2Q9::deallocate()
{
	// free array memory (host):
	free(uH);
	free(vH);
	free(rAH);
	free(rBH);
	free(sH);
	free(xH);
	free(yH);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);	
	free(ioletsH);	
		
	// free array memory (device):
	cudaFree(u);
	cudaFree(v);
	cudaFree(rA);
	cudaFree(rB);
	cudaFree(f1A);
	cudaFree(f2A);	
	cudaFree(f1B);
	cudaFree(f2B);
	cudaFree(FxA);
	cudaFree(FxB);
	cudaFree(FyA);
	cudaFree(FyB);
	cudaFree(s);
	cudaFree(x);
	cudaFree(y);
	cudaFree(nList);
	cudaFree(voxelType);
	cudaFree(streamIndex);
	cudaFree(iolets);
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void mcmp_D2Q9::memcopy_host_to_device()
{
    cudaMemcpy(u, uH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(v, vH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(rA, rAH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(rB, rBH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(s, sH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(x, xH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(y, yH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(nList, nListH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(iolets, ioletsH, sizeof(iolet2D)*numIolets, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void mcmp_D2Q9::memcopy_device_to_host()
{
    cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(rAH, rA, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(rBH, rB, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Initialize lattice as a "box":
// --------------------------------------------------------

void mcmp_D2Q9::create_lattice_box()
{
	GetPot inputParams("input.dat");	
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

void mcmp_D2Q9::create_lattice_box_periodic()
{
	build_box_lattice_D2Q9(nVoxels,Nx,Ny,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice from "file":
// --------------------------------------------------------

void mcmp_D2Q9::create_lattice_file()
{
	
}



// --------------------------------------------------------
// Build the streamIndex[] array for PUSH streaming:
// --------------------------------------------------------

void mcmp_D2Q9::stream_index_push()
{
	stream_index_push_D2Q9(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PULL streaming:
// --------------------------------------------------------

void mcmp_D2Q9::stream_index_pull()
{
	stream_index_pull_D2Q9(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PUSH streaming:
// --------------------------------------------------------

void mcmp_D2Q9::stream_index_push_bb()
{
	stream_index_push_bb_D2Q9(nVoxels,nListH,sH,streamIndexH);
}



// --------------------------------------------------------
// Read information about iolet:
// --------------------------------------------------------

void mcmp_D2Q9::read_iolet_info(int i, const char* name) 
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
// Swap the populations 1 and 2 for both A and B:
// --------------------------------------------------------

void mcmp_D2Q9::swap_populations()
{
	float* tempA = f1A;
	float* tempB = f1B;
	f1A = f2A;
	f1B = f2B;
	f2A = tempA;
	f2B = tempB;
}



// --------------------------------------------------------
// Setters for host arrays:
// --------------------------------------------------------

void mcmp_D2Q9::setU(int i, float val)
{
	uH[i] = val;
}

void mcmp_D2Q9::setV(int i, float val)
{
	vH[i] = val;
}

void mcmp_D2Q9::setS(int i, int val)
{
	sH[i] = val;
}

void mcmp_D2Q9::setX(int i, int val)
{
	xH[i] = val;
}

void mcmp_D2Q9::setY(int i, int val)
{
	yH[i] = val;
}

void mcmp_D2Q9::setRA(int i, float val)
{
	rAH[i] = val;
}

void mcmp_D2Q9::setRB(int i, float val)
{
	rBH[i] = val;
}

void mcmp_D2Q9::setVoxelType(int i, int val)
{
	voxelTypeH[i] = val;
}



// --------------------------------------------------------
// Getters for host arrays:
// --------------------------------------------------------

float mcmp_D2Q9::getU(int i)
{
	return uH[i];
}

float mcmp_D2Q9::getV(int i)
{
	return vH[i];
}

int mcmp_D2Q9::getS(int i)
{
	return sH[i];
}

float mcmp_D2Q9::getRA(int i)
{
	return rAH[i];
}

float mcmp_D2Q9::getRB(int i)
{
	return rBH[i];
}



// --------------------------------------------------------
// Calls to Kernels:
// --------------------------------------------------------

void mcmp_D2Q9::initial_equilibrium_bb(int nBlocks, int nThreads)
{
	mcmp_initial_equilibrium_bb_D2Q9 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,u,v,nVoxels);	
}

void mcmp_D2Q9::initial_equilibrium_psm(int nBlocks, int nThreads)
{
	mcmp_initial_equilibrium_psm_D2Q9 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,u,v,nVoxels);	
}

void mcmp_D2Q9::initial_particles_on_lattice(float* prx, float* pry, float* prad, int* pIDgrid, 
	int nParts, int nBlocks, int nThreads)
{
	mcmp_initial_particles_on_lattice_D2Q9 
	<<<nBlocks,nThreads>>> (prx,pry,prad,x,y,s,pIDgrid,nVoxels,nParts);	
}

void mcmp_D2Q9::compute_density_bb(int nBlocks, int nThreads)
{
	mcmp_compute_density_bb_D2Q9
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,nVoxels);
}

void mcmp_D2Q9::compute_density_psm(int nBlocks, int nThreads)
{
	mcmp_compute_density_psm_D2Q9
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,nVoxels);
}

void mcmp_D2Q9::update_particles_on_lattice(float* prx, float* pry, float* pvx, float* pvy,
	float* prad, int* pIDgrid, int nParts, int nBlocks, int nThreads)
{
	mcmp_update_particles_on_lattice_D2Q9
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,u,v,prx,pry,pvx,pvy,prad,x,y,s,pIDgrid,nList,nVoxels,nParts);
}

void mcmp_D2Q9::update_particles_on_lattice_psm(float* prx, float* pry, float* B, 
	float* rInner, float* rOuter, int* pIDgrid, int nParts, int nBlocks, int nThreads)
{
	mcmp_update_particles_on_lattice_psm_D2Q9
	<<<nBlocks,nThreads>>> (B,prx,pry,rOuter,rInner,x,y,pIDgrid,nu,nVoxels,nParts);
}

void mcmp_D2Q9::compute_SC_forces_bb(int nBlocks, int nThreads)
{
	mcmp_compute_SC_forces_bb_D2Q9 
	<<<nBlocks,nThreads>>> (rA,rB,FxA,FxB,FyA,FyB,s,nList,gAB,gAS,gBS,nVoxels);	
}

void mcmp_D2Q9::compute_SC_forces_psm(float* B, float* pfx, float* pfy, int* pIDgrid, 
	int nBlocks, int nThreads)
{
	mcmp_compute_SC_forces_psm_D2Q9 
	<<<nBlocks,nThreads>>> (rA,rB,B,FxA,FxB,FyA,FyB,pfx,pfy,nList,pIDgrid,gAB,gAS,gBS,omega,nVoxels);	
}

void mcmp_D2Q9::compute_velocity_bb(int nBlocks, int nThreads)
{
	mcmp_compute_velocity_bb_D2Q9 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,FxA,FxB,FyA,FyB,u,v,s,nVoxels);
}

void mcmp_D2Q9::compute_velocity_psm(float* pvx, float* pvy, float* pfx, float* pfy, 
	float* B, int* pIDgrid, int nBlocks, int nThreads)
{
	mcmp_compute_velocity_psm_D2Q9 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,FxA,FxB,FyA,FyB,u,v,pvx,pvy,pfx,pfy,B,pIDgrid,nVoxels);
}

void mcmp_D2Q9::set_boundary_velocity_psm(int nBlocks, int nThreads)
{
	mcmp_set_boundary_velocity_psm_D2Q9 
	<<<nBlocks,nThreads>>> (rA,rB,FxA,FxB,FyA,FyB,u,v,y,Ny,nVoxels);
}

void mcmp_D2Q9::collide_stream_bb(int nBlocks, int nThreads)
{
	mcmp_collide_stream_bb_D2Q9 
	<<<nBlocks,nThreads>>> (f1A,f1B,f2A,f2B,rA,rB,u,v,FxA,FxB,FyA,FyB,s,streamIndex,nu,nVoxels);
}

void mcmp_D2Q9::collide_stream_psm(float* pvx, float* pvy, float* B, int* pIDgrid, 
	float rApart, float rBpart, int nBlocks, int nThreads)
{
	mcmp_collide_stream_psm_D2Q9 
	<<<nBlocks,nThreads>>> (f1A,f1B,f2A,f2B,rA,rB,u,v,FxA,FxB,FyA,FyB,pvx,pvy,B,pIDgrid,streamIndex,
	                        rApart,rBpart,nu,nVoxels);
}

void mcmp_D2Q9::bounce_back(int nBlocks, int nThreads)
{
	mcmp_bounce_back_D2Q9
	<<<nBlocks,nThreads>>> (f2A,f2B,s,nList,streamIndex,nVoxels);
}

void mcmp_D2Q9::bounce_back_moving(int nBlocks, int nThreads)
{
	mcmp_bounce_back_moving_D2Q9
	<<<nBlocks,nThreads>>> (f2A,f2B,rA,rB,u,v,s,nList,streamIndex,nVoxels);
}



// --------------------------------------------------------
// Wrtie output:
// --------------------------------------------------------

void mcmp_D2Q9::write_output(std::string tagname, int step)
{
	write_vtk_structured_grid_2D(tagname,step,Nx,Ny,Nz,rAH,rBH,uH,vH);
	write_vtk_structured_grid_2D("rA",step,Nx,Ny,Nz,rAH,uH,vH);
	write_vtk_structured_grid_2D("rB",step,Nx,Ny,Nz,rBH,uH,vH);
}









