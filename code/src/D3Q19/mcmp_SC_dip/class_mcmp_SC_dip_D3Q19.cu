
# include "class_mcmp_SC_dip_D3Q19.cuh"
# include "../../IO/GetPot"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_mcmp_SC_dip_D3Q19::class_mcmp_SC_dip_D3Q19()
{
	Q = 19;
	GetPot inputParams("input.dat");	
	nVoxels = inputParams("Lattice/nVoxels",0);
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = inputParams("Lattice/Nz",1);
	if (nVoxels != Nx*Ny*Nz) cout << "nVoxels does not match Nx, Ny, Nz!" << endl;
	numIolets = inputParams("Lattice/numIolets",0);
	nParts = inputParams("Particles/nParts",0);
	nu = inputParams("LBM/nu",0.1666666);
	gAB = inputParams("LBM/gAB",6.0);
	gAS = inputParams("LBM/gAS",4.5);
	gBS = inputParams("LBM/gBS",4.5); 
	omega = inputParams("LBM/omega",0.0);
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_mcmp_SC_dip_D3Q19::~class_mcmp_SC_dip_D3Q19()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::allocate()
{
	// allocate array memory (host):
    uH = (float*)malloc(nVoxels*sizeof(float));
	vH = (float*)malloc(nVoxels*sizeof(float));
	wH = (float*)malloc(nVoxels*sizeof(float));
    rAH = (float*)malloc(nVoxels*sizeof(float));
	rBH = (float*)malloc(nVoxels*sizeof(float));
	xH = (int*)malloc(nVoxels*sizeof(int));
	yH = (int*)malloc(nVoxels*sizeof(int));
	zH = (int*)malloc(nVoxels*sizeof(int));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));	
	ioletsH = (iolet*)malloc(numIolets*sizeof(iolet));
	ptH = (particle3D_dip*)malloc(nParts*sizeof(particle3D_dip));
			
	// allocate array memory (device):
	cudaMalloc((void **) &u, nVoxels*sizeof(float));
	cudaMalloc((void **) &v, nVoxels*sizeof(float));
	cudaMalloc((void **) &w, nVoxels*sizeof(float));
	cudaMalloc((void **) &rA, nVoxels*sizeof(float));
	cudaMalloc((void **) &rB, nVoxels*sizeof(float));
	cudaMalloc((void **) &rS, nVoxels*sizeof(float));
	cudaMalloc((void **) &f1A, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2A, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &f1B, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2B, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &FxA, nVoxels*sizeof(float));
	cudaMalloc((void **) &FyA, nVoxels*sizeof(float));
	cudaMalloc((void **) &FzA, nVoxels*sizeof(float));
	cudaMalloc((void **) &FxB, nVoxels*sizeof(float));
	cudaMalloc((void **) &FyB, nVoxels*sizeof(float));
	cudaMalloc((void **) &FzB, nVoxels*sizeof(float));
	cudaMalloc((void **) &x, nVoxels*sizeof(int));	
	cudaMalloc((void **) &y, nVoxels*sizeof(int));	
	cudaMalloc((void **) &z, nVoxels*sizeof(int));	
	cudaMalloc((void **) &nList, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &pIDgrid, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &iolets, numIolets*sizeof(iolet));
	cudaMalloc((void **) &pt, nParts*sizeof(particle3D_dip));	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::deallocate()
{
	// free array memory (host):
	free(uH);
	free(vH);
	free(wH);
	free(rAH);
	free(rBH);
	free(xH);
	free(yH);
	free(zH);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);	
	free(ioletsH);	
	free(ptH);
		
	// free array memory (device):
	cudaFree(u);
	cudaFree(v);
	cudaFree(w);	
	cudaFree(rA);
	cudaFree(rB);
	cudaFree(rS);
	cudaFree(f1A);
	cudaFree(f2A);	
	cudaFree(f1B);
	cudaFree(f2B);
	cudaFree(FxA);
	cudaFree(FxB);
	cudaFree(FyA);
	cudaFree(FyB);
	cudaFree(FzA);
	cudaFree(FzB);
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
	cudaFree(nList);
	cudaFree(voxelType);
	cudaFree(streamIndex);
	cudaFree(iolets);
	cudaFree(pt);
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::memcopy_host_to_device()
{
    cudaMemcpy(u, uH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(v, vH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(w, wH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(rA, rAH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(rB, rBH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(x, xH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(y, yH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(z, zH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(nList, nListH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(iolets, ioletsH, sizeof(iolet)*numIolets, cudaMemcpyHostToDevice);
	cudaMemcpy(pt, ptH, sizeof(particle3D_dip)*nParts, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::memcopy_device_to_host()
{
    cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(wH, w, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(rAH, rA, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(rBH, rB, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(ptH, pt, sizeof(particle3D_dip)*nParts, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::memcopy_device_to_host_particles()
{
    cudaMemcpy(ptH, pt, sizeof(particle3D_dip)*nParts, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Initialize lattice as a "box":
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::create_lattice_box()
{
	GetPot inputParams("input.dat");	
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = inputParams("Lattice/Nz",1);
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

void class_mcmp_SC_dip_D3Q19::create_lattice_box_periodic()
{
	build_box_lattice_D3Q19(nVoxels,Nx,Ny,Nz,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice from "file":
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::create_lattice_file()
{
	
}



// --------------------------------------------------------
// Build the streamIndex[] array for PUSH streaming:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::stream_index_push()
{
	stream_index_push_D3Q19(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PULL streaming:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::stream_index_pull()
{
	stream_index_pull_D3Q19(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Read information about iolet:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::read_iolet_info(int i, const char* name) 
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
// Swap the populations 1 and 2 for both A and B:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::swap_populations()
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

void class_mcmp_SC_dip_D3Q19::setU(int i, float val)
{
	uH[i] = val;
}

void class_mcmp_SC_dip_D3Q19::setV(int i, float val)
{
	vH[i] = val;
}

void class_mcmp_SC_dip_D3Q19::setW(int i, float val)
{
	wH[i] = val;
}

void class_mcmp_SC_dip_D3Q19::setX(int i, int val)
{
	xH[i] = val;
}

void class_mcmp_SC_dip_D3Q19::setY(int i, int val)
{
	yH[i] = val;
}

void class_mcmp_SC_dip_D3Q19::setZ(int i, int val)
{
	zH[i] = val;
}

void class_mcmp_SC_dip_D3Q19::setRA(int i, float val)
{
	rAH[i] = val;
}

void class_mcmp_SC_dip_D3Q19::setRB(int i, float val)
{
	rBH[i] = val;
}

void class_mcmp_SC_dip_D3Q19::setVoxelType(int i, int val)
{
	voxelTypeH[i] = val;
}

void class_mcmp_SC_dip_D3Q19::setPrx(int i, float val)
{
	ptH[i].r.x = val;
}

void class_mcmp_SC_dip_D3Q19::setPry(int i, float val)
{
	ptH[i].r.y = val;
}

void class_mcmp_SC_dip_D3Q19::setPrz(int i, float val)
{
	ptH[i].r.z = val;
}

void class_mcmp_SC_dip_D3Q19::setPvx(int i, float val)
{
	ptH[i].v.x = val;
}

void class_mcmp_SC_dip_D3Q19::setPvy(int i, float val)
{
	ptH[i].v.y = val;
}

void class_mcmp_SC_dip_D3Q19::setPvz(int i, float val)
{
	ptH[i].v.z = val;
}

void class_mcmp_SC_dip_D3Q19::setPrInner(int i, float val)
{
	ptH[i].rInner = val;
}

void class_mcmp_SC_dip_D3Q19::setPrOuter(int i, float val)
{
	ptH[i].rOuter = val;
}



// --------------------------------------------------------
// Getters for host arrays:
// --------------------------------------------------------

float class_mcmp_SC_dip_D3Q19::getU(int i)
{
	return uH[i];
}

float class_mcmp_SC_dip_D3Q19::getV(int i)
{
	return vH[i];
}

float class_mcmp_SC_dip_D3Q19::getW(int i)
{
	return wH[i];
}

float class_mcmp_SC_dip_D3Q19::getRA(int i)
{
	return rAH[i];
}

float class_mcmp_SC_dip_D3Q19::getRB(int i)
{
	return rBH[i];
}

float class_mcmp_SC_dip_D3Q19::getPrx(int i)
{
	return ptH[i].r.x;
}

float class_mcmp_SC_dip_D3Q19::getPry(int i)
{
	return ptH[i].r.y;
}

float class_mcmp_SC_dip_D3Q19::getPrz(int i)
{
	return ptH[i].r.z;
}

float class_mcmp_SC_dip_D3Q19::getPvx(int i)
{
	return ptH[i].v.x;
}

float class_mcmp_SC_dip_D3Q19::getPvy(int i)
{
	return ptH[i].v.y;
}

float class_mcmp_SC_dip_D3Q19::getPvz(int i)
{
	return ptH[i].v.z;
}

float class_mcmp_SC_dip_D3Q19::getPfx(int i)
{
	return ptH[i].f.x;
}

float class_mcmp_SC_dip_D3Q19::getPfy(int i)
{
	return ptH[i].f.y;
}

float class_mcmp_SC_dip_D3Q19::getPfz(int i)
{
	return ptH[i].f.z;
}

float class_mcmp_SC_dip_D3Q19::getPmass(int i)
{
	return ptH[i].mass;
}

float class_mcmp_SC_dip_D3Q19::getPrInner(int i)
{
	return ptH[i].rInner;
}

float class_mcmp_SC_dip_D3Q19::getPrOuter(int i)
{
	return ptH[i].rOuter;
}
 


// --------------------------------------------------------
// Calls to Kernels:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::initial_equilibrium_dip(int nBlocks, int nThreads)
{
	mcmp_initial_equilibrium_dip_D3Q19 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,u,v,w,nVoxels);	
}

void class_mcmp_SC_dip_D3Q19::compute_density_dip(int nBlocks, int nThreads)
{
	mcmp_compute_density_dip_D3Q19
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,nVoxels);
}

void class_mcmp_SC_dip_D3Q19::map_particles_to_lattice_dip(int nBlocks, int nThreads)
{
	mcmp_map_particles_to_lattice_dip_D3Q19
	<<<nBlocks,nThreads>>> (rS,pt,x,y,z,pIDgrid,nVoxels,nParts);
} 

void class_mcmp_SC_dip_D3Q19::compute_SC_forces_dip(int nBlocks, int nThreads)
{
	mcmp_compute_SC_forces_dip_D3Q19 
	<<<nBlocks,nThreads>>> (rA,rB,rS,FxA,FxB,FyA,FyB,FzA,FzB,pt,nList,pIDgrid,gAB,gAS,gBS,omega,nVoxels);	
}

void class_mcmp_SC_dip_D3Q19::compute_velocity_dip(int nBlocks, int nThreads)
{
	mcmp_compute_velocity_dip_D3Q19 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,rS,FxA,FxB,FyA,FyB,FzA,FzB,u,v,w,pt,pIDgrid,nVoxels);
}

void class_mcmp_SC_dip_D3Q19::compute_velocity_dip_2(int nBlocks, int nThreads)
{
	mcmp_compute_velocity_dip_2_D3Q19 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,rS,FxA,FxB,FyA,FyB,FzA,FzB,u,v,w,pt,pIDgrid,nVoxels);
}

void class_mcmp_SC_dip_D3Q19::set_boundary_velocity_dip(float uBC, float vBC, float wBC, int nBlocks, int nThreads)
{
	mcmp_set_boundary_velocity_dip_D3Q19 
	<<<nBlocks,nThreads>>> (uBC,vBC,wBC,rA,rB,FxA,FxB,FyA,FyB,FzA,FzB,u,v,w,y,Ny,nVoxels);
}

void class_mcmp_SC_dip_D3Q19::collide_stream_dip(int nBlocks, int nThreads)
{
	mcmp_collide_stream_dip_D3Q19 
	<<<nBlocks,nThreads>>> (f1A,f1B,f2A,f2B,rA,rB,u,v,w,FxA,FxB,FyA,FyB,FzA,FzB,streamIndex,nu,nVoxels);
}

void class_mcmp_SC_dip_D3Q19::move_particles_dip(int nBlocks, int nThreads)
{
	mcmp_move_particles_dip_D3Q19
	<<<nBlocks,nThreads>>> (pt,nParts);
}

void class_mcmp_SC_dip_D3Q19::fix_particle_velocity_dip(float pvel, int nBlocks, int nThreads)
{
	mcmp_fix_particle_velocity_dip_D3Q19
	<<<nBlocks,nThreads>>> (pt,pvel,nParts);
}

void class_mcmp_SC_dip_D3Q19::zero_particle_forces_dip(int nBlocks, int nThreads)
{
	mcmp_zero_particle_forces_dip_D3Q19
	<<<nBlocks,nThreads>>> (pt,nParts);
}



// --------------------------------------------------------
// Write output:
// --------------------------------------------------------

void class_mcmp_SC_dip_D3Q19::write_output(std::string tagname, int step,
                                           int iskip, int jskip, int kskip)
{
	write_vtk_structured_grid(tagname,step,Nx,Ny,Nz,rAH,rBH,uH,vH,wH,iskip,jskip,kskip);
}









