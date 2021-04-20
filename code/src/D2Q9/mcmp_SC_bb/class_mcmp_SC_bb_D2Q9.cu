
# include "class_mcmp_SC_bb_D2Q9.cuh"
# include "../../IO/GetPot"
# include "../../Utils/gpu_parallel_reduction.h"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_mcmp_SC_bb_D2Q9::class_mcmp_SC_bb_D2Q9()
{
	Q = 9;
	GetPot inputParams("input.dat");	
	nVoxels = inputParams("Lattice/nVoxels",0);
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = 1;
	if (nVoxels != Nx*Ny*Nz) cout << "nVoxels does not match Nx, Ny, Nz!" << endl;
	numIolets = inputParams("Lattice/numIolets",0);
	nParts = inputParams("Particles/nParts",0);
	nu = inputParams("LBM/nu",0.1666666);
	gAB = inputParams("LBM/gAB",6.0);
	gAS = inputParams("LBM/gAS",6.0);
	gBS = inputParams("LBM/gBS",6.0); 
	omega = inputParams("LBM/omega",0.0);
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_mcmp_SC_bb_D2Q9::~class_mcmp_SC_bb_D2Q9()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::allocate()
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
	ptH = (particle2D_bb*)malloc(nParts*sizeof(particle2D_bb));
			
	// allocate array memory (device):
	cudaMalloc((void **) &u, nVoxels*sizeof(float));
	cudaMalloc((void **) &v, nVoxels*sizeof(float));
	cudaMalloc((void **) &rA, nVoxels*sizeof(float));
	cudaMalloc((void **) &rB, nVoxels*sizeof(float));
	cudaMalloc((void **) &rAvirt, nVoxels*sizeof(float));
	cudaMalloc((void **) &rBvirt, nVoxels*sizeof(float));
	cudaMalloc((void **) &f1A, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2A, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &f1B, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2B, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &FxA, nVoxels*sizeof(float));
	cudaMalloc((void **) &FxB, nVoxels*sizeof(float));
	cudaMalloc((void **) &FyA, nVoxels*sizeof(float));
	cudaMalloc((void **) &FyB, nVoxels*sizeof(float));
	cudaMalloc((void **) &s, nVoxels*sizeof(int));
	cudaMalloc((void **) &sprev, nVoxels*sizeof(int));	
	cudaMalloc((void **) &x, nVoxels*sizeof(int));	
	cudaMalloc((void **) &y, nVoxels*sizeof(int));	
	cudaMalloc((void **) &nList, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &pIDgrid, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &iolets, numIolets*sizeof(iolet2D));	
	cudaMalloc((void **) &pt, nParts*sizeof(particle2D_bb));
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::deallocate()
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
	free(ptH);	
		
	// free array memory (device):
	cudaFree(u);
	cudaFree(v);
	cudaFree(rA);
	cudaFree(rB);
	cudaFree(rAvirt);
	cudaFree(rBvirt);
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
	cudaFree(pt);
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::memcopy_host_to_device()
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
	cudaMemcpy(pt, ptH, sizeof(particle2D_bb)*nParts, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::memcopy_device_to_host()
{
    cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(rAH, rA, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(rBH, rB, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(ptH, pt, sizeof(particle2D_bb)*nParts, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::memcopy_device_to_host_particles()
{
    cudaMemcpy(ptH, pt, sizeof(particle2D_bb)*nParts, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Initialize lattice as a "box":
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::create_lattice_box()
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

void class_mcmp_SC_bb_D2Q9::create_lattice_box_periodic()
{
	build_box_lattice_D2Q9(nVoxels,Nx,Ny,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice as a "box" set up for shear flow
// in the x-direction:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::create_lattice_box_shear()
{
	build_box_lattice_shear_D2Q9(nVoxels,Nx,Ny,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice from "file":
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::create_lattice_file()
{
	
}



// --------------------------------------------------------
// Build the streamIndex[] array for PUSH streaming:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::stream_index_push()
{
	stream_index_push_D2Q9(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PULL streaming:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::stream_index_pull()
{
	stream_index_pull_D2Q9(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PUSH streaming:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::stream_index_push_bb()
{
	stream_index_push_bb_D2Q9(nVoxels,nListH,streamIndexH);
	//stream_index_push_bb_D2Q9(nVoxels,nListH,sH,streamIndexH);
}



// --------------------------------------------------------
// Read information about iolet:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::read_iolet_info(int i, const char* name) 
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

void class_mcmp_SC_bb_D2Q9::swap_populations()
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

void class_mcmp_SC_bb_D2Q9::setU(int i, float val)
{
	uH[i] = val;
}

void class_mcmp_SC_bb_D2Q9::setV(int i, float val)
{
	vH[i] = val;
}

void class_mcmp_SC_bb_D2Q9::setS(int i, int val)
{
	sH[i] = val;
}

void class_mcmp_SC_bb_D2Q9::setX(int i, int val)
{
	xH[i] = val;
}

void class_mcmp_SC_bb_D2Q9::setY(int i, int val)
{
	yH[i] = val;
}

void class_mcmp_SC_bb_D2Q9::setRA(int i, float val)
{
	rAH[i] = val;
}

void class_mcmp_SC_bb_D2Q9::setRB(int i, float val)
{
	rBH[i] = val;
}

void class_mcmp_SC_bb_D2Q9::setVoxelType(int i, int val)
{
	voxelTypeH[i] = val;
}

void class_mcmp_SC_bb_D2Q9::setPrx(int i, float val)
{
	ptH[i].r.x = val;
}

void class_mcmp_SC_bb_D2Q9::setPry(int i, float val)
{
	ptH[i].r.y = val;
}

void class_mcmp_SC_bb_D2Q9::setPvx(int i, float val)
{
	ptH[i].v.x = val;
}

void class_mcmp_SC_bb_D2Q9::setPvy(int i, float val)
{
	ptH[i].v.y = val;
}

void class_mcmp_SC_bb_D2Q9::setPrad(int i, float val)
{
	ptH[i].rad = val;
}

void class_mcmp_SC_bb_D2Q9::setPmass(int i, float val)
{
	ptH[i].mass = val;
}



// --------------------------------------------------------
// Getters for host arrays:
// --------------------------------------------------------

float class_mcmp_SC_bb_D2Q9::getU(int i)
{
	return uH[i];
}

float class_mcmp_SC_bb_D2Q9::getV(int i)
{
	return vH[i];
}

int class_mcmp_SC_bb_D2Q9::getS(int i)
{
	return sH[i];
}

float class_mcmp_SC_bb_D2Q9::getRA(int i)
{
	return rAH[i];
}

float class_mcmp_SC_bb_D2Q9::getRB(int i)
{
	return rBH[i];
}

float class_mcmp_SC_bb_D2Q9::getPrx(int i)
{
	return ptH[i].r.x;
}

float class_mcmp_SC_bb_D2Q9::getPry(int i)
{
	return ptH[i].r.y;
}

float class_mcmp_SC_bb_D2Q9::getPfx(int i)
{
	return ptH[i].f.x;
}

float class_mcmp_SC_bb_D2Q9::getPfy(int i)
{
	return ptH[i].f.y;
}

float class_mcmp_SC_bb_D2Q9::getPrad(int i)
{
	return ptH[i].rad;
}

float class_mcmp_SC_bb_D2Q9::getPmass(int i)
{
	return ptH[i].mass;
}



// --------------------------------------------------------
// Calls to Kernels:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::initial_equilibrium_bb(int nBlocks, int nThreads)
{
	mcmp_initial_equilibrium_bb_D2Q9 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,u,v,nVoxels);	
}

void class_mcmp_SC_bb_D2Q9::map_particles_on_lattice_bb(int nBlocks, int nThreads)
{
	mcmp_map_particles_on_lattice_bb_D2Q9 
	<<<nBlocks,nThreads>>> (pt,x,y,s,sprev,pIDgrid,nVoxels,nParts);	
}

void class_mcmp_SC_bb_D2Q9::cover_uncover_bb(int nBlocks, int nThreads)
{
	mcmp_cover_uncover_bb_D2Q9 
	<<<nBlocks,nThreads>>> (s,sprev,nList,u,v,rA,rB,f1A,f1B,nVoxels);	
}

void class_mcmp_SC_bb_D2Q9::compute_density_bb(int nBlocks, int nThreads)
{
	mcmp_compute_density_bb_D2Q9
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,nVoxels);
}

void class_mcmp_SC_bb_D2Q9::compute_virtual_density_bb(int nBlocks, int nThreads)
{
	mcmp_compute_virtual_density_bb_D2Q9
	<<<nBlocks,nThreads>>> (rAvirt,rBvirt,rA,rB,s,nList,omega,nVoxels);
}

void class_mcmp_SC_bb_D2Q9::update_particles_on_lattice_bb(int nBlocks, int nThreads)
{
	mcmp_update_particles_on_lattice_D2Q9
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,u,v,pt,x,y,s,pIDgrid,nList,nVoxels,nParts);
}

void class_mcmp_SC_bb_D2Q9::compute_SC_forces_bb(int nBlocks, int nThreads)
{
	mcmp_compute_SC_forces_bb_D2Q9 
	<<<nBlocks,nThreads>>> (rA,rB,FxA,FxB,FyA,FyB,pt,pIDgrid,s,nList,gAB,gAS,gBS,nVoxels);	
}

void class_mcmp_SC_bb_D2Q9::compute_SC_forces_bb_2(int nBlocks, int nThreads)
{
	mcmp_compute_SC_forces_bb_2_D2Q9 
	<<<nBlocks,nThreads>>> (rA,rB,rAvirt,rBvirt,FxA,FxB,FyA,FyB,pt,pIDgrid,s,nList,gAB,nVoxels);	
}

void class_mcmp_SC_bb_D2Q9::compute_velocity_bb(int nBlocks, int nThreads)
{
	mcmp_compute_velocity_bb_D2Q9 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,FxA,FxB,FyA,FyB,u,v,s,pIDgrid,pt,nVoxels);
}

void class_mcmp_SC_bb_D2Q9::set_boundary_velocity_bb(float uBC, float vBC, int nBlocks, int nThreads)
{
	mcmp_set_boundary_velocity_bb_D2Q9 
	<<<nBlocks,nThreads>>> (uBC,vBC,rA,rB,FxA,FxB,FyA,FyB,u,v,y,Ny,nVoxels);
}

void class_mcmp_SC_bb_D2Q9::set_boundary_shear_velocity_bb(float uBot, float uTop, int nBlocks, int nThreads)
{
	mcmp_set_boundary_shear_velocity_bb_D2Q9 
	<<<nBlocks,nThreads>>> (uBot,uTop,rA,rB,FxA,FxB,FyA,FyB,u,v,y,Ny,nVoxels);
}

void class_mcmp_SC_bb_D2Q9::collide_stream_bb(int nBlocks, int nThreads)
{
	mcmp_collide_stream_bb_D2Q9 
	<<<nBlocks,nThreads>>> (f1A,f1B,f2A,f2B,rA,rB,u,v,FxA,FxB,FyA,FyB,s,streamIndex,nu,nVoxels);
}

void class_mcmp_SC_bb_D2Q9::bounce_back(int nBlocks, int nThreads)
{
	mcmp_bounce_back_D2Q9
	<<<nBlocks,nThreads>>> (f2A,f2B,s,nList,streamIndex,nVoxels);
}

void class_mcmp_SC_bb_D2Q9::bounce_back_moving(int nBlocks, int nThreads)
{
	mcmp_bounce_back_moving_D2Q9
	<<<nBlocks,nThreads>>> (f2A,f2B,rA,rB,u,v,pt,pIDgrid,s,nList,streamIndex,nVoxels);
}

void class_mcmp_SC_bb_D2Q9::move_particles_bb(int nBlocks, int nThreads)
{
	mcmp_move_particles_bb_D2Q9
	<<<nBlocks,nThreads>>> (pt,nParts);
}

void class_mcmp_SC_bb_D2Q9::fix_particle_velocity_bb(float pvel, int nBlocks, int nThreads)
{
	mcmp_fix_particle_velocity_bb_D2Q9
	<<<nBlocks,nThreads>>> (pt,pvel,nParts);
}

void class_mcmp_SC_bb_D2Q9::zero_particle_forces_bb(int nBlocks, int nThreads)
{
	mcmp_zero_particle_forces_bb_D2Q9
	<<<nBlocks,nThreads>>> (pt,nParts);
}

void class_mcmp_SC_bb_D2Q9::particle_particle_forces_bb(float K, float halo, int nBlocks, int nThreads)
{
	mcmp_particle_particle_forces_bb_D2Q9
	<<<nBlocks,nThreads>>> (pt,K,halo,nParts);
}



// --------------------------------------------------------
// Calls to Kernels: sum fluid density
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::sum_fluid_densities_bb(int nBlocks, int nThreads)
{
	// allocate memory for partialSum array:
	float* partialSum;
	cudaMalloc((void**)&partialSum, sizeof(float)*nBlocks);
	// fluid A:
	add_array_elements<<<nBlocks,nThreads,nThreads*sizeof(float)>>>(rA,        nVoxels,partialSum);
	add_array_elements<<<1,      nThreads,nThreads*sizeof(float)>>>(partialSum,nBlocks,partialSum);
	cudaDeviceSynchronize();
	cudaMemcpy(&rAsum, partialSum, sizeof(float), cudaMemcpyDeviceToHost);
	// fluid B:
	add_array_elements<<<nBlocks,nThreads,nThreads*sizeof(float)>>>(rB,        nVoxels,partialSum);
	add_array_elements<<<1,      nThreads,nThreads*sizeof(float)>>>(partialSum,nBlocks,partialSum);
	cudaDeviceSynchronize();
	cudaMemcpy(&rBsum, partialSum, sizeof(float), cudaMemcpyDeviceToHost);
	// deallocate memory:
	cudaFree(partialSum);
}



// --------------------------------------------------------
// Wrtie output:
// --------------------------------------------------------

void class_mcmp_SC_bb_D2Q9::write_output(std::string tagname, int step)
{
	write_vtk_structured_grid_2D(tagname,step,Nx,Ny,Nz,rAH,rBH,uH,vH);
	//write_vtk_structured_grid_2D("rA",step,Nx,Ny,Nz,rAH,uH,vH);
	//write_vtk_structured_grid_2D("rB",step,Nx,Ny,Nz,rBH,uH,vH);
}

void class_mcmp_SC_bb_D2Q9::write_density_sums(int step)
{
	cout << step << "  " << rAsum << "  " << rBsum << endl;
}









