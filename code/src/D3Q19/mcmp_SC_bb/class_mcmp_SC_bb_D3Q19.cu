
# include "class_mcmp_SC_bb_D3Q19.cuh"
# include "../../IO/GetPot"
# include "../../Utils/gpu_parallel_reduction.cuh"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_mcmp_SC_bb_D3Q19::class_mcmp_SC_bb_D3Q19()
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
	omega = inputParams("LBM/omega",0.0);
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_mcmp_SC_bb_D3Q19::~class_mcmp_SC_bb_D3Q19()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::allocate()
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
	sH = (int*)malloc(nVoxels*sizeof(int));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));	
	ioletsH = (iolet*)malloc(numIolets*sizeof(iolet));
	ptH = (particle3D_bb*)malloc(nParts*sizeof(particle3D_bb));
			
	// allocate array memory (device):
	cudaMalloc((void **) &u, nVoxels*sizeof(float));
	cudaMalloc((void **) &v, nVoxels*sizeof(float));
	cudaMalloc((void **) &w, nVoxels*sizeof(float));
	cudaMalloc((void **) &rA, nVoxels*sizeof(float));
	cudaMalloc((void **) &rB, nVoxels*sizeof(float));
	cudaMalloc((void **) &rAvirt, nVoxels*sizeof(float));
	cudaMalloc((void **) &rBvirt, nVoxels*sizeof(float));
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
	cudaMalloc((void **) &s, nVoxels*sizeof(int));	
	cudaMalloc((void **) &sprev, nVoxels*sizeof(int));	
	cudaMalloc((void **) &nList, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &pIDgrid, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &iolets, numIolets*sizeof(iolet));
	cudaMalloc((void **) &pt, nParts*sizeof(particle3D_bb));	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::deallocate()
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
	free(sH);
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
	cudaFree(FzA);
	cudaFree(FzB);
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
	cudaFree(s);
	cudaFree(sprev);
	cudaFree(nList);
	cudaFree(voxelType);
	cudaFree(streamIndex);
	cudaFree(iolets);
	cudaFree(pt);
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::memcopy_host_to_device()
{
    cudaMemcpy(u, uH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(v, vH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(w, wH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(rA, rAH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(rB, rBH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(x, xH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(y, yH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(z, zH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(s, sH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(nList, nListH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(iolets, ioletsH, sizeof(iolet)*numIolets, cudaMemcpyHostToDevice);
	cudaMemcpy(pt, ptH, sizeof(particle3D_bb)*nParts, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::memcopy_device_to_host()
{
    cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(wH, w, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(rAH, rA, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(rBH, rB, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(ptH, pt, sizeof(particle3D_bb)*nParts, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::memcopy_device_to_host_particles()
{
    cudaMemcpy(ptH, pt, sizeof(particle3D_bb)*nParts, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Initialize lattice as a "box":
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::create_lattice_box()
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

void class_mcmp_SC_bb_D3Q19::create_lattice_box_periodic()
{
	build_box_lattice_D3Q19(nVoxels,Nx,Ny,Nz,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Initialize lattice from "file":
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::create_lattice_file()
{
	
}



// --------------------------------------------------------
// Build the streamIndex[] array for PUSH streaming:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::stream_index_push()
{
	stream_index_push_D3Q19(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PULL streaming:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::stream_index_pull()
{
	stream_index_pull_D3Q19(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Read information about iolet:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::read_iolet_info(int i, const char* name) 
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

void class_mcmp_SC_bb_D3Q19::swap_populations()
{
	float* tempA = f1A;
	float* tempB = f1B;
	f1A = f2A;
	f1B = f2B;
	f2A = tempA;
	f2B = tempB;
}



// --------------------------------------------------------
// Calculate initial density sums for both A and B:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::calculate_initial_density_sums()
{
	rAsum0 = 0.0;
	rBsum0 = 0.0;
	for (int i=0; i<nVoxels; i++) {
		rAsum0 += rAH[i];
		rBsum0 += rBH[i];
	}
}



// --------------------------------------------------------
// Setters for host arrays:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::setU(int i, float val)
{
	uH[i] = val;
}

void class_mcmp_SC_bb_D3Q19::setV(int i, float val)
{
	vH[i] = val;
}

void class_mcmp_SC_bb_D3Q19::setW(int i, float val)
{
	wH[i] = val;
}

void class_mcmp_SC_bb_D3Q19::setX(int i, int val)
{
	xH[i] = val;
}

void class_mcmp_SC_bb_D3Q19::setY(int i, int val)
{
	yH[i] = val;
}

void class_mcmp_SC_bb_D3Q19::setZ(int i, int val)
{
	zH[i] = val;
}

void class_mcmp_SC_bb_D3Q19::setS(int i, int val)
{
	sH[i] = val;
}

void class_mcmp_SC_bb_D3Q19::setRA(int i, float val)
{
	rAH[i] = val;
}

void class_mcmp_SC_bb_D3Q19::setRB(int i, float val)
{
	rBH[i] = val;
}

void class_mcmp_SC_bb_D3Q19::setVoxelType(int i, int val)
{
	voxelTypeH[i] = val;
}

void class_mcmp_SC_bb_D3Q19::setPrx(int i, float val)
{
	ptH[i].r.x = val;
}

void class_mcmp_SC_bb_D3Q19::setPry(int i, float val)
{
	ptH[i].r.y = val;
}

void class_mcmp_SC_bb_D3Q19::setPrz(int i, float val)
{
	ptH[i].r.z = val;
}

void class_mcmp_SC_bb_D3Q19::setPvx(int i, float val)
{
	ptH[i].v.x = val;
}

void class_mcmp_SC_bb_D3Q19::setPvy(int i, float val)
{
	ptH[i].v.y = val;
}

void class_mcmp_SC_bb_D3Q19::setPvz(int i, float val)
{
	ptH[i].v.z = val;
}

void class_mcmp_SC_bb_D3Q19::setPrad(int i, float val)
{
	ptH[i].rad = val;
}

void class_mcmp_SC_bb_D3Q19::setPmass(int i, float val)
{
	ptH[i].mass = val;
}



// --------------------------------------------------------
// Getters for host arrays:
// --------------------------------------------------------

float class_mcmp_SC_bb_D3Q19::getU(int i)
{
	return uH[i];
}

float class_mcmp_SC_bb_D3Q19::getV(int i)
{
	return vH[i];
}

float class_mcmp_SC_bb_D3Q19::getW(int i)
{
	return wH[i];
}

int class_mcmp_SC_bb_D3Q19::getS(int i)
{
	return sH[i];
}

float class_mcmp_SC_bb_D3Q19::getRA(int i)
{
	return rAH[i];
}

float class_mcmp_SC_bb_D3Q19::getRB(int i)
{
	return rBH[i];
}

float class_mcmp_SC_bb_D3Q19::getPrx(int i)
{
	return ptH[i].r.x;
}

float class_mcmp_SC_bb_D3Q19::getPry(int i)
{
	return ptH[i].r.y;
}

float class_mcmp_SC_bb_D3Q19::getPrz(int i)
{
	return ptH[i].r.z;
}

float class_mcmp_SC_bb_D3Q19::getPvx(int i)
{
	return ptH[i].v.x;
}

float class_mcmp_SC_bb_D3Q19::getPvy(int i)
{
	return ptH[i].v.y;
}

float class_mcmp_SC_bb_D3Q19::getPvz(int i)
{
	return ptH[i].v.z;
}

float class_mcmp_SC_bb_D3Q19::getPfx(int i)
{
	return ptH[i].f.x;
}

float class_mcmp_SC_bb_D3Q19::getPfy(int i)
{
	return ptH[i].f.y;
}

float class_mcmp_SC_bb_D3Q19::getPfz(int i)
{
	return ptH[i].f.z;
}

float class_mcmp_SC_bb_D3Q19::getPmass(int i)
{
	return ptH[i].mass;
}

float class_mcmp_SC_bb_D3Q19::getPrad(int i)
{
	return ptH[i].rad;
}



// --------------------------------------------------------
// Calculate fluid "A" volume:
// --------------------------------------------------------

float class_mcmp_SC_bb_D3Q19::calculate_fluid_A_volume() 
{
	float sum = 0.0;
	for (int i=0; i<nVoxels; i++) {
		if (rAH[i] > 0.5) sum += 1.0; 
	}
	return sum;
}



// --------------------------------------------------------
// Calls to Kernels:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::initial_equilibrium_bb(int nBlocks, int nThreads)
{
	mcmp_initial_equilibrium_bb_D3Q19 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,u,v,w,nVoxels);	
}

void class_mcmp_SC_bb_D3Q19::compute_density_bb(int nBlocks, int nThreads)
{
	mcmp_compute_density_bb_D3Q19
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,nVoxels);
}

void class_mcmp_SC_bb_D3Q19::correct_density_totals_bb(int nBlocks, int nThreads)
{
	float delrA = rAsum - rAsum0;
	float delrB = rBsum - rBsum0;
	mcmp_correct_density_totals_D3Q19
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,delrA,delrB,nVoxels);
}

void class_mcmp_SC_bb_D3Q19::compute_virtual_density_bb(int nBlocks, int nThreads)
{
	mcmp_compute_virtual_density_bb_D3Q19
	<<<nBlocks,nThreads>>> (rAvirt,rBvirt,rA,rB,s,nList,omega,nVoxels);	
}

void class_mcmp_SC_bb_D3Q19::map_particles_to_lattice_bb(int nBlocks, int nThreads)
{
	mcmp_map_particles_to_lattice_bb_D3Q19
	<<<nBlocks,nThreads>>> (pt,x,y,z,s,sprev,pIDgrid,nVoxels,nParts);
}

void class_mcmp_SC_bb_D3Q19::cover_uncover_bb(int nBlocks, int nThreads)
{
	mcmp_cover_uncover_bb_D3Q19
	<<<nBlocks,nThreads>>> (s,sprev,nList,u,v,w,rA,rB,f1A,f1B,nVoxels);
}

void class_mcmp_SC_bb_D3Q19::update_particles_on_lattice_bb(int nBlocks, int nThreads)
{
	mcmp_update_particles_on_lattice_bb_D3Q19
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,u,v,w,pt,x,y,z,s,pIDgrid,nList,nVoxels,nParts);
} 

void class_mcmp_SC_bb_D3Q19::compute_SC_forces_bb(int nBlocks, int nThreads)
{
	mcmp_compute_SC_forces_bb_D3Q19 
	<<<nBlocks,nThreads>>> (rAvirt,rBvirt,rA,rB,FxA,FxB,FyA,FyB,FzA,FzB,pt,nList,pIDgrid,s,gAB,nVoxels);	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
}

void class_mcmp_SC_bb_D3Q19::compute_velocity_bb(int nBlocks, int nThreads)
{
	mcmp_compute_velocity_bb_D3Q19 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,FxA,FxB,FyA,FyB,FzA,FzB,u,v,w,pt,s,pIDgrid,nVoxels);
}

void class_mcmp_SC_bb_D3Q19::set_boundary_velocity_bb(float uBC, float vBC, float wBC, int nBlocks, int nThreads)
{
	mcmp_set_boundary_velocity_bb_D3Q19 
	<<<nBlocks,nThreads>>> (uBC,vBC,wBC,rA,rB,FxA,FxB,FyA,FyB,FzA,FzB,u,v,w,y,Ny,nVoxels);
}

void class_mcmp_SC_bb_D3Q19::set_boundary_shear_velocity_bb(float uBot, float uTop, int nBlocks, int nThreads)
{
	mcmp_set_boundary_shear_velocity_bb_D3Q19 
	<<<nBlocks,nThreads>>> (uBot,uTop,rA,rB,FxA,FxB,FyA,FyB,FzA,FzB,u,v,w,y,Ny,nVoxels);
}

void class_mcmp_SC_bb_D3Q19::collide_stream_bb(int nBlocks, int nThreads)
{
	mcmp_collide_stream_bb_D3Q19 
	<<<nBlocks,nThreads>>> (f1A,f1B,f2A,f2B,rA,rB,u,v,w,FxA,FxB,FyA,FyB,FzA,FzB,streamIndex,nu,nVoxels);
}

void class_mcmp_SC_bb_D3Q19::bounce_back(int nBlocks, int nThreads)
{
	mcmp_bounce_back_D3Q19
	<<<nBlocks,nThreads>>> (f2A,f2B,s,nList,streamIndex,nVoxels);
}

void class_mcmp_SC_bb_D3Q19::bounce_back_moving(int nBlocks, int nThreads)
{
	mcmp_bounce_back_moving_D3Q19
	<<<nBlocks,nThreads>>> (f2A,f2B,rA,rB,u,v,w,pt,pIDgrid,s,nList,streamIndex,nVoxels);
}

void class_mcmp_SC_bb_D3Q19::move_particles_bb(int nBlocks, int nThreads)
{
	mcmp_move_particles_bb_D3Q19
	<<<nBlocks,nThreads>>> (pt,nParts);
}

void class_mcmp_SC_bb_D3Q19::fix_particle_velocity_bb(float pvel, int nBlocks, int nThreads)
{
	mcmp_fix_particle_velocity_bb_D3Q19
	<<<nBlocks,nThreads>>> (pt,pvel,nParts);
}

void class_mcmp_SC_bb_D3Q19::zero_particle_forces_bb(int nBlocks, int nThreads)
{
	mcmp_zero_particle_forces_bb_D3Q19
	<<<nBlocks,nThreads>>> (pt,nParts);
}

void class_mcmp_SC_bb_D3Q19::particle_particle_forces_bb(float K, float halo, int nBlocks, int nThreads)
{
	mcmp_particle_particle_forces_bb_D3Q19
	<<<nBlocks,nThreads>>> (pt,K,halo,nParts);
}



// --------------------------------------------------------
// Calls to Kernels: sum fluid density
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::sum_fluid_densities_bb(int nBlocks, int nThreads)
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
// Write output:
// --------------------------------------------------------

void class_mcmp_SC_bb_D3Q19::write_output(std::string tagname, int step,
                                           int iskip, int jskip, int kskip)
{
	write_vtk_structured_grid(tagname,step,Nx,Ny,Nz,rAH,rBH,uH,vH,wH,iskip,jskip,kskip);
	write_vtk_particles("parts",step,ptH,nParts);
}









