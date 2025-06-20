 
# include "class_fibers_ibm3D.cuh"
# include "../../IO/GetPot"
# include "../../Utils/eig3.cuh"
# include <math.h>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <string>
# include <sstream>
# include <stdlib.h>
# include <time.h>
using namespace std;  





// --------------------------------------------------------
//
// This class implements the implicit finite-difference
// model for a flexible filament given by:
//
// Huang WX, Shin SJ, Sung HJ.  Simulation of flexible 
// filaments in a uniform flow by the immersed boundary
// method.  Journal of Computational Physics 226 (2007)
// 2206-2228.
// 
// --------------------------------------------------------










// **********************************************************************************************
// Constructor, destructor, and array allocations...
// **********************************************************************************************










// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_fibers_ibm3D::class_fibers_ibm3D()
{
	// get some parameters:
	GetPot inputParams("input.dat");
	
	// mesh attributes	
	nBeadsPerFiber = inputParams("IBM_FIBERS/nBeadsPerFiber",0);
	nEdgesPerFiber = inputParams("IBM_FIBERS/nEdgesPerFiber",0);
	nFibers = inputParams("IBM_FIBERS/nFibers",1);
	nBeads = nBeadsPerFiber*nFibers;
	nEdges = nEdgesPerFiber*nFibers;
	
	// fiber properties
	dt = inputParams("Time/dt",1.0);
	dS = inputParams("IBM_FIBERS/dS",1.0);
	repA = inputParams("IBM_FIBERS/repA",0.0);
	repD = inputParams("IBM_FIBERS/repD",0.0);
	beadFmax = inputParams("IBM_FIBERS/beadFmax",1000.0);
	gam = inputParams("IBM_FIBERS/gamma",0.1);
	fricBead = 6.0*M_PI*(1.0/6.0)*repD;  // friction coefficient per bead (assume visc=1/6)
	
	// domain attributes
	N.x = inputParams("Lattice/Nx",1);
	N.y = inputParams("Lattice/Ny",1);
	N.z = inputParams("Lattice/Nz",1);	
	Box.x = float(N.x);   // assume dx=1
	Box.y = float(N.y);
	Box.z = float(N.z);
	pbcFlag = make_int3(1,1,1);
	
	// initialize cuSparse handle
	cusparseCreate(&handle);	
	
	// if we need bins, do some calculations:
	binsFlag = false;
	if (nFibers > 1) binsFlag = true;
	if (binsFlag) {		
		bins.sizeBins = inputParams("IBM_FIBERS/sizeBins",2.0);
		bins.binMax = inputParams("IBM_FIBERS/binMax",1);			
		bins.numBins.x = int(floor(N.x/bins.sizeBins));
	    bins.numBins.y = int(floor(N.y/bins.sizeBins));
	    bins.numBins.z = int(floor(N.z/bins.sizeBins));
		bins.nBins = bins.numBins.x*bins.numBins.y*bins.numBins.z;
		bins.nnbins = 26;		
	}	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_fibers_ibm3D::~class_fibers_ibm3D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_fibers_ibm3D::allocate()
{
	// allocate array memory (host):
	beadsH = (beadfiber*)malloc(nBeads*sizeof(beadfiber));
	edgesH = (edgefiber*)malloc(nEdges*sizeof(edgefiber));
	fibersH = (fiber*)malloc(nFibers*sizeof(fiber));
							
	// allocate array memory (device):	
	cudaMalloc((void **) &beads, nBeads*sizeof(beadfiber));
	cudaMalloc((void **) &edges, nEdges*sizeof(edgefiber));
	cudaMalloc((void **) &fibers, nFibers*sizeof(fiber));
	cudaMalloc((void **) &rngState, nBeads*sizeof(curandState));
	cudaMalloc((void **) &xp1, nBeads*sizeof(float));
	cudaMalloc((void **) &yp1, nBeads*sizeof(float));
	cudaMalloc((void **) &zp1, nBeads*sizeof(float));
	cudaMalloc((void **) &AuTen, nEdges*sizeof(float));
	cudaMalloc((void **) &AcTen, nEdges*sizeof(float));
	cudaMalloc((void **) &AlTen, nEdges*sizeof(float));
	cudaMalloc((void **) &T, nEdges*sizeof(float));
	cudaMalloc((void **) &Au, nBeads*sizeof(float));
	cudaMalloc((void **) &Ac, nBeads*sizeof(float));
	cudaMalloc((void **) &Al, nBeads*sizeof(float));
	
	
	if (binsFlag) {		
		cudaMalloc((void **) &bins.binMembers, bins.nBins*bins.binMax*sizeof(int));
		cudaMalloc((void **) &bins.binOccupancy, bins.nBins*sizeof(int));
		cudaMalloc((void **) &bins.binMap, bins.nBins*26*sizeof(int));		
	}	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_fibers_ibm3D::deallocate()
{
	// free array memory (host):
	free(beadsH);
	free(edgesH);
	free(fibersH);
					
	// free array memory (device):
	cudaFree(beads);
	cudaFree(edges);
	cudaFree(fibers);
	cudaFree(rngState);
	cudaFree(xp1);
	cudaFree(yp1);
	cudaFree(zp1);
	cudaFree(AuTen);
	cudaFree(AcTen);
	cudaFree(AlTen);
	cudaFree(T);
	cudaFree(Au);
	cudaFree(Ac);
	cudaFree(Al);
	cudaFree(bufferTen);
	cusparseDestroy(handle);
	if (binsFlag) {		
		cudaFree(bins.binMembers);
		cudaFree(bins.binOccupancy);
		cudaFree(bins.binMap);				
	}		
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_fibers_ibm3D::memcopy_host_to_device()
{
	cudaMemcpy(beads, beadsH, sizeof(beadfiber)*nBeads, cudaMemcpyHostToDevice);	
	cudaMemcpy(edges, edgesH, sizeof(edgefiber)*nEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(fibers, fibersH, sizeof(fiber)*nFibers, cudaMemcpyHostToDevice);	
}
	


// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_fibers_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(beadsH, beads, sizeof(beadfiber)*nBeads, cudaMemcpyDeviceToHost);
	cudaMemcpy(fibersH, fibers, sizeof(fiber)*nFibers, cudaMemcpyDeviceToHost);
	
	// unwrap coordinate positions:
	unwrap_bead_coordinates(); 
}



// --------------------------------------------------------
// Determine cuSparse buffer sizes for tridiagonal solves:
// --------------------------------------------------------

void class_fibers_ibm3D::cuSparse_buffer_sizes()
{
	// buffer for tension solve:
	int m = nEdges;
	int n = 1;
	cusparseStatus_t Status = cusparseSgtsv2_bufferSizeExt(handle,m,n,AlTen,AcTen,AuTen,T,m,&bufferSizeTen);
    cudaMalloc(&bufferTen, bufferSizeTen);
	
	// buffer for position solve:
	m = nBeads;
	n = 1;
	cusparseStatus_t Status2 = cusparseSgtsv2_bufferSizeExt(handle,m,n,Al,Ac,Au,xp1,m,&bufferSize);
    cudaMalloc(&buffer, bufferSize);
}











// **********************************************************************************************
// Initialization Stuff...
// **********************************************************************************************












// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void class_fibers_ibm3D::create_first_fiber()
{
	// set up the bead information for first fiber:
	for (int i=0; i<nBeadsPerFiber; i++) {
		beadsH[i].r.x = 0.0 + float(i)*dS;
		beadsH[i].r.y = 0.0;
		beadsH[i].r.z = 0.0;
		beadsH[i].rm1 = beadsH[i].r;
		beadsH[i].v = make_float3(0.0f);
		beadsH[i].f = make_float3(0.0f);
		beadsH[i].fiberID = 0;
		beadsH[i].posID = 0;
		if (i==0) beadsH[i].posID = 1;                 // left end
		if (i==nBeadsPerFiber-1) beadsH[i].posID = 2;  // right end      
	}
	
	// set up the edge information for first fiber:
	for (int i=0; i<nEdgesPerFiber; i++) {
		edgesH[i].b0 = i;
		edgesH[i].b1 = i+1;
		edgesH[i].posID = 0;
		if (i==0) edgesH[i].posID = 1;                 // left end
		if (i==nEdgesPerFiber-1) edgesH[i].posID = 2;  // right end
	}
	
	// set up indices for ALL fiber:
	for (int f=0; f<nFibers; f++) {
		fibersH[f].nBeads = nBeadsPerFiber;
		fibersH[f].nEdges = nEdgesPerFiber;
		fibersH[f].indxB0 = f*nBeadsPerFiber;   // start index for beads
		fibersH[f].indxE0 = f*nEdgesPerFiber;   // start index for edges
		fibersH[f].headBead = f*nBeadsPerFiber; // head bead (first bead)
	}
}



// --------------------------------------------------------
// Setters:
// --------------------------------------------------------

void class_fibers_ibm3D::set_pbcFlag(int x, int y, int z)
{
	pbcFlag.x = x; pbcFlag.y = y; pbcFlag.z = z;
}

void class_fibers_ibm3D::set_gamma(float val)
{
	for (int f=0; f<nFibers; f++) fibersH[f].gam = val;
}

void class_fibers_ibm3D::set_fibers_radii(float rad)
{
	// set radius for ALL cells:
	for (int f=0; f<nFibers; f++) fibersH[f].rad = rad;
}

int class_fibers_ibm3D::get_max_array_size()
{
	// return the maximum array size:
	int maxSize = max(nBeads,nEdges);
	if (binsFlag) {
		if (bins.nBins > maxSize) maxSize = bins.nBins;
	}
	return maxSize;
}



// --------------------------------------------------------
// Assign the fiber ID to every node:
// --------------------------------------------------------

void class_fibers_ibm3D::assign_fiberIDs_to_beads()
{
	for (int f=0; f<nFibers; f++) {
		int istr = fibersH[f].indxB0;
		int iend = istr + fibersH[f].nBeads;
		for (int i=istr; i<iend; i++) beadsH[i].fiberID = f;
	}
}



// --------------------------------------------------------
// Duplicate the first cell mesh information to all fibers:
// --------------------------------------------------------

void class_fibers_ibm3D::duplicate_fibers()
{
	if (nFibers > 1) {
		for (int f=1; f<nFibers; f++) {
			
			// skip if fiber 0 is different than fiber f:
			if (fibersH[0].nBeads != fibersH[f].nBeads ||
				fibersH[0].nEdges != fibersH[f].nEdges) {
					cout << "duplicate filaments error: fibers have different nBeads, nEdges" << endl;
					continue;
			}
			
			// copy bead information:
			for (int i=0; i<fibersH[0].nBeads; i++) {
				int ii = i + fibersH[f].indxB0;
				beadsH[ii].r = beadsH[i].r;
				beadsH[ii].v = beadsH[i].v;
				beadsH[ii].f = beadsH[i].f;
				beadsH[ii].rm1 = beadsH[i].rm1;
				beadsH[ii].fiberID = f;
				beadsH[ii].posID = beadsH[i].posID;				
			}
			
			// copy edge info:
			for (int i=0; i<fibersH[0].nEdges; i++) {
				int ii = i + fibersH[f].indxE0;
				edgesH[ii].b0 = edgesH[i].b0 + fibersH[f].indxB0;
				edgesH[ii].b1 = edgesH[i].b1 + fibersH[f].indxB0;
				edgesH[ii].posID = edgesH[i].posID;
			}
		}
	}
	
}



// --------------------------------------------------------
// randomize fiber positions and orientations:
// --------------------------------------------------------

void class_fibers_ibm3D::randomize_fibers(float sepWall)
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beadfiber)*nBeads, cudaMemcpyDeviceToHost);
	
	// assign random position and orientation to each fiber:
	for (int f=0; f<nFibers; f++) {
		float3 shift = make_float3(0.0,0.0,0.0);
		// get random position
		shift.x = (float)rand()/RAND_MAX*Box.x;
		shift.y = sepWall + (float)rand()/RAND_MAX*(Box.y-2.0*sepWall);
		shift.z = sepWall + (float)rand()/RAND_MAX*(Box.z-2.0*sepWall);
		rotate_and_shift_bead_positions(f,shift.x,shift.y,shift.z);
	}
	
	// copy bead positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(beadfiber)*nBeads, cudaMemcpyHostToDevice);	
}



// --------------------------------------------------------
// randomize fiber positions, but all oriented in x-dir:
// --------------------------------------------------------

void class_fibers_ibm3D::randomize_fibers_xdir_alligned(float sepWall)
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beadfiber)*nBeads, cudaMemcpyDeviceToHost);
	
	// assign random position and orientation to each filament:
	for (int f=0; f<nFibers; f++) {
		float3 shift = make_float3(0.0,0.0,0.0);
		// get random position
		shift.x = (float)rand()/RAND_MAX*Box.x;
		shift.y = sepWall + (float)rand()/RAND_MAX*(Box.y-2.0*sepWall);
		shift.z = sepWall + (float)rand()/RAND_MAX*(Box.z-2.0*sepWall);
		shift_bead_positions(f,shift.x,shift.y,shift.z);
	}
	
	// copy bead positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(beadfiber)*nBeads, cudaMemcpyHostToDevice);	
}



// --------------------------------------------------------
// calculate separation distance using PBCs:
// --------------------------------------------------------

float class_fibers_ibm3D::calc_separation_pbc(float3 r1, float3 r2)
{
	float3 dr = r1 - r2;
	dr -= roundf(dr/Box)*Box;
	return length(dr);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_fibers_ibm3D::shift_bead_positions(int fID, float xsh, float ysh, float zsh)
{
	int istr = fibersH[fID].indxB0;
	int iend = istr + fibersH[fID].nBeads;
	for (int i=istr; i<iend; i++) {
		beadsH[i].r.x += xsh;
		beadsH[i].r.y += ysh;
		beadsH[i].r.z += zsh;
		beadsH[i].rm1 = beadsH[i].r;
	}
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_fibers_ibm3D::rotate_and_shift_bead_positions(int fID, float xsh, float ysh, float zsh)
{
	// random rotation angles:
	float a = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // alpha
	float b = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // beta
	float g = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // gamma
	
	// update node positions:
	int istr = fibersH[fID].indxB0;
	int iend = istr + fibersH[fID].nBeads;
	for (int i=istr; i<iend; i++) {
		// rotate:
		float xrot = beadsH[i].r.x*(cos(a)*cos(b)) + beadsH[i].r.y*(cos(a)*sin(b)*sin(g)-sin(a)*cos(g)) + beadsH[i].r.z*(cos(a)*sin(b)*cos(g)+sin(a)*sin(g));
		float yrot = beadsH[i].r.x*(sin(a)*cos(b)) + beadsH[i].r.y*(sin(a)*sin(b)*sin(g)+cos(a)*cos(g)) + beadsH[i].r.z*(sin(a)*sin(b)*cos(g)-cos(a)*sin(g));
		float zrot = beadsH[i].r.x*(-sin(b))       + beadsH[i].r.y*(cos(b)*sin(g))                      + beadsH[i].r.z*(cos(b)*cos(g));
		// shift:		 
		beadsH[i].r.x = xrot + xsh;
		beadsH[i].r.y = yrot + ysh;
		beadsH[i].r.z = zrot + zsh;
		beadsH[i].rm1 = beadsH[i].r;	
	}
}



// --------------------------------------------------------
// Initialize fiber with curved profile:
// --------------------------------------------------------

void class_fibers_ibm3D::initialize_fiber_curved()
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beadfiber)*nBeads, cudaMemcpyDeviceToHost);
	
	// initialize fiber (if there is only one) with a curved profile:
	if (nFibers == 1) {
		float theta = -15.0f*M_PI/180.0f;
		float deltatheta = -2.0f*theta/(nBeadsPerFiber - 2);
		
		beadsH[0].r = make_float3(100.0,7.5,7.5);
		beadsH[0].rm1 = beadsH[0].r;
		
		for (int i=1; i<nBeadsPerFiber; i++) {
			beadsH[i].r.x = beadsH[i-1].r.x + dS*cos(theta);
			beadsH[i].r.y = beadsH[i-1].r.y + dS*sin(theta);
			beadsH[i].r.z = beadsH[i-1].r.z;
			beadsH[i].rm1 = beadsH[i].r;
			theta += deltatheta;
		}
	}
	
	// initialize fiber (if there is only one) with a curved profile:
	if (nFibers == 2) {
		float theta = -15.0f*M_PI/180.0f;
		float deltatheta = -2.0f*theta/(nBeadsPerFiber - 2);
		
		beadsH[0].r = make_float3(100.0,7.5,7.5);
		beadsH[0].rm1 = beadsH[0].r;
		
		// first filament:		
		for (int i=1; i<nBeadsPerFiber; i++) {
			beadsH[i].r.x = beadsH[i-1].r.x + dS*cos(theta);
			beadsH[i].r.y = beadsH[i-1].r.y + dS*sin(theta);
			beadsH[i].r.z = beadsH[i-1].r.z;
			beadsH[i].rm1 = beadsH[i].r;
			theta += deltatheta;
		}
		
		// make a copy for the second filament:
		for (int i=nBeadsPerFiber; i<nBeads; i++) {
			beadsH[i].r.x = beadsH[i-nBeadsPerFiber].r.x + 50.0;
			beadsH[i].r.y = beadsH[i-nBeadsPerFiber].r.y;
			beadsH[i].r.z = beadsH[i-nBeadsPerFiber].r.z;
			beadsH[i].rm1 = beadsH[i].r;
		}
		
	}
	
	// copy bead positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(beadfiber)*nBeads, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Calculate wall forces:
// --------------------------------------------------------

void class_fibers_ibm3D::compute_wall_forces(int nBlocks, int nThreads)
{
	if (pbcFlag.y==0 && pbcFlag.z==1) wall_forces_ydir(nBlocks,nThreads);
	if (pbcFlag.y==1 && pbcFlag.z==0) wall_forces_zdir(nBlocks,nThreads);
	if (pbcFlag.y==0 && pbcFlag.z==0) wall_forces_ydir_zdir(nBlocks,nThreads);
} 



// --------------------------------------------------------
// Step forward in time:
// --------------------------------------------------------

void class_fibers_ibm3D::stepIBM(class_scsp_D3Q19& lbm, int nBlocks, int nThreads)
{	
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);
	
	// zero bead forces:
	zero_bead_forces(nBlocks,nThreads);	
	
	// calculate bead velocity: (r(n) - r(n-1))/dt
	calculate_bead_velocity(nBlocks,nThreads);
		
	// calculate hydrodynamic fluid forces:
	lbm.hydrodynamic_forces_fibers_IBM_LBM(nBlocks,nThreads,beads,nBeads); 
	
	// compute wall forces:
	compute_wall_forces(nBlocks,nThreads);
	
	// compute non-bonded forces:
	// ---------------------------
	
	// unwrap bead coordinates:
	unwrap_bead_coordinates(nBlocks,nThreads);
	
	// calculate r-star: 2r(n) - r(n-1)
	update_rstar(nBlocks,nThreads);
	
	// calculate bending forces:
	compute_Laplacian(nBlocks,nThreads);
	compute_bending_force(nBlocks,nThreads);
	
	// calculate tension in fibers:
	compute_tension_RHS(nBlocks,nThreads);
	compute_tension_tridiag(nBlocks,nThreads);
	solve_tridiagonal_tension();
	
	// calculate node positions at step n+1:
	compute_bead_update_matrices(nBlocks,nThreads);
	solve_tridiagonal_positions();
	
	// update bead positions:
	update_bead_positions(nBlocks,nThreads); 
	
	// re-wrap bead coordinates: 
	wrap_bead_coordinates(nBlocks,nThreads);			
} 













// **********************************************************************************************
// Calls to CUDA kernels for main calculations
// **********************************************************************************************














// --------------------------------------------------------
// Call to "zero_bead_forces_fibers_IBM3D" kernel:
// --------------------------------------------------------

void class_fibers_ibm3D::zero_bead_forces(int nBlocks, int nThreads)
{
	zero_bead_forces_fibers_IBM3D
	<<<nBlocks,nThreads>>> (beads,nBeads);
}



// --------------------------------------------------------
// Call to "calculate_bead_velocity_fibers_IBM3D" kernel:
// --------------------------------------------------------

void class_fibers_ibm3D::calculate_bead_velocity(int nBlocks, int nThreads)
{
	calculate_bead_velocity_fibers_IBM3D
	<<<nBlocks,nThreads>>> (beads,dt,nBeads);
}



// --------------------------------------------------------
// Call to "enforce_max_node_force_IBM3D" kernel:
// --------------------------------------------------------

void class_fibers_ibm3D::update_rstar(int nBlocks, int nThreads)
{
	update_rstar_fibers_IBM3D
	<<<nBlocks,nThreads>>> (beads,nBeads);
}



// --------------------------------------------------------
// Call to "update_bead_positions_fibers_IBM3D" kernel:
// --------------------------------------------------------

void class_fibers_ibm3D::update_bead_positions(int nBlocks, int nThreads)
{
	update_bead_positions_fibers_IBM3D
	<<<nBlocks,nThreads>>> (beads,xp1,yp1,zp1,nBeads);
}



// --------------------------------------------------------
// Call to "compute_Laplacian_fibers_IBM3D" kernel:
// --------------------------------------------------------

void class_fibers_ibm3D::compute_Laplacian(int nBlocks, int nThreads)
{
	compute_Laplacian_fibers_IBM3D
	<<<nBlocks,nThreads>>> (beads,dS,nBeads);
}



// --------------------------------------------------------
// Call to "compute_bending_force_fibers_IBM3D" kernel:
// --------------------------------------------------------

void class_fibers_ibm3D::compute_bending_force(int nBlocks, int nThreads)
{
	compute_bending_force_fibers_IBM3D
	<<<nBlocks,nThreads>>> (beads,dS,gam,nBeads);
}



// --------------------------------------------------------
// Call to "compute_tension_RHS_fibers_IBM3D" kernel:
// {Notice that we are sending the array 'T' to store the
// RHS values of the tension equation.  This is because
// cuSparse replaces the solution with the RHS array}
// --------------------------------------------------------

void class_fibers_ibm3D::compute_tension_RHS(int nBlocks, int nThreads)
{
	compute_tension_RHS_fibers_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,T,dS,dt,nEdges);
}



// --------------------------------------------------------
// Call to "compute_tension_tridiag_fibers_IBM3D" kernel:
// --------------------------------------------------------

void class_fibers_ibm3D::compute_tension_tridiag(int nBlocks, int nThreads)
{
	compute_tension_tridiag_fibers_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,AuTen,AcTen,AlTen,dS,dt,nEdges);
}



// --------------------------------------------------------
// Call to "compute_bead_update_matrices_fibers_IBM3D" kernel:
// {Notice that we are sending the arrays 'xp1', 'yp1', and 'zp1'
// to store the RHS values of the position equation.  This is because
// cuSparse replaces the solution with the RHS array}
// --------------------------------------------------------

void class_fibers_ibm3D::compute_bead_update_matrices(int nBlocks, int nThreads)
{
	compute_bead_update_matrices_fibers_IBM3D
	<<<nBlocks,nThreads>>> (beads,T,xp1,yp1,zp1,Au,Ac,Al,dS,dt,nBeads);
}



// --------------------------------------------------------
// Call to "unwrap_bead_coordinates_IBM3D" kernel:
// --------------------------------------------------------

void class_fibers_ibm3D::unwrap_bead_coordinates(int nBlocks, int nThreads)
{
	unwrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,fibers,Box,pbcFlag,nBeads);
}



// --------------------------------------------------------
// Call to "wrap_bead_coordinates_IBM3D" kernel:
// --------------------------------------------------------

void class_fibers_ibm3D::wrap_bead_coordinates(int nBlocks, int nThreads)
{
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);
}



// --------------------------------------------------------
// Call to kernel that builds the binMap array:
// --------------------------------------------------------

void class_fibers_ibm3D::build_binMap(int nBlocks, int nThreads)
{
	/*
	if (nFibers > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;	
		build_binMap_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (bins);		
	}
	*/	
}



// --------------------------------------------------------
// Call to kernel that resets bin lists:
// --------------------------------------------------------

void class_fibers_ibm3D::reset_bin_lists(int nBlocks, int nThreads)
{
	/*
	if (nFibers > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;		
		reset_bin_lists_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (bins);
	}
	*/	
}



// --------------------------------------------------------
// Call to kernel that builds bin lists:
// --------------------------------------------------------

void class_fibers_ibm3D::build_bin_lists(int nBlocks, int nThreads)
{
	/*
	if (nFibers > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;		
		build_bin_lists_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (beads,bins,nBeads);		
	}
	*/	
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_fibers_ibm3D::nonbonded_bead_interactions(int nBlocks, int nThreads)
{
	/*
	if (nFibers > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;								
		nonbonded_bead_interactions_IBM3D
		<<<nBlocks,nThreads>>> (beads,bins,repA,repD,nBeads,Box,pbcFlag);
	}
	*/	
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir:
// --------------------------------------------------------

void class_fibers_ibm3D::wall_forces_ydir(int nBlocks, int nThreads)
{
	bead_wall_forces_ydir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in z-dir:
// --------------------------------------------------------

void class_fibers_ibm3D::wall_forces_zdir(int nBlocks, int nThreads)
{
	bead_wall_forces_zdir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir
// and z-dir:
// --------------------------------------------------------

void class_fibers_ibm3D::wall_forces_ydir_zdir(int nBlocks, int nThreads)
{
	bead_wall_forces_ydir_zdir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}










// **********************************************************************************************
// Calls to CUSPARSE to solve for fiber tension and updated bead positions
// **********************************************************************************************











// --------------------------------------------------------
// tridiagonal solve for tension in fiber:
// {note that the solution is returned in the array 'T'
//  which stores the RHS values before the solve}
// --------------------------------------------------------

void class_fibers_ibm3D::solve_tridiagonal_tension()
{
	int m = nEdges;
	int n = 1;
	cusparseStatus_t Status = cusparseSgtsv2(handle,m,n,AlTen,AcTen,AuTen,T,m,bufferTen);
	//cudaDeviceSynchronize();
}



// --------------------------------------------------------
// tridiagonal solve for bead positions at step n+1:
// {note that the solution is returned in the arrays 'xp1', 
//  'yp1', and 'zp1' which store the RHS values before the solve}
// --------------------------------------------------------

void class_fibers_ibm3D::solve_tridiagonal_positions()
{
	int m = nBeads;
	int n = 1;
	cusparseStatus_t StatusX = cusparseSgtsv2(handle,m,n,Al,Ac,Au,xp1,m,buffer);
	cusparseStatus_t StatusY = cusparseSgtsv2(handle,m,n,Al,Ac,Au,yp1,m,buffer);
	cusparseStatus_t StatusZ = cusparseSgtsv2(handle,m,n,Al,Ac,Au,zp1,m,buffer);
	//cudaDeviceSynchronize();
}












// **********************************************************************************************
// Analysis and Geometry calculations done by the host (CPU)
// **********************************************************************************************












// --------------------------------------------------------
// Write IBM output to file:
// --------------------------------------------------------

void class_fibers_ibm3D::write_output(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_3D_fibers(tagname,tagnum,
	nBeads,nEdges,beadsH,edgesH);
}



// --------------------------------------------------------
// Unwrap bead coordinates based on difference between bead
// position and the filament's head bead position:
// --------------------------------------------------------

void class_fibers_ibm3D::unwrap_bead_coordinates()
{
	for (int i=0; i<nBeads; i++) {
		int f = beadsH[i].fiberID;
		int j = fibersH[f].headBead;
		float3 rij = beadsH[j].r - beadsH[i].r;
		beadsH[i].r = beadsH[i].r + roundf(rij/Box)*Box*pbcFlag; // PBC's
		beadsH[i].rm1 = beadsH[i].rm1 + roundf(rij/Box)*Box*pbcFlag; // PBC's
	}	
}


