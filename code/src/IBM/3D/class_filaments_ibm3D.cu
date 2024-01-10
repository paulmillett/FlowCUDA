 
# include "class_filaments_ibm3D.cuh"
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











// **********************************************************************************************
// Constructor, destructor, and array allocations...
// **********************************************************************************************










// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_filaments_ibm3D::class_filaments_ibm3D()
{
	// get some parameters:
	GetPot inputParams("input.dat");
	
	// mesh attributes	
	nBeadsPerFilam = inputParams("IBM_FILAMS/nBeadsPerFilam",0);
	nEdgesPerFilam = inputParams("IBM_FILAMS/nEdgesPerFilam",0);
	nFilams = inputParams("IBM_FILAMS/nFilams",1);
	nBeads = nBeadsPerFilam*nFilams;
	nEdges = nEdgesPerFilam*nFilams;
	
	// mechanical properties
	dt = inputParams("Time/dt",1.0);
	ks = inputParams("IBM_FILAMS/ks",0.0);
	kb = inputParams("IBM_FILAMS/kb",0.0);
	L0 = inputParams("IBM_FILAMS/L0",0.5);
	kT = inputParams("IBM_FILAMS/kT",0.0);
	repA = inputParams("IBM_FILAMS/repA",0.0);
	repD = inputParams("IBM_FILAMS/repD",0.0);
	repA_bn = inputParams("IBM_FILAMS/repA_bn",0.0);
	repD_bn = inputParams("IBM_FILAMS/repD_bn",0.0);
	beadFmax = inputParams("IBM_FILAMS/beadFmax",1000.0);
	gam = inputParams("IBM_FILAMS/gamma",0.1);
	Mob = inputParams("IBM_FILAMS/Mob",0.5);
	ibmUpdate = inputParams("IBM/ibmUpdate","verlet");
	forceModel = inputParams("IBM_FILAMS/forceModel","spring");
	fricBead = 6.0*M_PI*(1.0/6.0)*repD;  // friction coefficient per bead (assume visc=1/6)
	noisekT = sqrt(2.0*kT*gam/dt);       // thernal noise strength
	
	// domain attributes
	N.x = inputParams("Lattice/Nx",1);
	N.y = inputParams("Lattice/Ny",1);
	N.z = inputParams("Lattice/Nz",1);	
	Box.x = float(N.x);   // assume dx=1
	Box.y = float(N.y);
	Box.z = float(N.z);
	pbcFlag = make_int3(1,1,1);
			
	// if we need bins, do some calculations:
	binsFlag = false;
	if (nFilams > 1) binsFlag = true;
	if (binsFlag) {		
		bins.sizeBins = inputParams("IBM_FILAMS/sizeBins",2.0);
		bins.binMax = inputParams("IBM_FILAMS/binMax",1);			
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

class_filaments_ibm3D::~class_filaments_ibm3D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_filaments_ibm3D::allocate()
{
	// allocate array memory (host):
	beadsH = (bead*)malloc(nBeads*sizeof(bead));
	edgesH = (edgefilam*)malloc(nEdges*sizeof(edgefilam));
	filamsH = (filament*)malloc(nFilams*sizeof(filament));
							
	// allocate array memory (device):	
	cudaMalloc((void **) &beads, nBeads*sizeof(bead));
	cudaMalloc((void **) &edges, nEdges*sizeof(edgefilam));
	cudaMalloc((void **) &filams, nFilams*sizeof(filament));
	cudaMalloc((void **) &rngState, nBeads*sizeof(curandState));
	if (binsFlag) {		
		cudaMalloc((void **) &bins.binMembers, bins.nBins*bins.binMax*sizeof(int));
		cudaMalloc((void **) &bins.binOccupancy, bins.nBins*sizeof(int));
		cudaMalloc((void **) &bins.binMap, bins.nBins*26*sizeof(int));		
	}	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_filaments_ibm3D::deallocate()
{
	// free array memory (host):
	free(beadsH);
	free(edgesH);
	free(filamsH);
					
	// free array memory (device):
	cudaFree(beads);
	cudaFree(edges);
	cudaFree(filams);
	cudaFree(rngState);
	if (binsFlag) {		
		cudaFree(bins.binMembers);
		cudaFree(bins.binOccupancy);
		cudaFree(bins.binMap);				
	}		
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_filaments_ibm3D::memcopy_host_to_device()
{
	cudaMemcpy(beads, beadsH, sizeof(bead)*nBeads, cudaMemcpyHostToDevice);	
	cudaMemcpy(edges, edgesH, sizeof(edgefilam)*nEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(filams, filamsH, sizeof(filament)*nFilams, cudaMemcpyHostToDevice);	
}
	


// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_filaments_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(beadsH, beads, sizeof(bead)*nBeads, cudaMemcpyDeviceToHost);
	cudaMemcpy(filamsH, filams, sizeof(filament)*nFilams, cudaMemcpyDeviceToHost);
	
	// unwrap coordinate positions:
	unwrap_bead_coordinates(); 
}











// **********************************************************************************************
// Initialization Stuff...
// **********************************************************************************************












// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void class_filaments_ibm3D::create_first_filament()
{
	// set up the bead information for first filament:
	for (int i=0; i<nBeadsPerFilam; i++) {
		beadsH[i].r.x = 0.0 + float(i)*L0;
		beadsH[i].r.y = 0.0;
		beadsH[i].r.z = 0.0;
		beadsH[i].v = make_float3(0.0f);
		beadsH[i].f = make_float3(0.0f);
		beadsH[i].filamID = 0;
	}
	
	// set up the edge information for first filament:
	for (int i=0; i<nEdgesPerFilam; i++) {
		edgesH[i].b0 = i;
		edgesH[i].b1 = i+1;
		edgesH[i].length0 = L0;
	}
	
	// set up indices for ALL filament:
	for (int f=0; f<nFilams; f++) {
		filamsH[f].nBeads = nBeadsPerFilam;
		filamsH[f].nEdges = nEdgesPerFilam;
		filamsH[f].indxB0 = f*nBeadsPerFilam;   // start index for beads
		filamsH[f].indxE0 = f*nEdgesPerFilam;   // start index for edges
		filamsH[f].headBead = f*nBeadsPerFilam; // head bead (first bead)
	}
}



// --------------------------------------------------------
// Setters:
// --------------------------------------------------------

void class_filaments_ibm3D::set_pbcFlag(int x, int y, int z)
{
	pbcFlag.x = x; pbcFlag.y = y; pbcFlag.z = z;
}

void class_filaments_ibm3D::set_ks(float val)
{
	for (int f=0; f<nFilams; f++) filamsH[f].ks = val;
}

void class_filaments_ibm3D::set_kb(float val)
{
	for (int f=0; f<nFilams; f++) filamsH[f].kb = val;
}

void class_filaments_ibm3D::set_fp(float val)
{
	for (int f=0; f<nFilams; f++) filamsH[f].fp = val;
}

void class_filaments_ibm3D::set_up(float val)
{
	for (int f=0; f<nFilams; f++) filamsH[f].up = val;
}

void class_filaments_ibm3D::set_filams_mechanical_props(float ks, float kb)
{
	// set props for ALL cells:
	for (int f=0; f<nFilams; f++) {
		filamsH[f].ks = ks;
		filamsH[f].kb = kb;
	}
}

void class_filaments_ibm3D::set_filam_mechanical_props(int fID, float ks, float kb)
{
	// set props for ONE cell:
	filamsH[fID].ks = ks;
	filamsH[fID].kb = kb;
}

void class_filaments_ibm3D::set_filams_radii(float rad)
{
	// set radius for ALL cells:
	for (int f=0; f<nFilams; f++) filamsH[f].rad = rad;
}

void class_filaments_ibm3D::set_filam_radius(int fID, float rad)
{
	// set radius for ONE cell:
	filamsH[fID].rad = rad;
}

void class_filaments_ibm3D::set_filams_types(int val)
{
	// set filamType for ALL filaments:
	for (int f=0; f<nFilams; f++) filamsH[f].filamType = val;
}

void class_filaments_ibm3D::set_filam_type(int fID, int val)
{
	// set filamType for ONE filament:
	filamsH[fID].filamType = val;
}

int class_filaments_ibm3D::get_max_array_size()
{
	// return the maximum array size:
	int maxSize = max(nBeads,nEdges);
	if (binsFlag) {
		if (bins.nBins > maxSize) maxSize = bins.nBins;
	}
	return maxSize;
}



// --------------------------------------------------------
// Assign the cell ID to every node:
// --------------------------------------------------------

void class_filaments_ibm3D::assign_filamIDs_to_beads()
{
	for (int f=0; f<nFilams; f++) {
		int istr = filamsH[f].indxB0;
		int iend = istr + filamsH[f].nBeads;
		for (int i=istr; i<iend; i++) beadsH[i].filamID = f;
	}
}



// --------------------------------------------------------
// Duplicate the first cell mesh information to all cells:
// --------------------------------------------------------

void class_filaments_ibm3D::duplicate_filaments()
{
	if (nFilams > 1) {
		for (int f=1; f<nFilams; f++) {
			
			// skip if filam 0 is different than filam f:
			if (filamsH[0].nBeads != filamsH[f].nBeads ||
				filamsH[0].nEdges != filamsH[f].nEdges) {
					cout << "duplicate filaments error: filaments have different nBeads, nEdges" << endl;
					continue;
			}
			
			// copy bead information:
			for (int i=0; i<filamsH[0].nBeads; i++) {
				int ii = i + filamsH[f].indxB0;
				beadsH[ii].r = beadsH[i].r;
				beadsH[ii].v = beadsH[i].v;
				beadsH[ii].f = beadsH[i].f;
				beadsH[ii].filamID = f;
			}
			// copy edge info:
			for (int i=0; i<filamsH[0].nEdges; i++) {
				int ii = i + filamsH[f].indxE0;
				edgesH[ii].b0 = edgesH[i].b0 + filamsH[f].indxB0;
				edgesH[ii].b1 = edgesH[i].b1 + filamsH[f].indxB0;
				edgesH[ii].length0 = edgesH[i].length0;
			}
		}
	}
	
}



// --------------------------------------------------------
// randomize cell positions and orientations:
// --------------------------------------------------------

void class_filaments_ibm3D::randomize_filaments(float sepWall)
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(bead)*nBeads, cudaMemcpyDeviceToHost);
	
	// assign random position and orientation to each filament:
	for (int f=0; f<nFilams; f++) {
		float3 shift = make_float3(0.0,0.0,0.0);
		// get random position
		shift.x = (float)rand()/RAND_MAX*Box.x;
		shift.y = sepWall + (float)rand()/RAND_MAX*(Box.y-2.0*sepWall);
		shift.z = sepWall + (float)rand()/RAND_MAX*(Box.z-2.0*sepWall);
		rotate_and_shift_bead_positions(f,shift.x,shift.y,shift.z);
	}
	
	// copy node positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(bead)*nBeads, cudaMemcpyHostToDevice);	
}



// --------------------------------------------------------
// randomize cell positions and orientations:
// --------------------------------------------------------

void class_filaments_ibm3D::randomize_filaments_inside_sphere(float xs, float ys, float zs, float rs, float sepWall)
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(bead)*nBeads, cudaMemcpyDeviceToHost);
	
	// assign random position and orientation to each filament inside a sphere:
	float3 sphere = make_float3(xs,ys,zs);
	for (int f=0; f<nFilams; f++) {
		float3 shift = make_float3(0.0,0.0,0.0);
		bool outsideSphere = true;
		while (outsideSphere) {
			// get random position
			shift.x = (float)rand()/RAND_MAX*Box.x;
			shift.y = (float)rand()/RAND_MAX*Box.y;
			shift.z = (float)rand()/RAND_MAX*Box.z;
			float r = length(shift - sphere);
			if (r <= (rs - sepWall)) outsideSphere = false;
		}
		rotate_and_shift_bead_positions(f,shift.x,shift.y,shift.z);
	}
		
	// copy node positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(bead)*nBeads, cudaMemcpyHostToDevice);	
}



// --------------------------------------------------------
// calculate separation distance using PBCs:
// --------------------------------------------------------

float class_filaments_ibm3D::calc_separation_pbc(float3 r1, float3 r2)
{
	float3 dr = r1 - r2;
	dr -= roundf(dr/Box)*Box;
	return length(dr);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_filaments_ibm3D::shift_bead_positions(int fID, float xsh, float ysh, float zsh)
{
	int istr = filamsH[fID].indxB0;
	int iend = istr + filamsH[fID].nBeads;
	for (int i=istr; i<iend; i++) {
		beadsH[i].r.x += xsh;
		beadsH[i].r.y += ysh;
		beadsH[i].r.z += zsh;
	}
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_filaments_ibm3D::rotate_and_shift_bead_positions(int fID, float xsh, float ysh, float zsh)
{
	// random rotation angles:
	float a = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // alpha
	float b = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // beta
	float g = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // gamma
	
	// update node positions:
	int istr = filamsH[fID].indxB0;
	int iend = istr + filamsH[fID].nBeads;
	for (int i=istr; i<iend; i++) {
		// rotate:
		float xrot = beadsH[i].r.x*(cos(a)*cos(b)) + beadsH[i].r.y*(cos(a)*sin(b)*sin(g)-sin(a)*cos(g)) + beadsH[i].r.z*(cos(a)*sin(b)*cos(g)+sin(a)*sin(g));
		float yrot = beadsH[i].r.x*(sin(a)*cos(b)) + beadsH[i].r.y*(sin(a)*sin(b)*sin(g)+cos(a)*cos(g)) + beadsH[i].r.z*(sin(a)*sin(b)*cos(g)-cos(a)*sin(g));
		float zrot = beadsH[i].r.x*(-sin(b))       + beadsH[i].r.y*(cos(b)*sin(g))                      + beadsH[i].r.z*(cos(b)*cos(g));
		// shift:		 
		beadsH[i].r.x = xrot + xsh;
		beadsH[i].r.y = yrot + ysh;
		beadsH[i].r.z = zrot + zsh;			
	}
}



// --------------------------------------------------------
// Calculate wall forces:
// --------------------------------------------------------

void class_filaments_ibm3D::compute_wall_forces(int nBlocks, int nThreads)
{
	if (pbcFlag.y==0 && pbcFlag.z==1) wall_forces_ydir(nBlocks,nThreads);
	if (pbcFlag.y==1 && pbcFlag.z==0) wall_forces_zdir(nBlocks,nThreads);
	if (pbcFlag.y==0 && pbcFlag.z==0) wall_forces_ydir_zdir(nBlocks,nThreads);
} 



// --------------------------------------------------------
// Take step forward for IBM using LBM object:
// --------------------------------------------------------

void class_filaments_ibm3D::stepIBM_Verlet_no_fluid(int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	//  here, the velocity-Verlet algorithm is used to update the 
	//  bead positions - using a viscous drag force proportional
	//  to the bead velocity.
	// ----------------------------------------------------------
		
	// first step of IBM velocity verlet:
	update_bead_positions_verlet_1_drag(nBlocks,nThreads);
	
	// re-build bin lists for filament beads:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
			
	// update IBM:
	compute_bead_forces(nBlocks,nThreads);
	nonbonded_bead_interactions(nBlocks,nThreads);
	compute_wall_forces(nBlocks,nThreads);
	enforce_max_bead_force(nBlocks,nThreads);
	update_bead_positions_verlet_2_drag(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Take step forward for IBM using LBM object:
// --------------------------------------------------------

void class_filaments_ibm3D::stepIBM_Euler_no_fluid(int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	//  here, the Euler algorithm is used to update the 
	//  bead positions - using a viscous drag force proportional
	//  to the bead velocity.
	// ----------------------------------------------------------
		
	// re-build bin lists for filament beads:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
			
	// update IBM:
	compute_bead_forces(nBlocks,nThreads);
	add_drag_force_to_beads(nBlocks,nThreads);
	nonbonded_bead_interactions(nBlocks,nThreads);
	compute_wall_forces(nBlocks,nThreads);
	enforce_max_bead_force(nBlocks,nThreads);
	update_bead_positions_euler(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Take step forward for IBM using LBM object:
// --------------------------------------------------------

void class_filaments_ibm3D::stepIBM_Euler_pusher_no_fluid(int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	//  here, the Euler algorithm is used to update the 
	//  bead positions.  Each bead is assigned an active VELOCITY
	//  (rather than an active force) as well as conservative forces
	// ----------------------------------------------------------
		
	// re-build bin lists for filament beads:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
		
	// update IBM:
	compute_bead_forces_spring_overdamped(nBlocks,nThreads);	
	nonbonded_bead_interactions(nBlocks,nThreads);
	compute_wall_forces(nBlocks,nThreads);
	enforce_max_bead_force(nBlocks,nThreads);
	update_bead_positions_euler_overdamped(nBlocks,nThreads); 
		
}



// --------------------------------------------------------
// Take step forward for IBM using LBM object:
// --------------------------------------------------------

void class_filaments_ibm3D::stepIBM_Euler(class_scsp_D3Q19& lbm, int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	//  here, the Euler algorithm is used to update the 
	//  bead positions - using a viscous drag force proportional
	//  to the difference between the bead velocities and the 
	//  fluid velocities
	// ----------------------------------------------------------
	
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);
	
	// re-build bin lists for filament beads:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
	
	// calculate bead forces:
	compute_bead_forces(nBlocks,nThreads);
	nonbonded_bead_interactions(nBlocks,nThreads);
	compute_wall_forces(nBlocks,nThreads);	
	
	// calculate viscous drag forces:
	lbm.viscous_force_filaments_IBM_LBM(nBlocks,nThreads,gam,beads,nBeads); 
	enforce_max_bead_force(nBlocks,nThreads);
	
	// update beads:
	update_bead_positions_euler(nBlocks,nThreads);
	
}



// --------------------------------------------------------
// Take step forward for IBM using LBM object:
// --------------------------------------------------------

void class_filaments_ibm3D::stepIBM_capsules_filaments(class_scsp_D3Q19& lbm, class_capsules_ibm3D& cap, 
                                                       int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	// This method updates filaments AND capsules
	// ----------------------------------------------------------

	// ----------------------------------------------------------
	//  here, the velocity-Verlet algorithm is used to update the 
	//  bead AND node positions - using a viscous drag force proportional
	//  to the difference between the bead velocities and the 
	//  fluid velocities
	// ----------------------------------------------------------
	
	// zero fluid forces:
	//lbm.zero_forces(nBlocks,nThreads);
	
	// first step of IBM velocity verlet:
	update_bead_positions_verlet_1_drag(nBlocks,nThreads);
	cap.update_node_positions_verlet_1(nBlocks,nThreads);
	
	// re-build bin lists for filament beads & capsule nodes:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
	cap.reset_bin_lists(nBlocks,nThreads);
	cap.build_bin_lists(nBlocks,nThreads);
			
	// calculate bonded forces within filaments and capsules:
	compute_bead_forces(nBlocks,nThreads); 
	cap.compute_node_forces(nBlocks,nThreads);
	
	// calculate nonbonded forces for filaments and capsules:
	nonbonded_bead_interactions(nBlocks,nThreads);
	nonbonded_bead_node_interactions(cap,nBlocks,nThreads);
	//cap.nonbonded_node_interactions(nBlocks,nThreads);
	cap.nonbonded_node_bead_interactions(beads,bins,nBlocks,nThreads);
	
	// calculate wall forces:
	compute_wall_forces(nBlocks,nThreads);
	enforce_max_bead_force(nBlocks,nThreads);
	cap.compute_wall_forces(nBlocks,nThreads);	
	cap.enforce_max_node_force(nBlocks,nThreads);
	
	// calculate viscous drag forces:
	//add_drag_force_to_beads(gam,nBlocks,nThreads);
	//lbm.viscous_force_filaments_IBM_LBM(nBlocks,nThreads,gam,beads,nBeads); 
	lbm.viscous_force_IBM_LBM(nBlocks,nThreads,gam,cap.nodes,cap.nNodes);
	
	// second step of IBM velocity verlet:
	update_bead_positions_verlet_2_drag(nBlocks,nThreads);
	cap.update_node_positions_verlet_2(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Take step forward for IBM using LBM object:
// --------------------------------------------------------

void class_filaments_ibm3D::stepIBM_capsules_filaments_no_fluid(class_capsules_ibm3D& cap, 
                                                                int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	// This method updates filaments AND capsules
	// ----------------------------------------------------------

	// ----------------------------------------------------------
	//  here, the velocity-Verlet algorithm is used to update the 
	//  bead AND node positions 
	// ----------------------------------------------------------
		
	// first step of IBM velocity verlet:
	update_bead_positions_verlet_1_drag(nBlocks,nThreads);
	cap.update_node_positions_verlet_1_drag(nBlocks,nThreads);
	
	// re-build bin lists for filament beads & capsule nodes:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
	cap.reset_bin_lists(nBlocks,nThreads);
	cap.build_bin_lists(nBlocks,nThreads);
			
	// calculate bonded forces within filaments and capsules:
	compute_bead_forces(nBlocks,nThreads); 
	cap.compute_node_forces(nBlocks,nThreads);
	
	// calculate nonbonded forces for filaments and capsules:
	nonbonded_bead_interactions(nBlocks,nThreads);
	nonbonded_bead_node_interactions(cap,nBlocks,nThreads);
	//cap.nonbonded_node_interactions(nBlocks,nThreads);
	cap.nonbonded_node_bead_interactions(beads,bins,nBlocks,nThreads);
	
	// calculate wall forces:
	compute_wall_forces(nBlocks,nThreads);
	enforce_max_bead_force(nBlocks,nThreads);
	cap.compute_wall_forces(nBlocks,nThreads);	
	cap.enforce_max_node_force(nBlocks,nThreads);
		
	// second step of IBM velocity verlet:
	update_bead_positions_verlet_2_drag(nBlocks,nThreads);
	cap.update_node_positions_verlet_2_drag(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Take step forward for IBM using LBM object:
// --------------------------------------------------------

void class_filaments_ibm3D::stepIBM_capsules_filaments_overdamp_no_fluid(class_capsules_ibm3D& cap, 
                                                                         int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	// This method updates filaments AND capsules
	// ----------------------------------------------------------

	// ----------------------------------------------------------
	//  here, the overdamped Euler algorithm is used to update the 
	//  bead AND node positions 
	// ----------------------------------------------------------
		
	// re-build bin lists for filament beads & capsule nodes:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
	cap.reset_bin_lists(nBlocks,nThreads);
	cap.build_bin_lists(nBlocks,nThreads);
			
	// calculate bonded forces within filaments and capsules:
	compute_bead_forces(nBlocks,nThreads); 
	cap.compute_node_forces(nBlocks,nThreads);
	
	// calculate nonbonded forces for filaments and capsules:
	nonbonded_bead_interactions(nBlocks,nThreads);
	nonbonded_bead_node_interactions(cap,nBlocks,nThreads);
	//cap.nonbonded_node_interactions(nBlocks,nThreads);
	cap.nonbonded_node_bead_interactions(beads,bins,nBlocks,nThreads);
	
	// calculate wall forces:
	compute_wall_forces(nBlocks,nThreads);
	enforce_max_bead_force(nBlocks,nThreads);
	cap.compute_wall_forces(nBlocks,nThreads);	
	cap.enforce_max_node_force(nBlocks,nThreads);
		
	// second step of IBM velocity verlet:
	update_bead_positions_euler_overdamped(nBlocks,nThreads);
	cap.update_node_positions_euler_overdamped(fricBead,nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Take step forward for IBM using LBM object:
// --------------------------------------------------------

void class_filaments_ibm3D::stepIBM_capsules_filaments_pusher_no_fluid(class_capsules_ibm3D& cap, 
                                                                       int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	// This method updates filaments AND capsules.  Here,
	// the filaments are considered pushers with a prescribed 
	// active velocity (rather than active force)
	// ----------------------------------------------------------

	// ----------------------------------------------------------
	//  here, the velocity-Verlet algorithm is used to update the
	//  node positions, and the Euler method is used to update the
	//  bead positions 
	// ----------------------------------------------------------
		
	// first step of IBM velocity verlet:
	cap.update_node_positions_verlet_1_drag(nBlocks,nThreads);
	
	// re-build bin lists for filament beads & capsule nodes:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
	cap.reset_bin_lists(nBlocks,nThreads);
	cap.build_bin_lists(nBlocks,nThreads);
			
	// calculate bonded forces within filaments and capsules:
	compute_bead_forces_spring_overdamped(nBlocks,nThreads); 
	cap.compute_node_forces(nBlocks,nThreads);
	
	// calculate nonbonded forces for filaments and capsules:
	nonbonded_bead_interactions(nBlocks,nThreads);
	nonbonded_bead_node_interactions(cap,nBlocks,nThreads);
	//cap.nonbonded_node_interactions(nBlocks,nThreads);
	cap.nonbonded_node_bead_interactions(beads,bins,nBlocks,nThreads);
	
	// calculate wall forces:
	compute_wall_forces(nBlocks,nThreads);
	enforce_max_bead_force(nBlocks,nThreads);
	cap.compute_wall_forces(nBlocks,nThreads);	
	cap.enforce_max_node_force(nBlocks,nThreads);
		
	// second step of IBM velocity verlet:
	update_bead_positions_euler_overdamped(nBlocks,nThreads);
	cap.update_node_positions_verlet_2_drag(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Take step forward for IBM w/o fluid pushing beads
// into sphere:
// (note: this uses the velocity-Verlet algorithm)
// --------------------------------------------------------

void class_filaments_ibm3D::stepIBM_push_into_sphere(int nSteps, float xs, float ys, float zs, float rs, 
                                                     int nBlocks, int nThreads) 
{		
	for (int i=0; i<nSteps; i++) {
		// first step of IBM velocity verlet:
		update_bead_positions_verlet_1_drag(nBlocks,nThreads);

		// re-build bin lists for IBM nodes:
		reset_bin_lists(nBlocks,nThreads);
		build_bin_lists(nBlocks,nThreads);
		
		// update IBM:
		compute_bead_forces_no_propulsion(nBlocks,nThreads);
		nonbonded_bead_interactions(nBlocks,nThreads);
		compute_wall_forces(nBlocks,nThreads);
		push_beads_inside_sphere(xs,ys,zs,rs,nBlocks,nThreads);
		enforce_max_bead_force(nBlocks,nThreads);
		update_bead_positions_verlet_2_drag(nBlocks,nThreads);
	}
	zero_bead_velocities_forces(nBlocks,nThreads); 
}



// --------------------------------------------------------
// Determine which bead-force model to use:
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_forces(int nBlocks, int nThreads) 
{	
	if (forceModel == "FENE") {
		compute_bead_forces_FENE(nBlocks,nThreads);
	}
	else if (forceModel == "spring") {
		compute_bead_forces_spring(nBlocks,nThreads);
	}
	else {
		cout << "valid bead-force model not selected" << endl;
	}
}










// **********************************************************************************************
// Calls to CUDA kernels for main calculations
// **********************************************************************************************











// --------------------------------------------------------
// Call to initialize cuRand state:
// --------------------------------------------------------

void class_filaments_ibm3D::initialize_cuRand(int nBlocks, int nThreads)
{
	init_curand_IBM3D
	<<<nBlocks,nThreads>>> (rngState,1,nBeads);
}



// --------------------------------------------------------
// Call to "update_bead_position_euler_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::update_bead_positions_euler(int nBlocks, int nThreads)
{
	update_bead_position_euler_IBM3D
	<<<nBlocks,nThreads>>> (beads,dt,1.0,nBeads);
	
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);	
}



// --------------------------------------------------------
// Call to "update_bead_position_euler_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::update_bead_positions_euler_overdamped(int nBlocks, int nThreads)
{
	update_bead_position_euler_overdamped_IBM3D
	<<<nBlocks,nThreads>>> (beads,dt,fricBead,nBeads);
	
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);	
}



// --------------------------------------------------------
// Call to "update_bead_position_verlet_1_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::update_bead_positions_verlet_1(int nBlocks, int nThreads)
{
	update_bead_position_verlet_1_IBM3D
	<<<nBlocks,nThreads>>> (beads,dt,1.0,nBeads);
	
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);	
}



// --------------------------------------------------------
// Call to "update_bead_position_verlet_2_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::update_bead_positions_verlet_2(int nBlocks, int nThreads)
{
	update_bead_position_verlet_2_IBM3D
	<<<nBlocks,nThreads>>> (beads,dt,1.0,nBeads);
}



// --------------------------------------------------------
// Call to "update_bead_position_verlet_1_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::update_bead_positions_verlet_1_drag(int nBlocks, int nThreads)
{
	update_bead_position_verlet_1_drag_IBM3D
	<<<nBlocks,nThreads>>> (beads,dt,1.0,gam,nBeads);
	
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);	
}



// --------------------------------------------------------
// Call to "update_bead_position_verlet_2_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::update_bead_positions_verlet_2_drag(int nBlocks, int nThreads)
{
	update_bead_position_verlet_2_drag_IBM3D
	<<<nBlocks,nThreads>>> (beads,dt,1.0,gam,nBeads);
}



// --------------------------------------------------------
// Call to "zero_bead_velocities_forces_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::zero_bead_velocities_forces(int nBlocks, int nThreads)
{
	zero_bead_velocities_forces_IBM3D
	<<<nBlocks,nThreads>>> (beads,nBeads);
}



// --------------------------------------------------------
// Call to "zero_bead_forces_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::zero_bead_forces(int nBlocks, int nThreads)
{
	zero_bead_forces_IBM3D
	<<<nBlocks,nThreads>>> (beads,nBeads);
}



// --------------------------------------------------------
// Call to "enforce_max_node_force_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::enforce_max_bead_force(int nBlocks, int nThreads)
{
	enforce_max_bead_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,beadFmax,nBeads);
}



// --------------------------------------------------------
// Call to "add_drag_force_to_node_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::add_drag_force_to_beads(int nBlocks, int nThreads)
{
	add_drag_force_to_bead_IBM3D
	<<<nBlocks,nThreads>>> (beads,gam,nBeads);
}



// --------------------------------------------------------
// Call to "compute_thermal_force_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_thermal_force(int nBlocks, int nThreads)
{
	compute_thermal_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,rngState,noisekT,nBeads);
}



// --------------------------------------------------------
// Call to "compute_bead_force_spring_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_bond_force_spring(int nBlocks, int nThreads)
{
	compute_bead_force_spring_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,filams,nEdges);
}



// --------------------------------------------------------
// Call to "compute_bead_force_bending_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_bending_force(int nBlocks, int nThreads)
{
	compute_bead_force_bending_IBM3D
	<<<nBlocks,nThreads>>> (beads,filams,nBeads);
}



// --------------------------------------------------------
// Call to "compute_propulsion_force_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_propulsion_force(int nBlocks, int nThreads)
{
	compute_propulsion_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,filams,nEdges);
}



// --------------------------------------------------------
// Call to "compute_propulsion_velocity_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_propulsion_velocity(int nBlocks, int nThreads)
{
	compute_propulsion_velocity_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,filams,nEdges); 
}



// --------------------------------------------------------
// Call to "unwrap_bead_coordinates_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::unwrap_bead_coordinates(int nBlocks, int nThreads)
{
	unwrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,filams,Box,pbcFlag,nBeads);
}



// --------------------------------------------------------
// Call to "wrap_bead_coordinates_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::wrap_bead_coordinates(int nBlocks, int nThreads)
{
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);
}



// --------------------------------------------------------
// Call to kernel that builds the binMap array:
// --------------------------------------------------------

void class_filaments_ibm3D::build_binMap(int nBlocks, int nThreads)
{
	if (nFilams > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;	
		build_binMap_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (bins);		
	}	
}



// --------------------------------------------------------
// Call to kernel that resets bin lists:
// --------------------------------------------------------

void class_filaments_ibm3D::reset_bin_lists(int nBlocks, int nThreads)
{
	if (nFilams > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;		
		reset_bin_lists_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (bins);
	}	
}



// --------------------------------------------------------
// Call to kernel that builds bin lists:
// --------------------------------------------------------

void class_filaments_ibm3D::build_bin_lists(int nBlocks, int nThreads)
{
	if (nFilams > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;		
		build_bin_lists_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (beads,bins,nBeads);		
	}	
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_filaments_ibm3D::nonbonded_bead_interactions(int nBlocks, int nThreads)
{
	if (nFilams > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;								
		nonbonded_bead_interactions_IBM3D
		<<<nBlocks,nThreads>>> (beads,bins,repA,repD,nBeads,Box,pbcFlag);
	}	
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_filaments_ibm3D::nonbonded_bead_node_interactions(class_capsules_ibm3D& cap, int nBlocks, int nThreads)
{
	if (nFilams > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;								
		nonbonded_bead_node_interactions_IBM3D
		<<<nBlocks,nThreads>>> (beads,cap.nodes,cap.bins,repA_bn,repD_bn,nBeads,Box,pbcFlag);
	}	
}



// --------------------------------------------------------
// Calls to kernels that compute forces on beads based 
// on the chain-like mechanics model (Spring model):
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_forces_spring(int nBlocks, int nThreads)
{
	// First, zero the bead forces
	zero_bead_forces_IBM3D
	<<<nBlocks,nThreads>>> (beads,nBeads);
		
	// Second, unwrap node coordinates:
	unwrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,filams,Box,pbcFlag,nBeads);	
			
	// Third, compute the edge extension and bending force for each edge:
	compute_bead_force_spring_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,filams,nEdges);
	
	compute_bead_force_bending_IBM3D
	<<<nBlocks,nThreads>>> (beads,filams,nBeads);
	
	// Forth, compute propulsion and thermal forces:
	compute_propulsion_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,filams,nEdges);
	
	compute_thermal_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,rngState,noisekT,nBeads);
	
	// Fifth, re-wrap bead coordinates:
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);			
}



// --------------------------------------------------------
// Calls to kernels that compute forces on beads based 
// on the chain-like mechanics model (Spring model).
// Here, a propulsion VELOCITY is applied rather than
// a propulsion force.
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_forces_spring_overdamped(int nBlocks, int nThreads)
{
	zero_bead_velocities_forces(nBlocks,nThreads);	
	unwrap_bead_coordinates(nBlocks,nThreads);
	compute_bead_propulsion_velocity(nBlocks,nThreads);
	compute_bead_bond_force_spring(nBlocks,nThreads);
	compute_bead_bending_force(nBlocks,nThreads);
	compute_bead_thermal_force(nBlocks,nThreads);
	wrap_bead_coordinates(nBlocks,nThreads);	
}



// --------------------------------------------------------
// Calls to kernels that compute forces on beads based 
// on the chain-like mechanics model (FENE model):
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_forces_FENE(int nBlocks, int nThreads)
{
	// First, zero the bead forces
	zero_bead_forces_IBM3D
	<<<nBlocks,nThreads>>> (beads,nBeads);
		
	// Second, unwrap node coordinates:
	unwrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,filams,Box,pbcFlag,nBeads);	
			
	// Third, compute the edge extension and bending force for each edge:
	compute_bead_force_FENE_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,filams,0.4,nEdges);
	
	compute_bead_force_bending_IBM3D
	<<<nBlocks,nThreads>>> (beads,filams,nBeads);
	
	// Forth, compute propulsion and thermal forces:
	compute_propulsion_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,filams,nEdges);
	
	compute_thermal_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,rngState,noisekT,nBeads);
	
	// Fifth, re-wrap bead coordinates:
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);			
}



// --------------------------------------------------------
// Calls to kernels that compute forces on nodes based 
// on the membrane mechanics model (Spring model):
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_forces_no_propulsion(int nBlocks, int nThreads)
{
	// First, zero the bead forces
	zero_bead_forces_IBM3D
	<<<nBlocks,nThreads>>> (beads,nBeads);
		
	// Second, unwrap node coordinates:
	unwrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,filams,Box,pbcFlag,nBeads);	
			
	// Third, compute the edge extension and bending force for each edge:
	compute_bead_force_spring_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,filams,nEdges);
	
	//compute_bead_force_FENE_IBM3D
	//<<<nBlocks,nThreads>>> (beads,edges,filams,0.4,nEdges);
	
	compute_bead_force_bending_IBM3D
	<<<nBlocks,nThreads>>> (beads,filams,nBeads);
		
	// Forth, re-wrap bead coordinates:
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);			
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir:
// --------------------------------------------------------

void class_filaments_ibm3D::wall_forces_ydir(int nBlocks, int nThreads)
{
	bead_wall_forces_ydir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in z-dir:
// --------------------------------------------------------

void class_filaments_ibm3D::wall_forces_zdir(int nBlocks, int nThreads)
{
	bead_wall_forces_zdir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir
// and z-dir:
// --------------------------------------------------------

void class_filaments_ibm3D::wall_forces_ydir_zdir(int nBlocks, int nThreads)
{
	bead_wall_forces_ydir_zdir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir
// and z-dir:
// --------------------------------------------------------

void class_filaments_ibm3D::push_beads_inside_sphere(float xs, float ys, float zs, float rs, 
                                                     int nBlocks, int nThreads)
{
	push_beads_into_sphere_IBM3D
	<<<nBlocks,nThreads>>> (beads,xs,ys,zs,rs,nBeads);
}












// **********************************************************************************************
// Analysis and Geometry calculations done by the host (CPU)
// **********************************************************************************************












// --------------------------------------------------------
// Write IBM output to file:
// --------------------------------------------------------

void class_filaments_ibm3D::write_output(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_3D_filaments(tagname,tagnum,
	nBeads,nEdges,beadsH,edgesH);
}



// --------------------------------------------------------
// Unwrap bead coordinates based on difference between bead
// position and the filament's head bead position:
// --------------------------------------------------------

void class_filaments_ibm3D::unwrap_bead_coordinates()
{
	for (int i=0; i<nBeads; i++) {
		int f = beadsH[i].filamID;
		int j = filamsH[f].headBead;
		float3 rij = beadsH[j].r - beadsH[i].r;
		beadsH[i].r = beadsH[i].r + roundf(rij/Box)*Box*pbcFlag; // PBC's		
	}	
}



// --------------------------------------------------------
// Write filament data to file "vtkoutput/filament_data.dat"
// --------------------------------------------------------

void class_filaments_ibm3D::output_filament_data()
{
	
	/*	
	// -----------------------------------------
	// Define the file location and name:
	// -----------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << "filament_data.dat";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------
	// Write to file:
	// -----------------------------------------
	
	for (int c=0; c<nCells; c++) {
		
		outfile << fixed << setprecision(2) << setw(2) << cellsH[c].cellType << " " << setw(2) << cellsH[c].intrain << " "
			                                << setw(5) << cellsH[c].rad      << " " << setw(8) << cellsH[c].vol     << " " <<
							setprecision(3) << setw(7) << cellsH[c].Ca       << " " << setw(7) << cellsH[c].D       << " " << setw(7) << cellsH[c].maxT1 << " " <<
							setprecision(4) << setw(10) << cellsH[c].com.x    << " " << setw(10) << cellsH[c].com.y   << " " << setw(10) << cellsH[c].com.z << " " <<
							setprecision(6) << setw(10) << cellsH[c].vel.x    << " " << setw(10) << cellsH[c].vel.y   << " " << setw(10) << cellsH[c].vel.z << endl;
		
	}
	*/
	
}



