 
# include "class_rods_ibm3D.cuh"
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

class_rods_ibm3D::class_rods_ibm3D()
{
	// get some parameters:
	GetPot inputParams("input.dat");
	
	// mesh attributes	
	nBeadsPerRod = inputParams("IBM_RODS/nBeadsPerRod",0);
	nRods = inputParams("IBM_RODS/nRods",1);
	nBeads = nBeadsPerRod*nRods;
	
	// mechanical properties
	dt = inputParams("Time/dt",1.0);
	L0 = inputParams("IBM_RODS/L0",0.5);
	kT = inputParams("IBM_RODS/kT",0.0);
	repA = inputParams("IBM_RODS/repA",0.0);
	repD = inputParams("IBM_RODS/repD",0.0);
	repA_bn = inputParams("IBM_RODS/repA_bn",0.0);
	repD_bn = inputParams("IBM_RODS/repD_bn",0.0);
	beadFmax = inputParams("IBM_RODS/beadFmax",1000.0);
	rodFmax = inputParams("IBM_RODS/rodFmax",1000.0);
	rodTmax = inputParams("IBM_RODS/rodTmax",1000.0);
	gam = inputParams("IBM_RODS/gamma",0.1);
	noisekT = 6.0*kT*gam/dt;
	noisekTforce = 6.0*kT*gam/dt;
	noisekTtorque = 6.0*kT*gam/dt;
	
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
	if (nRods > 1) binsFlag = true;
	if (binsFlag) {		
		bins.sizeBins = inputParams("IBM_RODS/sizeBins",2.0);
		bins.binMax = inputParams("IBM_RODS/binMax",1);			
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

class_rods_ibm3D::~class_rods_ibm3D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_rods_ibm3D::allocate()
{
	// allocate array memory (host):
	beadsH = (beadrod*)malloc(nBeads*sizeof(beadrod));
	rodsH = (rod*)malloc(nRods*sizeof(rod));
							
	// allocate array memory (device):	
	cudaMalloc((void **) &beads, nBeads*sizeof(beadrod));
	cudaMalloc((void **) &rods, nRods*sizeof(rod));
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

void class_rods_ibm3D::deallocate()
{
	// free array memory (host):
	free(beadsH);
	free(rodsH);
					
	// free array memory (device):
	cudaFree(beads);
	cudaFree(rods);
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

void class_rods_ibm3D::memcopy_host_to_device()
{
	cudaMemcpy(beads, beadsH, sizeof(beadrod)*nBeads, cudaMemcpyHostToDevice);	
	cudaMemcpy(rods, rodsH, sizeof(rod)*nRods, cudaMemcpyHostToDevice);	
}
	


// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_rods_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(beadsH, beads, sizeof(beadrod)*nBeads, cudaMemcpyDeviceToHost);
	cudaMemcpy(rodsH, rods, sizeof(rod)*nRods, cudaMemcpyDeviceToHost);
	
	// unwrap coordinate positions:
	unwrap_bead_coordinates(); 
}











// **********************************************************************************************
// Initialization Stuff...
// **********************************************************************************************












// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void class_rods_ibm3D::create_first_rod()
{
	// set up the bead information for first filament:
	for (int i=0; i<nBeadsPerRod; i++) {
		beadsH[i].r.x = 0.0 + float(i)*L0;
		beadsH[i].r.y = 0.0;
		beadsH[i].r.z = 0.0;
		beadsH[i].f = make_float3(0.0f);
		beadsH[i].rodID = 0;
	}
		
	// set up indices for ALL rods:
	for (int f=0; f<nRods; f++) {
		rodsH[f].nBeads = nBeadsPerRod;
		rodsH[f].indxB0 = f*nBeadsPerRod;      // start index for beads
		rodsH[f].headBead = f*nBeadsPerRod;    // head bead (first bead)
		rodsH[f].centerBead = f*nBeadsPerRod + nBeadsPerRod/2 + 1;  // center-of-mass, assuming nBeadsPerRod is odd
	}
}



// --------------------------------------------------------
// Setters:
// --------------------------------------------------------

void class_rods_ibm3D::set_pbcFlag(int x, int y, int z)
{
	pbcFlag.x = x; pbcFlag.y = y; pbcFlag.z = z;
}

void class_rods_ibm3D::set_fp(float val)
{
	for (int f=0; f<nRods; f++) rodsH[f].fp = val;
}

void class_rods_ibm3D::set_up(float val)
{
	for (int f=0; f<nRods; f++) rodsH[f].up = val;
}

void class_rods_ibm3D::set_rods_radii(float rad)
{
	// set radius for ALL cells:
	for (int f=0; f<nRods; f++) rodsH[f].rad = rad;
}

void class_rods_ibm3D::set_rod_radius(int rID, float rad)
{
	// set radius for ONE rod:
	rodsH[rID].rad = rad;
}

void class_rods_ibm3D::set_rods_types(int val)
{
	// set filamType for ALL filaments:
	for (int r=0; r<nRods; r++) rodsH[r].rodType = val;
}

void class_rods_ibm3D::set_rod_type(int rID, int val)
{
	// set rodType for ONE rod:
	rodsH[rID].rodType = val;
}

int class_rods_ibm3D::get_max_array_size()
{
	// return the maximum array size:
	int maxSize = nBeads;
	if (binsFlag) {
		if (bins.nBins > maxSize) maxSize = bins.nBins;
	}
	return maxSize;
}



// --------------------------------------------------------
// Assign the cell ID to every node:
// --------------------------------------------------------

void class_rods_ibm3D::assign_rodIDs_to_beads()
{
	for (int r=0; r<nRods; r++) {
		int istr = rodsH[r].indxB0;
		int iend = istr + rodsH[r].nBeads;
		for (int i=istr; i<iend; i++) beadsH[i].rodID = r;
	}
}



// --------------------------------------------------------
// Duplicate the first cell mesh information to all cells:
// --------------------------------------------------------

void class_rods_ibm3D::duplicate_rods()
{
	if (nRods > 1) {
		for (int r=1; r<nRods; r++) {
			// skip if filam 0 is different than filam f:
			if (rodsH[0].nBeads != rodsH[r].nBeads) {
				cout << "duplicate rods error: rods have different nBeads" << endl;
				continue;
			}
			// copy bead information:
			for (int i=0; i<rodsH[0].nBeads; i++) {
				int ii = i + rodsH[r].indxB0;
				beadsH[ii].r = beadsH[i].r;
				beadsH[ii].f = beadsH[i].f;
				beadsH[ii].rodID = r;
			}
		}
	}
}



// --------------------------------------------------------
// randomize cell positions and orientations:
// --------------------------------------------------------

void class_rods_ibm3D::randomize_rods(float sepWall)
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beadrod)*nBeads, cudaMemcpyDeviceToHost);
	
	// assign random position and orientation to each filament:
	for (int f=0; f<nRods; f++) {
		float3 shift = make_float3(0.0,0.0,0.0);
		// get random position
		shift.x = (float)rand()/RAND_MAX*Box.x;
		shift.y = sepWall + (float)rand()/RAND_MAX*(Box.y-2.0*sepWall);
		shift.z = sepWall + (float)rand()/RAND_MAX*(Box.z-2.0*sepWall);
		rotate_and_shift_bead_positions(f,shift.x,shift.y,shift.z);
	}
	
	// copy node positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(beadrod)*nBeads, cudaMemcpyHostToDevice);	
}



// --------------------------------------------------------
// randomize cell positions and orientations:
// --------------------------------------------------------

void class_rods_ibm3D::randomize_rods_inside_sphere(float xs, float ys, float zs, float rs, float sepWall)
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beadrod)*nBeads, cudaMemcpyDeviceToHost);
	
	// assign random position and orientation to each filament inside a sphere:
	float3 sphere = make_float3(xs,ys,zs);
	for (int f=0; f<nRods; f++) {
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
	cudaMemcpy(beads, beadsH, sizeof(beadrod)*nBeads, cudaMemcpyHostToDevice);	
}



// --------------------------------------------------------
// calculate separation distance using PBCs:
// --------------------------------------------------------

float class_rods_ibm3D::calc_separation_pbc(float3 r1, float3 r2)
{
	float3 dr = r1 - r2;
	dr -= roundf(dr/Box)*Box;
	return length(dr);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_rods_ibm3D::shift_bead_positions(int fID, float xsh, float ysh, float zsh)
{
	int istr = rodsH[fID].indxB0;
	int iend = istr + rodsH[fID].nBeads;
	for (int i=istr; i<iend; i++) {
		beadsH[i].r.x += xsh;
		beadsH[i].r.y += ysh;
		beadsH[i].r.z += zsh;
	}
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_rods_ibm3D::rotate_and_shift_bead_positions(int fID, float xsh, float ysh, float zsh)
{
	// random rotation angles:
	float a = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // alpha
	float b = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // beta
	float g = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // gamma
	
	// update node positions:
	int istr = rodsH[fID].indxB0;
	int iend = istr + rodsH[fID].nBeads;
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

void class_rods_ibm3D::compute_wall_forces(int nBlocks, int nThreads)
{
	if (pbcFlag.y==0 && pbcFlag.z==1) wall_forces_ydir(nBlocks,nThreads);
	if (pbcFlag.y==1 && pbcFlag.z==0) wall_forces_zdir(nBlocks,nThreads);
	if (pbcFlag.y==0 && pbcFlag.z==0) wall_forces_ydir_zdir(nBlocks,nThreads);
} 



// --------------------------------------------------------
// Take step forward for rods IBM:
// --------------------------------------------------------

void class_rods_ibm3D::stepIBM_Euler_no_fluid(int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	//  here, the Euler algorithm is used to update the 
	//  bead positions - using a viscous drag force proportional
	//  to the bead velocity.
	// ----------------------------------------------------------
		
	// re-build bin lists for rod beads:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
		
	// update IBM:
	zero_bead_forces(nBlocks,nThreads);
	zero_rod_forces_torques_moments(nBlocks,nThreads);
	nonbonded_bead_interactions(nBlocks,nThreads);	
	compute_wall_forces(nBlocks,nThreads);		
	unwrap_bead_coordinates(nBlocks,nThreads);
	sum_rod_forces_torques_moments(nBlocks,nThreads);
	compute_thermal_force_torque_rod(nBlocks,nThreads);
	compute_rod_propulsion_force(nBlocks,nThreads);
	enforce_max_rod_force_torque(nBlocks,nThreads);
	update_rod_position_orientation(nBlocks,nThreads);
	update_bead_position_rods(nBlocks,nThreads);
	wrap_bead_coordinates(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Take step forward for rods & capsules IBM:
// --------------------------------------------------------

void class_rods_ibm3D::stepIBM_capsules_rods_no_fluid(class_capsules_ibm3D& cap, 
                                                      int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	// This method updates rods AND capsules
	// ----------------------------------------------------------

	// ----------------------------------------------------------
	//  here, the Euler algorithm is used to update the 
	//  beads, and the velocity-Verlet algorithm is used to 
	//  update the node positions 
	// ----------------------------------------------------------
	
	// first step of capsule velocity verlet:
	cap.update_node_positions_verlet_1_drag(nBlocks,nThreads);
	
	// re-build bin lists for rod beads & capsule nodes:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
	cap.reset_bin_lists(nBlocks,nThreads);
	cap.build_bin_lists(nBlocks,nThreads);
			
	// calculate bonded forces within rods and capsules:
	zero_bead_forces(nBlocks,nThreads);
	zero_rod_forces_torques_moments(nBlocks,nThreads);	
	cap.compute_node_forces(nBlocks,nThreads);
	
	// calculate nonbonded forces for rods and capsules:
	nonbonded_bead_interactions(nBlocks,nThreads);
	nonbonded_bead_node_interactions(cap,nBlocks,nThreads);
	//cap.nonbonded_node_interactions(nBlocks,nThreads);
	cap.nonbonded_node_bead_rod_interactions(beads,bins,nBlocks,nThreads);
	
	// calculate wall forces:
	compute_wall_forces(nBlocks,nThreads);
	cap.compute_wall_forces(nBlocks,nThreads);	
	cap.enforce_max_node_force(nBlocks,nThreads);
		
	// update IBM:
	unwrap_bead_coordinates(nBlocks,nThreads);
	sum_rod_forces_torques_moments(nBlocks,nThreads);
	compute_thermal_force_torque_rod(nBlocks,nThreads);
	compute_rod_propulsion_force(nBlocks,nThreads);
	enforce_max_rod_force_torque(nBlocks,nThreads);		
	update_rod_position_orientation(nBlocks,nThreads);
	update_bead_position_rods(nBlocks,nThreads);
	wrap_bead_coordinates(nBlocks,nThreads);
	cap.update_node_positions_verlet_2_drag(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Take step forward for IBM w/o fluid pushing beads
// into sphere:
// (note: this uses the velocity-Verlet algorithm)
// --------------------------------------------------------

void class_rods_ibm3D::stepIBM_push_into_sphere(int nSteps, float xs, float ys, float zs, float rs, 
                                                int nBlocks, int nThreads) 
{		
	for (int i=0; i<nSteps; i++) {
		// re-build bin lists for rod beads:
		reset_bin_lists(nBlocks,nThreads);
		build_bin_lists(nBlocks,nThreads);
			
		// update IBM:
		zero_bead_forces(nBlocks,nThreads);
		zero_rod_forces_torques_moments(nBlocks,nThreads);	
		nonbonded_bead_interactions(nBlocks,nThreads);
		compute_wall_forces(nBlocks,nThreads);	
		push_beads_inside_sphere(xs,ys,zs,rs,nBlocks,nThreads);
		unwrap_bead_coordinates(nBlocks,nThreads);
		sum_rod_forces_torques_moments(nBlocks,nThreads);
		enforce_max_rod_force_torque(nBlocks,nThreads);		
		update_rod_position_orientation(nBlocks,nThreads);
		update_bead_position_rods(nBlocks,nThreads);
		wrap_bead_coordinates(nBlocks,nThreads);	
	}
	zero_bead_forces(nBlocks,nThreads); 
}














// **********************************************************************************************
// Calls to CUDA kernels for main calculations
// **********************************************************************************************













// --------------------------------------------------------
// Call to initialize cuRand state:
// --------------------------------------------------------

void class_rods_ibm3D::initialize_cuRand(int nBlocks, int nThreads)
{
	init_curand_rods_IBM3D
	<<<nBlocks,nThreads>>> (rngState,1,nRods);
}



// --------------------------------------------------------
// Call to "zero_rod_forces_torques_moments_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::zero_rod_forces_torques_moments(int nBlocks, int nThreads)
{
	zero_rod_forces_torques_moments_IBM3D
	<<<nBlocks,nThreads>>> (rods,nRods);
}



// --------------------------------------------------------
// Call to "set_rod_position_orientation_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::set_rod_position_orientation(int nBlocks, int nThreads)
{
	set_rod_position_orientation_IBM3D
	<<<nBlocks,nThreads>>> (beads,rods,nRods);
}



// --------------------------------------------------------
// Call to "update_bead_positions_euler_rods_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::update_bead_position_rods(int nBlocks, int nThreads)
{
	update_bead_positions_rods_IBM3D
	<<<nBlocks,nThreads>>> (beads,rods,L0,nBeads);
	
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);	
}



// --------------------------------------------------------
// Call to "update_rod_position_orientation_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::update_rod_position_orientation(int nBlocks, int nThreads)
{
	update_rod_position_orientation_IBM3D
	<<<nBlocks,nThreads>>> (rods,dt,gam,nRods);
	
	wrap_rod_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (rods,Box,pbcFlag,nRods);	
}



// --------------------------------------------------------
// Call to "zero_bead_forces_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::zero_bead_forces(int nBlocks, int nThreads)
{
	zero_bead_forces_IBM3D
	<<<nBlocks,nThreads>>> (beads,nBeads);
}



// --------------------------------------------------------
// Call to "enforce_max_node_force_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::enforce_max_bead_force(int nBlocks, int nThreads)
{
	enforce_max_bead_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,beadFmax,nBeads);
}



// --------------------------------------------------------
// Call to "enforce_max_rod_force_torque_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::enforce_max_rod_force_torque(int nBlocks, int nThreads)
{
	enforce_max_rod_force_torque_IBM3D
	<<<nBlocks,nThreads>>> (rods,rodFmax,rodTmax,nRods);
}



// --------------------------------------------------------
// Call to "compute_thermal_force_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::compute_bead_thermal_force(int nBlocks, int nThreads)
{
	compute_thermal_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,rngState,noisekT,nBeads);
}



// --------------------------------------------------------
// Call to "compute_thermal_force_torque_rod_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::compute_thermal_force_torque_rod(int nBlocks, int nThreads)
{
	compute_thermal_force_torque_rod_IBM3D
	<<<nBlocks,nThreads>>> (rods,rngState,noisekTforce,noisekTtorque,nRods);
}



// --------------------------------------------------------
// Call to "compute_propulsion_force_rods_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::compute_rod_propulsion_force(int nBlocks, int nThreads)
{
	compute_propulsion_force_rods_IBM3D
	<<<nBlocks,nThreads>>> (rods,nRods);
}



// --------------------------------------------------------
// Call to "sum_rod_forces_torques_moments_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::sum_rod_forces_torques_moments(int nBlocks, int nThreads)
{
	sum_rod_forces_torques_moments_IBM3D
	<<<nBlocks,nThreads>>> (beads,rods,1.0,nBeads);
}



// --------------------------------------------------------
// Call to "unwrap_bead_coordinates_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::unwrap_bead_coordinates(int nBlocks, int nThreads)
{
	unwrap_bead_coordinates_rods_IBM3D
	<<<nBlocks,nThreads>>> (beads,rods,Box,pbcFlag,nBeads);
}



// --------------------------------------------------------
// Call to "wrap_bead_coordinates_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::wrap_bead_coordinates(int nBlocks, int nThreads)
{
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);
}



// --------------------------------------------------------
// Call to kernel that builds the binMap array:
// --------------------------------------------------------

void class_rods_ibm3D::build_binMap(int nBlocks, int nThreads)
{
	if (nRods > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;	
		build_binMap_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (bins);		
	}	
}



// --------------------------------------------------------
// Call to kernel that resets bin lists:
// --------------------------------------------------------

void class_rods_ibm3D::reset_bin_lists(int nBlocks, int nThreads)
{
	if (nRods > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;		
		reset_bin_lists_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (bins);
	}	
}



// --------------------------------------------------------
// Call to kernel that builds bin lists:
// --------------------------------------------------------

void class_rods_ibm3D::build_bin_lists(int nBlocks, int nThreads)
{
	if (nRods > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;		
		build_bin_lists_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (beads,bins,nBeads);		
	}	
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_rods_ibm3D::nonbonded_bead_interactions(int nBlocks, int nThreads)
{
	if (nRods > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;								
		nonbonded_bead_interactions_IBM3D
		<<<nBlocks,nThreads>>> (beads,bins,repA,repD,nBeads,Box,pbcFlag);
	}	
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_rods_ibm3D::nonbonded_bead_node_interactions(class_capsules_ibm3D& cap, int nBlocks, int nThreads)
{
	if (nRods > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;								
		nonbonded_bead_node_interactions_rods_IBM3D
		<<<nBlocks,nThreads>>> (beads,cap.nodes,cap.bins,repA_bn,repD_bn,nBeads,Box,pbcFlag);
	}	
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir:
// --------------------------------------------------------

void class_rods_ibm3D::wall_forces_ydir(int nBlocks, int nThreads)
{
	bead_wall_forces_ydir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in z-dir:
// --------------------------------------------------------

void class_rods_ibm3D::wall_forces_zdir(int nBlocks, int nThreads)
{
	bead_wall_forces_zdir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir
// and z-dir:
// --------------------------------------------------------

void class_rods_ibm3D::wall_forces_ydir_zdir(int nBlocks, int nThreads)
{
	bead_wall_forces_ydir_zdir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir
// and z-dir:
// --------------------------------------------------------

void class_rods_ibm3D::push_beads_inside_sphere(float xs, float ys, float zs, float rs, 
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

void class_rods_ibm3D::write_output(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_3D_rods(tagname,tagnum,
	nBeads,beadsH);
}



// --------------------------------------------------------
// Unwrap bead coordinates based on difference between bead
// position and the rod's center bead position:
// --------------------------------------------------------

void class_rods_ibm3D::unwrap_bead_coordinates()
{
	for (int i=0; i<nBeads; i++) {
		int f = beadsH[i].rodID;
		int j = rodsH[f].centerBead;
		float3 rij = beadsH[j].r - beadsH[i].r;
		beadsH[i].r = beadsH[i].r + roundf(rij/Box)*Box*pbcFlag; // PBC's		
	}	
}






