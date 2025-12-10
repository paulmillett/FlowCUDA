 
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
	repA = inputParams("IBM_RODS/repA",0.0);
	repD = inputParams("IBM_RODS/repD",0.0);
	beadFmax = inputParams("IBM_RODS/beadFmax",1000.0);
	rodFmax = inputParams("IBM_RODS/rodFmax",1000.0);
	rodTmax = inputParams("IBM_RODS/rodTmax",1000.0);
	gam = inputParams("IBM_RODS/gamma",0.1);
		
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
		beadsH[i].rm1 = beadsH[i].r;
		beadsH[i].f = make_float3(0.0f);
		beadsH[i].rodID = 0;
	}
		
	// set up indices for ALL rods:
	for (int f=0; f<nRods; f++) {
		rodsH[f].nBeads = nBeadsPerRod;
		rodsH[f].indxB0 = f*nBeadsPerRod;      // start index for beads
		rodsH[f].headBead = f*nBeadsPerRod;    // head bead (first bead)
		rodsH[f].tailBead = f*nBeadsPerRod + nBeadsPerRod - 1;  // tail bead (last bead)
		rodsH[f].centerBead = f*nBeadsPerRod + nBeadsPerRod/2;  // center-of-mass, assuming nBeadsPerRod is odd		
	}
}



// --------------------------------------------------------
// Setters:
// --------------------------------------------------------

void class_rods_ibm3D::set_pbcFlag(int x, int y, int z)
{
	pbcFlag.x = x; pbcFlag.y = y; pbcFlag.z = z;
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
	// set rodType for ALL rods:
	for (int r=0; r<nRods; r++) rodsH[r].rodType = val;
}

void class_rods_ibm3D::set_rod_type(int rID, int val)
{
	// set rodType for ONE rod:
	rodsH[rID].rodType = val;
}

void class_rods_ibm3D::set_aspect_ratio(float val)
{
	// set aspect ratio for ALL rods:
	for (int r=0; r<nRods; r++) rodsH[r].ar = val;
}

void class_rods_ibm3D::set_mobility_coefficients(float mPar, float mPerp, float mRot)
{
	// set mobility coefficients for ALL rods:
	for (int r=0; r<nRods; r++) {
		rodsH[r].mobPar = mPar;
		rodsH[r].mobPer = mPerp;
		rodsH[r].mobRot = mRot;
	}
}

void class_rods_ibm3D::set_friction_coefficient_translational(float val)
{
	fricT = val;
}

void class_rods_ibm3D::set_friction_coefficient_rotational(float val)
{
	fricR = val;
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
				beadsH[ii].rm1 = beadsH[i].rm1;
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
// randomize fiber positions, but all oriented in x-dir:
// --------------------------------------------------------

void class_rods_ibm3D::randomize_rods_xdir_alligned_cylinder(float chRad, float sepWall)
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beadrod)*nBeads, cudaMemcpyDeviceToHost);
	
	// assign random position and orientation to each filament:
	for (int f=0; f<nRods; f++) {
		float3 shift = make_float3(0.0,0.0,0.0);
		// get random position
		float rad = (float)rand()/RAND_MAX*(chRad - sepWall);
		float ang = (float)rand()/RAND_MAX*(2*M_PI);
		shift.x = (float)rand()/RAND_MAX*Box.x;		
		shift.y = rad*cos(ang) + (Box.y-1.0)/2.0;
		shift.z = rad*sin(ang) + (Box.z-1.0)/2.0;		
		shift_bead_positions(f,shift.x,shift.y,shift.z);
	}
	
	// copy bead positions from host to device:
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
		beadsH[i].rm1 = beadsH[i].r;
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
		beadsH[i].rm1 = beadsH[i].r;		
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

void class_rods_ibm3D::stepIBM_Euler(class_scsp_D3Q19& lbm, int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	//  here, the Euler algorithm is used to update the 
	//  bead positions - using a viscous drag force proportional
	//  to the bead velocity.
	// ----------------------------------------------------------
	
	// zero fluid forces and apply hydrodynamic force to fluid:
	lbm.zero_forces(nBlocks,nThreads);
	lbm.hydrodynamic_force_bead_rod(nBlocks,nThreads,beads,nBeads,nBeadsPerRod);
	
	
	
	// LOOP over the below code for IBM sub-steps...
	
	
	// re-build bin lists for rod beads:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
		
	// calculate IBM forces:
	zero_bead_forces(nBlocks,nThreads);
	zero_rod_forces_torques_moments(nBlocks,nThreads);
	lbm.interpolate_gradient_of_velocity_rod(nBlocks,nThreads,beads,nBeads);
	nonbonded_bead_interactions(nBlocks,nThreads);	
	compute_wall_forces(nBlocks,nThreads);		
	unwrap_bead_coordinates(nBlocks,nThreads);
	sum_rod_forces_torques_moments(nBlocks,nThreads);	
			
	// update IBM positions:
	enforce_max_rod_force_torque(nBlocks,nThreads);
	update_rod_position_orientation_fluid(nBlocks,nThreads);
	update_bead_position_rods(nBlocks,nThreads);
	update_bead_velocity_rods(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Take step forward for rods IBM:
// --------------------------------------------------------

void class_rods_ibm3D::stepIBM_Euler_cylindrical_channel(class_scsp_D3Q19& lbm, float chRad, int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	//  here, the Euler algorithm is used to update the 
	//  bead positions - using a viscous drag force proportional
	//  to the bead velocity.
	// ----------------------------------------------------------
	
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);
	
	// re-build bin lists for rod beads:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
		
	// calculate IBM forces:
	zero_bead_forces(nBlocks,nThreads);
	zero_rod_forces_torques_moments(nBlocks,nThreads);
	lbm.hydrodynamic_force_bead_rod(nBlocks,nThreads,beads,nBeads,nBeadsPerRod);
	nonbonded_bead_interactions(nBlocks,nThreads);	
	compute_wall_forces_cylinder(chRad,nBlocks,nThreads);		
	unwrap_bead_coordinates(nBlocks,nThreads);
	sum_rod_forces_torques_moments(nBlocks,nThreads);
	lbm.interpolate_gradient_of_velocity_rod(nBlocks,nThreads,beads,nBeads);
		
	// update IBM positions:
	enforce_max_rod_force_torque(nBlocks,nThreads);
	update_rod_position_orientation_fluid(nBlocks,nThreads);
	update_bead_position_rods(nBlocks,nThreads);
	update_bead_velocity_rods(nBlocks,nThreads);
		
}













// **********************************************************************************************
// Calls to CUDA kernels for main calculations
// **********************************************************************************************













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
// Call to "update_bead_positions_rods_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::update_bead_position_rods(int nBlocks, int nThreads)
{
	update_bead_positions_rods_IBM3D
	<<<nBlocks,nThreads>>> (beads,rods,L0,dt,nBeads);
	
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);	
}



// --------------------------------------------------------
// Call to "update_bead_positions_rods_singlet_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::update_bead_position_rods_singlet(int nBlocks, int nThreads)
{
	update_bead_positions_rods_singlet_IBM3D
	<<<nBlocks,nThreads>>> (beads,rods,dt,nBeads);
	
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);	
}



// --------------------------------------------------------
// Call to "update_bead_positions_rods_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::update_bead_velocity_rods(int nBlocks, int nThreads)
{
	update_bead_velocity_rods_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,dt,nBeads);
}



// --------------------------------------------------------
// Call to "update_rod_position_orientation_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::update_rod_position_orientation(int nBlocks, int nThreads)
{
	update_rod_position_orientation_IBM3D
	<<<nBlocks,nThreads>>> (rods,dt,fricT,fricR,nRods);
	
	wrap_rod_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (rods,Box,pbcFlag,nRods);	
}



// --------------------------------------------------------
// Call to "update_rod_position_orientation_fluid_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::update_rod_position_orientation_fluid(int nBlocks, int nThreads)
{
	update_rod_position_orientation_fluid_IBM3D
	<<<nBlocks,nThreads>>> (rods,dt,fricT,fricR,nRods);
	
	wrap_rod_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (rods,Box,pbcFlag,nRods);	
}



// --------------------------------------------------------
// Call to "update_rod_position_fluid_IBM3D" kernel: 
// --------------------------------------------------------

void class_rods_ibm3D::update_rod_position_fluid(int nBlocks, int nThreads)
{
	update_rod_position_fluid_IBM3D
	<<<nBlocks,nThreads>>> (rods,dt,fricT,nRods);
	
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
// Call to "sum_rod_forces_torques_moments_IBM3D" kernel:
// --------------------------------------------------------

void class_rods_ibm3D::sum_rod_forces_torques_moments(int nBlocks, int nThreads)
{
	sum_rod_forces_torques_moments_IBM3D
	<<<nBlocks,nThreads>>> (beads,rods,nBeadsPerRod,nBeads);
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
// Call to kernel that calculates wall forces in radial
// direction for cylindrical channel:
// --------------------------------------------------------

void class_rods_ibm3D::compute_wall_forces_cylinder(float chRad, int nBlocks, int nThreads)
{
	bead_wall_forces_cylinder_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,chRad,repA,repD/2.0,nBeads);
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
	nBeads,nBeadsPerRod,beadsH,rodsH);
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
		beadsH[i].rm1 = beadsH[i].rm1 + roundf(rij/Box)*Box*pbcFlag; // PBC's	
	}	
}






