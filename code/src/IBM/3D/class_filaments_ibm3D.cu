 
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
	repA = inputParams("IBM_FILAMS/repA",0.0);
	repD = inputParams("IBM_FILAMS/repD",0.0);
	beadFmax = inputParams("IBM_FILAMS/beadFmax",1000.0);
	gam = inputParams("IBM_FILAMS/gamma",0.1);
	ibmUpdate = inputParams("IBM/ibmUpdate","verlet");
	
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
		sizeBins = inputParams("IBM_FILAMS/sizeBins",2.0);
		binMax = inputParams("IBM_FILAMS/binMax",1);			
		numBins.x = int(floor(N.x/sizeBins));
	    numBins.y = int(floor(N.y/sizeBins));
	    numBins.z = int(floor(N.z/sizeBins));
		nBins = numBins.x*numBins.y*numBins.z;
		nnbins = 26;
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
	if (binsFlag) {
		cudaMalloc((void **) &binMembers, nBins*binMax*sizeof(int));
		cudaMalloc((void **) &binOccupancy, nBins*sizeof(int));
		cudaMalloc((void **) &binMap, nBins*26*sizeof(int));		
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
	if (binsFlag) {
		cudaFree(binMembers);
		cudaFree(binOccupancy);
		cudaFree(binMap);		
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

void class_filaments_ibm3D::read_ibm_information(std::string tagname)
{
	// read data from file:
	//read_ibm_filament_information(tagname,nBeadsPerFilam,nEdgesPerFilam,beadsH,edgesH);
	
	// set up indices for each filament:
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
	return max(nBeads,nEdges);
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
			}
			// copy edge info:
			for (int i=0; i<filamsH[0].nEdges; i++) {
				int ii = i + filamsH[f].indxE0;
				edgesH[ii].b0 = edgesH[i].b0 + filamsH[f].indxB0;
				edgesH[ii].b1 = edgesH[i].b1 + filamsH[f].indxB0;
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
	float a = M_PI*(float)rand()/RAND_MAX;  // alpha
	float b = M_PI*(float)rand()/RAND_MAX;  // beta
	float g = M_PI*(float)rand()/RAND_MAX;  // gamma
	
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
// Write IBM output to file:
// --------------------------------------------------------

void class_filaments_ibm3D::write_output(std::string tagname, int tagnum)
{
	//write_vtk_immersed_boundary_3D(tagname,tagnum,nNodes,nFaces,rH,facesH);
	//write_vtk_immersed_boundary_3D_cellID(tagname,tagnum,nNodes,nFaces,rH,facesH,cellsH);
}



// --------------------------------------------------------
// Write IBM output to file, including more information
// (edge angles):
// --------------------------------------------------------

void class_filaments_ibm3D::write_output_long(std::string tagname, int tagnum)
{
	//write_vtk_immersed_boundary_normals_3D(tagname,tagnum,
	//nNodes,nFaces,nEdges,rH,facesH,edgesH);
}



// --------------------------------------------------------
// Calculate rest geometries (Skalak model):
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

void class_filaments_ibm3D::stepIBM(class_scsp_D3Q19& lbm, int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	//  here, the velocity-Verlet algorithm is used to update the 
	//  node positions - using a viscous drag force proportional
	//  to the difference between the node velocities and the 
	//  fluid velocities
	// ----------------------------------------------------------
	
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);
	
	// first step of IBM velocity verlet:
	update_bead_positions_verlet_1(nBlocks,nThreads);
	
	// re-build bin lists for IBM nodes:
	reset_bin_lists(nBlocks,nThreads);
	build_bin_lists(nBlocks,nThreads);
			
	// update IBM:
	compute_bead_forces(nBlocks,nThreads);
	nonbonded_bead_interactions(nBlocks,nThreads);
	compute_wall_forces(nBlocks,nThreads);
	enforce_max_bead_force(nBlocks,nThreads);
	lbm.viscous_force_filaments_IBM_LBM(nBlocks,nThreads,gam,beads,nBeads);
	update_bead_positions_verlet_2(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Take step forward for IBM w/o fluid:
// (note: this uses the velocity-Verlet algorithm)
// --------------------------------------------------------

void class_filaments_ibm3D::stepIBM_no_fluid(int nSteps, bool zeroFlag, int nBlocks, int nThreads) 
{		
	// use distinct (smaller) value of repA:
	float repA0 = repA;
	repA = 0.0001;
	
	for (int i=0; i<nSteps; i++) {
		// first step of IBM velocity verlet:
		update_bead_positions_verlet_1(nBlocks,nThreads);

		// re-build bin lists for IBM nodes:
		reset_bin_lists(nBlocks,nThreads);
		build_bin_lists(nBlocks,nThreads);
		
		// update IBM:
		compute_bead_forces(nBlocks,nThreads);
		nonbonded_bead_interactions(nBlocks,nThreads);
		compute_wall_forces(nBlocks,nThreads);
		add_drag_force_to_beads(0.001,nBlocks,nThreads);
		enforce_max_bead_force(nBlocks,nThreads);
		update_bead_positions_verlet_2(nBlocks,nThreads);
	}
	if (zeroFlag) zero_bead_velocities_forces(nBlocks,nThreads); 
	
	// reset repA to intended value:
	repA = repA0;
}











// **********************************************************************************************
// Calls to CUDA kernels for main calculations
// **********************************************************************************************












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
// Call to "zero_bead_velocities_forces_IBM3D" kernel:
// --------------------------------------------------------

void class_filaments_ibm3D::zero_bead_velocities_forces(int nBlocks, int nThreads)
{
	zero_bead_velocities_forces_IBM3D
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

void class_filaments_ibm3D::add_drag_force_to_beads(float dragcoeff, int nBlocks, int nThreads)
{
	add_drag_force_to_bead_IBM3D
	<<<nBlocks,nThreads>>> (beads,dragcoeff,nBeads);
}



// --------------------------------------------------------
// Call to kernel that builds the binMap array:
// --------------------------------------------------------

void class_filaments_ibm3D::build_binMap(int nBlocks, int nThreads)
{
	if (nFilams > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;	
		cout << "nnbins = " << nnbins << endl;	
		build_binMap_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (binMap,numBins,nnbins,nBins);
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
		<<<nBlocks,nThreads>>> (binOccupancy,binMembers,binMax,nBins);
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
		<<<nBlocks,nThreads>>> (beads,binOccupancy,binMembers,numBins,sizeBins,nBeads,binMax);
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
		<<<nBlocks,nThreads>>> (beads,binOccupancy,binMembers,binMap,numBins,sizeBins,
		                        repA,repD,nBeads,binMax,nnbins,Box,pbcFlag);
	}	
}



// --------------------------------------------------------
// Calls to kernels that compute forces on nodes based 
// on the membrane mechanics model (Spring model):
// --------------------------------------------------------

void class_filaments_ibm3D::compute_bead_forces(int nBlocks, int nThreads)
{
	// First, zero the bead forces
	zero_bead_forces_IBM3D
	<<<nBlocks,nThreads>>> (beads,nBeads);
		
	// Second, unwrap node coordinates:
	unwrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,filams,Box,pbcFlag,nBeads);	
			
	// Forth, compute the edge extension and bending force for each edge:
	compute_bead_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,edges,filams,nEdges);
	
	compute_bead_force_bending_IBM3D
	<<<nBlocks,nThreads>>> (beads,filams,nBeads);
			
	// Seventh, re-wrap bead coordinates:
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














// **********************************************************************************************
// Analysis and Geometry calculations done by the host (CPU)
// **********************************************************************************************











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



