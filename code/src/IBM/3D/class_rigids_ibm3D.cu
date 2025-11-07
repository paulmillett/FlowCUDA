
# include "class_rigids_ibm3D.cuh"
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
# include <random>
using namespace std;  








// **********************************************************************************************
// Constructor, destructor, and array allocations...
// **********************************************************************************************








// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_rigids_ibm3D::class_rigids_ibm3D()
{
	// get some parameters:
	GetPot inputParams("input.dat");
	
	// mesh attributes	
	nNodesPerBody = inputParams("IBM/nNodesPerBody",0);
	nBodies = inputParams("IBM/nBodies",1);
	nNodes = nNodesPerBody*nBodies;
		
	// mechanical properties
	dt = inputParams("Time/dt",1.0);
	repA = inputParams("IBM/repA",0.0);
	repD = inputParams("IBM/repD",0.0);
	repFmax = inputParams("IBM/repFmax",1000.0);
	bodyFmax = inputParams("IBM/bodyFmax",1000.0);
	bodyTmax = inputParams("IBM/bodyTmax",1000.0);
	channelShape = inputParams("Lattice/channelShape","rectangle");
	chRad = inputParams("Lattice/chRad",10.0);
	
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
	if (nBodies > 1) binsFlag = true;
	if (binsFlag) {
		bins.sizeBins = inputParams("IBM/sizeBins",2.0);
		bins.binMax = inputParams("IBM/binMax",1);			
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

class_rigids_ibm3D::~class_rigids_ibm3D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_rigids_ibm3D::allocate()
{
	// allocate array memory (host):
	nodesH = (rigidnode*)malloc(nNodes*sizeof(rigidnode));
	bodiesH = (rigid*)malloc(nBodies*sizeof(rigid));		
					
	// allocate array memory (device):
	cudaMalloc((void **) &nodes, nNodes*sizeof(rigidnode));
	cudaMalloc((void **) &bodies, nBodies*sizeof(rigid));
	if (binsFlag) allocate_bin_arrays();
}



// --------------------------------------------------------
// Allocate bin arrays:
// --------------------------------------------------------

void class_rigids_ibm3D::allocate_bin_arrays()
{
	cudaMalloc((void **) &bins.binMembers, bins.nBins*bins.binMax*sizeof(int));
	cudaMalloc((void **) &bins.binOccupancy, bins.nBins*sizeof(int));
	cudaMalloc((void **) &bins.binMap, bins.nBins*26*sizeof(int));	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_rigids_ibm3D::deallocate()
{
	// free array memory (host):
	free(nodesH);
	free(bodiesH);
					
	// free array memory (device):
	cudaFree(nodes);	
	cudaFree(bodies);
	if (binsFlag) {
		cudaFree(bins.binMembers);
		cudaFree(bins.binOccupancy);
		cudaFree(bins.binMap);	
	}		
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_rigids_ibm3D::memcopy_host_to_device()
{
	cudaMemcpy(nodes, nodesH, sizeof(rigidnode)*nNodes, cudaMemcpyHostToDevice);	
	cudaMemcpy(bodies, bodiesH, sizeof(rigid)*nBodies, cudaMemcpyHostToDevice);
}
	


// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_rigids_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(nodesH, nodes, sizeof(rigidnode)*nNodes, cudaMemcpyDeviceToHost);
		
	// unwrap coordinate positions:
	unwrap_node_coordinates(); 
}














// **********************************************************************************************
// Initialization Stuff...
// **********************************************************************************************












// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void class_rigids_ibm3D::read_ibm_information(std::string tagname)
{
	// read data from file:
	read_ibm_information_just_nodes(tagname,nNodesPerBody,nodesH);
	
	// set up indices for each body:
	for (int c=0; c<nBodies; c++) {
		bodiesH[c].nNodes = nNodesPerBody;
		bodiesH[c].indxN0 = c*nNodesPerBody;  // here, all cells are identical,
	}
}



// --------------------------------------------------------
// Initialize bins ONLY if bins have not yet been 
// initialized (e.g. when nBodies = 1):
// --------------------------------------------------------

void class_rigids_ibm3D::initialize_bins()
{
	// only do this if it hasn't been done yet!
	if (!binsFlag) {
		binsFlag = true;
		GetPot inputParams("input.dat");
		bins.sizeBins = inputParams("IBM/sizeBins",2.0);
		bins.binMax = inputParams("IBM/binMax",1);			
		bins.numBins.x = int(floor(N.x/bins.sizeBins));
	    bins.numBins.y = int(floor(N.y/bins.sizeBins));
	    bins.numBins.z = int(floor(N.z/bins.sizeBins));
		bins.nBins = bins.numBins.x*bins.numBins.y*bins.numBins.z;
		bins.nnbins = 26;
		allocate_bin_arrays();
	}	
}



// --------------------------------------------------------
// Setters:
// --------------------------------------------------------

void class_rigids_ibm3D::set_pbcFlag(int x, int y, int z)
{
	pbcFlag.x = x; pbcFlag.y = y; pbcFlag.z = z;
}

void class_rigids_ibm3D::set_cells_types(int val)
{
	// set cellType for ALL cells:
	for (int c=0; c<nBodies; c++) {
		bodiesH[c].cellType = val;
	}
}

void class_rigids_ibm3D::set_cell_type(int cID, int val)
{
	// set cellType for ONE cell:
	bodiesH[cID].cellType = val;
}

int class_rigids_ibm3D::get_max_array_size()
{
	// return the maximum array size:
	int maxSize = nNodes;
	if (binsFlag) {
		if (bins.nBins > maxSize) maxSize = bins.nBins;
	}
	return maxSize;
}



// --------------------------------------------------------
// Assign the reference node to every cell.  The reference
// node is arbitrary (here we use the first node), but it
// is necessary for handling PBC's.
// --------------------------------------------------------

void class_rigids_ibm3D::assign_refNode_to_cells()
{
	for (int c=0; c<nBodies; c++) {
		bodiesH[c].refNode = bodiesH[c].indxN0;
	}
}	



// --------------------------------------------------------
// Assign the cell ID to every node:
// --------------------------------------------------------

void class_rigids_ibm3D::assign_cellIDs_to_nodes()
{
	for (int c=0; c<nBodies; c++) {
		int istr = bodiesH[c].indxN0;
		int iend = istr + bodiesH[c].nNodes;
		for (int i=istr; i<iend; i++) nodesH[i].cellID = c;
	}
}



// --------------------------------------------------------
// Duplicate the first cell mesh information to all cells:
// --------------------------------------------------------

void class_rigids_ibm3D::duplicate_cells()
{
	if (nBodies > 1) {
		for (int c=1; c<nBodies; c++) {
			
			// skip if cell 0 is different than cell c:
			if (bodiesH[0].nNodes != bodiesH[c].nNodes) {
				cout << "duplicate cells error: cells have different nNodes" << endl;
				continue;
			}
			
			// copy node positions:
			for (int i=0; i<bodiesH[0].nNodes; i++) {
				int ii = i + bodiesH[c].indxN0;
				nodesH[ii].r = nodesH[i].r;
			}
			
		}
	}
	
}



// --------------------------------------------------------
// Calculate a node's position relative to it's rigid-body
// center-of-mass.  It is assumed that at this point, the
// body center-of-mass is at the origin (from the read-file):
// --------------------------------------------------------

void class_rigids_ibm3D::relative_node_position_versus_com()
{
	for (int i=0; i<nNodes; i++) {
		nodesH[i].delta = nodesH[i].r;
	}	
}



// --------------------------------------------------------
// Calculate a cylindrical rigid-body's mass and 
// principal moments of inertia:
// --------------------------------------------------------

void class_rigids_ibm3D::cell_mass_moment_of_inertia_cylinder(float L, float R)
{
	for (int c=0; c<nBodies; c++) {
		// assume body density = fluid density
		float rho = 1.0;   // LBM fluid density
		float vol = L*(M_PI*R*R);
		float mass = rho*vol;
		bodiesH[c].mass = mass;
		// assume cylinder is aligned with the x-axis
		bodiesH[c].I.x = (mass*R*R)/2.0;
		bodiesH[c].I.y = (mass*(3.0*R*R + L*L))/12.0;
		bodiesH[c].I.z = (mass*(3.0*R*R + L*L))/12.0;
		// initialize other parameters to zero
		bodiesH[c].L = make_float3(0.0f,0.0f,0.0f);
		bodiesH[c].f = make_float3(0.0f,0.0f,0.0f);
		bodiesH[c].t = make_float3(0.0f,0.0f,0.0f);
		bodiesH[c].vel = make_float3(0.0f,0.0f,0.0f);
		bodiesH[c].omega = make_float3(0.0f,0.0f,0.0f);
		bodiesH[c].q.set_values(1.0f,0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// randomize cell positions and orientations:
// --------------------------------------------------------

void class_rigids_ibm3D::randomize_cells(float sepWall)
{
	// copy node positions from device to host:
	cudaMemcpy(nodesH, nodes, sizeof(rigidnode)*nNodes, cudaMemcpyDeviceToHost);
	
	// assign random position and orientation to each cell:
	for (int c=0; c<nBodies; c++) {
		float3 shift = make_float3(0.0,0.0,0.0);
		// get random position
		shift.x = (float)rand()/RAND_MAX*Box.x;
		shift.y = sepWall + (float)rand()/RAND_MAX*(Box.y-2.0*sepWall);
		shift.z = sepWall + (float)rand()/RAND_MAX*(Box.z-2.0*sepWall);
		rotate_and_shift_node_positions(c,shift.x,shift.y,shift.z);
	}
	
	// copy node positions from host to device:
	cudaMemcpy(nodes, nodesH, sizeof(rigidnode)*nNodes, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// randomize cylindrical capsules positions, but all oriented
// in x-direction inside a cylindrical channel:
// --------------------------------------------------------

void class_rigids_ibm3D::randomize_capsules_xdir_alligned_cylinder(float L, float R, float sepWall, float sepMin)
{
	// copy node positions from device to host:
	cudaMemcpy(nodesH, nodes, sizeof(rigidnode)*nNodes, cudaMemcpyDeviceToHost);	
	
	// assign random position and orientation to each filament:
	float3* cellCOM = (float3*)malloc(nBodies*sizeof(float3));
	for (int c=0; c<nBodies; c++) {
		cellCOM[c] = make_float3(0.0);
		float3 shift = make_float3(0.0,0.0,0.0);	
		bool tooClose = true;
		while (tooClose) {
			// reset tooClose to false
			tooClose = false;
			// get random position
			float rad = (float)rand()/RAND_MAX*(chRad - sepWall);
			float ang = (float)rand()/RAND_MAX*(2*M_PI);
			shift.x = (float)rand()/RAND_MAX*Box.x;		
			shift.y = rad*cos(ang) + (Box.y-1.0)/2.0;
			shift.z = rad*sin(ang) + (Box.z-1.0)/2.0;		
			// check with other cells
			for (int d=0; d<c; d++) {
				if (cylinder_overlap(shift,cellCOM[d],L,R,sepMin)) {                    
					tooClose = true;
                    break;
				}				
			}			
		}
		cellCOM[c] = shift;	
		bodiesH[c].com = shift;	
		shift_node_positions(c,shift.x,shift.y,shift.z);
	}
	
	// last, copy node positions from host to device:
	cudaMemcpy(nodes, nodesH, sizeof(rigidnode)*nNodes, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// randomize cylindrical capsules positions, but all oriented
// in x-direction inside a cylindrical channel.  Here,
// cylindrical capsules are put in groups with the same
// x-position and random radial positions.
// --------------------------------------------------------

void class_rigids_ibm3D::semi_randomize_capsules_xdir_alligned_cylinder(float L, float R, float sepWall, float sepMin)
{
	// copy node positions from device to host:
	cudaMemcpy(nodesH, nodes, sizeof(rigidnode)*nNodes, cudaMemcpyDeviceToHost);	
	
	// calculate 'group' parameters:
	int nGroups = int(Box.x/(L+sepMin));
	int nBodies_per_group = nBodies/nGroups;
	if (nBodies%nGroups > 0) nBodies_per_group += 1;  // in case nBodies/nGroups is not a round number
	float groupLength = Box.x/float(nGroups);
	
	cout << "number of groups = " << nGroups << endl;
	cout << "number of capsules per group = " << nBodies_per_group << endl;
	cout << "group length = " << groupLength << endl;
	
	// arrays needed for positioning cells:
	float3* cellCOM = (float3*)malloc(nBodies*sizeof(float3));
	int* cellGroup = (int*)malloc(nBodies*sizeof(int));
	
	// assign random position and orientation to each filament:
	for (int c=0; c<nBodies; c++) {
		cellCOM[c] = make_float3(0.0);
		cellGroup[c] = c/nBodies_per_group;
		float3 shift = make_float3(0.0,0.0,0.0);	
		bool tooClose = true;
		while (tooClose) {
			// reset tooClose to false
			tooClose = false;
			// get random position
			float rad = (float)rand()/RAND_MAX*(chRad - sepWall);
			float ang = (float)rand()/RAND_MAX*(2*M_PI);
			shift.x = groupLength/2.0 + cellGroup[c]*groupLength;
			
			if (shift.x > Box.x) cout << shift.x << " " << cellGroup[c] << endl;
			
			shift.y = rad*cos(ang) + (Box.y-1.0)/2.0;
			shift.z = rad*sin(ang) + (Box.z-1.0)/2.0;		
			// check with other cells
			for (int d=0; d<c; d++) {
				// only check overlap for cells in the same group
				if (cellGroup[d] != cellGroup[c]) continue;
				if (cylinder_overlap(shift,cellCOM[d],L,R,sepMin)) {                    
					tooClose = true;
                    break;
				}				
			}			
		}
		cellCOM[c] = shift;	
		bodiesH[c].com = shift;	
		shift_node_positions(c,shift.x,shift.y,shift.z);
	}
	
	// last, copy node positions from host to device:
	cudaMemcpy(nodes, nodesH, sizeof(rigidnode)*nNodes, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// calculate separation distance using PBCs:
// --------------------------------------------------------

float class_rigids_ibm3D::calc_separation_pbc(float3 r1, float3 r2)
{
	float3 dr = r1 - r2;
	dr -= roundf(dr/Box)*Box;
	return length(dr);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_rigids_ibm3D::shift_node_positions(int cID, float xsh, float ysh, float zsh)
{
	int istr = bodiesH[cID].indxN0;
	int iend = istr + bodiesH[cID].nNodes;
	for (int i=istr; i<iend; i++) {
		nodesH[i].r.x += xsh;
		nodesH[i].r.y += ysh;
		nodesH[i].r.z += zsh;
	}
	bodiesH[cID].com = make_float3(xsh,ysh,zsh);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_rigids_ibm3D::rotate_and_shift_node_positions(int cID, float xsh, float ysh, float zsh, float a, float b, float g)
{
	// update node positions:
	int istr = bodiesH[cID].indxN0;
	int iend = istr + bodiesH[cID].nNodes;
	for (int i=istr; i<iend; i++) {
		// rotate:
		float xrot = nodesH[i].r.x*(cos(a)*cos(b)) + nodesH[i].r.y*(cos(a)*sin(b)*sin(g)-sin(a)*cos(g)) + nodesH[i].r.z*(cos(a)*sin(b)*cos(g)+sin(a)*sin(g));
		float yrot = nodesH[i].r.x*(sin(a)*cos(b)) + nodesH[i].r.y*(sin(a)*sin(b)*sin(g)+cos(a)*cos(g)) + nodesH[i].r.z*(sin(a)*sin(b)*cos(g)-cos(a)*sin(g));
		float zrot = nodesH[i].r.x*(-sin(b))       + nodesH[i].r.y*(cos(b)*sin(g))                      + nodesH[i].r.z*(cos(b)*cos(g));
		// shift:		 
		nodesH[i].r.x = xrot + xsh;
		nodesH[i].r.y = yrot + ysh;
		nodesH[i].r.z = zrot + zsh;			
	}
	bodiesH[cID].com = make_float3(xsh,ysh,zsh);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_rigids_ibm3D::rotate_and_shift_node_positions(int cID, float xsh, float ysh, float zsh)
{
	// random rotation angles:
	float a = M_PI*(float)rand()/RAND_MAX;  // alpha
	float b = M_PI*(float)rand()/RAND_MAX;  // beta
	float g = M_PI*(float)rand()/RAND_MAX;  // gamma
	
	// update node positions:
	int istr = bodiesH[cID].indxN0;
	int iend = istr + bodiesH[cID].nNodes;
	for (int i=istr; i<iend; i++) {
		// rotate:
		float xrot = nodesH[i].r.x*(cos(a)*cos(b)) + nodesH[i].r.y*(cos(a)*sin(b)*sin(g)-sin(a)*cos(g)) + nodesH[i].r.z*(cos(a)*sin(b)*cos(g)+sin(a)*sin(g));
		float yrot = nodesH[i].r.x*(sin(a)*cos(b)) + nodesH[i].r.y*(sin(a)*sin(b)*sin(g)+cos(a)*cos(g)) + nodesH[i].r.z*(sin(a)*sin(b)*cos(g)-cos(a)*sin(g));
		float zrot = nodesH[i].r.x*(-sin(b))       + nodesH[i].r.y*(cos(b)*sin(g))                      + nodesH[i].r.z*(cos(b)*cos(g));
		// shift:		 
		nodesH[i].r.x = xrot + xsh;
		nodesH[i].r.y = yrot + ysh;
		nodesH[i].r.z = zrot + zsh;			
	}
	bodiesH[cID].com = make_float3(xsh,ysh,zsh);
}



// --------------------------------------------------------
// Write IBM output to file:
// --------------------------------------------------------

void class_rigids_ibm3D::write_output(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_3D_rigid_bodies(tagname,tagnum,nNodes,nodesH);
}



// --------------------------------------------------------
// Calculate wall forces:
// --------------------------------------------------------

void class_rigids_ibm3D::compute_wall_forces(int nBlocks, int nThreads)
{
	if (channelShape == "rectangle") {
		if (pbcFlag.y==0 && pbcFlag.z==1) wall_forces_ydir(nBlocks,nThreads);
		if (pbcFlag.y==1 && pbcFlag.z==0) wall_forces_zdir(nBlocks,nThreads);
		if (pbcFlag.y==0 && pbcFlag.z==0) wall_forces_ydir_zdir(nBlocks,nThreads);
	}
	
	if (channelShape == "cylinder") {
		wall_forces_cylinder(chRad,nBlocks,nThreads);
	}
	
} 



// --------------------------------------------------------
// Take step forward for IBM using LBM object:
// --------------------------------------------------------

void class_rigids_ibm3D::stepIBM(class_scsp_D3Q19& lbm, int nBlocks, int nThreads) 
{
			
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);

	// re-build bin lists for IBM nodes:
	if (nBodies > 1) {
		reset_bin_lists(nBlocks,nThreads);
		build_bin_lists(nBlocks,nThreads);
	}		
		
	// zero forces/torques for nodes and rigid bodies:
	zero_node_forces(nBlocks,nThreads);
	zero_rigid_body_forces_torques(nBlocks,nThreads);
	
	// calculate forces on nodes (hydrodynamic, interactive, wall, etc.):
	lbm.hydrodynamic_force_rigid_node(nBlocks,nThreads,nodes,nNodes);
	if (nBodies > 1) nonbonded_node_interactions(nBlocks,nThreads);
	compute_wall_forces(nBlocks,nThreads);
	
	// sum node forces/torques on rigid-bodies:
	sum_rigid_forces_torques(nBlocks,nThreads);
	
	// update rigid-body positions and orientations:
	update_rigid_body(nBlocks,nThreads);
	
	// update node positions & velocities based on new rigid-body data:
	update_node_positions_velocities(nBlocks,nThreads);
			
}











// **********************************************************************************************
// Calls to CUDA kernels for main calculations
// **********************************************************************************************













// --------------------------------------------------------
// Call to "zero_node_forces_rigid_IBM3D" kernel:
// --------------------------------------------------------

void class_rigids_ibm3D::zero_node_forces(int nBlocks, int nThreads)
{
	zero_node_forces_rigid_IBM3D
	<<<nBlocks,nThreads>>> (nodes,nNodes);	
}



// --------------------------------------------------------
// Call to "zero_rigid_forces_torques_IBM3D" kernel:
// --------------------------------------------------------

void class_rigids_ibm3D::zero_rigid_body_forces_torques(int nBlocks, int nThreads)
{
	zero_rigid_forces_torques_IBM3D
	<<<nBlocks,nThreads>>> (bodies,nBodies);	
}



// --------------------------------------------------------
// Call to "enforce_max_rigid_force_torque_IBM3D" kernel:
// --------------------------------------------------------

void class_rigids_ibm3D::enforce_rigid_body_max_forces_torques(int nBlocks, int nThreads)
{
	enforce_max_rigid_force_torque_IBM3D
	<<<nBlocks,nThreads>>> (bodies,bodyFmax,bodyTmax,nBodies);	
}



// --------------------------------------------------------
// Call to "update_node_positions_velocities_rigids_IBM3D" kernel:
// --------------------------------------------------------

void class_rigids_ibm3D::update_node_positions_velocities(int nBlocks, int nThreads)
{
	update_node_positions_velocities_rigids_IBM3D
	<<<nBlocks,nThreads>>> (nodes,bodies,nNodes);
	
	wrap_node_coordinates_rigid_IBM3D
	<<<nBlocks,nThreads>>> (nodes,Box,pbcFlag,nNodes);		
}



// --------------------------------------------------------
// Call to "update_rigid_position_orientation_IBM3D" kernel:
// --------------------------------------------------------

void class_rigids_ibm3D::update_rigid_body(int nBlocks, int nThreads)
{
	update_rigid_position_orientation_IBM3D
	<<<nBlocks,nThreads>>> (bodies,dt,nBodies);	
}



// --------------------------------------------------------
// Call to "sum_rigid_forces_torques_IBM3D" kernel:
// --------------------------------------------------------

void class_rigids_ibm3D::sum_rigid_forces_torques(int nBlocks, int nThreads)
{
	sum_rigid_forces_torques_IBM3D
	<<<nBlocks,nThreads>>> (nodes,bodies,nNodes);	
}



// --------------------------------------------------------
// Call to "unwrap_node_coordinates_rigid_IBM3D" kernel:
// --------------------------------------------------------

void class_rigids_ibm3D::unwrap_node_coordinates(int nBlocks, int nThreads)
{
	unwrap_node_coordinates_rigid_IBM3D
	<<<nBlocks,nThreads>>> (nodes,bodies,Box,pbcFlag,nNodes);	
}



// --------------------------------------------------------
// Call to "wrap_node_coordinates_rigid_IBM3D" kernel:
// --------------------------------------------------------

void class_rigids_ibm3D::wrap_node_coordinates(int nBlocks, int nThreads)
{
	wrap_node_coordinates_rigid_IBM3D
	<<<nBlocks,nThreads>>> (nodes,Box,pbcFlag,nNodes);	
}



// --------------------------------------------------------
// Call to "wrap_rigid_coordinates_IBM3D" kernel:
// --------------------------------------------------------

void class_rigids_ibm3D::wrap_rigid_body_coordinates(int nBlocks, int nThreads)
{
	wrap_rigid_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (bodies,Box,pbcFlag,nBodies);	
}



// --------------------------------------------------------
// Call to kernel that builds the binMap array:
// --------------------------------------------------------

void class_rigids_ibm3D::build_binMap(int nBlocks, int nThreads)
{
	if (binsFlag) {			
		build_binMap_IBM3D
		<<<nBlocks,nThreads>>> (bins);
	} else {
		cout << "IBM bin arrays have not been initialized" << endl;
	}
}



// --------------------------------------------------------
// Call to kernel that resets bin lists:
// --------------------------------------------------------

void class_rigids_ibm3D::reset_bin_lists(int nBlocks, int nThreads)
{
	if (binsFlag) {	
		reset_bin_lists_IBM3D
		<<<nBlocks,nThreads>>> (bins);
	} else {
		cout << "IBM bin arrays have not been initialized" << endl;
	}
}



// --------------------------------------------------------
// Call to kernel that builds bin lists:
// --------------------------------------------------------

void class_rigids_ibm3D::build_bin_lists(int nBlocks, int nThreads)
{
	if (binsFlag) {	
		build_bin_lists_for_rigid_nodes_IBM3D
		<<<nBlocks,nThreads>>> (nodes,bins,nNodes);	
	} else {
		cout << "IBM bin arrays have not been initialized" << endl;
	}
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_rigids_ibm3D::nonbonded_node_interactions(int nBlocks, int nThreads)
{
	if (binsFlag) {	
		nonbonded_rigid_node_interactions_IBM3D
		<<<nBlocks,nThreads>>> (nodes,bins,repA,repD,nNodes,Box,pbcFlag);
	} else {
		cout << "IBM bin arrays have not been initialized" << endl;
	}
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_rigids_ibm3D::nonbonded_node_lubrication_interactions(float Rad, float nu, int nBlocks, int nThreads)
{
	/*
	if (binsFlag) {	
		nonbonded_node_lubrication_interactions_IBM3D
		<<<nBlocks,nThreads>>> (nodes,cells,bins,Rad,Rad,nu,repD,nNodes,Box,pbcFlag);		
	} else {
		cout << "IBM bin arrays have not been initialized" << endl;
	}
	*/
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir:
// --------------------------------------------------------

void class_rigids_ibm3D::wall_forces_ydir(int nBlocks, int nThreads)
{
	rigid_node_wall_forces_ydir_IBM3D
	<<<nBlocks,nThreads>>> (nodes,Box,repA,repD,nNodes);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in z-dir:
// --------------------------------------------------------

void class_rigids_ibm3D::wall_forces_zdir(int nBlocks, int nThreads)
{
	rigid_node_wall_forces_zdir_IBM3D
	<<<nBlocks,nThreads>>> (nodes,Box,repA,repD,nNodes);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir
// and z-dir:
// --------------------------------------------------------

void class_rigids_ibm3D::wall_forces_ydir_zdir(int nBlocks, int nThreads)
{
	rigid_node_wall_forces_ydir_zdir_IBM3D
	<<<nBlocks,nThreads>>> (nodes,Box,repA,repD,nNodes);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in radial
// direction (assuming channel cross section is yz-plane):
// --------------------------------------------------------

void class_rigids_ibm3D::wall_forces_cylinder(float chRad, int nBlocks, int nThreads)
{
	rigid_node_wall_forces_cylinder_IBM3D
	<<<nBlocks,nThreads>>> (nodes,Box,chRad,repA,repD,nNodes);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in radial
// direction (assuming channel cross section is yz-plane):
// --------------------------------------------------------

void class_rigids_ibm3D::wall_lubrication_forces_cylinder(float chRad, float cellRad, float nu, int nBlocks, int nThreads)
{
	/*
	wall_lubrication_forces_cylinder_IBM3D
	<<<nBlocks,nThreads>>> (nodes,Box,chRad,cellRad,nu,repD,nNodes);
	*/
}












// **********************************************************************************************
// Analysis and Geometry calculations done by the host (CPU)
// **********************************************************************************************











// --------------------------------------------------------
// Unwrap node coordinates based on difference between node
// position and the cell's reference node position:
// --------------------------------------------------------

void class_rigids_ibm3D::unwrap_node_coordinates()
{
	for (int i=0; i<nNodes; i++) {
		int c = nodesH[i].cellID;
		int j = bodiesH[c].refNode;
		float3 rij = nodesH[j].r - nodesH[i].r;
		nodesH[i].r = nodesH[i].r + roundf(rij/Box)*Box*pbcFlag; // PBC's		
	}	
}



// --------------------------------------------------------
// Write capsule data to file "vtkoutput/capsule_data.dat"
// --------------------------------------------------------

void class_rigids_ibm3D::output_capsule_data()
{
		
	// -----------------------------------------
	// Define the file location and name:
	// -----------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << "capsule_data.dat";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------
	// Write to file:
	// -----------------------------------------
	
	for (int c=0; c<nBodies; c++) {
		
		outfile << fixed << setprecision(2) << setw(2)  << bodiesH[c].cellType << " " <<			                                
							setprecision(4) << setw(10) << bodiesH[c].com.x    << " " << setw(10) << bodiesH[c].com.y   << " " << setw(10) << bodiesH[c].com.z << " " <<
							setprecision(6) << setw(10) << bodiesH[c].vel.x    << " " << setw(10) << bodiesH[c].vel.y   << " " << setw(10) << bodiesH[c].vel.z << endl;
		
	}
	
}



// --------------------------------------------------------
// Calculate the orientation of the capsules (used for
// cylindrically-shaped capsules).
// --------------------------------------------------------

void class_rigids_ibm3D::capsule_orientation_cylinders(int nNodesLength, int step)
{
	
	// -----------------------------------------
	// Define the file location and name:
	// -----------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << "cylinder_orientation.dat";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------
	// Loop over the capsules  
	// -----------------------------------------
		
	for (int c=0; c<nBodies; c++) {		
		// the first node (at one side of cylinder): 
		int N0 = bodiesH[c].indxN0;		
		// the second node (at the other side of cylinder):
		int N1 = N0 + nNodesLength - 1;
		// orientation vector:
		bodiesH[c].p = normalize(nodesH[N0].r - nodesH[N1].r);
		// radial distance to channel centerline:
		float ymid = (Box.y-1.0)/2.0;
		float zmid = (Box.z-1.0)/2.0;
		float yi = bodiesH[c].com.y - ymid;
		float zi = bodiesH[c].com.z - zmid;
		float ri = sqrt(yi*yi + zi*zi);
		// print data:
		outfile << fixed << setprecision(4) << step << "  " << c << "  " << bodiesH[c].p.x << "  " 
			                                                             << bodiesH[c].p.y << "  " 
																		 << bodiesH[c].p.z << "  "
																		 << bodiesH[c].com.x << "  "
																		 << bodiesH[c].com.y << "  " 
																		 << bodiesH[c].com.z << "  "
																		 << ri << endl;		
	}

}



// --------------------------------------------------------
// Check if two cylindrically-shaped capsules are overlapping,
// assuming their major axis is alligned in the x-direction.
//
// Also, assuming PBC's in the x-direction ONLY.
//
// --------------------------------------------------------

bool class_rigids_ibm3D::cylinder_overlap(float3 iCOM, float3 jCOM, float L, float R, float sepMin)
{
	bool flag = false;
		
	// 1st: check how close cylinders are in the x-dir:
	float3 dr = iCOM - jCOM;
	dr -= roundf(dr/Box)*Box;	
	if (abs(dr.x) > (L + sepMin)) {
		return flag;  // (return if cylinders are not close enough in x-dir)
	}
	
	// 2nd: check how close cylinders are in the yz-plane:
	float ryz = sqrt(dr.y*dr.y + dr.z*dr.z);	
	if (ryz < (2*R + sepMin)) {
		flag = true;
	}
	
	return flag;	
}




