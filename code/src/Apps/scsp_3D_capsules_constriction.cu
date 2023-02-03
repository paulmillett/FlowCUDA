
# include "scsp_3D_capsules_constriction.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_capsules_constriction::scsp_3D_capsules_constriction() : lbm(),ibm()
{		
	
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	
	// ----------------------------------------------
	// lattice parameters:
	// ----------------------------------------------
	
	nVoxels = inputParams("Lattice/nVoxels",0);
	Q = inputParams("Lattice/Q",19);
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = inputParams("Lattice/Nz",1);	
	
	// ----------------------------------------------
	// GPU parameters:
	// ----------------------------------------------
	
	nThreads = inputParams("GPU/nThreads",512);
	nBlocks = (nVoxels+(nThreads-1))/nThreads;  // integer division
	
	// ----------------------------------------------
	// time parameters:
	// ----------------------------------------------
	
	nSteps = inputParams("Time/nSteps",0);
	
	// ----------------------------------------------
	// Lattice Boltzmann parameters:
	// ----------------------------------------------
	
	nu = inputParams("LBM/nu",0.1666666);
	bodyForx = inputParams("LBM/bodyForx",0.0);
	
	// ----------------------------------------------
	// Immersed-Boundary parameters:
	// ----------------------------------------------
	
	int nNodesPerCell = inputParams("IBM/nNodesPerCell",0);
	int nCells = inputParams("IBM/nCells",1);
	nNodes = nNodesPerCell*nCells;
	
	// ----------------------------------------------
	// IBM set flags for PBC's:
	// ----------------------------------------------
	
	ibm.set_pbcFlag(1,0,0);
		
	// ----------------------------------------------
	// iolets parameters:
	// ----------------------------------------------
	
	numIolets = inputParams("Lattice/numIolets",2);
	
	// ----------------------------------------------
	// output parameters:
	// ----------------------------------------------
	
	vtkFormat = inputParams("Output/format","polydata");
	iskip = inputParams("Output/iskip",1);
	jskip = inputParams("Output/jskip",1);
	kskip = inputParams("Output/kskip",1);
	
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	lbm.allocate_forces();
	lbm.allocate_solid();
	ibm.allocate();	
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_capsules_constriction::~scsp_3D_capsules_constriction()
{
	lbm.deallocate();
	ibm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_capsules_constriction::initSystem()
{
		
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
	
	// ----------------------------------------------
	// define solid voxels:
	// ----------------------------------------------	
	
	int iNarBegin = inputParams("Lattice/iNarBegin",0);
	int iNarEnd = inputParams("Lattice/iNarEnd",Nx-1);
	int jNarBegin = inputParams("Lattice/jNarBegin",0);
	int jNarEnd = inputParams("Lattice/jNarEnd",Ny-1);
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {		
				int ndx = k*Nx*Ny + j*Nx + i;	
				// default fluid nodes:
				lbm.setR(ndx,1.0);
				lbm.setS(ndx,0);
				// top/bottom walls:
				if (k == 0 || k == Nz-1) {
					lbm.setR(ndx,0.0);
					lbm.setS(ndx,1);
				}
				if (j == 0 || j == Ny-1) {
					lbm.setR(ndx,0.0);
					lbm.setS(ndx,1);
				}
				// constriction:
				if (i >= iNarBegin && i <= iNarEnd && j < jNarBegin) {
					lbm.setR(ndx,0.0);
					lbm.setS(ndx,1);
				}
				if (i >= iNarBegin && i <= iNarEnd && j > jNarEnd) {
					lbm.setR(ndx,0.0);
					lbm.setS(ndx,1);
				}				
			}
		}
	}
	
	// ----------------------------------------------
	// create the lattice taking into account solid
	// walls:
	// ----------------------------------------------	
	
	lbm.create_lattice_box_periodic_solid_walls();
	
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// ----------------------------------------------
		
	lbm.stream_index_pull();
			
	// ----------------------------------------------			
	// initialize velocities: 
	// ----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		lbm.setU(i,0.0);
		lbm.setV(i,0.0);
		lbm.setW(i,0.0);		
	}
	
	// ----------------------------------------------			
	// initialize immersed boundary info: 
	// ----------------------------------------------
		
	ibm.read_ibm_information("sphere.dat");
	ibm.duplicate_cells();
	ibm.assign_cellIDs_to_nodes();
	ibm.assign_refNode_to_cells();	
	ibm.shift_node_positions(0,30.0,49.5,12.5);
	ibm.shift_node_positions(1,90.0,49.5,12.5);
		
	// ----------------------------------------------
	// build the binMap array for neighbor lists: 
	// ----------------------------------------------
	
	ibm.build_binMap(nBlocks,nThreads); 
		
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	lbm.memcopy_host_to_device();
	lbm.memcopy_host_to_device_solid();
	ibm.memcopy_host_to_device();
		
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	lbm.initial_equilibrium(nBlocks,nThreads);	
	
	// ----------------------------------------------
	// calculate rest geometries for membrane: 
	// ----------------------------------------------
	
	ibm.rest_geometries_skalak(nBlocks,nThreads);
	
	// ----------------------------------------------
	// shrink and randomly disperse cells: 
	// ----------------------------------------------
	
	/*
	float scale = 0.7;
	ibm.shrink_and_randomize_cells(scale,16.0,11.0);
	ibm.scale_equilibrium_cell_size(scale,nBlocks,nThreads);
	*/
	
	// ----------------------------------------------
	// relax node positions: 
	// ----------------------------------------------
	
	/*	
	scale = 1.0/0.7;
	ibm.relax_node_positions_skalak(90000,scale,0.02,nBlocks,nThreads);	
	ibm.relax_node_positions_skalak(90000,1.0,0.02,nBlocks,nThreads);
	*/
				
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	ibm.memcopy_device_to_host();
	writeOutput("macros",0);
		
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_capsules_constriction::cycleForward(int stepsPerCycle, int currentCycle)
{
		
	// ----------------------------------------------
	// determine the cummulative number of steps at the
	// beginning of this cycle:
	// ----------------------------------------------
	
	int cummulativeSteps = stepsPerCycle*currentCycle;	
	
	// ----------------------------------------------
	// loop through this cycle:
	// ----------------------------------------------
	
	for (int step=0; step<stepsPerCycle; step++) {
		cummulativeSteps++;	
				
		// zero fluid forces:
		lbm.zero_forces(nBlocks,nThreads);
		
		// re-build bin lists for IBM nodes:
		ibm.reset_bin_lists(nBlocks,nThreads);
		ibm.build_bin_lists(nBlocks,nThreads);
				
		// compute IBM node forces:
		ibm.compute_node_forces_skalak(nBlocks,nThreads);
		ibm.nonbonded_node_interactions(nBlocks,nThreads);
		//ibm.wall_forces_ydir(nBlocks,nThreads);
				
		// update fluid:
		lbm.extrapolate_forces_from_IBM(nBlocks,nThreads,ibm.r,ibm.f,nNodes);
		lbm.add_body_force_with_solid(bodyForx,0.0,0.0,nBlocks,nThreads);
		lbm.stream_collide_save_forcing_solid(nBlocks,nThreads);
		
		// update membrane:
		lbm.interpolate_velocity_to_IBM(nBlocks,nThreads,ibm.r,ibm.v,nNodes);
		ibm.update_node_positions(nBlocks,nThreads);
				
		cudaDeviceSynchronize();

	}
	
	cout << cummulativeSteps << endl;	
		
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
	lbm.memcopy_device_to_host();
	ibm.memcopy_device_to_host();    
	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	writeOutput("macros",cummulativeSteps);
		
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_capsules_constriction::writeOutput(std::string tagname, int step)
{	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// ----------------------------------------------
	
	int precision = 3;
	lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
	ibm.write_output("ibm",step);		
}









