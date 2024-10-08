
# include "scsp_3D_hemisphere_3.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_hemisphere_3::scsp_3D_hemisphere_3() : lbm(),ibm()
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
	
	// ----------------------------------------------
	// Immersed-Boundary parameters:
	// ----------------------------------------------
	
	nNodes = inputParams("IBM/nNodes",0);
	nFaces = inputParams("IBM/nFaces",0);
	nBlocksIB = (nNodes+(nThreads-1))/nThreads; // integer division	
	
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
	lbm.allocate_IB_velocities();
	lbm.allocate_inout();
	ibm.allocate();
	ibm.allocate_faces();
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_hemisphere_3::~scsp_3D_hemisphere_3()
{
	lbm.deallocate();
	ibm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_hemisphere_3::initSystem()
{
	
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
	
	// ----------------------------------------------
	// create the lattice using "box" function.
	// function location:
	// "lattice/lattice_builders_D2Q9.cuh"	 
	// ----------------------------------------------	
	
	if (latticeSource == "box") {
		lbm.create_lattice_box();
	}	
	
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// ----------------------------------------------
		
	lbm.stream_index_pull();
	
	// ----------------------------------------------			
	// initialize inlets/outlets: 
	// ----------------------------------------------
	
	lbm.read_iolet_info(0,"Iolet1");
	lbm.read_iolet_info(1,"Iolet2");
	lbm.read_iolet_info(2,"Iolet3");	
			
	// ----------------------------------------------			
	// edit inlet condition: 
	// ----------------------------------------------
	
	Nx = inputParams("Lattice/Nx",0);
	Ny = inputParams("Lattice/Ny",0);
	Nz = inputParams("Lattice/Nz",0);
	
	// reset top z-surface to bounce-back
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			int k = Nz - 1;
			int ndx = k*Nx*Ny + j*Nx + i;
			lbm.setVoxelType(ndx,0);
		}
	}
	
	// iolet #2:
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			int k = Nz - 1;
			int ndx = k*Nx*Ny + j*Nx + i;
			float rx = float(i - 84);
			float ry = float(j - 60);
			float rr = sqrt(rx*rx + ry*ry);
			if (rr <= 10.0) {
				lbm.setVoxelType(ndx,2);
			} 
		}	
	}
	
	// iolet #3:
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			int k = Nz - 1;
			int ndx = k*Nx*Ny + j*Nx + i;
			float rx = float(i - 36);
			float ry = float(j - 60);
			float rr = sqrt(rx*rx + ry*ry);
			if (rr <= 10.0) {
				lbm.setVoxelType(ndx,3);
			} 
		}	
	}
		
	// ----------------------------------------------			
	// initialize macros: 
	// ----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		lbm.setU(i,0.0);
		lbm.setV(i,0.0);
		lbm.setW(i,0.0);
		lbm.setR(i,1.0);		
	}
	
	// ----------------------------------------------			
	// initialize immersed boundary info: 
	// ----------------------------------------------
	
	ibm.read_ibm_start_positions("hemisphere1.dat");
	ibm.read_ibm_end_positions("hemisphere2.dat");
	
	ibm.shift_start_positions(-41.0,-41.0,-81.0);
	ibm.shift_end_positions(-40.0,-40.0,-81.0);
	
	ibm.initialize_positions_to_start();
	
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	writeOutput("macros",0);
	
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	lbm.memcopy_host_to_device();
	ibm.memcopy_host_to_device();
	
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	lbm.initial_equilibrium(nBlocks,nThreads);
				
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_hemisphere_3::cycleForward(int stepsPerCycle, int currentCycle)
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
		lbm.zero_forces_with_IBM(nBlocks,nThreads);
		lbm.extrapolate_velocity_from_IBM(nBlocksIB,nThreads,ibm.nodes,ibm.nNodes);		
		lbm.stream_collide_save_IBforcing(nBlocks,nThreads);
		ibm.update_node_positions(nBlocksIB,nThreads,cummulativeSteps,nSteps);
		cudaDeviceSynchronize();
	}
	
	cout << cummulativeSteps << endl;	
	
	// ----------------------------------------------
	// determine voxels inside hemisphere:
	// (kernel defined below)
	// ----------------------------------------------
	
	lbm.inside_hemisphere(nBlocks,nThreads); 
	lbm.memcopy_device_to_host_inout();	
	
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

void scsp_3D_hemisphere_3::writeOutput(std::string tagname, int step)
{	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// ----------------------------------------------
	
	lbm.vtk_structured_output_iuvw_inout(tagname,step,iskip,jskip,kskip);
	ibm.write_output("ibm",step);		
}









