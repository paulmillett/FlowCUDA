
# include "scsp_2D_expand.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_2D_expand::scsp_2D_expand() : lbm(),ibm()
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
	nBlocksIB = (nNodes+(nThreads-1))/nThreads; // integer division	
	
	// ----------------------------------------------
	// iolets parameters:
	// ----------------------------------------------
	
	numIolets = inputParams("Lattice/numIolets",2);
	
	// ----------------------------------------------
	// output parameters:
	// ----------------------------------------------
	
	vtkFormat = inputParams("Output/format","polydata");
	
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	lbm.allocate_forces();
	lbm.allocate_IB_velocities();
	ibm.allocate();
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_2D_expand::~scsp_2D_expand()
{	
	lbm.deallocate();
	ibm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_2D_expand::initSystem()
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
			
	// ----------------------------------------------			
	// edit inlet condition: 
	// ----------------------------------------------
	
	Nx = inputParams("Lattice/Nx",0);
	Ny = inputParams("Lattice/Ny",0);
	
	for (int i=0; i<Nx; i++) {
		int j = Ny - 1;
		int ndx = j*Nx + i;
		if (i < 120 || i > 140) {
			lbm.setVoxelType(ndx,0);
		} 
	}	
		
	// ----------------------------------------------			
	// initialize macros: 
	// ----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		lbm.setU(i,0.0);
		lbm.setV(i,0.0);
		lbm.setR(i,1.0);		
	}
		
	// ----------------------------------------------			
	// initialize immersed boundary info: 
	// ----------------------------------------------
			
	float xcent = 99.5;
	float ycent = 198.5;
	float radiusx = 50.0;
	float radiusy = 50.0;
	for (int i=0; i<nNodes; i++) { 
		float xst = xcent - radiusx*cos(1.0*M_PI*float(i)/(nNodes-1));
		float yst = ycent - radiusy*sin(1.0*M_PI*float(i)/(nNodes-1));
		ibm.setXStart(i,xst);
		ibm.setYStart(i,yst);
	}
	radiusx = 50.0;
	radiusy = 100.0;
	for (int i=0; i<nNodes; i++) { 
		float xend = xcent - radiusx*cos(1.0*M_PI*float(i)/(nNodes-1));
		float yend = ycent - radiusy*sin(1.0*M_PI*float(i)/(nNodes-1));
		ibm.setXEnd(i,xend);
		ibm.setYEnd(i,yend);		
	}
	
	ibm.set_positions_to_start_positions();	
	
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
		
	// ----------------------------------------------
	// define reference IBM node positions: 
	// ----------------------------------------------
	
	ibm.set_reference_node_positions(nBlocksIB,nThreads);

}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_2D_expand::cycleForward(int stepsPerCycle, int currentCycle)
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
		ibm.update_node_ref_position(nBlocksIB,nThreads,cummulativeSteps,nSteps);		
		lbm.zero_forces(nBlocks,nThreads);		
		ibm.compute_node_forces(nBlocksIB,nThreads);		
		lbm.extrapolate_forces_from_IBM(nBlocksIB,nThreads,ibm.x,ibm.y,ibm.fx,ibm.fy,ibm.nNodes);			
		lbm.stream_collide_save_forcing(nBlocks,nThreads);		
		lbm.interpolate_velocity_to_IBM(nBlocksIB,nThreads,ibm.x,ibm.y,ibm.vx,ibm.vy,ibm.nNodes);		
		ibm.update_node_positions(nBlocksIB,nThreads);		
		cudaDeviceSynchronize();
	}
	
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

void scsp_2D_expand::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// function location:
	// "io/write_vtk_output.cuh"
	// ----------------------------------------------
	
	lbm.write_output(tagname,step);
	ibm.write_output("ibm",step);		
		
}







