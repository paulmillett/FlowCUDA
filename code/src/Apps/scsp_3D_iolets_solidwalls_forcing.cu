
# include "scsp_3D_iolets_solidwalls_forcing.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_iolets_solidwalls_forcing::scsp_3D_iolets_solidwalls_forcing() : lbm()
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
	
	bodyForx = 0.00001;
	
	// ----------------------------------------------
	// GPU parameters:
	// ----------------------------------------------
	
	nThreads = inputParams("GPU/nThreads",512);
	nBlocks = (nVoxels+(nThreads-1))/nThreads;  // integer division
		
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
	lbm.allocate_solid();
	lbm.allocate_forces();
		
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_iolets_solidwalls_forcing::~scsp_3D_iolets_solidwalls_forcing()
{	
	lbm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_iolets_solidwalls_forcing::initSystem()
{
	
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
	
	// ----------------------------------------------
	// define the solid walls:
	// ----------------------------------------------
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				int Si = 0;				
				// set up solid walls
				if (k < 4 || k > Nz-4) Si = 1;
				if (j < 4 || j > Nz-4) Si = 1; 
				/*
				if (k == 0) Si = 1;
				if (k == Nz-1) Si = 1;
				if (j == 0) Si = 1;
				if (j == Ny-1) Si = 1;
				*/
				lbm.setS(ndx,Si);
			}
		}
	}
	
	// ----------------------------------------------
	// create the lattice using "box" function.
	// ----------------------------------------------	
	
	lbm.create_lattice_box_iolets_solid_walls();
		
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
	// initialize macros: 
	// ----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		lbm.setU(i,0.0);
		lbm.setV(i,0.0);
		lbm.setW(i,0.0);
		lbm.setR(i,1.0);		
	}
		
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	writeOutput("macros",0);
	
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	lbm.memcopy_host_to_device();
	lbm.memcopy_host_to_device_solid();
		
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

void scsp_3D_iolets_solidwalls_forcing::cycleForward(int stepsPerCycle, int currentCycle)
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
		lbm.zero_forces(nBlocks,nThreads);
		lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
		lbm.stream_collide_save_forcing_solid(nBlocks,nThreads);
		cudaDeviceSynchronize();
	}
	    	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	lbm.memcopy_device_to_host();
	writeOutput("macros",cummulativeSteps);
	
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_iolets_solidwalls_forcing::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// ----------------------------------------------
	
	if (vtkFormat == "structured") {
		int precision = 3;
		lbm.vtk_structured_output_ruvw(tagname,step,1,1,1,precision);
	}
	
	else if (vtkFormat == "polydata") {
		lbm.vtk_polydata_output_ruvw(tagname,step);
	}
	
}







