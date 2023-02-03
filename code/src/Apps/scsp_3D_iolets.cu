
# include "scsp_3D_iolets.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_iolets::scsp_3D_iolets() : lbm()
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
	lbm.allocate_voxel_positions();
		
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_iolets::~scsp_3D_iolets()
{	
	lbm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_iolets::initSystem()
{
	
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
	
	// ----------------------------------------------
	// create the lattice using "box" function.
	// ----------------------------------------------	
	
	if (latticeSource == "box") {
		lbm.create_lattice_box();
	}	
	
	// ----------------------------------------------
	// create the lattice by reading from file.	
	// input integer = 1 = read x[],y[],z[],voxelType[],nList[]
	//               = 2 = read x[],y[],z[],voxelType[]
	//               = 3 = read x[],y[],z[] 
	// ----------------------------------------------	
	
	if (latticeSource == "file") {
		lbm.read_lattice_geometry(1);
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

void scsp_3D_iolets::cycleForward(int stepsPerCycle, int currentCycle)
{
	
	// ----------------------------------------------
	// determine the cummulative number of steps at the
	// beginning of this cycle:
	// ----------------------------------------------
	
	int cummulativeSteps = stepsPerCycle*currentCycle;
	bool save = false;
	
	// ----------------------------------------------
	// loop through this cycle:
	// ----------------------------------------------
	
	for (int step=0; step<stepsPerCycle; step++) {
		cummulativeSteps++;		
		if (step == (stepsPerCycle-1)) save = true;		
		lbm.stream_collide_save(nBlocks,nThreads,save);
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

void scsp_3D_iolets::writeOutput(std::string tagname, int step)
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







