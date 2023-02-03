
# include "scsp_3D_bulge.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_bulge::scsp_3D_bulge() : lbm()
{		
	
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	
	// ----------------------------------------------
	// Read input parameters:
	// ----------------------------------------------
	
	nVoxels = inputParams("Lattice/nVoxels",0);
	Q = inputParams("Lattice/Q",19);	
	nThreads = inputParams("GPU/nThreads",512);
	nBlocks = (nVoxels+(nThreads-1))/nThreads;  // integer division
	numIolets = inputParams("Lattice/numIolets",2);
	vtkFormat = inputParams("Output/format","polydata");
	nSteps = inputParams("Time/nSteps",0);
	
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	lbm.allocate_voxel_positions();
		
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_bulge::~scsp_3D_bulge()
{	
	lbm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_bulge::initSystem()
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
	
	// read x[], y[], and z[] arrays:
	if (latticeSource == "file") {
		lbm.read_lattice_geometry(3);
	}
	
	// construct nList[] array:
	lbm.bounding_box_nList_construct();
	
	// find lowest and highest x[] values:
	int xmin = 1000;
	int xmax = -1000;
	for (int i=0; i<nVoxels; i++) {
		int xi = lbm.getX(i);
		if (xi < xmin) xmin = xi;
		if (xi > xmax) xmax = xi;
	}
	
	// assign voxelType[] arrays:
	for (int i=0; i<nVoxels; i++) {
		lbm.setVoxelType(i,0); // default
		int nnabors = 0;
		for (int j=0; j<19; j++) {
			int nabor = lbm.getNList(i*19+j);
			if (nabor != -1) nnabors++;			
		}
		int xi = lbm.getX(i);
		if (xi == xmin && nnabors == 14) lbm.setVoxelType(i,1); // iolet1
		if (xi == xmax && nnabors == 14) lbm.setVoxelType(i,2); // iolet2
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

void scsp_3D_bulge::cycleForward(int stepsPerCycle, int currentCycle)
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
		// calculate time-dependent inlet velocity:
		float inrho = 1.0;
		if (cummulativeSteps < nSteps/2) {
			inrho = 1.0 + 0.03*float(cummulativeSteps)/float(nSteps/2);
		}
		else {
			inrho = 1.00;
		}		
		lbm.setIoletR(0,inrho);
		lbm.memcopy_host_to_device_iolets();
		// calculate LBM:	
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

void scsp_3D_bulge::writeOutput(std::string tagname, int step)
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







