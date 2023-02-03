
# include "scsp_3D_swell_ibm.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_swell_ibm::scsp_3D_swell_ibm() : lbm(),ibm()
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
	ibm.allocate();
	ibm.allocate_faces();
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_swell_ibm::~scsp_3D_swell_ibm()
{
	lbm.deallocate();
	ibm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_swell_ibm::initSystem()
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
	Nz = inputParams("Lattice/Nz",0);
	
	// inlet
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			int i = 0;
			int ndx = k*Nx*Ny + j*Nx + i;
			float rz = float(k - 60);
			float ry = float(j - 60);
			float rr = sqrt(rz*rz + ry*ry);
			if (rr > 19.0) lbm.setVoxelType(ndx,0);
		}	
	}
	
	// outlet
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			int i = Nx-1;
			int ndx = k*Nx*Ny + j*Nx + i;
			float rz = float(k - 60);
			float ry = float(j - 60);
			float rr = sqrt(rz*rz + ry*ry);
			if (rr > 19.0) lbm.setVoxelType(ndx,0);
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
	
	ibm.read_ibm_start_positions("input_Swell_trimesh.txt");
	ibm.read_ibm_end_positions("input_Swell_trimesh.txt");
	
	ibm.shift_start_positions(0.0,5.0,5.0);
	ibm.shift_end_positions(0.0,5.0,5.0);
	
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

void scsp_3D_swell_ibm::cycleForward(int stepsPerCycle, int currentCycle)
{
		
	// ----------------------------------------------
	// determine the cummulative number of steps at the
	// beginning of this cycle:
	// ----------------------------------------------
	
	int cummulativeSteps = stepsPerCycle*currentCycle;	
	
	// ----------------------------------------------
	// Get the Iolet velocity:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	float velBC1 = inputParams("Iolet1/uBC",0.0);
	float velBC2 = inputParams("Iolet2/uBC",0.0);
	
	// ----------------------------------------------
	// loop through this cycle:
	// ----------------------------------------------
	
	for (int step=0; step<stepsPerCycle; step++) {
		cummulativeSteps++;	
		
		// reset LBM forces, etc.:
		lbm.zero_forces_with_IBM(nBlocks,nThreads);
		
		// extrapolate IBM node velocities to LBM voxels:
		lbm.extrapolate_velocity_from_IBM(nBlocksIB,nThreads,ibm.r,ibm.v,ibm.nNodes);
		
		// calculate time-dependent inlet velocity:
		float invel = sin(float(cummulativeSteps)/float(nSteps)*M_PI);
		lbm.setIoletU(0,invel*velBC1);
		lbm.setIoletU(1,invel*velBC2);
		lbm.memcopy_host_to_device_iolets();	
				
		// update LBM:	
		lbm.stream_collide_save_IBforcing(nBlocks,nThreads);
		
		// update IBM:
		ibm.update_node_positions(nBlocksIB,nThreads,cummulativeSteps,nSteps);
		cudaDeviceSynchronize();
	}	
	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	cout << cummulativeSteps << endl;	
	lbm.memcopy_device_to_host();
	ibm.memcopy_device_to_host();
	writeOutput("macros",cummulativeSteps);
		
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_swell_ibm::writeOutput(std::string tagname, int step)
{	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// ----------------------------------------------
	
	int precision = 3;
	lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision);
	ibm.write_output("ibm",step);		
}









