
# include "scsp_3D_rods_fluid_spinners.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_rods_fluid_spinners::scsp_3D_rods_fluid_spinners() : lbm(),rods()
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
	
	int sizeROD = rods.get_max_array_size();	
	int sizeMAX = max(nVoxels,sizeROD);	
	nThreads = inputParams("GPU/nThreads",512);
	nBlocks = (sizeMAX+(nThreads-1))/nThreads;  // integer division
	
	cout << "largest array size = " << sizeMAX << endl;
	cout << "nBlocks = " << nBlocks << ", nThreads = " << nThreads << endl;
		
	// ----------------------------------------------
	// time parameters:
	// ----------------------------------------------
	
	nSteps = inputParams("Time/nSteps",0);
	nStepsEquilibrate = inputParams("Time/nStepsEquilibrate",0);
	
	// ----------------------------------------------
	// Lattice Boltzmann parameters:
	// ----------------------------------------------
	
	nu = inputParams("LBM/nu",0.1666666);
	shearVel = inputParams("LBM/shearVel",0.0);
	float Re = inputParams("LBM/Re",2.0);
	shearVel = 2.0*Re*nu/float(Nz);
	
	// ----------------------------------------------
	// Rods Immersed-Boundary parameters:
	// ----------------------------------------------
		
	int nBeadsPerRod = inputParams("IBM_RODS/nBeadsPerRod",1);
	nRods = inputParams("IBM_RODS/nRods",1);
	fp = inputParams("IBM_RODS/fp",0.0);
	L0 = inputParams("IBM_RODS/L0",0.5);
	Pe = inputParams("IBM_RODS/Pe",0.0);
	kT = inputParams("IBM_RODS/kT",0.0);
	gam = inputParams("IBM_RODS/gamma",0.1);
	Drod = inputParams("IBM_RODS/diam",1.0);
	nBeads = nBeadsPerRod*nRods;
	Lrod = float(nBeadsPerRod)*L0;
	if (nBeadsPerRod == 1) Lrod = Drod;
	
	// ----------------------------------------------
	// IBM set flags for PBC's:
	// ----------------------------------------------
	
	rods.set_pbcFlag(1,1,1);
		
	// ----------------------------------------------
	// iolets parameters:
	// ----------------------------------------------
	
	numIolets = inputParams("Lattice/numIolets",2);
	
	// ----------------------------------------------
	// output parameters:
	// ----------------------------------------------
	
	iskip = inputParams("Output/iskip",1);
	jskip = inputParams("Output/jskip",1);
	kskip = inputParams("Output/kskip",1);
	nVTKOutputs = inputParams("Output/nVTKOutputs",0);
	precision = inputParams("Output/precision",3);
		
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	lbm.allocate_forces();
	rods.allocate();	
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_rods_fluid_spinners::~scsp_3D_rods_fluid_spinners()
{
	lbm.deallocate();
	rods.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_rods_fluid_spinners::initSystem()
{
		
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
	
	// ----------------------------------------------
	// create the lattice assuming shear flow.
	// ----------------------------------------------	
	
	lbm.create_lattice_box_periodic();
	
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// ----------------------------------------------
		
	lbm.stream_index_pull();
			
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
	// initialize rod immersed boundary info: 
	// ----------------------------------------------
	
	rods.create_first_rod();
	rods.duplicate_rods();
	rods.assign_rodIDs_to_beads();
	
	fp = Pe*kT/Lrod;
	up = 0.0;  //fp/gam;
	rods.set_fp(fp);
	rods.set_up(up);
	rods.set_rods_radii(Drod/2.0);
	cout << "  " << endl;
	cout << "Rod kT = " << kT << endl;
	cout << "Rod fp = " << fp << endl;
	cout << "Rod up = " << up << endl;
	
	// ----------------------------------------------			
	// drag friction coefficients (assume spherical
	// particle)
	// ----------------------------------------------
	
	// translational:
	float DT = kT/(3.0*M_PI*nu*Lrod);  // diffusivity (assume fluid density = 1)
	float fricT = kT/DT;
	float noiseT = sqrt(2.0*fricT*kT);
	rods.set_friction_coefficient_translational(fricT);
	rods.set_noise_strength_translational(noiseT);
	cout << "Rod fricT = " << fricT << endl;
	cout << "Rod noiseT = " << noiseT << endl;	
	
	// rotational:
	float DR = kT/(M_PI*nu*Lrod*Lrod*Lrod);  // rotational diffusivity
	float fricR = kT/DR;
	float noiseR = sqrt(2.0*fricR*kT);
	rods.set_friction_coefficient_rotational(fricR);
	rods.set_noise_strength_rotational(noiseR);
	cout << "Rod fricR = " << fricR << endl;
	cout << "Rod noiseR = " << noiseR << endl;	
	
	// ----------------------------------------------
	// build the binMap array for neighbor lists: 
	// ----------------------------------------------
	
	rods.build_binMap(nBlocks,nThreads);	
		
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	lbm.memcopy_host_to_device();
	rods.memcopy_host_to_device();
		
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	lbm.initial_equilibrium(nBlocks,nThreads);	
		
	// ----------------------------------------------
	// set the random number seed: 
	// ----------------------------------------------
	
	//srand(time(NULL));
	
	// ----------------------------------------------
	// randomly disperse filaments: 
	// ----------------------------------------------
		
	rods.randomize_rods(0.0);
	rods.set_rod_position_orientation(nBlocks,nThreads);
		
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	rods.memcopy_device_to_host();
	writeOutput("macros",0);
	
	// ----------------------------------------------
	// set IBM velocities & forces to zero: 
	// ----------------------------------------------
	
	rods.zero_bead_forces(nBlocks,nThreads);
	
	// ----------------------------------------------
	// initialize cuRand state for the thermal noise
	// force:
	// ----------------------------------------------
	
	rods.initialize_cuRand(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_rods_fluid_spinners::cycleForward(int stepsPerCycle, int currentCycle)
{
		
	// ----------------------------------------------
	// determine the cummulative number of steps at the
	// beginning of this cycle:
	// ----------------------------------------------
	
	int cummulativeSteps = stepsPerCycle*currentCycle;
	
	// ----------------------------------------------
	// if simulation just started, perform 
	// equilibration:
	// ----------------------------------------------
	
	if (cummulativeSteps == 0) {
		cout << " " << endl;
		cout << "-----------------------------------------------" << endl;
		cout << "Equilibrating for " << nStepsEquilibrate << " steps..." << endl;
		for (int i=0; i<nStepsEquilibrate; i++) {
			if (i%10000 == 0) cout << "equilibration step " << i << endl;
			rods.stepIBM_Euler_Spinners(lbm,nBlocks,nThreads);
			lbm.stream_collide_save_forcing(nBlocks,nThreads);
			cudaDeviceSynchronize();	
		}
		cout << " " << endl;
		cout << "... done equilibrating!" << endl;
		cout << "-----------------------------------------------" << endl;
		cout << " " << endl;
	}
	
	// ----------------------------------------------
	// loop through this cycle:
	// ----------------------------------------------
		
	for (int step=0; step<stepsPerCycle; step++) {
		cummulativeSteps++;
		rods.stepIBM_Euler_Spinners(lbm,nBlocks,nThreads);
		lbm.stream_collide_save_forcing(nBlocks,nThreads);
		cudaDeviceSynchronize();
	}
	
	cout << cummulativeSteps << endl;	
		
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
	lbm.memcopy_device_to_host();
	rods.memcopy_device_to_host();    
	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	writeOutput("macros",cummulativeSteps);
		
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_rods_fluid_spinners::writeOutput(std::string tagname, int step)
{				
	
	if (step == 0) {
		// only print out vtk files
		lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		rods.write_output("rods",step);
	}
	
	if (step > 0) { 					
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision);
			rods.write_output("rods",step);
		}
	}	
}








