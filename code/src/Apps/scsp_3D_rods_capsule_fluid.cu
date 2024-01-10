
# include "scsp_3D_rods_capsule_fluid.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_rods_capsule_fluid::scsp_3D_rods_capsule_fluid() : lbm(),rods(),ibm()
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
	
	int sizeIBM = ibm.get_max_array_size();
	int sizeROD = rods.get_max_array_size();	
	int sizeMAX = max(nVoxels,max(sizeIBM,sizeROD));	
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
		
	int nBeadsPerRod = inputParams("IBM_RODS/nBeadsPerRod",0);
	nRods = inputParams("IBM_RODS/nRods",1);
	fp = inputParams("IBM_RODS/fp",0.0);
	L0 = inputParams("IBM_RODS/L0",0.5);
	Pe = inputParams("IBM_RODS/Pe",0.0);
	kT = inputParams("IBM_RODS/kT",0.0);
	gam = inputParams("IBM_RODS/gamma",0.1);
	Drod = inputParams("IBM_RODS/diam",1.0);
	nBeads = nBeadsPerRod*nRods;
	Lrod = float(nBeadsPerRod)*L0;
	
	// ----------------------------------------------
	// Capsules Immersed-Boundary parameters:
	// ----------------------------------------------
		
	int nNodesPerCell = inputParams("IBM/nNodesPerCell",0);
	nCells = inputParams("IBM/nCells",1);
	nNodes = nNodesPerCell*nCells;
	a = inputParams("IBM/a",6.0);
	C = inputParams("IBM/C",1.0);
	Ca = inputParams("IBM/Ca",1.0);
	La = inputParams("IBM/La",1.0);
	gam = inputParams("IBM/gamma",0.1);
	ibmFile = inputParams("IBM/ibmFile","sphere.dat");
	ibmUpdate = inputParams("IBM/ibmUpdate","verlet");	
	
	// ----------------------------------------------
	// IBM set flags for PBC's:
	// ----------------------------------------------
	
	ibm.set_pbcFlag(1,1,0);
	rods.set_pbcFlag(1,1,0);
		
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
	
	ibm.allocate();
	lbm.allocate();
	lbm.allocate_forces();
	rods.allocate();	
	
	// ----------------------------------------------
	// even though there is only one capsule, 
	// set up bins for it:
	// ----------------------------------------------
	
	ibm.initialize_bins();
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_rods_capsule_fluid::~scsp_3D_rods_capsule_fluid()
{
	ibm.deallocate();
	lbm.deallocate();
	rods.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_rods_capsule_fluid::initSystem()
{
		
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
	
	// ----------------------------------------------
	// create the lattice assuming shear flow.
	// ----------------------------------------------	
	
	lbm.create_lattice_box_slit();
	
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
	// initialize immersed boundary info: 
	// ----------------------------------------------
	
	ibm.read_ibm_information(ibmFile);
	ibm.duplicate_cells();
	ibm.assign_cellIDs_to_nodes();
	ibm.assign_refNode_to_cells();
	ibm.set_cells_types(1);
	ibm.shift_node_positions(0,float(Nx)/2.0,float(Ny)/2.0,float(Nz)/2.0);
		
	float rho = 1.0;
	float Ks = La*rho*nu*nu/a;
	float Kb = Ks*a*a*0.003;
	float Kv = inputParams("IBM/kv",0.0000);
	ibm.set_cells_mechanical_props(Ks,Kb,Kv,C,Ca);
	cout << "Capsule Ks = " << Ks << endl;
	cout << "Capsule Kb = " << Kb << endl;
	
	// ----------------------------------------------			
	// initialize rod immersed boundary info: 
	// ----------------------------------------------
	
	rods.create_first_rod();
	rods.duplicate_rods();
	rods.assign_rodIDs_to_beads();
	
	fp = Pe*kT/Lrod;
	up = fp/gam;
	rods.set_fp(fp);
	rods.set_up(up);
	rods.set_rods_radii(Drod/2.0);
	cout << "  " << endl;
	cout << "Rod kT = " << kT << endl;
	cout << "Rod fp = " << fp << endl;
	cout << "Rod up = " << up << endl;
	
	// ----------------------------------------------			
	// drag friction coefficients using Broersma's
	// relations.  See Tsay et al. J. Amer. Chem. Soc.
	// 128:1639(2006)
	// ----------------------------------------------
	
	// translational:
	float delt = log(2*Lrod/Drod);  // this is natural log
	float g1 = 0.807 + 0.15/delt + 13.5/delt/delt - 37.0/delt/delt/delt + 22.0/delt/delt/delt/delt;
	float g2 = -0.193 + 0.15/delt + 8.1/delt/delt - 18.0/delt/delt/delt + 9.0/delt/delt/delt/delt;
	float pref = delt - 0.5*(g1 + g2);
	if (pref < 1.0) pref = 1.0;
	float DT = pref*kT/(3.0*M_PI*nu*Lrod);  // diffusivity (assume fluid density = 1)
	float fricT = kT/DT;
	float noiseT = sqrt(2.0*fricT*kT);
	rods.set_friction_coefficient_translational(fricT);
	rods.set_noise_strength_translational(noiseT);
	cout << "Rod fricT = " << fricT << endl;
	cout << "Rod noiseT = " << noiseT << endl;	
	
	// rotational:
	g1 = 1.14 + 0.2/delt + 16.0/delt/delt - 63.0/delt/delt/delt + 62.0/delt/delt/delt/delt;
	pref = delt - g1;
	if (pref < 0.5) pref = 0.5;
	float DR = pref*3.0*kT/(M_PI*nu*Lrod*Lrod*Lrod);  // rotational diffusivity
	float fricR = kT/DR;
	float noiseR = sqrt(2.0*fricR*kT);
	rods.set_friction_coefficient_rotational(fricR);
	rods.set_noise_strength_rotational(noiseR);
	cout << "Rod fricR = " << fricR << endl;
	cout << "Rod noiseR = " << noiseR << endl;	
				
	// ----------------------------------------------
	// build the binMap array for neighbor lists: 
	// ----------------------------------------------
	
	ibm.build_binMap(nBlocks,nThreads);
	rods.build_binMap(nBlocks,nThreads);
		
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	ibm.memcopy_host_to_device();
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
	
	rods.randomize_rods_inside_sphere(float(Nx)/2.0,float(Ny)/2.0,float(Nz)/2.0,a+6.0,Lrod);
	rods.set_rod_position_orientation(nBlocks,nThreads);
	rods.stepIBM_push_into_sphere(30000,float(Nx)/2.0,float(Ny)/2.0,float(Nz)/2.0,a,nBlocks,nThreads);	
		
	// ----------------------------------------------
	// calculate rest geometries for capsule: 
	// ----------------------------------------------
	
	ibm.rest_geometries(nBlocks,nThreads);
	
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	ibm.memcopy_device_to_host();
	rods.memcopy_device_to_host();
	writeOutput("macros",0);
	
	// ----------------------------------------------
	// set IBM velocities & forces to zero: 
	// ----------------------------------------------
	
	ibm.zero_velocities_forces(nBlocks,nThreads);
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

void scsp_3D_rods_capsule_fluid::cycleForward(int stepsPerCycle, int currentCycle)
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
			rods.stepIBM_capsules_rods(ibm,lbm,nBlocks,nThreads);
			lbm.stream_collide_save_forcing(nBlocks,nThreads);	
			lbm.set_boundary_shear_velocity(-shearVel,shearVel,nBlocks,nThreads);
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
		rods.stepIBM_capsules_rods(ibm,lbm,nBlocks,nThreads);
		lbm.stream_collide_save_forcing(nBlocks,nThreads);
		lbm.set_boundary_shear_velocity(-shearVel,shearVel,nBlocks,nThreads);
		cudaDeviceSynchronize();
	}
	
	cout << cummulativeSteps << endl;	
		
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
	ibm.memcopy_device_to_host();
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

void scsp_3D_rods_capsule_fluid::writeOutput(std::string tagname, int step)
{				
	
	if (step == 0) {
		// only print out vtk files
		ibm.write_output("ibm",step);
		lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		rods.write_output("rods",step);
	}
	
	if (step > 0) { 					
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			ibm.write_output("ibm",step);
			lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision);
			rods.write_output("rods",step);
		}
	}	
}








