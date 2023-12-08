
# include "scsp_3D_filaments_capsule_pusher.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_filaments_capsule_pusher::scsp_3D_filaments_capsule_pusher() : lbm(),ibm(),filams()
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
	int sizeFIL = filams.get_max_array_size();	
	int sizeMAX = max(sizeIBM,sizeFIL);	
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
	Re = inputParams("LBM/Re",2.0);
	float h = float(Nz)/2.0;
	shearVel = Re*nu/h;
	shearVel = 0.0;
		
	// ----------------------------------------------
	// Filaments Immersed-Boundary parameters:
	// ----------------------------------------------
		
	int nBeadsPerFilam = inputParams("IBM_FILAMS/nBeadsPerFilam",0);
	nFilams = inputParams("IBM_FILAMS/nFilams",1);
	ks = inputParams("IBM_FILAMS/ks",0.1);
	kb = inputParams("IBM_FILAMS/kb",0.1);
	fp = inputParams("IBM_FILAMS/fp",0.0);
	L0 = inputParams("IBM_FILAMS/L0",0.5);
	Pe = inputParams("IBM_FILAMS/Pe",0.0);
	PL = inputParams("IBM_FILAMS/PL",1.0);  // non-dimensional persistence length
	kT = inputParams("IBM_FILAMS/kT",0.0);
	nBeads = nBeadsPerFilam*nFilams;
	Lfil = float(nBeadsPerFilam)*L0;
	
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
	
	ibm.set_pbcFlag(1,1,1);
	filams.set_pbcFlag(1,1,1);
		
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
	//lbm.allocate();
	//lbm.allocate_forces();
	filams.allocate();	
	
	// ----------------------------------------------
	// even though there is only one capsule, 
	// set up bins for it:
	// ----------------------------------------------
	
	ibm.initialize_bins();
		
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_filaments_capsule_pusher::~scsp_3D_filaments_capsule_pusher()
{
	ibm.deallocate();
	//lbm.deallocate();
	filams.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_filaments_capsule_pusher::initSystem()
{
		
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
	
	// ----------------------------------------------
	// create the lattice assuming shear flow.
	// ----------------------------------------------	
	
	//lbm.create_lattice_box_shear();
	
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// ----------------------------------------------
		
	//lbm.stream_index_pull();
			
	// ----------------------------------------------			
	// initialize macros: 
	// ----------------------------------------------
	
	/*
	for (int i=0; i<nVoxels; i++) {
		lbm.setU(i,0.0);
		lbm.setV(i,0.0);
		lbm.setW(i,0.0);
		lbm.setR(i,1.0);		
	}
	*/
	
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
	// initialize filament immersed boundary info: 
	// ----------------------------------------------
	
	filams.create_first_filament();
	filams.duplicate_filaments();
	filams.assign_filamIDs_to_beads();
	
	fp = Pe*kT/Lfil/Lfil;
	kb = PL*kT;
	up = fp*L0/gam;  // active velocity per bead
	filams.set_ks(ks);
	filams.set_kb(kb);
	filams.set_fp(fp);
	filams.set_up(up);
	filams.set_filams_radii(0.5);
	cout << "  " << endl;
	cout << "Filament kT = " << kT << endl;
	cout << "Filament ks = " << ks << endl;
	cout << "Filament kb = " << kb << endl;
	cout << "Filament fp = " << fp << endl;
	cout << "Filament up = " << up << endl;
			
	// ----------------------------------------------
	// build the binMap array for neighbor lists: 
	// ----------------------------------------------
	
	ibm.build_binMap(nBlocks,nThreads);
	filams.build_binMap(nBlocks,nThreads);
		
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	ibm.memcopy_host_to_device();
	//lbm.memcopy_host_to_device();
	filams.memcopy_host_to_device();
		
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	//lbm.initial_equilibrium(nBlocks,nThreads);	
		
	// ----------------------------------------------
	// set the random number seed: 
	// ----------------------------------------------
	
	//srand(time(NULL));
	
	// ----------------------------------------------
	// randomly disperse filaments: 
	// ----------------------------------------------
			
	filams.randomize_filaments_inside_sphere(float(Nx)/2.0,float(Ny)/2.0,float(Nz)/2.0,11.0,Lfil);
	filams.stepIBM_push_into_sphere(20000,float(Nx)/2.0,float(Ny)/2.0,float(Nz)/2.0,6.0,nBlocks,nThreads);	
	
	// ----------------------------------------------
	// calculate rest geometries for membrane: 
	// ----------------------------------------------
	
	ibm.rest_geometries(nBlocks,nThreads);
	//ibm.set_edge_rest_angles(0.0,nBlocks,nThreads);
	
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	ibm.memcopy_device_to_host();
	filams.memcopy_device_to_host();
	writeOutput("macros",0);
	
	// ----------------------------------------------
	// set IBM velocities & forces to zero: 
	// ----------------------------------------------
	
	ibm.zero_velocities_forces(nBlocks,nThreads);
	filams.zero_bead_velocities_forces(nBlocks,nThreads);
	
	// ----------------------------------------------
	// initialize cuRand state for the thermal noise
	// force:
	// ----------------------------------------------
	
	filams.initialize_cuRand(nBlocks,nThreads);
	
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_filaments_capsule_pusher::cycleForward(int stepsPerCycle, int currentCycle)
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
			filams.stepIBM_capsules_filaments_pusher_no_fluid(ibm,nBlocks,nThreads);
			//filams.stepIBM_capsules_filaments(lbm,ibm,nBlocks,nThreads);
			//lbm.stream_collide_save_forcing(nBlocks,nThreads);	
			//lbm.set_boundary_shear_velocity(-shearVel,shearVel,nBlocks,nThreads);
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
		filams.stepIBM_capsules_filaments_pusher_no_fluid(ibm,nBlocks,nThreads);
		//filams.stepIBM_capsules_filaments(lbm,ibm,nBlocks,nThreads);	
		//lbm.stream_collide_save_forcing(nBlocks,nThreads);
		//lbm.set_boundary_shear_velocity(-shearVel,shearVel,nBlocks,nThreads);
		cudaDeviceSynchronize();
	}
	
	cout << cummulativeSteps << endl;	
		
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
	ibm.memcopy_device_to_host();
	//lbm.memcopy_device_to_host();
	filams.memcopy_device_to_host();    
	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	writeOutput("macros",cummulativeSteps);
		
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_filaments_capsule_pusher::writeOutput(std::string tagname, int step)
{				
	
	if (step == 0) {
		// only print out vtk files
		ibm.write_output("ibm",step);
		//lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		filams.write_output("filaments",step);
	}
	
	if (step > 0) { 					
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			ibm.write_output("ibm",step);
			//lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision);
			filams.write_output("filaments",step);
		}
	}	
}








