
# include "scsp_3D_capsules_slit_Janus.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_capsules_slit_Janus::scsp_3D_capsules_slit_Janus() : lbm(),ibm()
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
	nStepsEquilibrate = inputParams("Time/nStepsEquilibrate",0);
	
	// ----------------------------------------------
	// Lattice Boltzmann parameters:
	// ----------------------------------------------
	
	nu = inputParams("LBM/nu",0.1666666);
	bodyForx = inputParams("LBM/bodyForx",0.0);
	Re = inputParams("LBM/Re",2.0);
	
	// ----------------------------------------------
	// Immersed-Boundary parameters:
	// ----------------------------------------------
	
	int nNodesPerCell = inputParams("IBM/nNodesPerCell",0);
	int nCells = inputParams("IBM/nCells",1);
	nNodes = nNodesPerCell*nCells;
	a = inputParams("IBM/a",10.0);
	CaSoft = inputParams("IBM/CaSoft",1.0);
	CaHard = inputParams("IBM/CaHard",1.0);
	gam = inputParams("IBM/gamma",0.1);
	psi = inputParams("IBM/psi",0.0);
	C = inputParams("IBM/C",1.0);
	ibmUpdate = inputParams("IBM/ibmUpdate","verlet");
	initRandom = inputParams("IBM/initRandom",1);
	ibmFile = inputParams("IBM/ibmFile","sphere.dat");
	
	// ----------------------------------------------
	// IBM set flags for PBC's:
	// ----------------------------------------------
	
	ibm.set_pbcFlag(1,1,0);
		
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
	nVTKOutputs = inputParams("Output/nVTKOutputs",0);
	
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	lbm.allocate_forces();
	ibm.allocate();	
	
	// ----------------------------------------------
	// even though there is only one capsule, 
	// set up bins for it:
	// ----------------------------------------------
	
	ibm.initialize_bins();
		
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_capsules_slit_Janus::~scsp_3D_capsules_slit_Janus()
{
	lbm.deallocate();
	ibm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_capsules_slit_Janus::initSystem()
{
		
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
		
	// ----------------------------------------------
	// create the lattice for shear flow (same as slit):
	// ----------------------------------------------	
	
	lbm.create_lattice_box_slit();
	
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// ----------------------------------------------
		
	lbm.stream_index_pull();
			
	// ----------------------------------------------			
	// initialize velocities: 
	// ----------------------------------------------
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				lbm.setU(ndx,0.0);
				lbm.setV(ndx,0.0);
				lbm.setW(ndx,0.0);
				lbm.setR(ndx,1.0);
			}
		}
	}
	
	// ----------------------------------------------			
	// initialize immersed boundary info: 
	// ----------------------------------------------
		
	ibm.read_ibm_information(ibmFile);
	ibm.duplicate_cells();
	ibm.assign_cellIDs_to_nodes();
	ibm.assign_refNode_to_cells();
	ibm.set_cells_types(1);	
	ibm.define_Janus_capsule_geometry(a,psi);	
	ibm.shift_node_positions(0,float(Nx)/2.0,float(Ny)/2.0,float(Nz)/2.0+2.0);
	
	float rho = 1.0;
	float h = float(Nz)/2.0;
	float umax = Re*nu/h;		
	bodyForx = 2.0*rho*umax*umax/(Re*h);
	ksSoft = rho*umax*umax*a/(CaSoft*Re);
	ksHard = rho*umax*umax*a/(CaHard*Re);	
	
	float Ks = ksSoft;
	float Kb = Ks*a*a*0.003;
	float Kv = inputParams("IBM/kv",0.0000);
	ibm.set_cells_mechanical_props(Ks,Kb,Kv,C,CaSoft);
	
	cout << "  " << endl;
	cout << "Capsule ks (Soft) = " << ksSoft << endl;
	cout << "Capsule ks (Hard) = " << ksHard << endl;
	cout << "Capsule Kb = " << Kb << endl;
	cout << "  " << endl;
	cout << "H = " << h << endl;
	cout << "umax = " << umax << endl;
	cout << "fx = " << bodyForx << endl;
	cout << "Re = " << umax*h/nu << endl;
	cout << "  " << endl;
			
	// ----------------------------------------------
	// build the binMap array for neighbor lists: 
	// ----------------------------------------------
	
	ibm.build_binMap(nBlocks,nThreads); 
		
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
	// calculate rest geometries for membrane: 
	// ----------------------------------------------
	
	ibm.rest_geometries_skalak(nBlocks,nThreads);
	
	// ----------------------------------------------
	// set the random number seed: 
	// ----------------------------------------------
	
	srand(time(NULL));
	
	// ----------------------------------------------
	// shrink and randomly disperse cells: 
	// ----------------------------------------------
				
	//float scale = 1.0;  //0.7
	//ibm.shrink_and_randomize_cells(scale,16.0,a+2.0);
	//ibm.scale_equilibrium_cell_size(scale,nBlocks,nThreads);
			
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	ibm.memcopy_device_to_host();
	writeOutput("macros",0);
	
	// ----------------------------------------------
	// set IBM velocities & forces to zero: 
	// ----------------------------------------------
	
	ibm.zero_velocities_forces(nBlocks,nThreads);
	
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_capsules_slit_Janus::cycleForward(int stepsPerCycle, int currentCycle)
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
			// decide on update type:
			if (ibmUpdate == "ibm") {
				stepIBM();
			} else if (ibmUpdate == "verlet") {
				stepVerlet();
			}			
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
		// decide on update type:
		if (ibmUpdate == "ibm") {
			stepIBM();
		} else if (ibmUpdate == "verlet") {
			stepVerlet();
		}		
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
// Take a time-step with the traditional IBM approach:
// --------------------------------------------------------

void scsp_3D_capsules_slit_Janus::stepIBM()
{
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);
	
	// re-build bin lists for IBM nodes:
	ibm.reset_bin_lists(nBlocks,nThreads);
	ibm.build_bin_lists(nBlocks,nThreads);
			
	// compute IBM node forces:
	ibm.compute_node_forces_skalak_Janus(ksSoft,ksHard,nBlocks,nThreads);
	ibm.nonbonded_node_interactions(nBlocks,nThreads);
	ibm.wall_forces_zdir(nBlocks,nThreads);
	lbm.interpolate_velocity_to_IBM(nBlocks,nThreads,ibm.nodes,nNodes);
			
	// update fluid:
	lbm.extrapolate_forces_from_IBM(nBlocks,nThreads,ibm.nodes,nNodes);
	lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
	lbm.stream_collide_save_forcing(nBlocks,nThreads);
	//lbm.set_boundary_slit_density(nBlocks,nThreads);
	
	// update membrane:
	//ibm.update_node_positions(nBlocks,nThreads);
	//ibm.update_node_positions_verlet_1(nBlocks,nThreads);
	ibm.update_node_positions_include_force(nBlocks,nThreads);
	
	// CUDA sync
	cudaDeviceSynchronize();
}



// --------------------------------------------------------
// Take a time-step with the velocity-Verlet approach for IBM:
// --------------------------------------------------------

void scsp_3D_capsules_slit_Janus::stepVerlet()
{
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);
	
	// first step of IBM velocity verlet:
	ibm.update_node_positions_verlet_1(nBlocks,nThreads);
	
	// re-build bin lists for IBM nodes:
	ibm.reset_bin_lists(nBlocks,nThreads);
	ibm.build_bin_lists(nBlocks,nThreads);
			
	// compute IBM node forces:
	ibm.compute_node_forces_skalak_Janus(ksSoft,ksHard,nBlocks,nThreads);
	ibm.nonbonded_node_interactions(nBlocks,nThreads);
	ibm.wall_forces_zdir(nBlocks,nThreads);
			
	// update fluid:
	lbm.viscous_force_IBM_LBM(nBlocks,nThreads,gam,ibm.nodes,nNodes);
	lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
	lbm.stream_collide_save_forcing(nBlocks,nThreads);
	//lbm.set_boundary_slit_velocity(0.0,nBlocks,nThreads);
	lbm.set_boundary_slit_density(nBlocks,nThreads);
	
	// second step of IBM velocity verlet:
	ibm.update_node_positions_verlet_2(nBlocks,nThreads);
	
	// CUDA sync		
	cudaDeviceSynchronize();
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_capsules_slit_Janus::writeOutput(std::string tagname, int step)
{				
	
	int precision = 3;
	
	if (step == 0) {
		// only print out vtk files
		lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		ibm.write_output("ibm",step);
	}
	
	if (step > 0) { 
		// analyze membrane geometry:
		ibm.capsule_geometry_analysis(step);
		ibm.output_capsule_data();
			
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
			ibm.write_output("ibm",step);
		}
	}	
}








