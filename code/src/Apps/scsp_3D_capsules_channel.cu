
# include "scsp_3D_capsules_channel.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_capsules_channel::scsp_3D_capsules_channel() : lbm(),ibm()
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
	float Re = inputParams("LBM/Re",2.0);
	float umax = inputParams("LBM/umax",0.1);
	
	// ----------------------------------------------
	// Immersed-Boundary parameters:
	// ----------------------------------------------
	
	int nNodesPerCell = inputParams("IBM/nNodesPerCell",0);
	int nCells = inputParams("IBM/nCells",1);
	nNodes = nNodesPerCell*nCells;
	a = inputParams("IBM/a",10.0);
	float Ca = inputParams("IBM/Ca",1.0);
	float ksmax = inputParams("IBM/ksmax",0.002);
	gam = inputParams("IBM/gamma",0.1);
	ibmUpdate = inputParams("IBM/ibmUpdate","verlet");
	
	// ----------------------------------------------
	// IBM set flags for PBC's:
	// ----------------------------------------------
	
	ibm.set_pbcFlag(1,0,0);
		
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
	// determine membrane parameters (see function
	// below), then calculate reference flux for no
	// capsules:
	// ----------------------------------------------
	
	calcMembraneParams(Re,Ca,umax,ksmax);
	calcRefFlux();
	Q0 = inputParams("LBM/Q0",0.0);
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_capsules_channel::~scsp_3D_capsules_channel()
{
	lbm.deallocate();
	ibm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_capsules_channel::initSystem()
{
		
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
		
	// ----------------------------------------------
	// create the lattice for channel flow:
	// ----------------------------------------------	
	
	lbm.create_lattice_box_channel();
	
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// ----------------------------------------------
		
	lbm.stream_index_pull();
			
	// ----------------------------------------------			
	// initialize velocities: 
	// ----------------------------------------------
	
	float h = float(Nz)/2.0;
	float w = float(Ny)/2.0;
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				
				// calculate analytical value for x-vel:
				float y = float(j) - w;
				float z = float(k) - h;
				float sumval = 0.0;
				// take first 40 terms of infinite sum
				for (int n = 1; n<80; n=n+2) {
					float nf = float(n);
					float pref = pow(-1.0,(nf-1.0)/2)/(nf*nf*nf);
					float term = pref*(1 - cosh(nf*M_PI*z/2/w) / cosh(nf*M_PI*h/2/w)) * cos(nf*M_PI*y/2/w);
					sumval += term;
				}
				float xvel = (16*bodyForx*w*w/nu/pow(M_PI,3))*sumval;
				
				// set values:
				//lbm.setU(ndx,xvel);
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
		
	ibm.read_ibm_information("sphere.dat");
	ibm.duplicate_cells();
	ibm.assign_cellIDs_to_nodes();
	ibm.assign_refNode_to_cells();	
			
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
			
	float scale = 1.0;  //0.7;  
	ibm.shrink_and_randomize_cells(scale,2.0*a+1.0,a+2.0);
	ibm.scale_equilibrium_cell_size(scale,nBlocks,nThreads);
		
	// ----------------------------------------------
	// relax node positions: 
	// ----------------------------------------------
		
	cout << " " << endl;
	cout << "-----------------------------------------------" << endl;
	cout << "Relaxing capsules..." << endl;
		
	scale = 1.0/scale;
	ibm.relax_node_positions_skalak(90000,scale,0.02,nBlocks,nThreads);	
	ibm.relax_node_positions_skalak(90000,1.0,0.02,nBlocks,nThreads);
		
	cout << "... done relaxing" << endl;
	cout << "-----------------------------------------------" << endl;
	cout << " " << endl;
					
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

void scsp_3D_capsules_channel::cycleForward(int stepsPerCycle, int currentCycle)
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

void scsp_3D_capsules_channel::stepIBM()
{
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);
	
	// re-build bin lists for IBM nodes:
	ibm.reset_bin_lists(nBlocks,nThreads);
	ibm.build_bin_lists(nBlocks,nThreads);
			
	// compute IBM node forces:
	ibm.compute_node_forces_skalak(nBlocks,nThreads);
	ibm.nonbonded_node_interactions(nBlocks,nThreads);
	ibm.wall_forces_ydir_zdir(nBlocks,nThreads);
	lbm.interpolate_velocity_to_IBM(nBlocks,nThreads,ibm.r,ibm.v,nNodes);
			
	// update fluid:
	lbm.extrapolate_forces_from_IBM(nBlocks,nThreads,ibm.r,ibm.f,nNodes);
	lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
	lbm.stream_collide_save_forcing(nBlocks,nThreads);
	//lbm.set_channel_wall_velocity(0.0,nBlocks,nThreads);
	lbm.set_boundary_duct_density(nBlocks,nThreads);
	
	// update membrane:
	//ibm.update_node_positions(nBlocks,nThreads);
	ibm.update_node_positions_verlet_1(nBlocks,nThreads);
	
	// CUDA sync
	cudaDeviceSynchronize();
}



// --------------------------------------------------------
// Take a time-step with the velocity-Verlet approach for IBM:
// --------------------------------------------------------

void scsp_3D_capsules_channel::stepVerlet()
{
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);
	
	// first step of IBM velocity verlet:
	ibm.update_node_positions_verlet_1(nBlocks,nThreads);
	
	// re-build bin lists for IBM nodes:
	ibm.reset_bin_lists(nBlocks,nThreads);
	ibm.build_bin_lists(nBlocks,nThreads);
			
	// compute IBM node forces:
	ibm.compute_node_forces_skalak(nBlocks,nThreads);
	ibm.nonbonded_node_interactions(nBlocks,nThreads);
	ibm.wall_forces_ydir_zdir(nBlocks,nThreads);
			
	// update fluid:
	lbm.viscous_force_IBM_LBM(nBlocks,nThreads,gam,ibm.r,ibm.v,ibm.f,nNodes);
	lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
	lbm.stream_collide_save_forcing(nBlocks,nThreads);
	//lbm.set_channel_wall_velocity(0.0,nBlocks,nThreads);
	lbm.set_boundary_duct_density(nBlocks,nThreads);
	
	// second step of IBM velocity verlet:
	ibm.update_node_positions_verlet_2(nBlocks,nThreads);
	
	// CUDA sync		
	cudaDeviceSynchronize();
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_capsules_channel::writeOutput(std::string tagname, int step)
{				
	
	int precision = 3;
	
	if (step == 0) {
		// only print out vtk files
		lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		ibm.write_output("ibm",step);
	}
	
	if (step > 0) { 
		// analyze membrane geometry:
		ibm.membrane_geometry_analysis("capdata",step);
		ibm.capsule_train_fraction(4*a,30.0,step);
	
		// calculate relative viscosity:
		lbm.calculate_relative_viscosity("relative_viscosity_thru_time",Q0,step);
		
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
			ibm.write_output("ibm",step);
		}
		
		// print out final averaged flow profile:
		if (step == nSteps) {
			lbm.print_flow_rate_xdir("flow_data",step);			
		}
	}	
}



// --------------------------------------------------------
// Calculate membrane elastic parameters.  Here, we
// calculate the appropriate values of nu, ks, and bodyForx
// that satisfy the given Re and Ca subject to the 
// conditions that maximum u < umax and ks < ksmax:
// --------------------------------------------------------

void scsp_3D_capsules_channel::calcMembraneParams(float Re, float Ca, float umax, float Ksmax)
{
	// assumed parameters:
	float rho = 1.0;
	float w = float(Ny)/2.0;
	float h = float(Nz)/2.0;
	float Dh = 4.0*(4.0*w*h)/(4.0*(w+h));
	float infsum = calcInfSum(w,h);	
	
	// my calculations:
	nu = umax*Dh/(2.0*Re);
	bodyForx = umax*nu*M_PI*M_PI*M_PI/(16.0*w*w*infsum);
	float Ks = rho*nu*umax*a/(h*Ca);
	
	/*	
	// set Ks to some large number:
	float Ks = 1000.0;

	// loop until parameters are acceptable:
	while (Ks > Ksmax) {		
		// step 1: calculate nu:
		nu = umax*Dh/Re;
		while (nu > 1.0/6.0) {
		    umax *= 0.9999;
		    nu = umax*Dh/Re;
		}	    
		// step 2: calculate fx:
		bodyForx = umax*nu*pow(M_PI,3)/(16.0*w*w*infsum);
		// step 3: calculate Es:
		Ks = rho*nu*umax*a/(h*Ca);
		// step 4: if Es > Esmax, reduce umax
		if (Ks > Ksmax) umax *= 0.9999;
	}
	*/
	
	// assign values for ks and nu:
	ibm.set_ks(Ks); 
	ibm.set_kb(Ks*a*a*0.00287*sqrt(3));
	//ibm.set_kv(0.5);
	ibm.set_ka(0.0007);
	ibm.set_kag(0.0);
	ibm.set_C(2.0);
	lbm.setNu(nu);   
	
	// output the results:
	cout << "  " << endl;
	cout << "hydraulic diameter = " << Dh << endl;
	cout << "umax = " << umax << endl;
	cout << "ks = " << Ks << endl;
	cout << "nu = " << nu << endl;
	cout << "fx = " << bodyForx << endl;
	cout << "  " << endl;
	cout << "Re = " << Re << endl;
	cout << "Ca = " << rho*nu*umax*a/h/Ks << endl;
	cout << "  " << endl;
		
}



// --------------------------------------------------------
// Calculate infinite sum associated with solution
// to velocity profile in rectanglular channel:
// --------------------------------------------------------

float scsp_3D_capsules_channel::calcInfSum(float w, float h)
{
	float outval = 0.0;
	// take first 40 terms of infinite sum
	for (int n = 1; n<80; n=n+2) {
		float nf = float(n);
		float pref = pow(-1.0,(nf-1.0)/2)/(nf*nf*nf);
		float term = pref*(1 - 1/cosh(nf*M_PI*h/2.0/w));
		outval += term;
	}
	return outval;
}



// --------------------------------------------------------
// Calculate reference flux for the chosen values of w, h,
// bodyForx, and nu:
// --------------------------------------------------------

void scsp_3D_capsules_channel::calcRefFlux()
{
	// parameters:
	float w = float(Ny)/2.0;
	float h = float(Nz)/2.0;
	Q0 = 0.0;
	
	// calculate solution for velocity at every
	// site in the y-z plane:
	for (int j=0; j<Ny; j++) {
		for (int k=0; k<Nz; k++) {
			float y = float(j) - w;
			float z = float(k) - h;
			float sumval = 0.0;
			// take first 40 terms of infinite sum
			for (int n = 1; n<80; n=n+2) {
				float nf = float(n);
				float pref = pow(-1.0,(nf-1.0)/2)/(nf*nf*nf);
				float term = pref*(1 - cosh(nf*M_PI*z/2/w) / cosh(nf*M_PI*h/2/w)) * cos(nf*M_PI*y/2/w);
				sumval += term;
			}
			float u0 = (16*bodyForx*w*w/nu/pow(M_PI,3))*sumval;
			Q0 += u0;
		}
	}
	
	// output the results:
	cout << "reference flux = " << Q0 << endl;
	cout << "  " << endl;		
}




