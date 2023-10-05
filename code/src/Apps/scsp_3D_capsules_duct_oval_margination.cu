
# include "scsp_3D_capsules_duct_oval_margination.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_capsules_duct_oval_margination::scsp_3D_capsules_duct_oval_margination() : lbm(),ibm(),poissonRBC(),poissonPLT()
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
	chA = inputParams("Lattice/chA",float(Ny-1));
	chB = inputParams("Lattice/chB",float(Nz-1));
	chA /= 2.0;
	chB /= 2.0;
	
	// ----------------------------------------------
	// GPU parameters:
	// ----------------------------------------------
	
	int sizeIBM = ibm.get_max_array_size();
	int sizeMAX = max(sizeIBM,nVoxels);	
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
	bodyForx = inputParams("LBM/bodyForx",0.0);
	Re = inputParams("LBM/Re",2.0);
	umax = inputParams("LBM/umax",0.1);
	Q0 = inputParams("LBM/Q0",0.0);
	
	// ----------------------------------------------
	// Immersed-Boundary parameters:
	// ----------------------------------------------
		
	a1 = inputParams("IBM/a1",10.0);
	a2 = inputParams("IBM/a2",10.0);
	Ca1 = inputParams("IBM/Ca1",1.0);
	Ca2 = inputParams("IBM/Ca2",1.0);
	float ksmax = inputParams("IBM/ksmax",0.002);
	gam = inputParams("IBM/gamma",0.1);
	ibmFile1 = inputParams("IBM/ibmFile1","rbc.dat");
	ibmFile2 = inputParams("IBM/ibmFile2","sphere.dat");
	ibmUpdate = inputParams("IBM/ibmUpdate","verlet");
	initRandom = inputParams("IBM/initRandom",1);
	nu_in = 5.0/6.0;   // internal RBC visc
	nu_out = 1.0/6.0;  // plasma visc
	
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
	precision = inputParams("Output/precision",3);
	
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	lbm.allocate_forces();
	lbm.allocate_solid();
	ibm.allocate();	
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_capsules_duct_oval_margination::~scsp_3D_capsules_duct_oval_margination()
{
	lbm.deallocate();
	ibm.deallocate();	
	poissonRBC.deallocate();
	poissonPLT.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_capsules_duct_oval_margination::initSystem()
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
				float y = float(j) - (float(Ny/2) + 1.0);
				float z = float(k) - (float(Nz/2) + 1.0);
				if (y*y/chA/chA + z*z/chB/chB > 1.0) Si = 1;				
				lbm.setS(ndx,Si);
			}
		}
	}
		
	// ----------------------------------------------
	// create the lattice for channel flow:
	// ----------------------------------------------	
	
	lbm.create_lattice_box_periodic_solid_walls();
	
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// ----------------------------------------------
		
	lbm.stream_index_pull();
			
	// ----------------------------------------------			
	// initialize velocities: 
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
		
	ibm.read_ibm_information(ibmFile1,ibmFile2);		
	ibm.duplicate_cells();
	ibm.assign_cellIDs_to_nodes();
	ibm.assign_refNode_to_cells();	
	ibm.set_cells_radii_binary();
	ibm.set_cells_types_binary();
		
	// ----------------------------------------------
	// determine membrane parameters (see function
	// below), then calculate reference flux for no
	// capsules:
	// ----------------------------------------------
	
	calcMembraneParams();
			
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
		
	if (initRandom) {		
		
		float a = max(a1,a2);
		ibm.randomize_platelets_and_rbcs(3.0,a+2.0);
		ibm.stepIBM_no_fluid_rbcs_platelets(20000,false,nBlocks,nThreads);  // here, only RBC's move
		ibm.stepIBM_no_fluid(10000,true,nBlocks,nThreads);   // here, both RBC's and PLT's move
			
		/*
		float scale = 0.2;   // 0.7;
		float a = max(a1,a2);
		ibm.shrink_and_randomize_cells(scale,2.0,a+2.0);
		ibm.scale_equilibrium_cell_size(scale,nBlocks,nThreads);
		
		
		cout << " " << endl;
		cout << "-----------------------------------------------" << endl;
		cout << "Relaxing capsules..." << endl;
		
		int relaxSteps = 0;  // 90000
		scale = 1.0/scale;
		ibm.relax_node_positions_skalak(relaxSteps,scale,0.1,nBlocks,nThreads);	
		ibm.relax_node_positions_skalak(relaxSteps,1.0,0.1,nBlocks,nThreads);
		
		cout << "... done relaxing" << endl;
		cout << "-----------------------------------------------" << endl;
		cout << " " << endl;	
		*/
	}
	
	// ----------------------------------------------
	// initialize poisson solver:
	// ----------------------------------------------
	
	poissonRBC.initialize(Nx,Ny,Nz);
	poissonPLT.initialize(Nx,Ny,Nz);
	poissonRBC.solve_poisson(ibm.faces,ibm.r,ibm.cells,ibm.nFaces,1,nBlocks,nThreads);
	poissonRBC.write_output("indicatorRBC",0,iskip,jskip,kskip,precision);
	poissonPLT.solve_poisson(ibm.faces,ibm.r,ibm.cells,ibm.nFaces,2,nBlocks,nThreads);
	poissonPLT.write_output("indicatorPLT",0,iskip,jskip,kskip,precision);
			
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

void scsp_3D_capsules_duct_oval_margination::cycleForward(int stepsPerCycle, int currentCycle)
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
			if (i%5 == 0) poissonRBC.solve_poisson(ibm.faces,ibm.r,ibm.cells,ibm.nFaces,1,nBlocks,nThreads);
			ibm.stepIBM(lbm,nBlocks,nThreads);
			lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
			lbm.stream_collide_save_forcing_varvisc(poissonRBC.indicator,nu_in,nu_out,nBlocks,nThreads);
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
		if (cummulativeSteps%5 == 0) poissonRBC.solve_poisson(ibm.faces,ibm.r,ibm.cells,ibm.nFaces,1,nBlocks,nThreads);
		ibm.stepIBM(lbm,nBlocks,nThreads);
		lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
		lbm.stream_collide_save_forcing_varvisc(poissonRBC.indicator,nu_in,nu_out,nBlocks,nThreads);
		cudaDeviceSynchronize();
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
// Write output to file
// --------------------------------------------------------

void scsp_3D_capsules_duct_oval_margination::writeOutput(std::string tagname, int step)
{				
		
	if (step == 0) {
		// only print out vtk files
		lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		ibm.write_output("ibm",step);
		poissonRBC.volume_fraction_analysis("vol_frac_RBC",0.4);
		poissonPLT.volume_fraction_analysis("vol_frac_PLT",0.4);
	}
	
	if (step > 0) { 
		// analyze membrane geometry:
		ibm.capsule_geometry_analysis(step);
		ibm.output_capsule_data();
		// need to perform PLT poisson solver because it is not performed during
		// regular time steps
		poissonPLT.solve_poisson(ibm.faces,ibm.r,ibm.cells,ibm.nFaces,2,nBlocks,nThreads);
		poissonRBC.volume_fraction_analysis("vol_frac_RBC",0.4);
		poissonPLT.volume_fraction_analysis("vol_frac_PLT",0.4);
	
		// calculate relative viscosity:
		lbm.calculate_relative_viscosity("relative_viscosity_thru_time",Q0,step);
		
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision);
			ibm.write_output("ibm",step);
			//poissonRBC.write_output("indicatorRBC",step,iskip,jskip,kskip,precision);
			//poissonPLT.write_output("indicatorPLT",step,iskip,jskip,kskip,precision);
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

void scsp_3D_capsules_duct_oval_margination::calcMembraneParams()
{
	// 'GetPot' object containing input parameters:
	GetPot inputParams("input.dat");
	float Kv = inputParams("IBM/kv",0.0);
	float C = inputParams("IBM/C",10.0);
	int nCells1 = inputParams("IBM/nCells1",1);
	int nCells2 = inputParams("IBM/nCells2",0);
	
	// assumed parameters:
	float rho = 1.0;
	float w = float(Ny)/2.0;
	float h = float(Nz)/2.0;
	float Dh = 4.0*(4.0*w*h)/(4.0*(w+h));
	float infsum = calcInfSum(w,h);	
	
	// per cell calculations:
	umax = 2.0*Re*nu_out/Dh;
	bodyForx = umax*nu_out*M_PI*M_PI*M_PI/(16.0*w*w*infsum);
	for (int i=0; i<nCells1+nCells2; i++) {
		float rad_i = ibm.cellsH[i].rad;
		float Ca_i = 0.1;
		if (i<nCells1)  Ca_i = Ca1;
		if (i>=nCells1) Ca_i = Ca2;
		float Ks = rho*umax*umax*rad_i/(Ca_i*Re);
		float Kb = Ks*rad_i*rad_i*0.00287*sqrt(3);		
		ibm.set_cell_mechanical_props(i,Ks,Kb,Kv,C,Ca_i);
	}
	
	// shear rates:
	float gamma_aver_ydir = umax/w;
	float gamma_wall_ydir = wall_shear_rate("y");
	float gamma_aver_zdir = umax/h;
	float gamma_wall_zdir = wall_shear_rate("z");
	
	// reference flux:
	calcRefFlux();
	
	// output the results:
	cout << "  " << endl;
	cout << "hydraulic diameter = " << Dh << endl;
	cout << "umax (bare fluid) = " << umax << endl;
	cout << "fx = " << bodyForx << endl;
	cout << "aver shear stress in z-dir = " << gamma_aver_zdir << endl;
	cout << "wall shear stress in z-dir = " << gamma_wall_zdir << endl;
	cout << "aver shear stress in y-dir = " << gamma_aver_ydir << endl;
	cout << "wall shear stress in y-dir = " << gamma_wall_ydir << endl;
	cout << "  " << endl;
	
}



// --------------------------------------------------------
// Calculate infinite sum associated with solution
// to velocity profile in rectanglular channel:
// --------------------------------------------------------

float scsp_3D_capsules_duct_oval_margination::calcInfSum(float w, float h)
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

void scsp_3D_capsules_duct_oval_margination::calcRefFlux()
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
			float u0 = velocity_at_point(y,z,w,h);
			Q0 += u0;
		}
	}
	
	// output the results:
	cout << "reference flux = " << Q0 << endl;
	cout << "  " << endl;		
}



// --------------------------------------------------------
// Calculate wall shear rate:
// --------------------------------------------------------

float scsp_3D_capsules_duct_oval_margination::wall_shear_rate(std::string dir)
{
	// parameters:
	float w = float(Ny)/2.0;
	float h = float(Nz)/2.0;	
	// calculate solution for velocity at two points
	// and determine forward finite difference for shear rate
	if (dir == "z") {
		float u0 = velocity_at_point(0.0,h,w,h);
		float u1 = velocity_at_point(0.0,h-1.0,w,h);
		return (u1-u0)/1.0;
	} else if (dir == "y") {
		float u0 = velocity_at_point(w,0.0,w,h);
		float u1 = velocity_at_point(w-1,0.0,w,h);
		return (u1-u0)/1.0;
	} else {
		return 0.0;
	}
}



// --------------------------------------------------------
// Calculate velocity at point:
// --------------------------------------------------------

float scsp_3D_capsules_duct_oval_margination::velocity_at_point(float y, float z, float w, float h)
{
	float sumval = 0.0;
	// take first 40 terms of infinite sum
	for (int n = 1; n<80; n=n+2) {
		float nf = float(n);
		float pref = pow(-1.0,(nf-1.0)/2)/(nf*nf*nf);
		float term = pref*(1 - cosh(nf*M_PI*z/2/w) / cosh(nf*M_PI*h/2/w)) * cos(nf*M_PI*y/2/w);
		sumval += term;
	}
	return (16*bodyForx*w*w/nu/pow(M_PI,3))*sumval;
}

