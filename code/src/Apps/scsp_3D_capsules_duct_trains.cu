
# include "scsp_3D_capsules_duct_trains.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_capsules_duct_trains::scsp_3D_capsules_duct_trains() : lbm(),ibm()
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
	umax = inputParams("LBM/umax",0.1);
	pulsatile = inputParams("LBM/pulsatile",0);
	wavelength = inputParams("LBM/wavelength",100000.0);  // for pulsatile flow
	
	// ----------------------------------------------
	// Immersed-Boundary parameters:
	// ----------------------------------------------
	
	int nNodesPerCell = inputParams("IBM/nNodesPerCell",0);
	nCells = inputParams("IBM/nCells",1);
	nNodes = nNodesPerCell*nCells;
	a = inputParams("IBM/a",10.0);
	float Ca = inputParams("IBM/Ca",1.0);
	float ksmax = inputParams("IBM/ksmax",0.002);
	gam = inputParams("IBM/gamma",0.1);
	ibmFile = inputParams("IBM/ibmFile","sphere.dat");
	ibmUpdate = inputParams("IBM/ibmUpdate","verlet");
	initRandom = inputParams("IBM/initRandom",1);
	trainRij = inputParams("IBM/trainRij",2.8*a);
	trainAng = inputParams("IBM/trainAng",15.0);
	sepMin = inputParams("IBM/sepMin",0.9);
	sepWallY = inputParams("IBM/sepWallY",10.0);
	sepWallZ = inputParams("IBM/sepWallZ",1.3);
	sepWallY += a;
	sepWallZ += a;
	
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
	ibm.allocate();	
	
	// ----------------------------------------------
	// determine membrane parameters (see function
	// below), then calculate reference flux for no
	// capsules:
	// ----------------------------------------------
	
	calcMembraneParams(Re,Ca,ksmax);
	calcRefFlux();
	Q0 = inputParams("LBM/Q0",0.0);
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_capsules_duct_trains::~scsp_3D_capsules_duct_trains()
{
	lbm.deallocate();
	ibm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_capsules_duct_trains::initSystem()
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
	
	// ----------------------------------------------			
	// rescale capsule sizes for normal distribution: 
	// ----------------------------------------------
	
	cellSizes = inputParams("IBM/cellSizes","uniform");
	float stddevA = inputParams("IBM/stddevA",0.0);
	ibm.rescale_cell_radii(a,stddevA,cellSizes);	
					
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
		float scale = 1.0;   // 0.7;
		ibm.shrink_and_randomize_cells(scale,sepMin,sepWallY,sepWallZ);
		ibm.scale_equilibrium_cell_size(scale,nBlocks,nThreads);
			
		cout << " " << endl;
		cout << "-----------------------------------------------" << endl;
		cout << "Relaxing capsules..." << endl;
		
		scale = 1.0/scale;
		ibm.relax_node_positions_skalak(90000,scale,0.1,nBlocks,nThreads);	
		ibm.relax_node_positions_skalak(90000,1.0,0.1,nBlocks,nThreads);
		
		cout << "... done relaxing" << endl;
		cout << "-----------------------------------------------" << endl;
		cout << " " << endl;	
		
	}
	
	
	// ----------------------------------------------
	// line up cells in a single-file line: 
	// ----------------------------------------------
	
	if (!initRandom) {
		float cellSpacingX = inputParams("IBM/cellSpacingX",float(Nx));
		float offsetY = inputParams("IBM/offsetY",0.0);
		ibm.single_file_cells(Nx,Ny,Nz,cellSpacingX,offsetY);		
	}
	
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

void scsp_3D_capsules_duct_trains::cycleForward(int stepsPerCycle, int currentCycle)
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
			ibm.stepIBM(lbm,nBlocks,nThreads);
			lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
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
		// if pulsatile flow, calculate bodyforce:
		float bodyForxPul = bodyForx;
		if (pulsatile) bodyForxPul *= sin(2*M_PI*float(cummulativeSteps)/wavelength);
		// update IBM & LBM:
		ibm.stepIBM(lbm,nBlocks,nThreads);
		lbm.add_body_force(bodyForxPul,0.0,0.0,nBlocks,nThreads);
		lbm.stream_collide_save_forcing(nBlocks,nThreads);	
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

void scsp_3D_capsules_duct_trains::writeOutput(std::string tagname, int step)
{				
	
	float h = float(Nz)/2.0;
	float scale = 1.0/umax;
	
	if (step == 0) {
		// only print out vtk files
		lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		ibm.write_output("ibm",step);
	}
	
	if (step > 0) { 
		// analyze membrane geometry:
		ibm.capsule_geometry_analysis(step);
		ibm.capsule_train_fraction(trainRij*a,trainAng,step);
		ibm.output_capsule_data();
	
		// calculate relative viscosity:
		//lbm.calculate_relative_viscosity("relative_viscosity_thru_time",Q0,step);
		
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision);
			ibm.write_output("ibm",step);
		}
		
		// print out final averaged flow profile:
		//if (step == nSteps) {
		//	lbm.print_flow_rate_xdir("flow_data",step);			
		//}
	}	
}



// --------------------------------------------------------
// Calculate membrane elastic parameters.  Here, we
// calculate the appropriate values of nu, ks, and bodyForx
// that satisfy the given Re and Ca subject to the 
// conditions that maximum u < umax and ks < ksmax:
// --------------------------------------------------------

void scsp_3D_capsules_duct_trains::calcMembraneParams(float Re, float Ca, float Ksmax)
{
	// 'GetPot' object containing input parameters:
	GetPot inputParams("input.dat");
	cellProps = inputParams("IBM/cellProps","uniform");
	float stddevCa = inputParams("IBM/stddevCa",0.0);
	float Kv = inputParams("IBM/kv",0.0);
	float C = inputParams("IBM/C",2.0);
	float rho = 1.0;
	float h = float(Nz)/2.0;
	float w = float(Ny)/2.0;
	
	// calculate umax and required body force:
	float Dh = 4.0*(4.0*w*h)/(4.0*(w+h));
	float infsum = calcInfSum(w,h);	
	umax = 2.0*Re*nu/Dh;      //Re*nu/h;
	// modify if umax is too high due to high Re:
	if (umax > 0.03) {
		umax = 0.03;
		nu = umax*Dh/(2.0*Re);
		lbm.setNu(nu);
		cout << "  " << endl;
		cout << "nu = " << nu << endl;	
	}
	bodyForx = umax*nu*M_PI*M_PI*M_PI/(16.0*w*w*infsum);
		
	// set the mechanical properties:
	ibm.calculate_cell_membrane_props(Re,Ca,stddevCa,a,h,rho,umax,Kv,C,cellProps);
}



// --------------------------------------------------------
// Calculate reference flux for the chosen values of w, h,
// bodyForx, and nu:
// --------------------------------------------------------

void scsp_3D_capsules_duct_trains::calcRefFlux()
{
	// parameters:
	float w = float(Ny);   // PBC's in y-dir
	//float h = float(Nz-1)/2.0;
	float h = float(Nz)/2.0;
	Q0 = 2.0*bodyForx*h*h*h*w/3.0/nu;
		
	// output the results:
	cout << "reference flux = " << Q0 << endl;
	cout << "  " << endl;		
}



// --------------------------------------------------------
// Calculate infinite sum associated with solution
// to velocity profile in rectanglular channel:
// --------------------------------------------------------

float scsp_3D_capsules_duct_trains::calcInfSum(float w, float h)
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
