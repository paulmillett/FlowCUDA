
# include "scsp_3D_cylinders_channel.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_cylinders_channel::scsp_3D_cylinders_channel() : lbm(),ibm()
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
	chRad = inputParams("Lattice/chRad",float(Nz-1)/2.0);	
	
	// ----------------------------------------------
	// GPU parameters:
	// ----------------------------------------------
	
	int sizeCAP = ibm.get_max_array_size();	
	int sizeMAX = max(nVoxels,sizeCAP);	
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
	float Re = inputParams("LBM/Re",2.0);
	umax = inputParams("LBM/umax",0.01);
	
	// ----------------------------------------------
	// Immersed-Boundary parameters:
	// ----------------------------------------------
	
	int nNodesPerCell = inputParams("IBM/nNodesPerCell",0);
	nCells = inputParams("IBM/nCells",1);
	nNodes = nNodesPerCell*nCells;
	a = inputParams("IBM/a",10.0);
	L = inputParams("IBM/L",10.0);
	R = inputParams("IBM/R",1.0);
	nNodesLength = inputParams("IBM/nNodesLength",10);
	float Ca = inputParams("IBM/Ca",1.0);
	float ksmax = inputParams("IBM/ksmax",0.002);
	gam = inputParams("IBM/gamma",0.1);
	ibmFile = inputParams("IBM/ibmFile","sphere.dat");
	ibmUpdate = inputParams("IBM/ibmUpdate","verlet");
	initRandom = inputParams("IBM/initRandom",1);
	sepMin = inputParams("IBM/sepMin",0.9);
	sepWallY = inputParams("IBM/sepWallY",10.0);
	sepWallZ = inputParams("IBM/sepWallZ",1.3);
	sepWallY += a;
	sepWallZ += a;
	
	// ----------------------------------------------
	// calculate particle volume fraction:
	// ----------------------------------------------
	
	float Vp = float(nCells)*(M_PI*R*R*L);
	float V = M_PI*chRad*chRad*float(Nx);
	float phi = Vp/V;
	cout << "particle volume fraction = " << phi << endl;
	
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
	
	// ----------------------------------------------
	// determine membrane parameters (see function
	// below), then calculate reference flux for no
	// capsules:
	// ----------------------------------------------
	
	//calcMembraneParams_Skalak(Re,Ca,ksmax);
	//calcMembraneParams_Spring(Re);
		
	float C = inputParams("IBM/C",2.0);
	float ks = inputParams("IBM/ks",0.0);
	float kb = inputParams("IBM/kb",0.0);
	float ka = inputParams("IBM/ka",0.0);
	float kag = inputParams("IBM/kag",0.0);
	float kv = inputParams("IBM/kv",0.0);
	ibm.set_cells_mechanical_props(ks,kb,kv,C,Ca);
	
	cout << "  " << endl;
	cout << "ks = " << ks << endl;
	cout << "kb = " << kb << endl;
	cout << "ka = " << ka << endl;
	cout << "kv = " << kv << endl;
	cout << "kag = " << kag << endl;
	cout << "  " << endl;
	cout << "Ca = " << Ca << endl;
	cout << "  " << endl;
	cout << "cylinder particle length = " << L << endl;
	cout << "cylinder particle radius = " << R << endl;
	
	// ----------------------------------------------
	// calculate body-force depending on Re:
	// ----------------------------------------------
	
	float Dh = 2.0*chRad;
	umax = Re*nu/Dh;
	
	// modify if umax is too high due to high Re:
	if (umax > 0.03) {
		umax = 0.03;
		nu = umax*Dh/Re;
		lbm.setNu(nu);
		cout << "  " << endl;
		cout << "nu = " << nu << endl;	
	}
	bodyForx = umax*(4*nu)/chRad/chRad;
	
	cout << "  " << endl;
	cout << "Re = " << Re << endl;
	cout << "Body Force X-dir = " << bodyForx << endl;
	cout << "nu = " << nu << endl;
	cout << "umax = " << umax << endl;
	cout << "  " << endl;	
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_cylinders_channel::~scsp_3D_cylinders_channel()
{
	lbm.deallocate();
	ibm.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_cylinders_channel::initSystem()
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
				float y = float(j) - float(Ny-1)/2.0;
				float z = float(k) - float(Nz-1)/2.0;
				if ((y*y + z*z)/chRad/chRad > 1.0) Si = 1;				
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
	
	ibm.rest_geometries_spring(nBlocks,nThreads);
	//ibm.rest_geometries_skalak(nBlocks,nThreads);
	
	// ----------------------------------------------
	// set the random number seed: 
	// ----------------------------------------------
	
	srand(time(NULL));
	
	// ----------------------------------------------
	// randomly disperse cells: 
	// ----------------------------------------------
	
	float sepMin = inputParams("IBM/sepMin",2.0);	
	ibm.randomize_capsules_xdir_alligned_cylinder(L,R,sepMin,sepMin);
			
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

void scsp_3D_cylinders_channel::cycleForward(int stepsPerCycle, int currentCycle)
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
			//ibm.stepIBM_spring(lbm,nBlocks,nThreads);	
			ibm.stepIBM_spring_cylinders(lbm,R,nu,nBlocks,nThreads);		
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
		// update IBM & LBM:
		//ibm.stepIBM_spring(lbm,nBlocks,nThreads);
		ibm.stepIBM_spring_cylinders(lbm,R,nu,nBlocks,nThreads);
		lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
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

void scsp_3D_cylinders_channel::writeOutput(std::string tagname, int step)
{				
	
	float h = float(Nz)/2.0;
	float scale = 1.0/umax;
	
	if (step == 0) {
		// only print out vtk files
		ibm.capsule_orientation_cylinders(nNodesLength,step);
		lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		ibm.write_output_cylinders("ibm",step);
	}
	
	if (step > 0) { 
		// analyze membrane geometry: 
		ibm.capsule_geometry_analysis(step);
		ibm.capsule_orientation_cylinders(nNodesLength,step);
		ibm.output_capsule_data();
			
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision);
			ibm.write_output_cylinders("ibm",step);
		}
				
	}	
}



// --------------------------------------------------------
// Calculate membrane elastic parameters.  Here, we
// calculate the appropriate values of nu, ks, and bodyForx
// that satisfy the given Re and Ca subject to the 
// conditions that maximum u < umax and ks < ksmax:
// --------------------------------------------------------

void scsp_3D_cylinders_channel::calcMembraneParams_Skalak(float Re, float Ca, float Ksmax)
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
	float Ks = rho*umax*umax*a/(Ca*Re);    //rho*nu*umax*a/(h*Ca);
	float Kb = Ks*a*a*0.00287*sqrt(3); 
	Kb *= 1.0; 
	ibm.set_cells_mechanical_props(Ks,Kb,Kv,C,Ca);
			
	// output the results:
	cout << "  " << endl;
	cout << "H = " << h << endl;
	cout << "umax = " << umax << endl;
	cout << "ks = " << Ks << endl;
	cout << "kb = " << Kb << endl;
	cout << "Body force in x-dir = " << bodyForx << endl;
	cout << "  " << endl;
	cout << "Ca = " << Ca << endl;
	cout << "  " << endl;
	
}



// --------------------------------------------------------
// Calculate membrane elastic parameters, assuming the 
// spring model:
// --------------------------------------------------------

void scsp_3D_cylinders_channel::calcMembraneParams_Spring(float Re)
{
	// 'GetPot' object containing input parameters:
	GetPot inputParams("input.dat");
	float C = inputParams("IBM/C",2.0);
	float ks = inputParams("IBM/ks",0.0);
	float kb = inputParams("IBM/kb",0.0);
	float ka = inputParams("IBM/ka",0.0);
	float kag = inputParams("IBM/kag",0.0);
	float kv = inputParams("IBM/kv",0.0);
	float Ca = inputParams("IBM/Ca",1.0);
	ibm.set_cells_mechanical_props(ks,kb,kv,C,Ca);
		
	// calculate umax and required body force:
	float h = float(Nz)/2.0;
	float w = float(Ny)/2.0;
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
			
	// output the results:
	cout << "  " << endl;
	cout << "H = " << h << endl;
	cout << "umax = " << umax << endl;
	cout << "ks = " << ks << endl;
	cout << "kb = " << kb << endl;
	cout << "ka = " << ka << endl;
	cout << "kv = " << kv << endl;
	cout << "kag = " << kag << endl;
	cout << "Body force in x-dir = " << bodyForx << endl;
	cout << "  " << endl;
	cout << "Ca = " << Ca << endl;
	cout << "  " << endl;
	
}



// --------------------------------------------------------
// Calculate reference flux for the chosen values of w, h,
// bodyForx, and nu:
// --------------------------------------------------------

void scsp_3D_cylinders_channel::calcRefFlux()
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

float scsp_3D_cylinders_channel::calcInfSum(float w, float h)
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
