
# include "scsp_3D_fibers_duct.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_fibers_duct::scsp_3D_fibers_duct() : lbm(),fibers()
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
	
	int sizeFIL = fibers.get_max_array_size();	
	int sizeMAX = max(nVoxels,sizeFIL);	
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
	umax = inputParams("LBM/umax",0.1);
		
	// ----------------------------------------------
	// Fibers Immersed-Boundary parameters:
	// ----------------------------------------------
		
	int nBeadsPerFiber = inputParams("IBM_FIBERS/nBeadsPerFilam",0);
	nFibers = inputParams("IBM_FIBERS/nFibers",1);
	gam = inputParams("IBM_FIBERS/gamma",0.1);
	dS = inputParams("IBM_FIBERS/dS",1.0);
	nBeads = nBeadsPerFiber*nFibers;
	Lfib = float(nBeadsPerFiber)*dS;
	
	// ----------------------------------------------
	// IBM set flags for PBC's:
	// ----------------------------------------------
	
	fibers.set_pbcFlag(1,0,0);
		
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
	fibers.allocate();	
	
	// ----------------------------------------------
	// calculate body-force depending on Re:
	// ----------------------------------------------
	
	float w = float(Ny)/2.0;
	float h = float(Nz)/2.0;	
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
	cout << "  " << endl;
	cout << "Re = " << Re << endl;
	cout << "Body Force X-dir = " << bodyForx << endl;
	cout << "nu = " << nu << endl;
	cout << "  " << endl;	
	
	// ----------------------------------------------
	// set up buffer sizes for cuSparse:
	// ----------------------------------------------
	
	fibers.cuSparse_buffer_sizes();
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_fibers_duct::~scsp_3D_fibers_duct()
{
	lbm.deallocate();
	fibers.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_fibers_duct::initSystem()
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
	// initialize macros: 
	// ----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		lbm.setU(i,0.0);
		lbm.setV(i,0.0);
		lbm.setW(i,0.0);
		lbm.setR(i,1.0);		
	}
	
	// ----------------------------------------------			
	// initialize filament immersed boundary info: 
	// ----------------------------------------------
	
	fibers.create_first_fiber();
	fibers.duplicate_fibers();
	fibers.assign_fiberIDs_to_beads();
		
	fibers.set_gamma(gam);
	cout << "  " << endl;
	cout << "Filament gamma = " << gam << endl;
				
	// ----------------------------------------------
	// build the binMap array for neighbor lists: 
	// ----------------------------------------------
	
	fibers.build_binMap(nBlocks,nThreads);
		
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	lbm.memcopy_host_to_device();
	fibers.memcopy_host_to_device();
		
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	lbm.initial_equilibrium(nBlocks,nThreads);	
		
	// ----------------------------------------------
	// set the random number seed: 
	// ----------------------------------------------
	
	srand(time(NULL));
	
	// ----------------------------------------------
	// randomly disperse filaments: 
	// ----------------------------------------------
		
	if (nFibers == 1) {
		float xsh = inputParams("IBM_FIBERS/xsh",float(Nx)/2.0);
		float ysh = inputParams("IBM_FIBERS/ysh",float(Ny)/2.0);
		float zsh = inputParams("IBM_FIBERS/zsh",float(Nz)/2.0);
		float alpha = inputParams("IBM_FIBERS/alpha",0.0);
		float beta = inputParams("IBM_FIBERS/beta",0.0);
		float gammarot = inputParams("IBM_FIBERS/gammarot",0.0);
		fibers.rotate_and_shift_bead_positions(0,xsh,ysh,zsh,alpha,beta,gammarot);
		//fibers.initialize_fiber_curved();
	}
	
	if (nFibers > 1) {
		fibers.randomize_fibers_xdir_alligned(4.0);	
		//fibers.randomize_fibers(Lfib+2.0);
	}
			
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	fibers.memcopy_device_to_host();
	writeOutput("macros",0);
			
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_fibers_duct::cycleForward(int stepsPerCycle, int currentCycle)
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
			fibers.stepIBM(lbm,nBlocks,nThreads);
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
		fibers.stepIBM(lbm,nBlocks,nThreads);
		lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
		lbm.stream_collide_save_forcing(nBlocks,nThreads);
		cudaDeviceSynchronize();
	}
	
	cout << cummulativeSteps << endl;	
		
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
	lbm.memcopy_device_to_host();
	fibers.memcopy_device_to_host();    
	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	writeOutput("macros",cummulativeSteps);
		
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_fibers_duct::writeOutput(std::string tagname, int step)
{				
	
	if (step == 0) {
		// only print out vtk files
		lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		fibers.write_output("fibers",step);
	}
	
	if (step > 0) { 					
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision);
			fibers.write_output("fibers",step);
		}
	}	
}



// --------------------------------------------------------
// Calculate infinite sum associated with solution
// to velocity profile in rectanglular channel:
// --------------------------------------------------------

float scsp_3D_fibers_duct::calcInfSum(float w, float h)
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






