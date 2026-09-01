
# include "scsp_3D_discs_cylinder.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_discs_cylinder::scsp_3D_discs_cylinder() : lbm(),discs()
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
	
	int sizeDisc = discs.get_max_array_size();	
	int sizeMAX = max(nVoxels,sizeDisc);	
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
	umax = inputParams("LBM/umax",0.03);
	
	// ----------------------------------------------
	// Discs Immersed-Boundary parameters:
	// ----------------------------------------------
		
	int nBeadsPerDisc = inputParams("IBM_DISCS/nBeadsPerDisc",0);
	nDiscs = inputParams("IBM_DISCS/nDiscs",1);
	gam = inputParams("IBM_DISCS/gamma",0.1);
	Ddisc = inputParams("IBM_DISCS/diam",1.0);
	Hdisc = inputParams("IBM_DISCS/thickness",1.0);
	nBeads = nBeadsPerDisc*nDiscs;	
		
	// ----------------------------------------------
	// calculate particle volume fraction:
	// ----------------------------------------------
	
	float Vp = float(nDiscs)*(M_PI*Ddisc*Ddisc*Hdisc/4.0);
	float V = M_PI*chRad*chRad*float(Nx);
	float phi = Vp/V;
	cout << " " << endl;
	cout << "particle volume fraction = " << phi << endl;
	cout << " " << endl;
	
	// ----------------------------------------------
	// IBM set flags for PBC's:
	// ----------------------------------------------
	
	discs.set_pbcFlag(1,0,0);
		
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
	lbm.allocate_solid();
	discs.allocate();
	
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
	Q0 = M_PI*chRad*chRad*chRad*chRad*bodyForx/(8.0*nu);
	
	cout << "  " << endl;
	cout << "Re = " << Re << endl;
	cout << "Body Force X-dir = " << bodyForx << endl;
	cout << "nu = " << nu << endl;
	cout << "umax = " << umax << endl;
	cout << "Q0 = " << Q0 << endl; 
	cout << "  " << endl;	
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_discs_cylinder::~scsp_3D_discs_cylinder()
{
	lbm.deallocate();
	discs.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_discs_cylinder::initSystem()
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
	// initialize macros: 
	// ----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		lbm.setU(i,0.0);
		lbm.setV(i,0.0);
		lbm.setW(i,0.0);
		lbm.setR(i,1.0);		
	}
	
	// ----------------------------------------------			
	// initialize disc immersed boundary info: 
	// ----------------------------------------------
	
	discs.create_first_disc();
	discs.duplicate_discs();
	discs.assign_discIDs_to_beads();
	discs.set_discs_radii(Ddisc/2.0);
	float ar = Hdisc/Ddisc; // aspect ratio
	discs.set_aspect_ratio(ar);
	discs.set_discs_half_thickness(Hdisc/2.0);
	discs.set_mobility_coefficients(nu,ar,Ddisc/2.0);	
	
	if (nDiscs == 1) {
		discs.rotate_and_shift_bead_positions(0,float(Nx-1)/2.0,float(Ny-1)/4.0,float(Nz-1)/2.0,0.0,M_PI/2.0,0.0);
		//discs.rotate_and_shift_bead_positions(0,float(Nx-1)/2.0,float(Ny-1)/2.0,float(Nz-1)/4.0,0.0,0.0,0.0);
	}
		
	// ----------------------------------------------
	// build the binMap array for neighbor lists: 
	// ----------------------------------------------
	
	discs.build_binMap(nBlocks,nThreads);	
		
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	lbm.memcopy_host_to_device();
	lbm.memcopy_host_to_device_solid();
	discs.memcopy_host_to_device();
		
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	lbm.initial_equilibrium(nBlocks,nThreads);	
		
	// ----------------------------------------------
	// set the random number seed: 
	// ----------------------------------------------
	
	srand(time(NULL));
	
	// ----------------------------------------------
	// randomly disperse discs: 
	// ----------------------------------------------
			
	if (nDiscs > 1) {
		discs.randomize_discs_cylinder(Ddisc); 
	}
	discs.set_disc_position_orientation(nBlocks,nThreads);
	
	// ----------------------------------------------
	// push discs inside slit (if 'random'), then
	// relax discs to eliminate any overlap:
	// ----------------------------------------------
	
	if (nDiscs > 1) {
		//discs.stepIBM_Euler_push_inside_slit(1000,nBlocks,nThreads);
		//discs.stepIBM_Euler_relax_discs_in_slit(1000,nBlocks,nThreads);
	}
		
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	discs.memcopy_device_to_host();
	writeOutput("macros",0);
	
	// ----------------------------------------------
	// set IBM velocities & forces to zero: 
	// ----------------------------------------------
	
	discs.zero_bead_forces(nBlocks,nThreads);
	
	cout << "  " << endl;
	cout << "Done with Initializtion" << endl;
			
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_discs_cylinder::cycleForward(int stepsPerCycle, int currentCycle)
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
			discs.stepIBM_Euler_cylindrical_channel(lbm,chRad,nBlocks,nThreads);
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
		discs.stepIBM_Euler_cylindrical_channel(lbm,chRad,nBlocks,nThreads);
		lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
		lbm.stream_collide_save_forcing(nBlocks,nThreads);
		cudaDeviceSynchronize();
	}
	
	cout << cummulativeSteps << endl;	
		
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
	lbm.memcopy_device_to_host();
	discs.memcopy_device_to_host();    
	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	writeOutput("macros",cummulativeSteps);
		
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_discs_cylinder::writeOutput(std::string tagname, int step)
{				
	
	if (step == 0) {
		// only print out vtk files
		discs.orientation_in_cylindrical_channel(step);
		lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		discs.write_output("discs",step);
	}
	
	if (step > 0) {
		// output rod position & orientation: 
		discs.orientation_in_cylindrical_channel(step);
						
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision);
			discs.write_output("discs",step);
		}
	}	
}








