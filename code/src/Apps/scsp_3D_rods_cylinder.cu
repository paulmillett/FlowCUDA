
# include "scsp_3D_rods_cylinder.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_rods_cylinder::scsp_3D_rods_cylinder() : lbm(),rods()
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
	float Re = inputParams("LBM/Re",2.0);
	umax = inputParams("LBM/umax",0.03);
		
	// ----------------------------------------------
	// Rods Immersed-Boundary parameters:
	// ----------------------------------------------
		
	int nBeadsPerRod = inputParams("IBM_RODS/nBeadsPerRod",0);
	nRods = inputParams("IBM_RODS/nRods",1);
	L0 = inputParams("IBM_RODS/L0",0.5);
	gam = inputParams("IBM_RODS/gamma",0.1);
	Drod = inputParams("IBM_RODS/diam",1.0);
	nBeads = nBeadsPerRod*nRods;
	Lrod = float(nBeadsPerRod-1)*L0;
	
	// ----------------------------------------------
	// calculate particle volume fraction:
	// ----------------------------------------------
	
	float Vp = float(nRods)*(M_PI*Drod*Drod*Lrod/4.0);
	float V = M_PI*chRad*chRad*float(Nx);
	float phi = Vp/V;
	cout << " " << endl;
	cout << "particle volume fraction = " << phi << endl;
	cout << " " << endl;
	
	// ----------------------------------------------
	// IBM set flags for PBC's:
	// ----------------------------------------------
	
	rods.set_pbcFlag(1,0,0);
		
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
	rods.allocate();	
	
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

scsp_3D_rods_cylinder::~scsp_3D_rods_cylinder()
{
	lbm.deallocate();
	rods.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_rods_cylinder::initSystem()
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
	// initialize rod immersed boundary info: 
	// ----------------------------------------------
	
	rods.create_first_rod();
	rods.duplicate_rods();
	rods.assign_rodIDs_to_beads();
	rods.set_rods_radii(Drod/2.0);
	
	// ----------------------------------------------			
	// mobility coefficients.  See Luders et al. 
	// J. Chem. Phys. 159:054901 (2023) Eqs. (15-17)
	// (note: mobility = diffusivity/kT)
	// (assume fluid density = 1)
	// ----------------------------------------------
	
	float ar = Lrod/Drod;  // aspect ratio
	float mobPar = (log(ar) - 0.207 + 0.980/ar - 0.133/(ar*ar)) / (2.0*M_PI*nu*Lrod); 
	float mobPer = (log(ar) + 0.839 + 0.185/ar + 0.233/(ar*ar)) / (4.0*M_PI*nu*Lrod);
	float mobRot = (log(ar) - 0.662 + 0.917/ar - 0.050/(ar*ar)) / (M_PI*nu*Lrod*Lrod*Lrod);
	
	rods.set_aspect_ratio(ar);
	rods.set_mobility_coefficients(mobPar,mobPer,mobRot);
	
	cout << " " << endl;
	cout << "Rod aspect ratio = " << ar << endl;
	cout << "Rod mobility coeff (parallel) = " << mobPar << endl;
	cout << "Rod mobility coeff (perpendicular) = " << mobPer << endl;
	cout << "Rod mobility coeff (rotational) = " << mobRot << endl;	
		
	// ----------------------------------------------
	// build the binMap array for neighbor lists: 
	// ----------------------------------------------
		
	rods.build_binMap(nBlocks,nThreads);	
		
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
		
	rods.memcopy_host_to_device();
	lbm.memcopy_host_to_device();
		
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
		
	//rods.randomize_rods_xdir_alligned_cylinder(Lrod,Drod/2.0,1.0,1.0);
	rods.semi_randomize_rods_xdir_alligned_cylinder(Lrod,Drod/2.0,Drod/2.0+0.1,Drod/2.0+0.1);
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
	
	cout << "  " << endl;
	cout << "Done with Initializtion" << endl;
	
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_rods_cylinder::cycleForward(int stepsPerCycle, int currentCycle)
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
			rods.stepIBM_Euler_cylindrical_channel(lbm,chRad,nBlocks,nThreads);
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
		rods.stepIBM_Euler_cylindrical_channel(lbm,chRad,nBlocks,nThreads);
		lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
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

void scsp_3D_rods_cylinder::writeOutput(std::string tagname, int step)
{				
	
	if (step == 0) {
		// only print out vtk files
		rods.orientation_in_cylindrical_channel(step);
		lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
		rods.write_output("rods",step);
	}
	
	if (step > 0) { 					
		// output rod position & orientation: 
		rods.orientation_in_cylindrical_channel(step);
		
		// write vtk output for LBM and IBM:
		int intervalVTK = nSteps/nVTKOutputs;
		if (nVTKOutputs == 0) intervalVTK = nSteps;
		if (step%intervalVTK == 0) {
			lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision);
			rods.write_output("rods",step);
		}
	}	
}








