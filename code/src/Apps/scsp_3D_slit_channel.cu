
# include "scsp_3D_slit_channel.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_slit_channel::scsp_3D_slit_channel() : lbm()
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
	
	// ----------------------------------------------
	// Lattice Boltzmann parameters:
	// ----------------------------------------------
	
	nu = inputParams("LBM/nu",0.1666666);
	bodyForx = inputParams("LBM/bodyForx",0.0);
	numIolets = inputParams("Lattice/numIolets",2);
	Re = inputParams("LBM/Re",2.0);
	umax = inputParams("LBM/umax",0.1);
	
	float rho = 1.0;
	float h = float(Nz)/2.0;  // float(Nz-1)/2.0;
	nu = umax*h/Re;
	bodyForx = 2.0*rho*umax*umax/(Re*h);
	
	lbm.setNu(nu);  
	
	// ----------------------------------------------
	// Calculate reference flux:
	// ----------------------------------------------
	
	float w = float(Ny);
	Q0 = 2.0*bodyForx*h*h*h*w/3.0/nu;
	cout << "reference flux = " << Q0 << endl;
	cout << "  " << endl;		
	
	// ----------------------------------------------
	// output parameters:
	// ----------------------------------------------
	
	vtkFormat = inputParams("Output/format","polydata");
	iskip = inputParams("Output/iskip",1);
	jskip = inputParams("Output/jskip",1);
	kskip = inputParams("Output/kskip",1);
	
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	lbm.allocate_forces();
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_slit_channel::~scsp_3D_slit_channel()
{
	lbm.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_slit_channel::initSystem()
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
		
	float h = float(Nz)/2.0;  //float(Nz-1)/2.0;
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				// analytical solution for Poiseuille flow
				// between parallel plates:
				double xvel0 = bodyForx*h*h/(2.0*nu);  // assume rho=1
				double z = float(k) - h;
				double xvel = xvel0*(1.0 - pow(z/h,2));
				lbm.setU(ndx,xvel);
				lbm.setV(ndx,0.0);
				lbm.setW(ndx,0.0);
				lbm.setR(ndx,1.0);
			}
		}
	}
				
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	lbm.memcopy_host_to_device();
		
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	lbm.initial_equilibrium(nBlocks,nThreads);	
						
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	writeOutput("macros",0);
		
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_slit_channel::cycleForward(int stepsPerCycle, int currentCycle)
{
		
	// ----------------------------------------------
	// determine the cummulative number of steps at the
	// beginning of this cycle:
	// ----------------------------------------------
	
	int cummulativeSteps = stepsPerCycle*currentCycle;	
	
	// ----------------------------------------------
	// loop through this cycle:
	// ----------------------------------------------
	
	for (int step=0; step<stepsPerCycle; step++) {
		cummulativeSteps++;	
				
		// zero fluid forces:
		lbm.zero_forces(nBlocks,nThreads);
						
		// update fluid:
		lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
		lbm.stream_collide_save_forcing(nBlocks,nThreads);
		//lbm.set_boundary_slit_velocity(0.0,nBlocks,nThreads);
		lbm.set_boundary_slit_density(nBlocks,nThreads);
		cudaDeviceSynchronize();

	}
	
	cout << cummulativeSteps << endl;	
		
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
	lbm.memcopy_device_to_host();
	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	writeOutput("macros",cummulativeSteps);
		
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_slit_channel::writeOutput(std::string tagname, int step)
{	
	// calculate relative viscosity:
	lbm.calculate_relative_viscosity("relative_viscosity_thru_time",Q0,step);
	
	// analyze the system:
	lbm.calculate_flow_rate_xdir("flowdata",step);
		
	// write output for LBM and IBM:	
	int precision = 3;
	lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
}









