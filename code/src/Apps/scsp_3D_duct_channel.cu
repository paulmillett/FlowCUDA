
# include "scsp_3D_duct_channel.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_duct_channel::scsp_3D_duct_channel() : lbm()
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
		
	float h = float(Nz)/2.0;
	float w = float(Ny)/2.0;
	float Dh = 4.0*(4.0*w*h) / (4.0*(h+w));
	nu = umax*Dh/(2.0*Re);
	lbm.setNu(nu);
	
	float infsum = calcInfSum(w,h);
	bodyForx = umax*nu*M_PI*M_PI*M_PI/(16.0*w*w*infsum);
		
	// ----------------------------------------------
	// Calculate reference flux:
	// ----------------------------------------------
		
	Q0 = calcRefFlux(w,h);
	cout << "reference flux = " << Q0 << endl;
	cout << "viscosity = " << nu << endl;
	cout << "body force = " << bodyForx << endl;
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

scsp_3D_duct_channel::~scsp_3D_duct_channel()
{
	lbm.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_duct_channel::initSystem()
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

void scsp_3D_duct_channel::cycleForward(int stepsPerCycle, int currentCycle)
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
		//lbm.set_channel_wall_velocity(0.0,nBlocks,nThreads);
		lbm.set_boundary_duct_density(nBlocks,nThreads);
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

void scsp_3D_duct_channel::writeOutput(std::string tagname, int step)
{	
	// calculate relative viscosity:
	lbm.calculate_relative_viscosity("relative_viscosity_thru_time",Q0,step);
	
	// analyze the system:
	lbm.calculate_flow_rate_xdir("flowdata",step);
	
	// write output for LBM and IBM:	
	int precision = 3;
	lbm.vtk_structured_output_ruvw(tagname,step,iskip,jskip,kskip,precision); 
}



// --------------------------------------------------------
// Calculate infinite sum associated with solution
// to velocity profile in rectanglular channel:
// --------------------------------------------------------

float scsp_3D_duct_channel::calcInfSum(float w, float h)
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

float scsp_3D_duct_channel::calcRefFlux(float w, float h)
{
	float Q00 = 0.0;
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
			Q00 += u0;
		}
	}
	return Q00;			
}




