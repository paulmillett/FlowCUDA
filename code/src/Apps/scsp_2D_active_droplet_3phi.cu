
# include "scsp_2D_active_droplet_3phi.cuh"
# include "../IO/GetPot"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_2D_active_droplet_3phi::scsp_2D_active_droplet_3phi() : lbm()
{		
	
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	
	// ----------------------------------------------
	// lattice parameters:
	// ----------------------------------------------
	
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	nVoxels = inputParams("Lattice/nVoxels",0);
	Q = inputParams("Lattice/Q",9);	
	
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
	dropRad1 = inputParams("LBM/dropRad1",10.0);
	dropRad2 = inputParams("LBM/dropRad2",10.0);
	activity = inputParams("LBM/activity",-0.0005);
	
	// ----------------------------------------------
	// Output parameters:
	// ----------------------------------------------
	
	iskip = inputParams("Output/iskip",1);
	jskip = inputParams("Output/jskip",1);
	nVTKOutputs = inputParams("Output/nVTKOutputs",0);
	
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_2D_active_droplet_3phi::~scsp_2D_active_droplet_3phi()
{	
	lbm.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_2D_active_droplet_3phi::initSystem()
{
		
	// ----------------------------------------------
	// create the lattice using "box" function. 
	// ----------------------------------------------	
		
	lbm.create_lattice_box_periodic();
	
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
		lbm.setR(i,1.0);	
	}
	
	// ----------------------------------------------			
	// initialize order parameter phi: 
	// ----------------------------------------------
	
	float phisum1 = 0.0;
	float phisum2 = 0.0;
	float phisum3 = 0.0;
	
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {		
			int ndx = j*Nx + i;	
			float phi1 = 0.0;
			float phi2 = 0.0;
			// phi 1 domain:
			float centx = float(Nx)/2.0 + (2.0/3.0)*dropRad1;
			float centy = float(Ny)/2.0;
			float dx = float(i) - centx;
			float dy = float(j) - centy;
			float r = sqrt(dx*dx + dy*dy);
			if (r < dropRad1 && i > Nx/2) phi1 = 1.0;
			// phi 2 domain:
			centx = float(Nx)/2.0 - (2.0/3.0)*dropRad2;
			centy = float(Ny)/2.0;
			dx = float(i) - centx;
			dy = float(j) - centy;
			r = sqrt(dx*dx + dy*dy);
			if (r < dropRad2 && i < Nx/2) phi2 = 1.0;
			// set values:
			lbm.setPhi1(ndx,phi1);
			lbm.setPhi2(ndx,phi2);
			lbm.setPhi3(ndx,1.0 - phi1 - phi2);	
			phisum1 += phi1;
			phisum2 += phi2;
			phisum3 += 1.0 - phi1 - phi2;		
		}
	}
	
	lbm.setPhiSum(phisum1,phisum2,phisum3);
	
	// ----------------------------------------------			
	// relax phi fields:
	// ----------------------------------------------
	
	lbm.memcopy_host_to_device();	
	for (int i=0; i<10000; i++) {
		lbm.scsp_active_fluid_chemical_potential(nBlocks,nThreads);
		lbm.scsp_active_fluid_update_phi(nBlocks,nThreads);
	}	
	lbm.memcopy_device_to_host();
	
	// ----------------------------------------------			
	// initialize orientation: 
	// ----------------------------------------------
		
	for (int i=0; i<nVoxels; i++) {
		float phi = lbm.getPhi1(i);
		float theta = 0.0;
		float px = 1.0;
		float py = 0.0;
		if (activity > 0.0) {  // if extensile activity
			px = 0.0;
			py = 1.0;
		}		
		float pxr = px*cos(theta) - py*sin(theta);
		float pyr = px*sin(theta) + py*cos(theta);
		lbm.setPx(i,pxr*phi);
		lbm.setPy(i,pyr*phi);	
	}
		
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	writeOutput("macros",0);
	
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	lbm.memcopy_host_to_device();
	
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	lbm.initial_equilibrium(nBlocks,nThreads);
	
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_2D_active_droplet_3phi::cycleForward(int stepsPerCycle, int currentCycle)
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
		lbm.zero_forces(nBlocks,nThreads);
		lbm.scsp_active_fluid_chemical_potential(nBlocks,nThreads);
		lbm.scsp_active_fluid_molecular_field_with_phi(nBlocks,nThreads);		
		lbm.scsp_active_fluid_stress(nBlocks,nThreads);
		lbm.scsp_active_fluid_forces(nBlocks,nThreads);
		lbm.scsp_active_fluid_capillary_force(nBlocks,nThreads);
		lbm.scsp_active_update_orientation(nBlocks,nThreads);
		lbm.scsp_active_fluid_update_phi(nBlocks,nThreads);
		//lbm.stream_collide_save_forcing(nBlocks,nThreads);		
		lbm.stream_collide_save_forcing_varvisc(nBlocks,nThreads);
		lbm.set_wall_velocity_ydir(0.0,nBlocks,nThreads);
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

void scsp_2D_active_droplet_3phi::writeOutput(std::string tagname, int step)
{	
	
	// ----------------------------------------------
	// write VTK file
	// ----------------------------------------------
	
	int intervalVTK = nSteps/nVTKOutputs;
	if (step%intervalVTK == 0) {
		lbm.write_output(tagname,step,iskip,jskip);
	}
		
	// ----------------------------------------------
	// write droplet data to file
	// ----------------------------------------------
	
	lbm.write_output_droplet_properties(step);
			
}







