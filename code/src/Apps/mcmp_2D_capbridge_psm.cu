
# include "mcmp_2D_capbridge_psm.cuh"
# include "../IO/GetPot"
# include <math.h>
# include <string> 
using namespace std;   



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

mcmp_2D_capbridge_psm::mcmp_2D_capbridge_psm() : lbm(), parts()
{	
	
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	
	// ----------------------------------------------
	// lattice parameters:
	// ----------------------------------------------
	
	nVoxels = inputParams("Lattice/nVoxels",0);
	Q = inputParams("Lattice/Q",9);	
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	
	// ----------------------------------------------
	// GPU parameters:
	// ----------------------------------------------
	
	nThreads = inputParams("GPU/nThreads",512);
	nBlocks = (nVoxels+(nThreads-1))/nThreads;  // integer division
	
	// ----------------------------------------------
	// Lattice Boltzmann parameters:
	// ----------------------------------------------
	
	nu = inputParams("LBM/nu",0.1666666);
	gAB = inputParams("LBM/gAB",6.0);
		
	// ----------------------------------------------
	// Particles parameters:
	// ----------------------------------------------
	
	nParts = inputParams("Particles/nParts",1);	
	rApart = inputParams("Particles/rApart",0.5);
	rBpart = inputParams("Particles/rBpart",0.5);			
	
	// ----------------------------------------------
	// output parameters:
	// ----------------------------------------------
	
	vtkFormat = inputParams("Output/format","structured");
	
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	parts.allocate();	
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

mcmp_2D_capbridge_psm::~mcmp_2D_capbridge_psm()
{
	lbm.deallocate();
	parts.deallocate();	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void mcmp_2D_capbridge_psm::initSystem()
{
	
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
	
	// ----------------------------------------------
	// create the periodic lattice:
	// ----------------------------------------------	
	
	lbm.create_lattice_box_periodic();	
			
	// ----------------------------------------------			
	// initialize particles: 
	// ----------------------------------------------
		
	parts.xH[0] = 420.0;
	parts.yH[0] = 250.0;
	parts.rInnerH[0] = 40.0;
	parts.rOuterH[0] = 45.0;
	
	parts.xH[1] = 580.0;
	parts.yH[1] = 250.0;
	parts.rInnerH[1] = 70.0;
	parts.rOuterH[1] = 75.0;
		
	// ----------------------------------------------			
	// initialize macros: 
	// ----------------------------------------------
	
	// initialize solid field:
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {		
			int ndx = j*Nx + i;	
			lbm.setX(ndx,i);
			lbm.setY(ndx,j);
			float Bi = 0.0;
			for (int k=0; k<nParts; k++) {
				float dx = float(i) - parts.xH[k];
				float dy = float(j) - parts.yH[k];
				float rr = sqrt(dx*dx + dy*dy);				
				if (rr <= parts.rOuterH[k]) {
					if (rr < parts.rInnerH[k]) {
						Bi = 1.0;
					}
					else {
						float rsc = rr - parts.rInnerH[k];
						Bi = 1.0 - rsc/(parts.rOuterH[k] - parts.rInnerH[k]);
					}
				}	
			}			
			if (i > 400 && i < 600 && j > 220 && j < 280) {
				lbm.setRA(ndx,1.0*(1.0-Bi) + rApart*Bi);
				lbm.setRB(ndx,0.02*(1.0-Bi) + rBpart*Bi);				
			}
			else {
				lbm.setRA(ndx,0.02*(1.0-Bi) + rApart*Bi);
				lbm.setRB(ndx,1.0*(1.0-Bi) + rBpart*Bi);
				
			}		 
		}
	}
			
	// initialize velocity fields
	for (int i=0; i<nVoxels; i++) {
		lbm.setU(i,0.0);
		lbm.setV(i,0.0);
	}	
	
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// ----------------------------------------------
	
	lbm.stream_index_push();	
			
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	writeOutput("macros",0);
	
	// ----------------------------------------------	
	// copy arrays from host to device: 
	// ----------------------------------------------
	
	lbm.memcopy_host_to_device();
	parts.memcopy_host_to_device();
	
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	lbm.initial_equilibrium_psm(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Step forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void mcmp_2D_capbridge_psm::cycleForward(int stepsPerCycle, int currentCycle)
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
		
		// ------------------------------
		// zero particle forces:
		// ------------------------------
		
		parts.zero_forces(nBlocks,nThreads);
		
		// ------------------------------
		// update density fields:
		// ------------------------------
		
		lbm.update_particles_on_lattice_psm(parts.x,parts.y,parts.B,parts.rInner,parts.rOuter,
		                                    parts.pIDgrid,nParts,nBlocks,nThreads);
		lbm.compute_density_psm(nBlocks,nThreads);
		cudaDeviceSynchronize();
		
		// ------------------------------
		// update fluid fields:											   
		// ------------------------------ 
		
		lbm.compute_SC_forces_psm(parts.B,parts.fx,parts.fy,parts.pIDgrid,nBlocks,nThreads);
		lbm.compute_velocity_psm(parts.vx,parts.vy,parts.fx,parts.fy,parts.B,parts.pIDgrid,nBlocks,nThreads);
		lbm.collide_stream_psm(parts.vx,parts.vy,parts.B,parts.pIDgrid,rApart,rBpart,nBlocks,nThreads);  		
		lbm.swap_populations();	
		
		// ------------------------------
		// update particles:											   
		// ------------------------------ 
		
		parts.move_particles(nBlocks,nThreads);			
		cudaDeviceSynchronize();
				
	}
	
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
// Write output:
// --------------------------------------------------------

void mcmp_2D_capbridge_psm::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// function location:
	// "io/write_vtk_output.cuh"
	// ----------------------------------------------
	
	lbm.write_output(tagname,step); 

}










