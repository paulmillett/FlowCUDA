
# include "mcmp_2D_drag_dip.cuh"
# include "../IO/GetPot"
# include <math.h>
# include <string> 
# include <iostream>
using namespace std;   



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

mcmp_2D_drag_dip::mcmp_2D_drag_dip() : lbm()
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
		
	// ----------------------------------------------
	// Particles parameters:
	// ----------------------------------------------
	
	nParts = inputParams("Particles/nParts",1);	
	rApart = inputParams("Particles/rApart",0.5);
	rBpart = inputParams("Particles/rBpart",0.5);
	pvel = inputParams("Particles/pvel",0.0);			
	
	// ----------------------------------------------
	// output parameters:
	// ----------------------------------------------
	
	vtkFormat = inputParams("Output/format","structured");
	
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	
	// ----------------------------------------------
	// delete "drag.txt" file from previous simulations:
	// ----------------------------------------------
	
	std::system("exec rm -rf drag.txt"); 
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

mcmp_2D_drag_dip::~mcmp_2D_drag_dip()
{
	lbm.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void mcmp_2D_drag_dip::initSystem()
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
	
	float xpos = inputParams("Particles/xpos",100.0);
	float ypos = inputParams("Particles/ypos",100.0);
	float rInner = inputParams("Particles/rInner",10.0);
	float rOuter = inputParams("Particles/rOuter",15.0);
	
	lbm.setPrx(0,xpos);
	lbm.setPry(0,ypos);
	lbm.setPrInner(0,rInner);
	lbm.setPrOuter(0,rOuter);	
	
	// ----------------------------------------------			
	// initialize macros: 
	// ----------------------------------------------
		
	// initialize solid field:
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {		
			int ndx = j*Nx + i;	
			lbm.setX(ndx,i);
			lbm.setY(ndx,j);
			float dx = float(i) - lbm.getPrx(0);
			float dy = float(j) - lbm.getPry(0);
			float rr = sqrt(dx*dx + dy*dy);
			float Bi = 0.0;
			if (rr <= lbm.getPrOuter(0)) {
				if (rr < lbm.getPrInner(0)) {
					Bi = 1.0;
				}
				else {
					float rsc = rr - lbm.getPrInner(0);
					Bi = 1.0 - rsc/(lbm.getPrOuter(0) - lbm.getPrInner(0));
				}
			}				
			float rhoA = 1.00;
			float rhoB = 0.02;	
			lbm.setRA(ndx,rhoA*(1.0-Bi) + rApart*Bi);
			lbm.setRB(ndx,rhoB*(1.0-Bi) + rBpart*Bi);			 
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
	
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	lbm.initial_equilibrium_dip(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Step forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// -------------------------------------------------------- 

void mcmp_2D_drag_dip::cycleForward(int stepsPerCycle, int currentCycle)
{
	
	// ----------------------------------------------
	// determine the cummulative number of steps at the
	// beginning of this cycle:
	// ----------------------------------------------
		
	int cummulativeSteps = stepsPerCycle*currentCycle;
	
	// ----------------------------------------------
	// open file that stores drag force data:
	// ----------------------------------------------
	
	ofstream outfile;
	outfile.open ("drag.txt", ios::out | ios::app);
	
	// ----------------------------------------------
	// loop through this cycle:
	// ----------------------------------------------

	for (int step=0; step<stepsPerCycle; step++) {
		
		cummulativeSteps++;
		
		// ------------------------------
		// zero particle forces:
		// ------------------------------
		
		lbm.zero_particle_forces_dip(nBlocks,nThreads);
		
		// ------------------------------
		// update density fields:
		// ------------------------------
		
		lbm.map_particles_to_lattice_dip(nBlocks,nThreads);
		lbm.compute_density_dip(nBlocks,nThreads);
		cudaDeviceSynchronize();
		
		// ------------------------------
		// update fluid fields:											   
		// ------------------------------ 
		
		lbm.compute_SC_forces_dip(nBlocks,nThreads);
		lbm.compute_velocity_dip_2(nBlocks,nThreads);
		lbm.set_boundary_velocity_dip(0.0,0.0,nBlocks,nThreads);
		lbm.collide_stream_dip(nBlocks,nThreads);  		
		lbm.swap_populations();	
		
		// ------------------------------
		// write particle drag force to file:											   
		// ------------------------------ 
		
		lbm.memcopy_device_to_host_particles();		
		outfile << lbm.getPfx(0) << endl;		

		// ------------------------------
		// update particles:											   
		// ------------------------------ 

		lbm.fix_particle_velocity_dip(pvel,nBlocks,nThreads);
		lbm.move_particles_dip(nBlocks,nThreads);
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
	
	// ----------------------------------------------
	// close file that stores drag force data:
	// ----------------------------------------------
	
	outfile.close();

}



// --------------------------------------------------------
// Write output:
// --------------------------------------------------------

void mcmp_2D_drag_dip::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// function location:
	// "io/write_vtk_output.cuh"
	// ----------------------------------------------
	
	lbm.write_output(tagname,step); 

}










