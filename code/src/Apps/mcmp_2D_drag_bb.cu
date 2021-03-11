
# include "mcmp_2D_drag_bb.cuh"
# include "../IO/GetPot"
# include <math.h>
# include <string> 
using namespace std;   



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

mcmp_2D_drag_bb::mcmp_2D_drag_bb() : lbm()
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
	gAS = inputParams("LBM/gAS",6.0);
	gBS = inputParams("LBM/gBS",6.0); 	
	
	// ----------------------------------------------
	// Particles parameters:
	// ----------------------------------------------
	
	nParts = inputParams("Particles/nParts",1);	
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

mcmp_2D_drag_bb::~mcmp_2D_drag_bb()
{
	lbm.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void mcmp_2D_drag_bb::initSystem()
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
	// initialize particle information:
	// ----------------------------------------------
	
	float xpos = inputParams("Particles/xpos",100.0);
	float ypos = inputParams("Particles/ypos",100.0);
	float rad = inputParams("Particles/rad",10.0);
	
	lbm.setPrx(0,xpos);
	lbm.setPry(0,ypos);
	lbm.setPrad(0,rad);
	lbm.setPmass(0,3.14159*rad*rad);
			
	// ----------------------------------------------			
	// initialize macros: 
	// ----------------------------------------------	
	
	// initialize solid field:	
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {		
			int ndx = j*Nx + i;	
			lbm.setX(ndx,i);
			lbm.setY(ndx,j);
			lbm.setS(ndx,0);
			float dx = float(i) - lbm.getPrx(0);
			float dy = float(j) - lbm.getPry(0);
			float rr = sqrt(dx*dx + dy*dy);
			if (rr <= lbm.getPrad(0)) lbm.setS(ndx,1); 
		}
	}
		
	// initialize density fields: 
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			int ndx = j*Nx + i;
			int sij = lbm.getS(ndx);			
			float rhoA = 1.00;
			float rhoB = 0.00;					
			lbm.setRA(ndx,rhoA*float(1 - sij));
			lbm.setRB(ndx,rhoB*float(1 - sij));						
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
	
	lbm.initial_equilibrium_bb(nBlocks,nThreads);
	lbm.map_particles_on_lattice_bb(nBlocks,nThreads);
		
}



// --------------------------------------------------------
// Step forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void mcmp_2D_drag_bb::cycleForward(int stepsPerCycle, int currentCycle)
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
		
		lbm.zero_particle_forces_bb(nBlocks,nThreads);
		
		// ------------------------------
		// update density fields:
		// ------------------------------
		
		lbm.update_particles_on_lattice_bb(nBlocks,nThreads);
		lbm.compute_density_bb(nBlocks,nThreads);
		cudaDeviceSynchronize();
		
		// ------------------------------
		// update fluid fields:											   
		// ------------------------------ 
		
		lbm.compute_SC_forces_bb(nBlocks,nThreads);
		lbm.compute_velocity_bb(nBlocks,nThreads);
		lbm.set_boundary_velocity_bb(0.0,0.0,nBlocks,nThreads);
		lbm.collide_stream_bb(nBlocks,nThreads);
		lbm.bounce_back_moving(nBlocks,nThreads);
		lbm.swap_populations();	
		
		// ------------------------------
		// write particle drag force to file:											   
		// ------------------------------ 
		
		lbm.memcopy_device_to_host_particles();		
		outfile << lbm.getPfx(0) << endl;	
		
		// ------------------------------
		// update particles:											   
		// ------------------------------ 
		
		lbm.fix_particle_velocity_bb(pvel,nBlocks,nThreads);
		lbm.move_particles_bb(nBlocks,nThreads);
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

void mcmp_2D_drag_bb::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// function location:
	// "io/write_vtk_output.cuh"
	// ----------------------------------------------
	
	lbm.write_output(tagname,step); 

}










