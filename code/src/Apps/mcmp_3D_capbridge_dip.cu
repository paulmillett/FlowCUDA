
# include "mcmp_3D_capbridge_dip.cuh"
# include "../IO/GetPot"
# include <math.h>
# include <string> 
# include <iostream>
# include <iomanip>
# include <fstream>
# include <sstream>
using namespace std;   



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

mcmp_3D_capbridge_dip::mcmp_3D_capbridge_dip() : lbm()
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
	// Lattice Boltzmann parameters:
	// ----------------------------------------------
	
	nu = inputParams("LBM/nu",0.1666666);
		
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
	
	// ----------------------------------------------
	// delete "forces.txt" file from previous simulations:
	// ----------------------------------------------
	
	std::system("exec rm -rf forces.txt"); 
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

mcmp_3D_capbridge_dip::~mcmp_3D_capbridge_dip()
{
	lbm.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void mcmp_3D_capbridge_dip::initSystem()
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
	
	lbm.setPrx(0,65.0);
	lbm.setPry(0,49.5);
	lbm.setPrz(0,49.5);
	lbm.setPvx(0,0.0);
	lbm.setPvy(0,0.0);
	lbm.setPvz(0,0.0);
	lbm.setPrInner(0,27.5);
	lbm.setPrOuter(0,32.5);
	
	lbm.setPrx(1,134.0);
	lbm.setPry(1,49.5);
	lbm.setPrz(1,49.5);
	lbm.setPvx(1,0.0);
	lbm.setPvy(1,0.0);
	lbm.setPvz(1,0.0);
	lbm.setPrInner(1,27.5);
	lbm.setPrOuter(1,32.5);
			
	// ----------------------------------------------			
	// initialize macros: 
	// ----------------------------------------------
	
	// initialize solid field:
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {		
				int ndx = k*Nx*Ny + j*Nx + i;	
				lbm.setX(ndx,i);
				lbm.setY(ndx,j);
				lbm.setZ(ndx,k);
				float Bi = 0.0;
				for (int p=0; p<nParts; p++) {
					float dx = float(i) - lbm.getPrx(p);
					float dy = float(j) - lbm.getPry(p);
					float dz = float(k) - lbm.getPrz(p);
					float rr = sqrt(dx*dx + dy*dy + dz*dz);				
					if (rr <= lbm.getPrOuter(p)) {
						if (rr < lbm.getPrInner(p)) {
							Bi = 1.0;
						}
						else {
							float rsc = rr - lbm.getPrInner(p);
							Bi = 1.0 - rsc/(lbm.getPrOuter(p) - lbm.getPrInner(p));
						}
					}
				}
				if (i > 60 && i < 139 && j > 30 && j < 70 && k > 30 && k < 70) {
					lbm.setRA(ndx,1.015*(1.0-Bi) + rApart*Bi);
					lbm.setRB(ndx,0.015*(1.0-Bi) + rBpart*Bi);				
				}
				else {
					lbm.setRA(ndx,0.04*(1.0-Bi) + rApart*Bi);
					lbm.setRB(ndx,0.99*(1.0-Bi) + rBpart*Bi);
				
				}		 
			}
		}
	}	
			
	// initialize velocity fields
	for (int i=0; i<nVoxels; i++) {
		lbm.setU(i,0.0);
		lbm.setV(i,0.0);
		lbm.setW(i,0.0);
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

void mcmp_3D_capbridge_dip::cycleForward(int stepsPerCycle, int currentCycle)
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
	outfile.open ("forces.txt", ios::out | ios::app);
	
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
		lbm.collide_stream_dip(nBlocks,nThreads);  		
		lbm.swap_populations();	
		
		// ------------------------------
		// write particle drag force to file:											   
		// ------------------------------ 
		
		lbm.memcopy_device_to_host_particles();		
		outfile << fixed << setprecision(3) << lbm.getPfx(0) << "   " << lbm.getPfx(1) << endl;	
		
		// ------------------------------
		// update particles:											   
		// ------------------------------ 
		
		lbm.fix_particle_velocity_dip(0.0,nBlocks,nThreads);
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

void mcmp_3D_capbridge_dip::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// function location:
	// "io/write_vtk_output.cuh"
	// ----------------------------------------------
	
	lbm.write_output(tagname,step,1,1,1); 

}










