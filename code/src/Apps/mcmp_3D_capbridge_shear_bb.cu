
# include "mcmp_3D_capbridge_shear_bb.cuh"
# include "../IO/GetPot"
# include <math.h>
# include <string> 
using namespace std;   



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

mcmp_3D_capbridge_shear_bb::mcmp_3D_capbridge_shear_bb() : lbm()
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
	gAB = inputParams("LBM/gAB",6.0);
	shearVel = inputParams("LBM/shearVel",0.0);
		
	// ----------------------------------------------
	// Particles parameters:
	// ----------------------------------------------
	
	nParts = inputParams("Particles/nParts",2);	
	pvel = inputParams("Particles/pvel",0.0);
	Khertz = inputParams("Particles/Khertz",0.0);
	halo = inputParams("Particles/halo",0.0);			
	
	// ----------------------------------------------
	// output parameters:
	// ----------------------------------------------
	
	vtkFormat = inputParams("Output/format","structured");
	iskip = inputParams("Output/iskip",1);
	jskip = inputParams("Output/jskip",1);
	kskip = inputParams("Output/kskip",1);	
	
	// ----------------------------------------------
	// allocate array memory (host & device):
	// ----------------------------------------------
	
	lbm.allocate();
	
	// ----------------------------------------------
	// delete "drag.txt" file from previous simulations:
	// ----------------------------------------------
	
	std::system("exec rm -rf force.txt"); 
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

mcmp_3D_capbridge_shear_bb::~mcmp_3D_capbridge_shear_bb()
{
	lbm.deallocate();
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void mcmp_3D_capbridge_shear_bb::initSystem()
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
	
	float xpos0 = inputParams("Particles/xpos0",100.0);
	float ypos0 = inputParams("Particles/ypos0",100.0);
	float zpos0 = inputParams("Particles/zpos0",100.0);
	float rad0 = inputParams("Particles/rad0",10.0);
	lbm.set_particle_sphere(0,xpos0,ypos0,zpos0,rad0);
		
	float xpos1 = inputParams("Particles/xpos1",200.0);
	float ypos1 = inputParams("Particles/ypos1",100.0);
	float zpos1 = inputParams("Particles/zpos1",100.0);
	float rad1 = inputParams("Particles/rad1",10.0);
	lbm.set_particle_sphere(1,xpos1,ypos1,zpos1,rad1);
				
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
				lbm.setS(ndx,0);
				for (int p=0; p<nParts; p++) {
					float dx = float(i) - lbm.getPrx(p);
					float dy = float(j) - lbm.getPry(p);
					float dz = float(k) - lbm.getPrz(p);
					float rr = sqrt(dx*dx + dy*dy + dz*dz);
					if (rr <= lbm.getPrad(p)) lbm.setS(ndx,1); 
				}			
			}
		}
	}	
		
	// initialize density fields: 
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {		
				int ndx = k*Nx*Ny + j*Nx + i;	
				float sij = float(lbm.getS(ndx));
				if (i > 46 && i < 54 && j > 32 && j < 68 && k > 32 && k < 68) {			
					lbm.setRA(ndx,1.00*(1.0-sij));
					lbm.setRB(ndx,0.02*(1.0-sij));				
				}
				else {
					lbm.setRA(ndx,0.02*(1.0-sij));
					lbm.setRB(ndx,1.00*(1.0-sij));
				}
				/*								
				if (i > 96 && i < 104 && j > 22 && j < 78 && k > 22 && k < 78) {			
					lbm.setRA(ndx,1.00*(1.0-sij));
					lbm.setRB(ndx,0.02*(1.0-sij));				
				}
				else {
					lbm.setRA(ndx,0.02*(1.0-sij));
					lbm.setRB(ndx,1.00*(1.0-sij));
				}
				*/								
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
	// calculate initial density sums:  
	// ----------------------------------------------
	
	lbm.calculate_initial_density_sums();
	
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

void mcmp_3D_capbridge_shear_bb::cycleForward(int stepsPerCycle, int currentCycle)
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
	outfile.open ("force.txt", ios::out | ios::app);
	
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
				
		lbm.map_particles_on_lattice_bb(nBlocks,nThreads);
		cudaDeviceSynchronize();
		lbm.cover_uncover_bb(nBlocks,nThreads);
		cudaDeviceSynchronize();
		lbm.compute_density_bb(nBlocks,nThreads);
		cudaDeviceSynchronize();
		lbm.compute_virtual_density_bb(nBlocks,nThreads);
		cudaDeviceSynchronize();
		lbm.sum_fluid_densities_bb(nBlocks,nThreads);
		lbm.correct_density_totals_bb(nBlocks,nThreads);
				
		// ------------------------------
		// update fluid fields:											   
		// ------------------------------ 
				
		lbm.compute_SC_forces_bb(nBlocks,nThreads);
		lbm.compute_velocity_bb(nBlocks,nThreads);
		lbm.set_boundary_shear_velocity_bb(-shearVel,shearVel,nBlocks,nThreads);
		lbm.collide_stream_bb(nBlocks,nThreads);
		lbm.bounce_back_moving(nBlocks,nThreads);
		lbm.swap_populations();	
				
		// ------------------------------
		// update particles:											   
		// ------------------------------ 
		
		lbm.particle_particle_forces_bb(Khertz,halo,nBlocks,nThreads);
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

void mcmp_3D_capbridge_shear_bb::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// function location:
	// "io/write_vtk_output.cuh"
	// ----------------------------------------------
	
	lbm.write_output(tagname,step,iskip,jskip,kskip); 
	float rATotal = lbm.calculate_fluid_A_volume();
	cout << rATotal << endl;

}
