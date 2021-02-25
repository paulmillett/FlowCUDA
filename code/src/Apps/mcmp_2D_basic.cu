
# include "mcmp_2D_basic.cuh"
# include "../D2Q9/mcmp_SC/kernels_mcmp_SC_D2Q9.cuh"
# include "../D2Q9/init/stream_index_builder_D2Q9.cuh"
# include "../D2Q9/init/lattice_builders_D2Q9.cuh"
# include "../IO/GetPot"
# include "../IO/write_vtk_output.cuh"
# include <math.h>
# include <string> 
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

mcmp_2D_basic::mcmp_2D_basic() 
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
	potType = inputParams("LBM/potType",1);
	
	// ----------------------------------------------
	// output parameters:
	// ----------------------------------------------
	
	vtkFormat = inputParams("Output/format","structured");
	
	// ----------------------------------------------
	// allocate array memory (host):
	// ----------------------------------------------
	
    uH = (float*)malloc(nVoxels*sizeof(float));
    vH = (float*)malloc(nVoxels*sizeof(float));
    rAH = (float*)malloc(nVoxels*sizeof(float));
	rBH = (float*)malloc(nVoxels*sizeof(float));
	prH = (float*)malloc(nVoxels*sizeof(float));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));	
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));
	
	// ----------------------------------------------
	// allocate array memory (device):
	// ----------------------------------------------
	
	cudaMalloc((void **) &u, nVoxels*sizeof(float));
	cudaMalloc((void **) &v, nVoxels*sizeof(float));
	cudaMalloc((void **) &rA, nVoxels*sizeof(float));
	cudaMalloc((void **) &rB, nVoxels*sizeof(float));
	cudaMalloc((void **) &pr, nVoxels*sizeof(float));
	cudaMalloc((void **) &f1A, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f1B, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2A, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2B, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &FxA, nVoxels*sizeof(float));
	cudaMalloc((void **) &FxB, nVoxels*sizeof(float));
	cudaMalloc((void **) &FyA, nVoxels*sizeof(float));
	cudaMalloc((void **) &FyB, nVoxels*sizeof(float));	
	cudaMalloc((void **) &nList, nVoxels*Q*sizeof(int));
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

mcmp_2D_basic::~mcmp_2D_basic()
{
	
	// ----------------------------------------------
	// free array memory (host):
	// ----------------------------------------------
	
	free(uH);
	free(vH);
	free(rAH);
	free(rBH);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);
	
	// ----------------------------------------------
	// free array memory (device):
	// ----------------------------------------------
	
	cudaFree(u);
	cudaFree(v);
	cudaFree(rA);
	cudaFree(rB);
	cudaFree(f1A);
	cudaFree(f2A);
	cudaFree(f1B);
	cudaFree(f2B);
	cudaFree(FxA);
	cudaFree(FxB);
	cudaFree(FyA);
	cudaFree(FyB);
	cudaFree(nList);
	cudaFree(voxelType);
	cudaFree(streamIndex);
	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void mcmp_2D_basic::initSystem()
{
	
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
	
	// ----------------------------------------------
	// create the lattice using "box" function.
	// function location:
	// "lattice/lattice_builders_D2Q9.cuh"	 
	// ----------------------------------------------	
	
	if (latticeSource == "box") {
		Nx = inputParams("Lattice/Nx",0);
		Ny = inputParams("Lattice/Ny",0);
		Nz = inputParams("Lattice/Nz",0);
		build_box_lattice_D2Q9(nVoxels,Nx,Ny,voxelTypeH,nListH);
	}	
		
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// function location:
	// "D2Q9/stream_index_builder_D2Q9.cuh"
	// ----------------------------------------------
		
	stream_index_push_D2Q9(nVoxels,nListH,streamIndexH);
	
	// ----------------------------------------------			
	// initialize macros: 
	// ----------------------------------------------
	
	std::string initType = inputParams("LBM/initType","random");
	
	if (initType == "random") {
		srand(time(NULL));   // set the random seed
		for (int i=0; i<nVoxels; i++) {
			uH[i] = 0.0;
			vH[i] = 0.0;
			float ranA = (float)rand()/RAND_MAX;
			float ranB = (float)rand()/RAND_MAX;
			rAH[i] = 0.5 + 0.1*(ranA-0.5);
			rBH[i] = 0.5 + 0.1*(ranB-0.5);
		}
	}
	
	if (initType == "droplet") {
		float rInner = inputParams("LBM/rInner",10.0);	
		float rOuter = inputParams("LBM/rOuter",15.0);	
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = j*Nx + i;
				rAH[ndx] = 0.0;
				float dx = float(i) - float(Nx/2);
				float dy = float(j) - float(Ny/2);		
				float r2 = dx*dx + dy*dy;
				float r = sqrt(r2);
				if (r <= rOuter) {
					if (r < rInner) {
						rAH[ndx] = 1.0;
					}
					else {
						float rsc = r - rInner;
						rAH[ndx] = 1.0*exp(-rsc*rsc/5.0);
					}
				}
				rBH[ndx] = 1.0 - rAH[ndx];				
			}
		}
	}	
	
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	writeOutput("macros",0);
	
	// ----------------------------------------------	
	// copy arrays from host to device: 
	// ----------------------------------------------
	
    cudaMemcpy(u, uH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(v, vH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(rA, rAH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(rB, rBH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(nList, nListH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	mcmp_initial_equilibrium_D2Q9 
	<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,u,v,nVoxels);	
		
}



// --------------------------------------------------------
// Step forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void mcmp_2D_basic::cycleForward(int stepsPerCycle, int currentCycle)
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
		
		mcmp_compute_density_D2Q9 
		<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,nVoxels);
				
		cudaDeviceSynchronize();
		
		if (potType == 1) {
			mcmp_compute_SC_forces_1_D2Q9 
			<<<nBlocks,nThreads>>> (rA,rB,FxA,FxB,FyA,FyB,nList,gAB,nVoxels);	
		}
		else if (potType == 2) {
			mcmp_compute_SC_forces_2_D2Q9 
			<<<nBlocks,nThreads>>> (rA,rB,FxA,FxB,FyA,FyB,nList,gAB,nVoxels);
		}													  
		
		mcmp_compute_velocity_D2Q9 
		<<<nBlocks,nThreads>>> (f1A,f1B,rA,rB,FxA,FxB,FyA,FyB,u,v,nVoxels);
		
		mcmp_collide_stream_D2Q9 
		<<<nBlocks,nThreads>>> (f1A,f1B,f2A,f2B,rA,rB,u,v,FxA,FxB,FyA,FyB,streamIndex,nu,nVoxels);
														 
		float* tempA = f1A;
		float* tempB = f1B;
		f1A = f2A;
		f1B = f2B;
		f2A = tempA;
		f2B = tempB;
		
		cudaDeviceSynchronize();
		
	}
	
	// ----------------------------------------------
	// compute EOS pressure:
	// ----------------------------------------------
	
	mcmp_compute_SC_pressure_D2Q9 
	<<<nBlocks,nThreads>>> (rA,rB,pr,gAB,nVoxels);
	
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
    cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(rAH, rA, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(prH, pr, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	writeOutput("macros",cummulativeSteps);	
	
}



// --------------------------------------------------------
// Write output:
// --------------------------------------------------------

void mcmp_2D_basic::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// function location:
	// "io/write_vtk_output.cuh"
	// ----------------------------------------------
	
	if (vtkFormat == "structured") {
		write_vtk_structured_grid_2D(tagname,step,Nx,Ny,Nz,prH,uH,vH);
		//write_vtk_structured_grid_2D(tagname,step,Nx,Ny,Nz,rAH,rBH,uH,vH);
	}
	
}










