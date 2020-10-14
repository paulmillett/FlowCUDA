
# include "scsp_3D_iolets.cuh"
# include "../D3Q19/scsp/scsp_initial_equilibrium_D3Q19.cuh"
# include "../D3Q19/scsp/scsp_stream_collide_save_D3Q19.cuh"
# include "../D3Q19/init/stream_index_builder_D3Q19.cuh"
# include "../IO/GetPot"
# include "../IO/read_lattice_geometry.cuh"
# include "../IO/write_vtk_output.cuh"
# include "../Lattice/lattice_builders_D3Q19.cuh"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_iolets::scsp_3D_iolets()
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
	// iolets parameters:
	// ----------------------------------------------
	
	numIolets = inputParams("Lattice/numIolets",2);
	
	// ----------------------------------------------
	// output parameters:
	// ----------------------------------------------
	
	vtkFormat = inputParams("Output/format","polydata");
	
	// ----------------------------------------------
	// allocate array memory (host):
	// ----------------------------------------------
	
    uH = (float*)malloc(nVoxels*sizeof(float));
    vH = (float*)malloc(nVoxels*sizeof(float));
	wH = (float*)malloc(nVoxels*sizeof(float));
    rH = (float*)malloc(nVoxels*sizeof(float));
	pH = (float*)malloc(nVoxels*sizeof(float));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));
	xH = (int*)malloc(nVoxels*sizeof(int));
	yH = (int*)malloc(nVoxels*sizeof(int));
	zH = (int*)malloc(nVoxels*sizeof(int));
	ioletsH = (iolet*)malloc(numIolets*sizeof(iolet));
	
	// ----------------------------------------------
	// allocate array memory (device):
	// ----------------------------------------------
	
	cudaMalloc((void **) &u, nVoxels*sizeof(float));
	cudaMalloc((void **) &v, nVoxels*sizeof(float));
	cudaMalloc((void **) &w, nVoxels*sizeof(float));
	cudaMalloc((void **) &r, nVoxels*sizeof(float));
	cudaMalloc((void **) &p, nVoxels*sizeof(float));
	cudaMalloc((void **) &f1, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &iolets, numIolets*sizeof(iolet));
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_iolets::~scsp_3D_iolets()
{
	
	// ----------------------------------------------
	// free array memory (host):
	// ----------------------------------------------
	
	free(uH);
	free(vH);
	free(wH);
	free(rH);
	free(pH);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);
	free(xH);
	free(yH);
	free(zH);
	free(ioletsH);
		
	// ----------------------------------------------
	// free array memory (device):
	// ----------------------------------------------
	
	cudaFree(u);
	cudaFree(v);
	cudaFree(w);
	cudaFree(r);
	cudaFree(p);
	cudaFree(f1);
	cudaFree(f2);
	cudaFree(voxelType);
	cudaFree(streamIndex);
	cudaFree(iolets);
	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_iolets::initSystem()
{
	
	// ----------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------
	
	GetPot inputParams("input.dat");
	string latticeSource = inputParams("Lattice/source","box");	
	
	// ----------------------------------------------
	// create the lattice using "box" function.
	// function location:
	// "lattice/lattice_builders_D3Q19.cuh"	 
	// ----------------------------------------------	
	
	if (latticeSource == "box") {
		Nx = inputParams("Lattice/Nx",0);
		Ny = inputParams("Lattice/Ny",0);
		Nz = inputParams("Lattice/Nz",0);
		int flowDir = inputParams("Lattice/flowDir",0);
		int xLBC = inputParams("Lattice/xLBC",0);
		int xUBC = inputParams("Lattice/xUBC",0);
		int yLBC = inputParams("Lattice/yLBC",0);
		int yUBC = inputParams("Lattice/yUBC",0);
		int zLBC = inputParams("Lattice/zLBC",0);
		int zUBC = inputParams("Lattice/zUBC",0);		
		build_box_lattice_D3Q19(nVoxels,flowDir,Nx,Ny,Nz,
		                        xLBC,xUBC,yLBC,yUBC,zLBC,zUBC,
								voxelTypeH,nListH);
	}	
	
	// ----------------------------------------------
	// create the lattice by reading from file.
	// function location:
	// "io/read_lattice_geometry.cuh":	 
	// ----------------------------------------------	
	
	if (latticeSource == "file") {
		read_lattice_geometry_D3Q19(nVoxels,xH,yH,zH,voxelTypeH,nListH);
	}		
	
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// function location:
	// "D3Q19/stream_index_builder_D3Q19.cuh"
	// ----------------------------------------------
		
	stream_index_pull_D3Q19(nVoxels,nListH,streamIndexH);
	
	// ----------------------------------------------			
	// initialize inlets/outlets: 
	// ----------------------------------------------
	
	// I'm assuming there are 2 iolets!!!!
	ioletsH[0].type = inputParams("Iolet1/type",1);
	ioletsH[0].uBC = inputParams("Iolet1/uBC",0.0);
	ioletsH[0].vBC = inputParams("Iolet1/vBC",0.0);
	ioletsH[0].wBC = inputParams("Iolet1/wBC",0.0);
	ioletsH[0].rBC = inputParams("Iolet1/rBC",1.0);
	ioletsH[0].pBC = inputParams("Iolet1/pBC",0.0);
	
	ioletsH[1].type = inputParams("Iolet2/type",1);
	ioletsH[1].uBC = inputParams("Iolet2/uBC",0.0);
	ioletsH[1].vBC = inputParams("Iolet2/vBC",0.0);
	ioletsH[1].wBC = inputParams("Iolet2/wBC",0.0);
	ioletsH[1].rBC = inputParams("Iolet2/rBC",1.0);
	ioletsH[1].pBC = inputParams("Iolet2/pBC",0.0);	
		
	// ----------------------------------------------			
	// initialize macros: 
	// ----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		uH[i] = 0.00;
		vH[i] = 0.00;
		wH[i] = 0.00;
		rH[i] = 1.0;
	}
	/*
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				//float rjk = float(j-19)*float(j-19) + float(k-19)*float(k-19);
				uH[ndx] = 0.00; //0.1*(1.0 - (rjk/324.0)*(rjk/324.0)); // 0.1;
				vH[ndx] = 0.00;
				wH[ndx] = 0.00;
				rH[ndx] = 1.0;
			}
		}
	}
	*/
	
	// ----------------------------------------------
	// write initial output file:
	// ----------------------------------------------
	
	writeOutput("macros",0);
	
	// ----------------------------------------------		
	// copy arrays from host to device: 
	// ----------------------------------------------
	
    cudaMemcpy(u, uH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(v, vH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(w, wH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(r, rH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(iolets, ioletsH, sizeof(iolet)*numIolets, cudaMemcpyHostToDevice);
	
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	scsp_initial_equilibrium_D3Q19 
	<<<nBlocks,nThreads>>> (f1,r,u,v,w,nVoxels);	

}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_iolets::cycleForward(int stepsPerCycle, int currentCycle)
{
	
	// ----------------------------------------------
	// determine the cummulative number of steps at the
	// beginning of this cycle:
	// ----------------------------------------------
	
	int cummulativeSteps = stepsPerCycle*currentCycle;
	bool save = false;
	
	// ----------------------------------------------
	// loop through this cycle:
	// ----------------------------------------------
	
	for (int step=0; step<stepsPerCycle; step++) {
		cummulativeSteps++;
		if (step == (stepsPerCycle-1)) save = true;
		
		scsp_stream_collide_save_D3Q19 
		<<<nBlocks,nThreads>>> (f1,f2,r,u,v,w,streamIndex,voxelType,iolets,nu,nVoxels,save);
														 
		float* temp = f1;
		f1 = f2;
		f2 = temp;
		cudaDeviceSynchronize();
	}
	
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
    cudaMemcpy(rH, r, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(wH, w, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	writeOutput("macros",cummulativeSteps);
	
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_iolets::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// function location:
	// "io/write_vtk_output.cuh"
	// ----------------------------------------------
	
	if (vtkFormat == "structured") {
		write_vtk_structured_grid(tagname,step,Nx,Ny,Nz,rH,uH,vH,wH);
	}
	
	else if (vtkFormat == "polydata") {
		write_vtk_polydata(tagname,step,nVoxels,xH,yH,zH,rH,uH,vH,wH);
	}
	
}







