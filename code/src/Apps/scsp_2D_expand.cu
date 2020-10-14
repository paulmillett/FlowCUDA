
# include "scsp_2D_expand.cuh"
# include "../D2Q9/scsp/scsp_initial_equilibrium_D2Q9.cuh"
# include "../D2Q9/scsp/scsp_stream_collide_save_forcing_D2Q9.cuh"
# include "../D2Q9/scsp/scsp_zero_forces_D2Q9.cuh"
# include "../D2Q9/init/stream_index_builder_D2Q9.cuh"
# include "../IBM/2D/compute_node_force_IBM2D.cuh"
# include "../IBM/2D/extrapolate_force_IBM2D.cuh"
# include "../IBM/2D/interpolate_velocity_IBM2D.cuh"
# include "../IBM/2D/set_reference_node_positions_IBM2D.cuh"
# include "../IBM/2D/update_node_position_IBM2D.cuh"
# include "../IBM/2D/update_node_ref_position_IBM2D.cuh"
# include "../IO/GetPot"
# include "../IO/read_lattice_geometry.cuh"
# include "../IO/write_vtk_output.cuh"
# include "../Lattice/lattice_builders_D2Q9.cuh"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_2D_expand::scsp_2D_expand()
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
	// time parameters:
	// ----------------------------------------------
	
	nSteps = inputParams("Time/nSteps",0);
	
	// ----------------------------------------------
	// Lattice Boltzmann parameters:
	// ----------------------------------------------
	
	nu = inputParams("LBM/nu",0.1666666);
	
	// ----------------------------------------------
	// Immersed-Boundary parameters:
	// ----------------------------------------------
	
	nNodes = inputParams("IBM/nNodes",0);
	kstiff = inputParams("IBM/kstiff",0.0);
	nBlocksIB = (nNodes+(nThreads-1))/nThreads; // integer division	
	
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
    rH = (float*)malloc(nVoxels*sizeof(float));
	xIBH = (float*)malloc(nNodes*sizeof(float));
	yIBH = (float*)malloc(nNodes*sizeof(float));
	xIBH_start = (float*)malloc(nNodes*sizeof(float));
	yIBH_start = (float*)malloc(nNodes*sizeof(float));
	xIBH_end = (float*)malloc(nNodes*sizeof(float));
	yIBH_end = (float*)malloc(nNodes*sizeof(float));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));
	xH = (int*)malloc(nVoxels*sizeof(int));
	yH = (int*)malloc(nVoxels*sizeof(int));
	ioletsH = (iolet2D*)malloc(numIolets*sizeof(iolet2D));
	
	// ----------------------------------------------
	// allocate array memory (device):
	// ----------------------------------------------
	
	cudaMalloc((void **) &u, nVoxels*sizeof(float));
	cudaMalloc((void **) &v, nVoxels*sizeof(float));
	cudaMalloc((void **) &r, nVoxels*sizeof(float));
	cudaMalloc((void **) &f1, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &Fx, nVoxels*sizeof(float));
	cudaMalloc((void **) &Fy, nVoxels*sizeof(float));
	cudaMalloc((void **) &xIB, nNodes*sizeof(float));
	cudaMalloc((void **) &yIB, nNodes*sizeof(float));
	cudaMalloc((void **) &xIBref, nNodes*sizeof(float));
	cudaMalloc((void **) &yIBref, nNodes*sizeof(float));
	cudaMalloc((void **) &xIBref_start, nNodes*sizeof(float));
	cudaMalloc((void **) &yIBref_start, nNodes*sizeof(float));
	cudaMalloc((void **) &xIBref_end, nNodes*sizeof(float));
	cudaMalloc((void **) &yIBref_end, nNodes*sizeof(float));
	cudaMalloc((void **) &vxIB, nNodes*sizeof(float));
	cudaMalloc((void **) &vyIB, nNodes*sizeof(float));
	cudaMalloc((void **) &fxIB, nNodes*sizeof(float));
	cudaMalloc((void **) &fyIB, nNodes*sizeof(float));
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &iolets, numIolets*sizeof(iolet2D));
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_2D_expand::~scsp_2D_expand()
{
	
	// ----------------------------------------------
	// free array memory (host):
	// ----------------------------------------------
	
	free(uH);
	free(vH);
	free(rH);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);
	free(xH);
	free(yH);
	free(ioletsH);
	free(xIBH);
	free(yIBH);
	free(xIBH_start);
	free(yIBH_start);
	free(xIBH_end);
	free(yIBH_end);
		
	// ----------------------------------------------
	// free array memory (device):
	// ----------------------------------------------
	
	cudaFree(u);
	cudaFree(v);
	cudaFree(r);
	cudaFree(f1);
	cudaFree(f2);
	cudaFree(Fx);
	cudaFree(Fy);
	cudaFree(voxelType);
	cudaFree(streamIndex);
	cudaFree(iolets);
	cudaFree(xIB);
	cudaFree(yIB);
	cudaFree(xIBref);
	cudaFree(yIBref);
	cudaFree(xIBref_start);
	cudaFree(yIBref_start);
	cudaFree(xIBref_end);
	cudaFree(yIBref_end);
	cudaFree(vxIB);
	cudaFree(vyIB);
	cudaFree(fxIB);
	cudaFree(fyIB);
	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_2D_expand::initSystem()
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
		Nz = 1;
		int flowDir = inputParams("Lattice/flowDir",0);
		int xLBC = inputParams("Lattice/xLBC",0);
		int xUBC = inputParams("Lattice/xUBC",0);
		int yLBC = inputParams("Lattice/yLBC",0);
		int yUBC = inputParams("Lattice/yUBC",0);			
		build_box_lattice_D2Q9(nVoxels,flowDir,Nx,Ny,
		                       xLBC,xUBC,yLBC,yUBC,
		                       voxelTypeH,nListH);
	}	
		
	// ----------------------------------------------		
	// build the streamIndex[] array.  
	// function location:
	// "D2Q9/init/stream_index_builder_D2Q9.cuh"
	// ----------------------------------------------
		
	stream_index_pull_D2Q9(nVoxels,nListH,streamIndexH);
	
	// ----------------------------------------------			
	// initialize inlets/outlets: 
	// ----------------------------------------------
	
	// I'm assuming there are 2 iolets!!!!
	ioletsH[0].type = inputParams("Iolet1/type",1);
	ioletsH[0].uBC = inputParams("Iolet1/uBC",0.0);
	ioletsH[0].vBC = inputParams("Iolet1/vBC",0.0);
	ioletsH[0].rBC = inputParams("Iolet1/rBC",1.0);
	ioletsH[0].pBC = inputParams("Iolet1/pBC",0.0);
	
	ioletsH[1].type = inputParams("Iolet2/type",1);
	ioletsH[1].uBC = inputParams("Iolet2/uBC",0.0);
	ioletsH[1].vBC = inputParams("Iolet2/vBC",0.0);
	ioletsH[1].rBC = inputParams("Iolet2/rBC",1.0);
	ioletsH[1].pBC = inputParams("Iolet2/pBC",0.0);	
		
	// ----------------------------------------------			
	// edit inlet condition: 
	// ----------------------------------------------
	
	for (int i=0; i<Nx; i++) {
		int j = Ny - 1;
		int ndx = j*Nx + i;
		if (i < 120 || i > 140) {
			voxelTypeH[ndx] = 0;
		} 
	}	
	
	// ----------------------------------------------			
	// initialize macros: 
	// ----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		uH[i] = 0.00;
		vH[i] = 0.00;
		rH[i] = 1.0;
	} 
	
	// ----------------------------------------------			
	// initialize immersed boundary info: 
	// ----------------------------------------------
			
	float xcent = 99.5;
	float ycent = 198.5;
	float radiusx = 50.0;
	float radiusy = 50.0;
	for (int i=0; i<nNodes; i++) { 
		xIBH_start[i] = xcent - radiusx*cos(1.0*M_PI*float(i)/(nNodes-1));
		yIBH_start[i] = ycent - radiusy*sin(1.0*M_PI*float(i)/(nNodes-1));
	}
	radiusx = 50.0;
	radiusy = 100.0;
	for (int i=0; i<nNodes; i++) { 
		xIBH_end[i] = xcent - radiusx*cos(1.0*M_PI*float(i)/(nNodes-1));
		yIBH_end[i] = ycent - radiusy*sin(1.0*M_PI*float(i)/(nNodes-1));		
	}
	
	for (int i=0; i<nNodes; i++) { 
		xIBH[i] = xIBH_start[i];
		yIBH[i] = yIBH_start[i];		
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
	cudaMemcpy(r, rH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(xIB, xIBH, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(yIB, yIBH, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(xIBref_start, xIBH_start, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(yIBref_start, yIBH_start, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(xIBref_end, xIBH_end, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(yIBref_end, yIBH_end, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(iolets, ioletsH, sizeof(iolet2D)*numIolets, cudaMemcpyHostToDevice);
	
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	scsp_initial_equilibrium_D2Q9 
	<<<nBlocks,nThreads>>> (f1,r,u,v,nVoxels);	
	
	// ----------------------------------------------
	// define reference IBM node positions: 
	// ----------------------------------------------
	
	set_reference_node_positions_IBM2D
	<<<nBlocksIB,nThreads>>> (xIB,yIB,xIBref,yIBref,nNodes);
	
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_2D_expand::cycleForward(int stepsPerCycle, int currentCycle)
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
		
		update_node_ref_position_IBM2D
		<<<nBlocksIB,nThreads>>> 
		(xIBref,yIBref,xIBref_start,yIBref_start,xIBref_end,yIBref_end,cummulativeSteps,nSteps,nNodes);
		
		scsp_zero_forces_D2Q9
		<<<nBlocks,nThreads>>> (Fx,Fy,nVoxels);
		
		compute_node_force_IBM2D
		<<<nBlocksIB,nThreads>>> (xIB,yIB,xIBref,yIBref,fxIB,fyIB,kstiff,nNodes);
		
		extrapolate_force_IBM2D
		<<<nBlocksIB,nThreads>>> (xIB,yIB,fxIB,fyIB,Fx,Fy,Nx,nNodes);
				
		scsp_stream_collide_save_forcing_D2Q9 
		<<<nBlocks,nThreads>>> (f1,f2,r,u,v,Fx,Fy,streamIndex,voxelType,iolets,nu,nVoxels);
		float* temp = f1;
		f1 = f2;
		f2 = temp;
		
		interpolate_velocity_IBM2D
		<<<nBlocksIB,nThreads>>> (xIB,yIB,vxIB,vyIB,u,v,Nx,nNodes);
		
		update_node_position_IBM2D
		<<<nBlocksIB,nThreads>>> (xIB,yIB,vxIB,vyIB,nNodes);														 
		
		cudaDeviceSynchronize();
	}
	
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
    cudaMemcpy(rH, r, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(xIBH, xIB, sizeof(float)*nNodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(yIBH, yIB, sizeof(float)*nNodes, cudaMemcpyDeviceToHost);
		
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	writeOutput("macros",cummulativeSteps);
	
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_2D_expand::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// function location:
	// "io/write_vtk_output.cuh"
	// ----------------------------------------------
	
	write_vtk_structured_grid_2D(tagname,step,Nx,Ny,Nz,rH,uH,vH);
	write_vtk_immersed_boundary_2D("ibm",step,nNodes,xIBH,yIBH);
		
}







