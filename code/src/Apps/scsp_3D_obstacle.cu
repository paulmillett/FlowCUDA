
# include "scsp_3D_obstacle.cuh"
# include "../D3Q19/scsp/scsp_initial_equilibrium_D3Q19.cuh"
# include "../D3Q19/scsp/scsp_stream_collide_save_forcing_D3Q19.cuh"
# include "../D3Q19/scsp/scsp_zero_forces_D3Q19.cuh"
# include "../D3Q19/init/stream_index_builder_D3Q19.cuh"
# include "../IBM/3D/compute_node_force_IBM3D.cuh"
# include "../IBM/3D/extrapolate_force_IBM3D.cuh"
# include "../IBM/3D/interpolate_velocity_IBM3D.cuh"
# include "../IBM/3D/set_reference_node_positions_IBM3D.cuh"
# include "../IBM/3D/update_node_position_IBM3D.cuh"
# include "../IBM/3D/update_node_ref_position_IBM3D.cuh"
# include "../IO/GetPot"
# include "../IO/read_lattice_geometry.cuh"
# include "../IO/write_vtk_output.cuh"
# include "../IO/read_ibm_information.cuh"
# include "../Lattice/lattice_builders_D3Q19.cuh"
# include <string>
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

scsp_3D_obstacle::scsp_3D_obstacle()
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
	nFaces = inputParams("IBM/nFaces",0);
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
	iskip = inputParams("Output/iskip",1);
	jskip = inputParams("Output/jskip",1);
	kskip = inputParams("Output/kskip",1);
	
	// ----------------------------------------------
	// allocate array memory (host):
	// ----------------------------------------------
	
    uH = (float*)malloc(nVoxels*sizeof(float));
    vH = (float*)malloc(nVoxels*sizeof(float));
	wH = (float*)malloc(nVoxels*sizeof(float));
	rH = (float*)malloc(nVoxels*sizeof(float));
	xIBH = (float*)malloc(nNodes*sizeof(float));
	yIBH = (float*)malloc(nNodes*sizeof(float));
	zIBH = (float*)malloc(nNodes*sizeof(float));	
	faceV1 = (int*)malloc(nFaces*sizeof(int));
	faceV2 = (int*)malloc(nFaces*sizeof(int));
	faceV3 = (int*)malloc(nFaces*sizeof(int));	
	xIBH_start = (float*)malloc(nNodes*sizeof(float));
	yIBH_start = (float*)malloc(nNodes*sizeof(float));
	zIBH_start = (float*)malloc(nNodes*sizeof(float));	
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
	cudaMalloc((void **) &f1, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &Fx, nVoxels*sizeof(float));
	cudaMalloc((void **) &Fy, nVoxels*sizeof(float));
	cudaMalloc((void **) &Fz, nVoxels*sizeof(float));
	cudaMalloc((void **) &xIB, nVoxels*sizeof(float));
	cudaMalloc((void **) &yIB, nVoxels*sizeof(float));
	cudaMalloc((void **) &zIB, nVoxels*sizeof(float));
	cudaMalloc((void **) &xIBref, nVoxels*sizeof(float));
	cudaMalloc((void **) &yIBref, nVoxels*sizeof(float));
	cudaMalloc((void **) &zIBref, nVoxels*sizeof(float));
	cudaMalloc((void **) &vxIB, nVoxels*sizeof(float));
	cudaMalloc((void **) &vyIB, nVoxels*sizeof(float));
	cudaMalloc((void **) &vzIB, nVoxels*sizeof(float));
	cudaMalloc((void **) &fxIB, nVoxels*sizeof(float));
	cudaMalloc((void **) &fyIB, nVoxels*sizeof(float));
	cudaMalloc((void **) &fzIB, nVoxels*sizeof(float));
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &iolets, numIolets*sizeof(iolet));
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

scsp_3D_obstacle::~scsp_3D_obstacle()
{
	
	// ----------------------------------------------
	// free array memory (host):
	// ----------------------------------------------
	
	free(uH);
	free(vH);
	free(wH);
	free(rH);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);
	free(xH);
	free(yH);
	free(zH);
	free(ioletsH);
	free(xIBH);
	free(yIBH);
	free(zIBH);
	free(xIBH_start);
	free(yIBH_start);
	free(zIBH_start);	
		
	// ----------------------------------------------
	// free array memory (device):
	// ----------------------------------------------
	
	cudaFree(u);
	cudaFree(v);
	cudaFree(w);
	cudaFree(r);
	cudaFree(f1);
	cudaFree(f2);
	cudaFree(Fx);
	cudaFree(Fy);
	cudaFree(Fz);
	cudaFree(voxelType);
	cudaFree(streamIndex);
	cudaFree(iolets);
	cudaFree(xIB);
	cudaFree(yIB);
	cudaFree(zIB);
	cudaFree(xIBref);
	cudaFree(yIBref);
	cudaFree(zIBref);	
	cudaFree(vxIB);
	cudaFree(vyIB);
	cudaFree(vzIB);
	cudaFree(fxIB);
	cudaFree(fyIB);
	cudaFree(fzIB);
	
}



// --------------------------------------------------------
// Initialize system:
// --------------------------------------------------------

void scsp_3D_obstacle::initSystem()
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
	// build the streamIndex[] array.  
	// function location:
	// "D3Q19/init/stream_index_builder_D3Q19.cuh"
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
	
	// ----------------------------------------------			
	// initialize immersed boundary info: 
	// ----------------------------------------------
	
	read_ibm_information("sphere.dat",nNodes,nFaces,xIBH_start,yIBH_start,zIBH_start,
	                     faceV1,faceV2,faceV3);
												 
	for (int i=0; i<nNodes; i++) {
		xIBH[i] = xIBH_start[i];
		yIBH[i] = yIBH_start[i];
		zIBH[i] = zIBH_start[i];		
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
	cudaMemcpy(w, wH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(r, rH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(xIB, xIBH, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(yIB, yIBH, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(zIB, zIBH, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(iolets, ioletsH, sizeof(iolet)*numIolets, cudaMemcpyHostToDevice);
	
	// ----------------------------------------------
	// initialize equilibrium populations: 
	// ----------------------------------------------
	
	scsp_initial_equilibrium_D3Q19 
	<<<nBlocks,nThreads>>> (f1,r,u,v,w,nVoxels);	
	
	// ----------------------------------------------
	// define reference IBM node positions: 
	// ----------------------------------------------
	
	set_reference_node_positions_IBM3D
	<<<nBlocksIB,nThreads>>> (xIB,yIB,zIB,xIBref,yIBref,zIBref,nNodes);
		
}



// --------------------------------------------------------
// Cycle forward
// (this function iterates the system by a certain 
//  number of time steps between print-outs):
// --------------------------------------------------------

void scsp_3D_obstacle::cycleForward(int stepsPerCycle, int currentCycle)
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
				
		scsp_zero_forces_D3Q19
		<<<nBlocks,nThreads>>> (Fx,Fy,Fz,nVoxels);
		
		compute_node_force_IBM3D
		<<<nBlocksIB,nThreads>>> (xIB,yIB,zIB,xIBref,yIBref,zIBref,fxIB,fyIB,fzIB,kstiff,nNodes);
		
		extrapolate_force_IBM3D
		<<<nBlocksIB,nThreads>>> (xIB,yIB,zIB,fxIB,fyIB,fzIB,Fx,Fy,Fz,Nx,Ny,nNodes);
				
		scsp_stream_collide_save_forcing_D3Q19 
		<<<nBlocks,nThreads>>> (f1,f2,r,u,v,w,Fx,Fy,Fz,streamIndex,voxelType,iolets,nu,nVoxels);
		float* temp = f1;
		f1 = f2;
		f2 = temp;
		
		interpolate_velocity_IBM3D
		<<<nBlocksIB,nThreads>>> (xIB,yIB,zIB,vxIB,vyIB,vzIB,u,v,w,Nx,Ny,nNodes);
		
		update_node_position_IBM3D
		<<<nBlocksIB,nThreads>>> (xIB,yIB,zIB,vxIB,vyIB,vzIB,nNodes);														 
				
		cudaDeviceSynchronize();
	}
	
	// ----------------------------------------------
	// copy arrays from device to host:
	// ----------------------------------------------
	
    cudaMemcpy(rH, r, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(uH, u, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(wH, w, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(xIBH, xIB, sizeof(float)*nNodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(yIBH, yIB, sizeof(float)*nNodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(zIBH, zIB, sizeof(float)*nNodes, cudaMemcpyDeviceToHost);
	
	// ----------------------------------------------
	// write output from this cycle:
	// ----------------------------------------------
	
	writeOutput("macros",cummulativeSteps);
		
}



// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------

void scsp_3D_obstacle::writeOutput(std::string tagname, int step)
{
	
	// ----------------------------------------------
	// decide which VTK file format to use for output
	// function location:
	// "io/write_vtk_output.cuh"
	// ----------------------------------------------
	
	write_vtk_structured_grid(tagname,step,Nx,Ny,Nz,rH,uH,vH,wH,iskip,jskip,kskip);
	write_vtk_immersed_boundary_3D("ibm",step,nNodes,nFaces,xIBH,yIBH,zIBH,faceV1,faceV2,faceV3);
		
}







