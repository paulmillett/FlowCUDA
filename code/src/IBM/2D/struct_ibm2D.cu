
# include "struct_ibm2D.cuh"
# include "struct_ibm2D_includes.cuh"
# include "../../IO/GetPot"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

struct_ibm2D::struct_ibm2D()
{
	GetPot inputParams("input.dat");	
	nNodes = inputParams("IBM/nNodes",0);
	nFaces = inputParams("IBM/nFaces",0);	
	kstiff = inputParams("IBM/kstiff",0.0); 
	facesFlag = false;	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

struct_ibm2D::~struct_ibm2D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void struct_ibm2D::allocate()
{
	// allocate array memory (host):
	xH = (float*)malloc(nNodes*sizeof(float));
	yH = (float*)malloc(nNodes*sizeof(float));
	xH_start = (float*)malloc(nNodes*sizeof(float));
	yH_start = (float*)malloc(nNodes*sizeof(float));
	xH_end = (float*)malloc(nNodes*sizeof(float));
	yH_end = (float*)malloc(nNodes*sizeof(float));
			
	// allocate array memory (device):
	cudaMalloc((void **) &x, nNodes*sizeof(float));
	cudaMalloc((void **) &y, nNodes*sizeof(float));
	cudaMalloc((void **) &x_start, nNodes*sizeof(float));
	cudaMalloc((void **) &y_start, nNodes*sizeof(float));
	cudaMalloc((void **) &x_end, nNodes*sizeof(float));
	cudaMalloc((void **) &y_end, nNodes*sizeof(float));
	cudaMalloc((void **) &x_ref, nNodes*sizeof(float));
	cudaMalloc((void **) &y_ref, nNodes*sizeof(float));
	cudaMalloc((void **) &vx, nNodes*sizeof(float));
	cudaMalloc((void **) &vy, nNodes*sizeof(float));
	cudaMalloc((void **) &fx, nNodes*sizeof(float));
	cudaMalloc((void **) &fy, nNodes*sizeof(float));
}



// --------------------------------------------------------
// Allocate face vertice arrays:
// --------------------------------------------------------

void struct_ibm2D::allocate_faces()
{
	// allocate voxel position arrays (host):
	faceV1 = (int*)malloc(nFaces*sizeof(int));
	faceV2 = (int*)malloc(nFaces*sizeof(int));
	faceV3 = (int*)malloc(nFaces*sizeof(int));	
	facesFlag = true;	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void struct_ibm2D::deallocate()
{
	// free array memory (host):
	free(xH);
	free(yH);
	free(xH_start);
	free(yH_start);
	free(xH_end);
	free(yH_end);
	if (facesFlag) {
		free(faceV1);
		free(faceV2);
		free(faceV3);
	}
			
	// free array memory (device):
	cudaFree(x);
	cudaFree(y);
	cudaFree(x_start);
	cudaFree(y_start);
	cudaFree(x_end);
	cudaFree(y_end);
	cudaFree(x_ref);
	cudaFree(y_ref);
	cudaFree(vx);
	cudaFree(vy);
	cudaFree(fx);
	cudaFree(fy);
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void struct_ibm2D::memcopy_host_to_device()
{
	cudaMemcpy(x, xH, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(y, yH, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(x_start, xH_start, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(y_start, yH_start, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(x_end, xH_end, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(y_end, yH_end, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void struct_ibm2D::memcopy_device_to_host()
{
	cudaMemcpy(xH, x, sizeof(float)*nNodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(yH, y, sizeof(float)*nNodes, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Setters:
// --------------------------------------------------------

void struct_ibm2D::setXStart(int i, float val)
{
	xH_start[i] = val;
}

void struct_ibm2D::setYStart(int i, float val)
{
	yH_start[i] = val;
}

void struct_ibm2D::setXEnd(int i, float val)
{
	xH_end[i] = val;
}

void struct_ibm2D::setYEnd(int i, float val)
{
	yH_end[i] = val;
}

void struct_ibm2D::set_positions_to_start_positions()
{
	for (int i=0; i<nNodes; i++) { 
		xH[i] = xH_start[i];
		yH[i] = yH_start[i];		
	}
}



// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void struct_ibm2D::read_ibm_start_positions(std::string tagname)
{
	
}



// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void struct_ibm2D::read_ibm_end_positions(std::string tagname)
{
	
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void struct_ibm2D::shift_start_positions(float xsh, float ysh)
{
	for (int i=0; i<nNodes; i++) {
		xH_start[i] += xsh;
		yH_start[i] += ysh;
	}
}



// --------------------------------------------------------
// Shift IBM end positions by specified amount:
// --------------------------------------------------------

void struct_ibm2D::shift_end_positions(float xsh, float ysh)
{
	for (int i=0; i<nNodes; i++) {
		xH_end[i] += xsh;
		yH_end[i] += ysh;
	}
}



// --------------------------------------------------------
// Assign IBM positions to the start positions:
// --------------------------------------------------------

void struct_ibm2D::initialize_positions_to_start()
{
	for (int i=0; i<nNodes; i++) {
		xH[i] = xH_start[i];
		yH[i] = yH_start[i];
	}
}



// --------------------------------------------------------
// Write IBM output to file:
// --------------------------------------------------------

void struct_ibm2D::write_output(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_2D(tagname,tagnum,nNodes,xH,yH);
}



// --------------------------------------------------------
// Call to "set_reference_node_positions_IBM2D" kernel:
// --------------------------------------------------------

void struct_ibm2D::set_reference_node_positions(int nBlocks, int nThreads)
{
	set_reference_node_positions_IBM2D
	<<<nBlocks,nThreads>>> (x,y,x_ref,y_ref,nNodes);
}



// --------------------------------------------------------
// Call to "update_node_ref_position_IBM2D" kernel:
// --------------------------------------------------------

void struct_ibm2D::update_node_ref_position(int nBlocks, int nThreads)
{
	update_node_ref_position_IBM2D
	<<<nBlocks,nThreads>>> (x_ref,y_ref,nNodes);
}



// --------------------------------------------------------
// Call to "update_node_ref_position_IBM2D" kernel:
// --------------------------------------------------------

void struct_ibm2D::update_node_ref_position(int nBlocks, int nThreads,
                                            int currentStep, int nSteps)
{
	update_node_ref_position_IBM2D
	<<<nBlocks,nThreads>>> 
	(x_ref,y_ref,x_start,y_start,x_end,y_end,currentStep,nSteps,nNodes);
}



// --------------------------------------------------------
// Call to "compute_node_force_IBM2D" kernel:
// --------------------------------------------------------

void struct_ibm2D::compute_node_forces(int nBlocks, int nThreads)
{
	compute_node_force_IBM2D
	<<<nBlocks,nThreads>>> (x,y,x_ref,y_ref,fx,fy,kstiff,nNodes);
}



// --------------------------------------------------------
// Call to "update_node_position_IBM2D" kernel:
// --------------------------------------------------------

void struct_ibm2D::update_node_positions(int nBlocks, int nThreads)
{
	update_node_position_IBM2D
	<<<nBlocks,nThreads>>> (x,y,vx,vy,nNodes);	
}



// --------------------------------------------------------
// Call to "update_node_position_IBM2D" kernel:
// --------------------------------------------------------

void struct_ibm2D::update_node_positions(int nBlocks, int nThreads,
                                         int currentStep, int nSteps)
{
	update_node_position_IBM2D
	<<<nBlocks,nThreads>>> (x,y,x_start,y_start,
	x_end,y_end,vx,vy,currentStep,nSteps,nNodes);	
}






