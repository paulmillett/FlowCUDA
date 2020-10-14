
# include "struct_ibm3D.cuh"
# include "struct_ibm3D_includes.cuh"
# include "../../IO/GetPot"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

struct_ibm3D::struct_ibm3D()
{
	GetPot inputParams("input.dat");	
	nNodes = inputParams("IBM/nNodes",0);
	nFaces = inputParams("IBM/nFaces",0);	
	facesFlag = false;	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

struct_ibm3D::~struct_ibm3D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void struct_ibm3D::allocate()
{
	// allocate array memory (host):
	xH = (float*)malloc(nNodes*sizeof(float));
	yH = (float*)malloc(nNodes*sizeof(float));
	zH = (float*)malloc(nNodes*sizeof(float));		
	xH_start = (float*)malloc(nNodes*sizeof(float));
	yH_start = (float*)malloc(nNodes*sizeof(float));
	zH_start = (float*)malloc(nNodes*sizeof(float));
	xH_end = (float*)malloc(nNodes*sizeof(float));
	yH_end = (float*)malloc(nNodes*sizeof(float));
	zH_end = (float*)malloc(nNodes*sizeof(float));	    
			
	// allocate array memory (device):
	cudaMalloc((void **) &x, nNodes*sizeof(float));
	cudaMalloc((void **) &y, nNodes*sizeof(float));
	cudaMalloc((void **) &z, nNodes*sizeof(float));
	cudaMalloc((void **) &x_start, nNodes*sizeof(float));
	cudaMalloc((void **) &y_start, nNodes*sizeof(float));
	cudaMalloc((void **) &z_start, nNodes*sizeof(float));
	cudaMalloc((void **) &x_end, nNodes*sizeof(float));
	cudaMalloc((void **) &y_end, nNodes*sizeof(float));
	cudaMalloc((void **) &z_end, nNodes*sizeof(float));
	cudaMalloc((void **) &vx, nNodes*sizeof(float));
	cudaMalloc((void **) &vy, nNodes*sizeof(float));
	cudaMalloc((void **) &vz, nNodes*sizeof(float));
	cudaMalloc((void **) &fx, nNodes*sizeof(float));
	cudaMalloc((void **) &fy, nNodes*sizeof(float));
	cudaMalloc((void **) &fz, nNodes*sizeof(float));	
}



// --------------------------------------------------------
// Allocate face vertice arrays:
// --------------------------------------------------------

void struct_ibm3D::allocate_faces()
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

void struct_ibm3D::deallocate()
{
	// free array memory (host):
	free(xH);
	free(yH);
	free(zH);
	free(xH_start);
	free(yH_start);
	free(zH_start);
	free(xH_end);
	free(yH_end);
	free(zH_end);
	if (facesFlag) {
		free(faceV1);
		free(faceV2);
		free(faceV3);
	}
			
	// free array memory (device):
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
	cudaFree(x_start);
	cudaFree(y_start);
	cudaFree(z_start);
	cudaFree(x_end);
	cudaFree(y_end);
	cudaFree(z_end);
	cudaFree(vx);
	cudaFree(vy);
	cudaFree(vz);
	cudaFree(fx);
	cudaFree(fy);
	cudaFree(fz);
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void struct_ibm3D::memcopy_host_to_device()
{
	cudaMemcpy(x, xH, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(y, yH, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(z, zH, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(x_start, xH_start, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(y_start, yH_start, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(z_start, zH_start, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(x_end, xH_end, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(y_end, yH_end, sizeof(float)*nNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(z_end, zH_end, sizeof(float)*nNodes, cudaMemcpyHostToDevice);	
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void struct_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(xH, x, sizeof(float)*nNodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(yH, y, sizeof(float)*nNodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(zH, z, sizeof(float)*nNodes, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void struct_ibm3D::read_ibm_start_positions(std::string tagname)
{
	read_ibm_information(tagname,nNodes,nFaces,xH_start,
	yH_start,zH_start,faceV1,faceV2,faceV3);
}



// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void struct_ibm3D::read_ibm_end_positions(std::string tagname)
{
	read_ibm_information(tagname,nNodes,nFaces,xH_end,
	yH_end,zH_end,faceV1,faceV2,faceV3);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void struct_ibm3D::shift_start_positions(float xsh, float ysh, float zsh)
{
	for (int i=0; i<nNodes; i++) {
		xH_start[i] += xsh;
		yH_start[i] += ysh;
		zH_start[i] += zsh;
	}
}



// --------------------------------------------------------
// Shift IBM end positions by specified amount:
// --------------------------------------------------------

void struct_ibm3D::shift_end_positions(float xsh, float ysh, float zsh)
{
	for (int i=0; i<nNodes; i++) {
		xH_end[i] += xsh;
		yH_end[i] += ysh;
		zH_end[i] += zsh;
	}
}



// --------------------------------------------------------
// Assign IBM positions to the start positions:
// --------------------------------------------------------

void struct_ibm3D::initialize_positions_to_start()
{
	for (int i=0; i<nNodes; i++) {
		xH[i] = xH_start[i];
		yH[i] = yH_start[i];
		zH[i] = zH_start[i];
	}
}



// --------------------------------------------------------
// Write IBM output to file:
// --------------------------------------------------------

void struct_ibm3D::write_output(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_3D(tagname,tagnum,
	nNodes,nFaces,xH,yH,zH,faceV1,faceV2,faceV3);
}



// --------------------------------------------------------
// Call to "update_node_position_IBM3D" kernel:
// --------------------------------------------------------

void struct_ibm3D::update_node_positions(int nBlocks, int nThreads,
                                         int currentStep, int nSteps)
{
	update_node_position_IBM3D
	<<<nBlocks,nThreads>>> (x,y,z,x_start,y_start,z_start,
	x_end,y_end,z_end,vx,vy,vz,currentStep,nSteps,nNodes);	
}






