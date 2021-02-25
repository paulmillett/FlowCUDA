
# include "class_ibm3D.cuh"
# include "../../IO/GetPot"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_ibm3D::class_ibm3D()
{
	GetPot inputParams("input.dat");	
	nNodes = inputParams("IBM/nNodes",0);
	nFaces = inputParams("IBM/nFaces",0);	
	facesFlag = false;	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_ibm3D::~class_ibm3D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_ibm3D::allocate()
{
	// allocate array memory (host):
	rH = (float3*)malloc(nNodes*sizeof(float3));		
	rH_start = (float3*)malloc(nNodes*sizeof(float3));	
	rH_end = (float3*)malloc(nNodes*sizeof(float3));		    
			
	// allocate array memory (device):
	cudaMalloc((void **) &r, nNodes*sizeof(float3));	
	cudaMalloc((void **) &r_start, nNodes*sizeof(float3));	
	cudaMalloc((void **) &r_end, nNodes*sizeof(float3));	
	cudaMalloc((void **) &v, nNodes*sizeof(float3));	
	cudaMalloc((void **) &f, nNodes*sizeof(float3));	
}



// --------------------------------------------------------
// Allocate face vertice arrays:
// --------------------------------------------------------

void class_ibm3D::allocate_faces()
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

void class_ibm3D::deallocate()
{
	// free array memory (host):
	free(rH);	
	free(rH_start);	
	free(rH_end);	
	if (facesFlag) {
		free(faceV1);
		free(faceV2);
		free(faceV3);
	}
			
	// free array memory (device):
	cudaFree(r);	
	cudaFree(r_start);	
	cudaFree(r_end);	
	cudaFree(v);	
	cudaFree(f);	
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_ibm3D::memcopy_host_to_device()
{
	cudaMemcpy(r, rH, sizeof(float3)*nNodes, cudaMemcpyHostToDevice);	
	cudaMemcpy(r_start, rH_start, sizeof(float3)*nNodes, cudaMemcpyHostToDevice);	
	cudaMemcpy(r_end, rH_end, sizeof(float3)*nNodes, cudaMemcpyHostToDevice);
}
	


// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(rH, r, sizeof(float3)*nNodes, cudaMemcpyDeviceToHost);	
}



// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void class_ibm3D::read_ibm_start_positions(std::string tagname)
{
	read_ibm_information(tagname,nNodes,nFaces,rH_start,faceV1,faceV2,faceV3);
}



// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void class_ibm3D::read_ibm_end_positions(std::string tagname)
{
	read_ibm_information(tagname,nNodes,nFaces,rH_end,faceV1,faceV2,faceV3);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_ibm3D::shift_start_positions(float xsh, float ysh, float zsh)
{
	for (int i=0; i<nNodes; i++) {
		rH_start[i].x += xsh;
		rH_start[i].y += ysh;
		rH_start[i].z += zsh;
	}
}



// --------------------------------------------------------
// Shift IBM end positions by specified amount:
// --------------------------------------------------------

void class_ibm3D::shift_end_positions(float xsh, float ysh, float zsh)
{
	for (int i=0; i<nNodes; i++) {
		rH_end[i].x += xsh;
		rH_end[i].y += ysh;
		rH_end[i].z += zsh;
	}
}



// --------------------------------------------------------
// Assign IBM positions to the start positions:
// --------------------------------------------------------

void class_ibm3D::initialize_positions_to_start()
{
	for (int i=0; i<nNodes; i++) {
		rH[i] = rH_start[i];
	}
}



// --------------------------------------------------------
// Write IBM output to file:
// --------------------------------------------------------

void class_ibm3D::write_output(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_3D(tagname,tagnum,
	nNodes,nFaces,rH,faceV1,faceV2,faceV3);
}



// --------------------------------------------------------
// Call to "update_node_position_IBM3D" kernel:
// --------------------------------------------------------

void class_ibm3D::update_node_positions(int nBlocks, int nThreads,
                                         int currentStep, int nSteps)
{
	update_node_position_IBM3D
	<<<nBlocks,nThreads>>> (r,r_start,r_end,v,currentStep,nSteps,nNodes);	
}






