
# include "struct_membrane_ibm3D.cuh"
# include "../../IO/GetPot"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

struct_membrane_ibm3D::struct_membrane_ibm3D()
{
	GetPot inputParams("input.dat");	
	nNodes = inputParams("IBM/nNodes",0);
	nFaces = inputParams("IBM/nFaces",0);	
	nEdges = inputParams("IBM/nEdges",0);
	nCells = inputParams("IBM/nCells",0);
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

struct_membrane_ibm3D::~struct_membrane_ibm3D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void struct_membrane_ibm3D::allocate()
{
	// allocate array memory (host):
	rH = (float3*)malloc(nNodes*sizeof(float3));		
	facesH = (triangle*)malloc(nFaces*sizeof(triangle));
	edgesH = (edge*)malloc(nEdges*sizeof(edge));
	cellsH = (cell*)malloc(nCells*sizeof(cell));
				
	// allocate array memory (device):
	cudaMalloc((void **) &r, nNodes*sizeof(float3));	
	cudaMalloc((void **) &v, nNodes*sizeof(float3));	
	cudaMalloc((void **) &f, nNodes*sizeof(float3));
	cudaMalloc((void **) &faces, nFaces*sizeof(triangle));
	cudaMalloc((void **) &edges, nEdges*sizeof(edge));
	cudaMalloc((void **) &cells, nCells*sizeof(cell));	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void struct_membrane_ibm3D::deallocate()
{
	// free array memory (host):
	free(rH);	
	free(facesH);
	free(edgesH);
	free(cellsH);	
			
	// free array memory (device):
	cudaFree(r);	
	cudaFree(v);	
	cudaFree(f);
	cudaFree(faces);
	cudaFree(edges);
	cudaFree(cells);		
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void struct_membrane_ibm3D::memcopy_host_to_device()
{
	cudaMemcpy(r, rH, sizeof(float3)*nNodes, cudaMemcpyHostToDevice);	
	cudaMemcpy(faces, facesH, sizeof(triangle)*nFaces, cudaMemcpyHostToDevice);	
	cudaMemcpy(edges, edgesH, sizeof(edge)*nEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(cells, cellsH, sizeof(cell)*nCells, cudaMemcpyHostToDevice);
}
	


// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void struct_membrane_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(rH, r, sizeof(float3)*nNodes, cudaMemcpyDeviceToHost);	
}



// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void struct_membrane_ibm3D::read_ibm_information(std::string tagname)
{
	read_ibm_information_long(tagname,nNodes,nFaces,nEdges,rH,facesH,edgesH);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void struct_membrane_ibm3D::shift_node_positions(float xsh, float ysh, float zsh)
{
	for (int i=0; i<nNodes; i++) {
		rH[i].x += xsh;
		rH[i].y += ysh;
		rH[i].z += zsh;
	}
}



// --------------------------------------------------------
// Write IBM output to file:
// --------------------------------------------------------

void struct_membrane_ibm3D::write_output(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_3D(tagname,tagnum,
	nNodes,nFaces,rH,facesH);
}



// --------------------------------------------------------
// Call to "update_node_position_IBM3D" kernel:
// --------------------------------------------------------

void struct_membrane_ibm3D::update_node_positions(int nBlocks, int nThreads)
{
	update_node_position_IBM3D
	<<<nBlocks,nThreads>>> (r,v,nNodes);	
}






