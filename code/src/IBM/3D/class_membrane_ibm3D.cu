
# include "class_membrane_ibm3D.cuh"
# include "../../IO/GetPot"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_membrane_ibm3D::class_membrane_ibm3D()
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

class_membrane_ibm3D::~class_membrane_ibm3D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_membrane_ibm3D::allocate()
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

void class_membrane_ibm3D::deallocate()
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

void class_membrane_ibm3D::memcopy_host_to_device()
{
	cudaMemcpy(r, rH, sizeof(float3)*nNodes, cudaMemcpyHostToDevice);	
	cudaMemcpy(faces, facesH, sizeof(triangle)*nFaces, cudaMemcpyHostToDevice);	
	cudaMemcpy(edges, edgesH, sizeof(edge)*nEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(cells, cellsH, sizeof(cell)*nCells, cudaMemcpyHostToDevice);
}
	


// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_membrane_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(rH, r, sizeof(float3)*nNodes, cudaMemcpyDeviceToHost);	
}



// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void class_membrane_ibm3D::read_ibm_information(std::string tagname)
{
	read_ibm_information_long(tagname,nNodes,nFaces,nEdges,rH,facesH,edgesH);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_membrane_ibm3D::shift_node_positions(float xsh, float ysh, float zsh)
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

void class_membrane_ibm3D::write_output(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_3D(tagname,tagnum,
	nNodes,nFaces,rH,facesH);
}



// --------------------------------------------------------
// Call to "update_node_position_IBM3D" kernel:
// --------------------------------------------------------

void class_membrane_ibm3D::update_node_positions(int nBlocks, int nThreads)
{
	update_node_position_IBM3D
	<<<nBlocks,nThreads>>> (r,v,nNodes);	
}



// --------------------------------------------------------
// Call to "interpolate_velocity_IBM3D" kernel:
// --------------------------------------------------------

void class_membrane_ibm3D::interpolate_velocity(float* uLBM, float* vLBM, 
	float* wLBM, int Nx, int Ny, int nBlocks, int nThreads)
{
	interpolate_velocity_IBM3D
	<<<nBlocks,nThreads>>> (r,v,uLBM,vLBM,wLBM,Nx,Ny,nNodes);	
}



// --------------------------------------------------------
// Call to "extrapolate_force_IBM3D" kernel:
// --------------------------------------------------------

void class_membrane_ibm3D::extrapolate_force(float* fxLBM, float* fyLBM, 
	float* fzLBM, int Nx, int Ny, int nBlocks, int nThreads)
{
	extrapolate_force_IBM3D
	<<<nBlocks,nThreads>>> (r,v,fxLBM,fyLBM,fzLBM,Nx,Ny,nNodes);	
}



// --------------------------------------------------------
// Calls to kernels that compute forces on nodes based 
// on the membrane mechanics model:
// --------------------------------------------------------

void class_membrane_ibm3D::compute_node_forces(int nBlocks, int nThreads)
{
	// First, zero the nodes forces and the cell volumes:
	zero_node_forces_IBM3D
	<<<nBlocks,nThreads>>> (f,nNodes);
	
	zero_cell_volumes_IBM3D
	<<<nBlocks,nThreads>>> (cells,nCells);
	
	// Second, compute the area dilation force for each face:
	compute_node_force_membrane_area_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,cells,ka,nFaces);
	
	// Third, compute the edge extension and bending force for each edge:
	compute_node_force_membrane_edge_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,edges,ks,kb,nEdges);
	
	// Last, compute the volume expansion force for each face:
	compute_node_force_membrane_volume_IBM3D
	<<<nBlocks,nThreads>>> (faces,f,cells,kv,nFaces);
}








