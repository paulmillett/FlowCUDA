
# include "class_membrane_ibm3D.cuh"
# include "../../IO/GetPot"
# include <math.h>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_membrane_ibm3D::class_membrane_ibm3D()
{
	// get some parameters:
	GetPot inputParams("input.dat");	
	nNodes = inputParams("IBM/nNodes",0);
	nFaces = inputParams("IBM/nFaces",0);	
	nEdges = inputParams("IBM/nEdges",0);
	nCells = inputParams("IBM/nCells",0);
	ks = inputParams("IBM/ks",0.0);
	kb = inputParams("IBM/kb",0.0);
	ka = inputParams("IBM/ka",0.0);
	kag = inputParams("IBM/kag",0.0);
	kv = inputParams("IBM/kv",0.0);
	binsFlag = inputParams("IBM/binsFlag",false);
		
	// if we need bins, do some calculations:
	if (binsFlag) {
		sizeBins = inputParams("IBM/sizeBins",0.0);
		binMax = inputParams("IBM/binMax",0);
		int Nx = inputParams("Lattice/Nx",1);
		int Ny = inputParams("Lattice/Ny",1);
		int Nz = inputParams("Lattice/Nz",1);
		numBins.x = int(floor(Nx/sizeBins));
	    numBins.y = int(floor(Ny/sizeBins));
	    numBins.z = int(floor(Nz/sizeBins));
		nBins = numBins.x*numBins.y*numBins.z;
	}
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
	if (binsFlag) {
		cudaMalloc((void **) &binMembers, nBins*binMax*sizeof(int));
		cudaMalloc((void **) &binOccupancy, nBins*sizeof(int));
		cudaMalloc((void **) &binMap, nBins*26*sizeof(int));
	}	
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
	if (binsFlag) {
		cudaFree(binMembers);
		cudaFree(binOccupancy);
		cudaFree(binMap);
	}		
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
	cudaMemcpy(facesH, faces, sizeof(triangle)*nFaces, cudaMemcpyDeviceToHost);
	cudaMemcpy(edgesH, edges, sizeof(edge)*nEdges, cudaMemcpyDeviceToHost);
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
	//write_vtk_immersed_boundary_3D(tagname,tagnum,
	//nNodes,nFaces,rH,facesH);
	
	write_vtk_immersed_boundary_normals_3D(tagname,tagnum,
	nNodes,nFaces,nEdges,rH,facesH,edgesH);
}



// --------------------------------------------------------
// Calculate rest geometries:
// --------------------------------------------------------

void class_membrane_ibm3D::rest_geometries(int nBlocks, int nThreads)
{
	// rest edge lengths:
	rest_edge_lengths_IBM3D
	<<<nBlocks,nThreads>>> (r,edges,nEdges);
	
	// rest edge angles:
	rest_edge_angles_IBM3D
	<<<nBlocks,nThreads>>> (r,edges,faces,nEdges);
	
	// rest triangle area:
	rest_triangle_areas_IBM3D
	<<<nBlocks,nThreads>>> (r,faces,cells,nFaces);
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
// Call to kernel that builds the binMap array:
// --------------------------------------------------------

void class_membrane_ibm3D::build_binMap(int nBlocks, int nThreads)
{
	build_binMap_IBM3D
	<<<nBlocks,nThreads>>> (binMap,numBins,nnbins,nBins);
}



// --------------------------------------------------------
// Call to kernel that resets bin lists:
// --------------------------------------------------------

void class_membrane_ibm3D::reset_bin_lists(int nBlocks, int nThreads)
{
	reset_bin_lists_IBM3D
	<<<nBlocks,nThreads>>> (binOccupancy,binMembers,binMax,nBins);
}



// --------------------------------------------------------
// Call to kernel that builds bin lists:
// --------------------------------------------------------

void class_membrane_ibm3D::build_bin_lists(int nBlocks, int nThreads)
{
	build_bin_lists_IBM3D
	<<<nBlocks,nThreads>>> (r,binOccupancy,binMembers,numBins,sizeBins,nNodes,binMax);
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_membrane_ibm3D::nonbonded_node_interactions(int nBlocks, int nThreads)
{
	nonbonded_node_interactions_IBM3D
	<<<nBlocks,nThreads>>> (r,f,binOccupancy,binMembers,binMap,numBins,sizeBins,nNodes,binMax,nnbins);
}



// --------------------------------------------------------
// Calls to kernels that compute forces on nodes based 
// on the membrane mechanics model:
// --------------------------------------------------------

void class_membrane_ibm3D::compute_node_forces(int nBlocks, int nThreads)
{
	// First, zero the node forces and the cell volumes:
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
		
	// Forth, compute the volume conservation force for each face:
	compute_node_force_membrane_volume_IBM3D
	<<<nBlocks,nThreads>>> (faces,f,cells,kv,nFaces);
	
	// Fifth, compute the global area conservation force for each face:
	compute_node_force_membrane_globalarea_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,cells,kag,nFaces);
	
}



// --------------------------------------------------------
// Calls to kernels that changes the default cell volume:
// --------------------------------------------------------

void class_membrane_ibm3D::change_cell_volume(float change, int nBlocks, int nThreads)
{
	change_cell_volumes_IBM3D
	<<<nBlocks,nThreads>>> (cells,change,nCells);
}






