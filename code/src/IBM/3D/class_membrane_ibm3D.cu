
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
	nCells = inputParams("IBM/nCells",1);
	ks = inputParams("IBM/ks",0.0);
	kb = inputParams("IBM/kb",0.0);
	ka = inputParams("IBM/ka",0.0);
	kag = inputParams("IBM/kag",0.0);
	kv = inputParams("IBM/kv",0.0);
	N.x = inputParams("Lattice/Nx",1);
	N.y = inputParams("Lattice/Ny",1);
	N.z = inputParams("Lattice/Nz",1);	
	nNodesPerCell = nNodes/nCells;
	nFacesPerCell = nFaces/nCells;
	nEdgesPerCell = nEdges/nCells;
	Box.x = float(N.x);   // assume dx=1
	Box.y = float(N.y);
	Box.z = float(N.z);
		
	// if we need bins, do some calculations:
	binsFlag = false;
	if (nCells > 1) binsFlag = true;
	if (binsFlag) {
		sizeBins = inputParams("IBM/sizeBins",2.0);
		binMax = inputParams("IBM/binMax",1);			
		numBins.x = int(floor(N.x/sizeBins));
	    numBins.y = int(floor(N.y/sizeBins));
	    numBins.z = int(floor(N.z/sizeBins));
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
	cellIDsH = (int*)malloc(nNodes*sizeof(int));
					
	// allocate array memory (device):
	cudaMalloc((void **) &r, nNodes*sizeof(float3));	
	cudaMalloc((void **) &v, nNodes*sizeof(float3));	
	cudaMalloc((void **) &f, nNodes*sizeof(float3));
	cudaMalloc((void **) &faces, nFaces*sizeof(triangle));
	cudaMalloc((void **) &edges, nEdges*sizeof(edge));
	cudaMalloc((void **) &cells, nCells*sizeof(cell));
	cudaMalloc((void **) &cellIDs, nNodes*sizeof(int));
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
	free(cellIDsH);	
				
	// free array memory (device):
	cudaFree(r);	
	cudaFree(v);	
	cudaFree(f);
	cudaFree(faces);
	cudaFree(edges);
	cudaFree(cells);
	cudaFree(cellIDs);
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
	cudaMemcpy(cellIDs, cellIDsH, sizeof(int)*nNodes, cudaMemcpyHostToDevice);	
}
	


// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_membrane_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(rH, r, sizeof(float3)*nNodes, cudaMemcpyDeviceToHost);	
	//cudaMemcpy(facesH, faces, sizeof(triangle)*nFaces, cudaMemcpyDeviceToHost);
	//cudaMemcpy(edgesH, edges, sizeof(edge)*nEdges, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void class_membrane_ibm3D::read_ibm_information(std::string tagname)
{
	read_ibm_information_long(tagname,nNodesPerCell,nFacesPerCell,nEdgesPerCell,rH,facesH,edgesH);
}



// --------------------------------------------------------
// Assign the reference node to every cell.  The reference
// node is arbitrary (here we use the first node), but it
// is necessary for handling PBC's.
// --------------------------------------------------------

void class_membrane_ibm3D::assign_refNode_to_cells()
{
	for (int c=0; c<nCells; c++) {
		cellsH[c].refNode = c*nNodesPerCell;
	}
}	



// --------------------------------------------------------
// Assign the cell ID to every node:
// --------------------------------------------------------

void class_membrane_ibm3D::assign_cellIDs_to_nodes()
{
	for (int c=0; c<nCells; c++) {
		for (int i=0; i<nNodesPerCell; i++) {
			int ii = i + c*nNodesPerCell;
			cellIDsH[ii] = c;
		}
	}
}



// --------------------------------------------------------
// Duplicate the first cell mesh information to all cells:
// --------------------------------------------------------

void class_membrane_ibm3D::duplicate_cells()
{
	if (nCells > 1) {
		for (int c=1; c<nCells; c++) {
			// copy node positions:
			for (int i=0; i<nNodesPerCell; i++) {
				int ii = i + c*nNodesPerCell;
				rH[ii] = rH[i];
			}
			// copy edge info:
			for (int i=0; i<nEdgesPerCell; i++) {
				int ii = i + c*nEdgesPerCell;
				edgesH[ii].v0 = edgesH[i].v0 + c*nNodesPerCell;
				edgesH[ii].v1 = edgesH[i].v1 + c*nNodesPerCell;
				edgesH[ii].f0 = edgesH[i].f0 + c*nFacesPerCell;
				edgesH[ii].f1 = edgesH[i].f1 + c*nFacesPerCell;
			}
			// copy face info:
			for (int i=0; i<nFacesPerCell; i++) {
				int ii = i + c*nFacesPerCell;
				facesH[ii].v0 = facesH[i].v0 + c*nNodesPerCell;
				facesH[ii].v1 = facesH[i].v1 + c*nNodesPerCell;
				facesH[ii].v2 = facesH[i].v2 + c*nNodesPerCell;
				facesH[ii].cellID = c;								
			}
		}
	}
	
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_membrane_ibm3D::shift_node_positions(int cellID, float xsh, float ysh, float zsh)
{
	for (int i=0; i<nNodesPerCell; i++) {
		int indx = i + cellID*nNodesPerCell;
		rH[indx].x += xsh;
		rH[indx].y += ysh;
		rH[indx].z += zsh;
	}
}



// --------------------------------------------------------
// Write IBM output to file:
// --------------------------------------------------------

void class_membrane_ibm3D::write_output(std::string tagname, int tagnum)
{
	// first unwrap coordinate positions:
	unwrap_node_coordinates(); 
	// write ouput:
	write_vtk_immersed_boundary_3D(tagname,tagnum,
	nNodes,nFaces,rH,facesH);
	// below writes out more information (edge angles)
	//write_vtk_immersed_boundary_normals_3D(tagname,tagnum,
	//nNodes,nFaces,nEdges,rH,facesH,edgesH);
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
	
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,nNodes);	
}



// --------------------------------------------------------
// Call to "interpolate_velocity_IBM3D" kernel:
// --------------------------------------------------------

void class_membrane_ibm3D::interpolate_velocity(float* uLBM, float* vLBM, 
	float* wLBM, int nBlocks, int nThreads)
{
	interpolate_velocity_IBM3D
	<<<nBlocks,nThreads>>> (r,v,uLBM,vLBM,wLBM,N.x,N.y,N.z,nNodes);	
}



// --------------------------------------------------------
// Call to "extrapolate_force_IBM3D" kernel:
// --------------------------------------------------------

void class_membrane_ibm3D::extrapolate_force(float* fxLBM, float* fyLBM, 
	float* fzLBM, int nBlocks, int nThreads)
{
	extrapolate_force_IBM3D
	<<<nBlocks,nThreads>>> (r,v,fxLBM,fyLBM,fzLBM,N.x,N.y,N.z,nNodes);	
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
	<<<nBlocks,nThreads>>> (r,f,binOccupancy,binMembers,binMap,cellIDs,numBins,sizeBins,
	                        nNodes,binMax,nnbins,Box);
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
	
	// Second, unwrap node coordinates:
	unwrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,cells,cellIDs,Box,nNodes);	
					
	// Third, compute the area dilation force for each face:
	compute_node_force_membrane_area_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,cells,ka,nFaces);	
		
	// Forth, compute the edge extension and bending force for each edge:
	compute_node_force_membrane_edge_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,edges,ks,kb,nEdges);
		
	// Fifth, compute the volume conservation force for each face:
	compute_node_force_membrane_volume_IBM3D
	<<<nBlocks,nThreads>>> (faces,f,cells,kv,nFaces);
	
	// Sixth, compute the global area conservation force for each face:
	compute_node_force_membrane_globalarea_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,cells,kag,nFaces);
		
	// Seventh, re-wrap node coordinates:
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,nNodes);
			
}



// --------------------------------------------------------
// Calls to kernels that changes the default cell volume:
// --------------------------------------------------------

void class_membrane_ibm3D::change_cell_volume(float change, int nBlocks, int nThreads)
{
	change_cell_volumes_IBM3D
	<<<nBlocks,nThreads>>> (cells,change,nCells);
}



// --------------------------------------------------------
// Unwrap node coordinates based on difference between node
// position and the cell's reference node position:
// --------------------------------------------------------

void class_membrane_ibm3D::unwrap_node_coordinates()
{
	for (int i=0; i<nNodes; i++) {
		int c = cellIDsH[i];
		int j = cellsH[c].refNode;
		float3 rij = rH[j] - rH[i];
		rH[i] = rH[i] + roundf(rij/Box)*Box; // PBC's		
	}	
}




