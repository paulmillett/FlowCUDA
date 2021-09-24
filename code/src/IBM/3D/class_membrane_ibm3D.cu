
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
	nNodesPerCell = inputParams("IBM/nNodesPerCell",0);
	nFacesPerCell = inputParams("IBM/nFacesPerCell",0);	
	nEdgesPerCell = inputParams("IBM/nEdgesPerCell",0);
	nCells = inputParams("IBM/nCells",1);
	dt = inputParams("Time/dt",1.0);
	ks = inputParams("IBM/ks",0.0);
	kb = inputParams("IBM/kb",0.0);
	ka = inputParams("IBM/ka",0.0);
	kag = inputParams("IBM/kag",0.0);
	kv = inputParams("IBM/kv",0.0);
	C  = inputParams("IBM/C",0.0);
	N.x = inputParams("Lattice/Nx",1);
	N.y = inputParams("Lattice/Ny",1);
	N.z = inputParams("Lattice/Nz",1);	
	nNodes = nNodesPerCell*nCells;
	nFaces = nFacesPerCell*nCells;
	nEdges = nEdgesPerCell*nCells;
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
// With the Host, shrink cells and randomly shift them with
// the box:
// --------------------------------------------------------

void class_membrane_ibm3D::shrink_and_randomize_cells(float shrinkFactor, float sepMin, float sepWall)
{
	// copy node positions from device to host:
	cudaMemcpy(rH, r, sizeof(float3)*nNodes, cudaMemcpyDeviceToHost);
	
	// shrink cells by specified amount:
	for (int c=0; c<nCells; c++) {
		for (int i=0; i<nNodesPerCell; i++) {
			int indx = i + c*nNodesPerCell;
			rH[indx] *= shrinkFactor;
		}
	}
	
	// randomly shift cells, without overlapping previous cells:
	float3* cellCOM = (float3*)malloc(nCells*sizeof(float3));
	for (int c=0; c<nCells; c++) {
		cellCOM[c] = make_float3(0.0);
		float3 shift = make_float3(0.0);		
		bool tooClose = true;
		while (tooClose) {
			// reset tooClose to false
			tooClose = false;
			// get random position
			shift.x = (float)rand()/RAND_MAX*Box.x;
			shift.y = sepWall + (float)rand()/RAND_MAX*(Box.y-2.0*sepWall);
			shift.z = (float)rand()/RAND_MAX*Box.z;
			// check with other cells
			for (int d=0; d<c; d++) {
				float sep = calc_separation_pbc(shift,cellCOM[d]);
                if (sep < sepMin) 
                {
                    tooClose = true;
                    break;
                }
			}
			
		}
		cellCOM[c] = shift;		
		rotate_and_shift_node_positions(c,shift.x,shift.y,shift.z);
	}
	
	// last, copy node positions from host to device:
	cudaMemcpy(r, rH, sizeof(float3)*nNodes, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// calculate separation distance using PBCs:
// --------------------------------------------------------

float class_membrane_ibm3D::calc_separation_pbc(float3 r1, float3 r2)
{
	float3 dr = r1 - r2;
	dr -= roundf(dr/Box)*Box;
	return length(dr);
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
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_membrane_ibm3D::rotate_and_shift_node_positions(int cellID, float xsh, float ysh, float zsh)
{
	float a = M_PI*(float)rand()/RAND_MAX;  // alpha
	float b = M_PI*(float)rand()/RAND_MAX;  // beta
	float g = M_PI*(float)rand()/RAND_MAX;  // gamma
	for (int i=0; i<nNodesPerCell; i++) {
		int indx = i + cellID*nNodesPerCell;
		// rotate:
		float xrot = rH[indx].x*(cos(a)*cos(b)) + rH[indx].y*(cos(a)*sin(b)*sin(g)-sin(a)*cos(g)) + rH[indx].z*(cos(a)*sin(b)*cos(g)+sin(a)*sin(g));
		float yrot = rH[indx].x*(sin(a)*cos(b)) + rH[indx].y*(sin(a)*sin(b)*sin(g)+cos(a)*cos(g)) + rH[indx].z*(sin(a)*sin(b)*cos(g)-cos(a)*sin(g));
		float zrot = rH[indx].x*(-sin(b))       + rH[indx].y*(cos(b)*sin(g))                      + rH[indx].z*(cos(b)*cos(g));
		// shift:		 
		rH[indx].x = xrot + xsh;
		rH[indx].y = yrot + ysh;
		rH[indx].z = zrot + zsh;		
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
}



// --------------------------------------------------------
// Write IBM output to file, including more information
// (edge angles):
// --------------------------------------------------------

void class_membrane_ibm3D::write_output_long(std::string tagname, int tagnum)
{
	// first unwrap coordinate positions:
	unwrap_node_coordinates(); 
	// write ouput:
	write_vtk_immersed_boundary_normals_3D(tagname,tagnum,
	nNodes,nFaces,nEdges,rH,facesH,edgesH);
}



// --------------------------------------------------------
// Calculate rest geometries (Spring model):
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
// Calculate rest geometries (Skalak model):
// --------------------------------------------------------

void class_membrane_ibm3D::rest_geometries_skalak(int nBlocks, int nThreads)
{
	// rest triangle properties:
	rest_triangle_skalak_IBM3D
	<<<nBlocks,nThreads>>> (r,faces,cells,nFaces);
		
	// rest edge angles for bending:
	rest_edge_angles_IBM3D
	<<<nBlocks,nThreads>>> (r,edges,faces,nEdges);
}



// --------------------------------------------------------
// For a certain number of iterations, relax the 
// the node positions (for example, after cells are shrunk 
// to allow them to readjust to their regular volume):
// --------------------------------------------------------

void class_membrane_ibm3D::relax_node_positions(int nIts, float scale, float M, int nBlocks, int nThreads) 
{
	// per iteraction scale factor:
	float power = 1.0/float(nIts);
	float scalePerIter = powf(scale,power);
	
	// make sure node coordinates are wrapped for 
	// PBC's prior to building bin-lists the first time:
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,nNodes);	
	
	// iterate to relax node positions while scaling equilibirum
	// cell size:
	for (int i=0; i<nIts; i++) {
		if (i%10000 == 0) cout << "relax step " << i << endl;		
		scale_equilibrium_cell_size(scalePerIter,nBlocks,nThreads);		
		reset_bin_lists(nBlocks,nThreads);		
		build_bin_lists(nBlocks,nThreads);		
		compute_node_forces(nBlocks,nThreads);		
		nonbonded_node_interactions(nBlocks,nThreads);		
		wall_forces_ydir(nBlocks,nThreads);		
		update_node_positions_vacuum(M,nBlocks,nThreads);		
		cudaDeviceSynchronize();
	}	
}



// --------------------------------------------------------
// For a certain number of iterations, relax the 
// the node positions (for example, after cells are shrunk 
// to allow them to readjust to their regular volume):
// --------------------------------------------------------

void class_membrane_ibm3D::relax_node_positions_skalak(int nIts, float scale, float M, int nBlocks, int nThreads) 
{
	// per iteraction scale factor:
	float power = 1.0/float(nIts);
	float scalePerIter = powf(scale,power);
	
	// make sure node coordinates are wrapped for 
	// PBC's prior to building bin-lists the first time:
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,nNodes);	
	
	// iterate to relax node positions while scaling equilibirum
	// cell size:
	for (int i=0; i<nIts; i++) {
		if (i%10000 == 0) cout << "relax step " << i << endl;		
		scale_equilibrium_cell_size(scalePerIter,nBlocks,nThreads);		
		reset_bin_lists(nBlocks,nThreads);		
		build_bin_lists(nBlocks,nThreads);		
		compute_node_forces_skalak(nBlocks,nThreads);		
		nonbonded_node_interactions(nBlocks,nThreads);		
		wall_forces_ydir(nBlocks,nThreads);		
		update_node_positions_vacuum(M,nBlocks,nThreads);		
		cudaDeviceSynchronize();
	}	
}



// --------------------------------------------------------
// Call to "update_node_position_vacuum_IBM3D" kernel:
// --------------------------------------------------------

void class_membrane_ibm3D::update_node_positions_vacuum(float M, int nBlocks, int nThreads)
{
	update_node_position_vacuum_IBM3D
	<<<nBlocks,nThreads>>> (r,f,M,nNodes);
	
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,nNodes);	
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
// Call to "update_node_position_dt_IBM3D" kernel:
// --------------------------------------------------------

void class_membrane_ibm3D::update_node_positions_dt(int nBlocks, int nThreads)
{
	update_node_position_dt_IBM3D
	<<<nBlocks,nThreads>>> (r,v,dt,nNodes);
	
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
// on the membrane mechanics model (Spring model):
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
	<<<nBlocks,nThreads>>> (faces,r,f,edges,ks,nEdges);
	
	compute_node_force_membrane_bending_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,edges,kb,nEdges);
		
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
// Calls to kernels that compute forces on nodes based 
// on the membrane mechanics model (Skalak model):
// --------------------------------------------------------

void class_membrane_ibm3D::compute_node_forces_skalak(int nBlocks, int nThreads)
{
	// First, zero the node forces and the cell volumes:
	zero_node_forces_IBM3D
	<<<nBlocks,nThreads>>> (f,nNodes);
			
	zero_cell_volumes_IBM3D
	<<<nBlocks,nThreads>>> (cells,nCells);
	
	// Second, unwrap node coordinates:
	unwrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,cells,cellIDs,Box,nNodes);	
					
	// Third, compute the Skalak forces for each face:
	compute_node_force_membrane_skalak_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,cells,ks,C,nFaces);
	
	// Fourth, compute the bending force for each edge:		
	compute_node_force_membrane_bending_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,edges,kb,nEdges);
		
	// Fifth, compute the volume conservation force for each face:
	compute_node_force_membrane_volume_IBM3D
	<<<nBlocks,nThreads>>> (faces,f,cells,kv,nFaces);
			
	// Sixth, re-wrap node coordinates:
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,nNodes);
			
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_membrane_ibm3D::wall_forces_ydir(int nBlocks, int nThreads)
{
	wall_forces_ydir_IBM3D
	<<<nBlocks,nThreads>>> (r,f,Box,nNodes);
}



// --------------------------------------------------------
// Call to kernel that changes the default cell volume:
// --------------------------------------------------------

void class_membrane_ibm3D::change_cell_volume(float change, int nBlocks, int nThreads)
{
	change_cell_volumes_IBM3D
	<<<nBlocks,nThreads>>> (cells,change,nCells);
}



// --------------------------------------------------------
// Call to kernel that scales the default cell geometry:
// --------------------------------------------------------

void class_membrane_ibm3D::scale_equilibrium_cell_size(float scale, int nBlocks, int nThreads)
{
	// scale the equilibrium edge length:
	scale_edge_lengths_IBM3D
	<<<nBlocks,nThreads>>> (edges,scale,nEdges);		
	// scale the equilibrium local area:
	scale_face_areas_IBM3D
	<<<nBlocks,nThreads>>> (faces,scale,nFaces);
	// scale the equilibrium global area and volume:
	scale_cell_areas_volumes_IBM3D
	<<<nBlocks,nThreads>>> (cells,scale,nCells);		
}



// --------------------------------------------------------
// Call to kernel that scales the default edge lengths:
// --------------------------------------------------------

void class_membrane_ibm3D::scale_edge_lengths(float scale, int nBlocks, int nThreads)
{
	scale_edge_lengths_IBM3D
	<<<nBlocks,nThreads>>> (edges,scale,nEdges);
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




