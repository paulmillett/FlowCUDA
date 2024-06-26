# include "class_capsules_binary_ibm3D.cuh"
# include "../../IO/GetPot"
# include "../../Utils/eig3.cuh"
# include <math.h>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <string>
# include <sstream>
# include <stdlib.h>
# include <time.h>
using namespace std;  








// **********************************************************************************************
// Constructor, destructor, and array allocations...
// **********************************************************************************************








// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_capsules_binary_ibm3D::class_capsules_binary_ibm3D()
{
	// get some parameters:
	GetPot inputParams("input.dat");
	
	// mesh attributes	
	nNodesPerCell1 = inputParams("IBM/nNodesPerCell1",0);
	nFacesPerCell1 = inputParams("IBM/nFacesPerCell1",0);	
	nEdgesPerCell1 = inputParams("IBM/nEdgesPerCell1",0);
	nNodesPerCell2 = inputParams("IBM/nNodesPerCell2",0);
	nFacesPerCell2 = inputParams("IBM/nFacesPerCell2",0);	
	nEdgesPerCell2 = inputParams("IBM/nEdgesPerCell2",0);
	nCells1 = inputParams("IBM/nCells1",1);
	nCells2 = inputParams("IBM/nCells2",0);
	a1 = inputParams("IBM/a1",6.0);
	a2 = inputParams("IBM/a2",6.0);
	nCells = nCells1 + nCells2;
	nNodes = nNodesPerCell1*nCells1 + nNodesPerCell2*nCells2;
	nFaces = nFacesPerCell1*nCells1 + nFacesPerCell2*nCells2;
	nEdges = nEdgesPerCell1*nCells1 + nEdgesPerCell2*nCells2;
	
	// mechanical properties
	dt = inputParams("Time/dt",1.0);
	ks = inputParams("IBM/ks",0.0);
	kb = inputParams("IBM/kb",0.0);
	ka = inputParams("IBM/ka",0.0);
	kag = inputParams("IBM/kag",0.0);
	kv = inputParams("IBM/kv",0.0);
	C  = inputParams("IBM/C",0.0);
	repA = inputParams("IBM/repA",0.0);
	repD = inputParams("IBM/repD",0.0);
	repFmax = inputParams("IBM/repFmax",0.0);
	gam = inputParams("IBM/gamma",0.1);
	ibmUpdate = inputParams("IBM/ibmUpdate","verlet");
	
	// domain attributes
	N.x = inputParams("Lattice/Nx",1);
	N.y = inputParams("Lattice/Ny",1);
	N.z = inputParams("Lattice/Nz",1);	
	Box.x = float(N.x);   // assume dx=1
	Box.y = float(N.y);
	Box.z = float(N.z);
	pbcFlag = make_int3(1,1,1);
			
	// if we need bins, do some calculations:
	binsFlag = false;
	if (nCells > 1) binsFlag = true;
	if (binsFlag) {		
		bins.sizeBins = inputParams("IBM/sizeBins",2.0);
		bins.binMax = inputParams("IBM/binMax",1);			
		bins.numBins.x = int(floor(N.x/bins.sizeBins));
	    bins.numBins.y = int(floor(N.y/bins.sizeBins));
	    bins.numBins.z = int(floor(N.z/bins.sizeBins));
		bins.nBins = bins.numBins.x*bins.numBins.y*bins.numBins.z;
		bins.nnbins = 26;			
	}	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_capsules_binary_ibm3D::~class_capsules_binary_ibm3D()
{
		
}



// --------------------------------------------------------
// Read IBM information from file:
//
// Override from parent class
//
// --------------------------------------------------------

void class_capsules_binary_ibm3D::read_ibm_information(std::string tagname1, std::string tagname2)
{
	// read file 1
	int C0 = 0;  // start index for cells
	int N0 = 0;  // start index for nodes
	int F0 = 0;  // start index for faces
	int E0 = 0;  // start index for edges	
	read_ibm_file_long(tagname1,C0,N0,F0,E0,nNodesPerCell1,nFacesPerCell1,nEdgesPerCell1);	
	
	// read file 2
	if (nCells2 > 0) {
		C0 = nCells1;
		N0 = nCells1*nNodesPerCell1;  // start index for nodes
		F0 = nCells1*nFacesPerCell1;  // start index for faces
		E0 = nCells1*nEdgesPerCell1;  // start index for edges
		read_ibm_file_long(tagname2,C0,N0,F0,E0,nNodesPerCell2,nFacesPerCell2,nEdgesPerCell2);
	}
	
	// set up indices for each cell:
	for (int c=0; c<nCells; c++) {
		if (c<nCells1) {
			cellsH[c].nNodes = nNodesPerCell1;
			cellsH[c].nFaces = nFacesPerCell1;
			cellsH[c].nEdges = nEdgesPerCell1;
			cellsH[c].indxN0 = c*nNodesPerCell1;
			cellsH[c].indxF0 = c*nFacesPerCell1;
			cellsH[c].indxE0 = c*nEdgesPerCell1;
		}
		else {
			cellsH[c].nNodes = nNodesPerCell2;
			cellsH[c].nFaces = nFacesPerCell2;
			cellsH[c].nEdges = nEdgesPerCell2;
			cellsH[c].indxN0 = nCells1*nNodesPerCell1 + (c-nCells1)*nNodesPerCell2;
			cellsH[c].indxF0 = nCells1*nFacesPerCell1 + (c-nCells1)*nFacesPerCell2;
			cellsH[c].indxE0 = nCells1*nEdgesPerCell1 + (c-nCells1)*nEdgesPerCell2;
		}
		
	}
}



// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void class_capsules_binary_ibm3D::read_ibm_file_long(std::string fname, int C0, int N0, int F0, int E0,
                                                     int nNodes, int nFaces, int nEdges)
{
	
	// -----------------------------------------------
	// variables for reading:
	// -----------------------------------------------

	int nN, nF, nE;
	
	// -----------------------------------------------
	// open file:
	// -----------------------------------------------

	ifstream infile;
	infile.open(fname, ios::in);

	// -----------------------------------------------
	// read header:
	// -----------------------------------------------
	
	infile >> nN >> nF >> nE; 
	if (nN != nNodes) cout << "number of IBM nodes is not consistent with ibm input file = " << nN << endl;
	if (nF != nFaces) cout << "number of IBM faces is not consistent with ibm input file = " << nF << endl;
	if (nE != nEdges) cout << "number of IBM edges is not consistent with ibm input file = " << nE << endl;

	// -----------------------------------------------
	// read node positions:
	// -----------------------------------------------

	for (int i=0; i<nNodes; i++) {
		int n = i + N0;  // adjusted index
		infile >> nodesH[n].r.x >> nodesH[n].r.y >> nodesH[n].r.z;
	}
	
	// -----------------------------------------------
	// read face vertices:
	// -----------------------------------------------
	
	for (int i=0; i<nFaces; i++) {
		int n = i + F0;  // adjusted index
		infile >> facesH[n].v0 >> facesH[n].v1 >> facesH[n].v2;
		facesH[n].v0 += N0;     // adjust from offset
		facesH[n].v1 += N0;     // " "
		facesH[n].v2 += N0;     // " " 
		facesH[n].v0 -= 1;      // adjust from 1-based to 0-based indexing
		facesH[n].v1 -= 1;      // " "
		facesH[n].v2 -= 1;      // " " 
		facesH[n].cellID = C0;  // cell index
	}

	// -----------------------------------------------
	// read edge vertices:
	// -----------------------------------------------

	for (int i=0; i<nEdges; i++) {
		int n = i + E0;
		infile >> edgesH[n].v0 >> edgesH[n].v1;
		edgesH[n].v0 += N0;  // adjust from offset
		edgesH[n].v1 += N0;  // " "
		edgesH[n].v0 -= 1;   // adjust from 1-based to 0-based indexing
		edgesH[n].v1 -= 1;   // " "
	}
	
	// -----------------------------------------------
	// read edge faces:
	// -----------------------------------------------

	for (int i=0; i<nEdges; i++) {
		int n = i + E0;
		infile >> edgesH[n].f0 >> edgesH[n].f1;
		edgesH[n].f0 += F0;  // adjust from offset
		edgesH[n].f1 += F0;  // " "
		edgesH[n].f0 -= 1;   // adjust from 1-based to 0-based indexing
		edgesH[n].f1 -= 1;   // " "
	}
	
	// -----------------------------------------------
	// close file:
	// -----------------------------------------------

	infile.close();
	
}



// --------------------------------------------------------
// Duplicate the first cell mesh information to all cells:
//
// Override from parent class
//
// --------------------------------------------------------

void class_capsules_binary_ibm3D::duplicate_cells()
{
	// Cell population 1
	if (nCells1 > 1) {
		for (int c=1; c<nCells1; c++) {
			// copy node positions:
			for (int i=0; i<nNodesPerCell1; i++) {
				int ii = i + c*nNodesPerCell1;
				nodesH[ii].r = nodesH[i].r;
			}
			// copy edge info:
			for (int i=0; i<nEdgesPerCell1; i++) {
				int ii = i + c*nEdgesPerCell1;
				edgesH[ii].v0 = edgesH[i].v0 + c*nNodesPerCell1;
				edgesH[ii].v1 = edgesH[i].v1 + c*nNodesPerCell1;
				edgesH[ii].f0 = edgesH[i].f0 + c*nFacesPerCell1;
				edgesH[ii].f1 = edgesH[i].f1 + c*nFacesPerCell1;
			}
			// copy face info:
			for (int i=0; i<nFacesPerCell1; i++) {
				int ii = i + c*nFacesPerCell1;
				facesH[ii].v0 = facesH[i].v0 + c*nNodesPerCell1;
				facesH[ii].v1 = facesH[i].v1 + c*nNodesPerCell1;
				facesH[ii].v2 = facesH[i].v2 + c*nNodesPerCell1;
				facesH[ii].cellID = c;								
			}
		}
	}
	
	// Cell population 2
	if (nCells2 > 1) {
		int offsetN = nCells1*nNodesPerCell1;
		int offsetF = nCells1*nFacesPerCell1;
		int offsetE = nCells1*nEdgesPerCell1;
		for (int c=1; c<nCells2; c++) {
			// copy node positions:
			for (int i=0; i<nNodesPerCell2; i++) {
				int ii = i + c*nNodesPerCell2 + offsetN;
				nodesH[ii].r = nodesH[i+offsetN].r;
			}
			// copy edge info:
			for (int i=0; i<nEdgesPerCell2; i++) {
				int ii = i + c*nEdgesPerCell2 + offsetE;
				edgesH[ii].v0 = edgesH[i+offsetE].v0 + c*nNodesPerCell2;
				edgesH[ii].v1 = edgesH[i+offsetE].v1 + c*nNodesPerCell2;
				edgesH[ii].f0 = edgesH[i+offsetE].f0 + c*nFacesPerCell2;
				edgesH[ii].f1 = edgesH[i+offsetE].f1 + c*nFacesPerCell2;
			}
			// copy face info:
			for (int i=0; i<nFacesPerCell2; i++) {
				int ii = i + c*nFacesPerCell2 + offsetF;
				facesH[ii].v0 = facesH[i+offsetF].v0 + c*nNodesPerCell2;
				facesH[ii].v1 = facesH[i+offsetF].v1 + c*nNodesPerCell2;
				facesH[ii].v2 = facesH[i+offsetF].v2 + c*nNodesPerCell2;
				facesH[ii].cellID = c + nCells1;										
			}
		}
	}
	
}



// --------------------------------------------------------
// Set cell radii for each cell type:
// --------------------------------------------------------

void class_capsules_binary_ibm3D::set_cells_radii_binary()
{
	for (int i=0; i<nCells; i++) {
		if (i<nCells1) {
			set_cell_radius(i,a1);
		}
		else {
			set_cell_radius(i,a2);
		}
			
	}
}



// --------------------------------------------------------
// Set cell types for each cell:
// --------------------------------------------------------

void class_capsules_binary_ibm3D::set_cells_types_binary()
{
	for (int i=0; i<nCells; i++) {
		if (i<nCells1) {
			set_cell_type(i,1);
		}
		else {
			set_cell_type(i,2);
		}
			
	}
}



// --------------------------------------------------------
// With the Host, randomly position cells within the box:
// assumptions:
//   1.) RBC's can overlap each other and platelets
//   2.) Platelets cannot overlap each other
// --------------------------------------------------------

void class_capsules_binary_ibm3D::randomize_platelets_and_rbcs(float sepMin, float sepWall)
{
	// copy node positions from device to host:
	cudaMemcpy(nodesH, nodes, sizeof(node)*nNodes, cudaMemcpyDeviceToHost);
	
	// randomly shift cells, without overlapping previous cells:
	// assume that: cellType = 1 are RBC's and 
	//              cellType = 2 are Platelets
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
			shift.z = sepWall + (float)rand()/RAND_MAX*(Box.z-2.0*sepWall);
			// check with other cells
			for (int d=0; d<c; d++) {
				// only check separation if both cells are platelets
				if (cellsH[c].cellType==2 && cellsH[d].cellType==2) {
					float sep = calc_separation_pbc(shift,cellCOM[d]);
					sep -= cellsH[c].rad + cellsH[d].rad;
					// if both cells are platelets, check to see if they are too close
					if (sep < sepMin) 
	                {
	                    tooClose = true;
	                    break;
	                }
				}				
			}			
		}
		cellCOM[c] = shift;		
		rotate_and_shift_node_positions(c,shift.x,shift.y,shift.z);
	}
	
	// last, copy node positions from host to device:
	cudaMemcpy(nodes, nodesH, sizeof(node)*nNodes, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// With the Host, randomly position cells within the box:
// assumptions:
//   1.) RBC's (cellType==2) can overlap each other
//   2.) The probe (cellType==1) is already positioned
// --------------------------------------------------------

void class_capsules_binary_ibm3D::randomize_probe_and_rbcs(float sepMin, float sepWall)
{
	// copy node positions from device to host:
	cudaMemcpy(nodesH, nodes, sizeof(node)*nNodes, cudaMemcpyDeviceToHost);
	
	// randomly shift cells, without overlapping previous cells:
	// assume that: cellType = 1 are RBC's and 
	//              cellType = 2 are Platelets
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
			shift.z = sepWall + (float)rand()/RAND_MAX*(Box.z-2.0*sepWall);
			// check with other cells
			for (int d=0; d<c; d++) {
				// only check separation for rbc/probe combinations
				if (cellsH[c].cellType==2 && cellsH[d].cellType==1) {
					float sep = calc_separation_pbc(shift,cellCOM[d]);
					sep -= cellsH[c].rad + cellsH[d].rad;
					// if both cells are platelets, check to see if they are too close
					if (sep < sepMin) 
	                {
	                    tooClose = true;
	                    break;
	                }
				}				
			}			
		}
		// if this cell is the probe, then assign a center position
		if (cellsH[c].cellType==1) {
			shift.x = Box.x/2.0;
			shift.y = Box.y/2.0;
			shift.z = Box.z/2.0;
		}
		cellCOM[c] = shift;		
		rotate_and_shift_node_positions(c,shift.x,shift.y,shift.z);
	}
	
	// last, copy node positions from host to device:
	cudaMemcpy(nodes, nodesH, sizeof(node)*nNodes, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Take step forward for IBM w/o fluid:
// (note: this uses the velocity-Verlet algorithm)
// --------------------------------------------------------

void class_capsules_binary_ibm3D::stepIBM_no_fluid_rbcs_platelets(int nSteps, bool zeroFlag, int nBlocks, int nThreads) 
{		
	// use distinct (smaller) value of repA:
	float repA0 = repA;
	repA = 0.0001;
	
	for (int i=0; i<nSteps; i++) {
		// first step of IBM velocity verlet:
		// NOTE: cellType2 (platelets) is stationary
		update_node_positions_verlet_1_cellType2_stationary(nBlocks,nThreads);

		// re-build bin lists for IBM nodes:
		reset_bin_lists(nBlocks,nThreads);
		build_bin_lists(nBlocks,nThreads);
		
		// update IBM:
		compute_node_forces_skalak(nBlocks,nThreads);
		nonbonded_node_interactions(nBlocks,nThreads);
		compute_wall_forces(nBlocks,nThreads);
		add_drag_force_to_nodes(0.001,nBlocks,nThreads);
		enforce_max_node_force(nBlocks,nThreads);
		update_node_positions_verlet_2(nBlocks,nThreads);
	}
	if (zeroFlag) zero_velocities_forces(nBlocks,nThreads); 
	
	// reset repA to intended value:
	repA = repA0;
}



// --------------------------------------------------------
// Call to "update_node_position_verlet_1_IBM3D" kernel:
// --------------------------------------------------------

void class_capsules_binary_ibm3D::update_node_positions_verlet_1_cellType2_stationary(int nBlocks, int nThreads)
{
	update_node_position_verlet_1_cellType2_stationary_IBM3D
	<<<nBlocks,nThreads>>> (nodes,cells,dt,1.0,nNodes);
	
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (nodes,Box,pbcFlag,nNodes);	
}



