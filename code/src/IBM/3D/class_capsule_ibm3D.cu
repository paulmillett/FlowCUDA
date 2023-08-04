 
# include "class_capsule_ibm3D.cuh"
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

class_capsule_ibm3D::class_capsule_ibm3D()
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
	repA = inputParams("IBM/repA",0.0);
	repD = inputParams("IBM/repD",0.0);
	repFmax = inputParams("IBM/repFmax",0.0);
	gam = inputParams("IBM/gamma",0.1);
	ibmUpdate = inputParams("IBM/ibmUpdate","verlet");
	N.x = inputParams("Lattice/Nx",1);
	N.y = inputParams("Lattice/Ny",1);
	N.z = inputParams("Lattice/Nz",1);
	nNodes = nNodesPerCell*nCells;
	nFaces = nFacesPerCell*nCells;
	nEdges = nEdgesPerCell*nCells;
	Box.x = float(N.x);   // assume dx=1
	Box.y = float(N.y);
	Box.z = float(N.z);
	pbcFlag = make_int3(1,1,1);
			
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
		nnbins = 26;
	}
	
	// set up buckets for cell distributions in y-z plane:
	int buckets = inputParams("IBM/buckets",0);
	if (buckets == 0) bucketsFlag = false;
	if (buckets == 1) bucketsFlag = true;
	if (bucketsFlag) {
		numBuckets.y = inputParams("IBM/numBucketsY",1);
		numBuckets.z = inputParams("IBM/numBucketsZ",1);
		numBuckets.x = 1;
		sizeBuckets.y = float(N.y-1)/float(numBuckets.y);
		sizeBuckets.z = float(N.z-1)/float(numBuckets.z);
		sizeBuckets.x = 0.0;
		totalBucketCnt = 0;
	}	
	
	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_capsule_ibm3D::~class_capsule_ibm3D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_capsule_ibm3D::allocate()
{
	// allocate array memory (host):
	rH = (float3*)malloc(nNodes*sizeof(float3));
	vH = (float3*)malloc(nNodes*sizeof(float3));
	facesH = (triangle*)malloc(nFaces*sizeof(triangle));
	edgesH = (edge*)malloc(nEdges*sizeof(edge));
	cellsH = (cell*)malloc(nCells*sizeof(cell));
	cellIDsH = (int*)malloc(nNodes*sizeof(int));
	if (bucketsFlag) {
		bucketCnt = (int*)malloc(numBuckets.y*numBuckets.z*sizeof(int));
		for (int i=0; i<numBuckets.y*numBuckets.z; i++) bucketCnt[i] = 0;
	}
	
	// assign membrane properties to all cells:
	GetPot inputParams("input.dat");
	float Ca = inputParams("IBM/Ca",1.0); 
	set_cells_mechanical_props(ks,kb,kv,C,Ca);
					
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

void class_capsule_ibm3D::deallocate()
{
	// free array memory (host):
	free(rH);
	free(vH);
	free(facesH);
	free(edgesH);
	free(cellsH);
	free(cellIDsH);
	if (bucketsFlag) {
		free(bucketCnt);
	}
				
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

void class_capsule_ibm3D::memcopy_host_to_device()
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

void class_capsule_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(rH, r, sizeof(float3)*nNodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(vH, v, sizeof(float3)*nNodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(facesH, faces, sizeof(triangle)*nFaces, cudaMemcpyDeviceToHost);
	//cudaMemcpy(edgesH, edges, sizeof(edge)*nEdges, cudaMemcpyDeviceToHost);
	
	// unwrap coordinate positions:
	unwrap_node_coordinates(); 
}











// **********************************************************************************************
// Initialization Stuff...
// **********************************************************************************************











// --------------------------------------------------------
// Read IBM information from file:
// --------------------------------------------------------

void class_capsule_ibm3D::read_ibm_information(std::string tagname)
{
	read_ibm_information_long(tagname,nNodesPerCell,nFacesPerCell,nEdgesPerCell,rH,facesH,edgesH);
}



// --------------------------------------------------------
// Setters:
// --------------------------------------------------------

void class_capsule_ibm3D::set_pbcFlag(int x, int y, int z)
{
	pbcFlag.x = x; pbcFlag.y = y; pbcFlag.z = z;
}

void class_capsule_ibm3D::set_ks(float val)
{
	ks = val;
	for (int c=0; c<nCells; c++) cellsH[c].ks = ks;
}

void class_capsule_ibm3D::set_ka(float val)
{
	ka = val;
}

void class_capsule_ibm3D::set_kb(float val)
{
	kb = val;
	for (int c=0; c<nCells; c++) cellsH[c].kb = kb;
}

void class_capsule_ibm3D::set_kv(float val)
{
	kv = val;
	for (int c=0; c<nCells; c++) cellsH[c].kv = kv;
}

void class_capsule_ibm3D::set_kag(float val)
{
	kag = val;
}

void class_capsule_ibm3D::set_C(float val)
{
	C = val;
	for (int c=0; c<nCells; c++) cellsH[c].C = C;
}

void class_capsule_ibm3D::set_cells_mechanical_props(float ks, float kb, float kv, float C, float Ca)
{
	// set props for ALL cells:
	for (int c=0; c<nCells; c++) {
		cellsH[c].ks = ks;
		cellsH[c].kb = kb;
		cellsH[c].kv = kv;
		cellsH[c].C  = C;
		cellsH[c].Ca = Ca;
	}
}

void class_capsule_ibm3D::set_cell_mechanical_props(int cID, float ks, float kb, float kv, float C, float Ca)
{
	// set props for ONE cell:
	cellsH[cID].ks = ks;
	cellsH[cID].kb = kb;
	cellsH[cID].kv = kv;
	cellsH[cID].C  = C;
	cellsH[cID].Ca = Ca;
}

void class_capsule_ibm3D::resize_cell_radius(int cID, float scale)
{
	for (int i=0; i<nNodesPerCell; i++) {
		int indx = i + cID*nNodesPerCell;
		rH[indx] *= scale;
	}
}

void class_capsule_ibm3D::set_cells_radii(float rad)
{
	// set radius for ALL cells:
	for (int c=0; c<nCells; c++) {
		cellsH[c].rad = rad;
	}
}

void class_capsule_ibm3D::set_cell_radius(int cID, float rad)
{
	// set radius for ONE cell:
	cellsH[cID].rad = rad;
}



// --------------------------------------------------------
// Calculate cell mechanical properties based 
// on a given distribution:
// --------------------------------------------------------

void class_capsule_ibm3D::calculate_cell_membrane_props(float Re, float Ca, float stddevCa, float a,
                                                         float h, float rho, float umax, float Kv, float C,
														 std::string cellPropsDist)
{
	// UNIFORM mechanical properties:
	if (cellPropsDist == "uniform") {		
		float Ks = rho*umax*umax*a/(Ca*Re);    //rho*nu*umax*a/(h*Ca);
		float Kb = Ks*a*a*0.00287*sqrt(3);  
		for (int i=0; i<nCells; i++) {
			set_cell_mechanical_props(i,Ks,Kb,Kv,C,Ca);
		}
		// output the results:
		cout << "  " << endl;
		cout << "H = " << h << endl;
		cout << "umax = " << umax << endl;
		cout << "ks = " << Ks << endl;
		cout << "kb = " << Kb << endl;
		cout << "  " << endl;
		cout << "Ca = " << Ca << endl;
		cout << "  " << endl;
	}
	
	// GAUSSIAN 'normal' mechanical properties:
	else if (cellPropsDist == "normal") {
		// output file:
		ofstream outfile;
		std::stringstream filenamecombine;
		filenamecombine << "vtkoutput/" << "distribution_Ca.dat";
		string filename = filenamecombine.str();
		outfile.open(filename.c_str(), ios::out | ios::app);
		// random number generator:
		std::default_random_engine generator;
		std::normal_distribution<double> distribution(Ca,stddevCa);
		// loop over cells:
		for (int i=0; i<nCells; i++) {
			float Ca_i = distribution(generator);
			if (Ca_i < 0.008) Ca_i = 0.008;
			float Ks = rho*umax*umax*a/(Ca_i*Re);  //rho*nu*umax*a/(h*Ca);
			float Kb = Ks*a*a*0.00287*sqrt(3);
			set_cell_mechanical_props(i,Ks,Kb,Kv,C,Ca_i);
			outfile << fixed << setprecision(5) << Ca_i << endl;
		}
		outfile.close();
		// output the results:
		cout << "  " << endl;
		cout << "H = " << h << endl;
		cout << "umax = " << umax << endl;
		cout << "  " << endl;
		cout << "Ca = " << Ca << endl;
		cout << "Ca std dev = " << stddevCa << endl;
		cout << "  " << endl;
	}
	
	// BIMODAL mechanical properties:
	else if (cellPropsDist == "bimodal") {
		// assign membrane properties to all cells:
		GetPot inputParams("input.dat");
		float Ca1 = inputParams("IBM/Ca1",0.1); 
		float Ca2 = inputParams("IBM/Ca2",0.1);
		float xCa1 = inputParams("IBM/xCa1",0.5);
		srand(time(NULL));
		for (int i=0; i<nCells; i++) {
			float Ca_i = Ca2;
			float R = ((float) rand()/(RAND_MAX));			 
			if (R <= xCa1) Ca_i = Ca1;
			float Ks = rho*umax*umax*a/(Ca_i*Re);  //rho*nu*umax*a/(h*Ca);
			float Kb = Ks*a*a*0.00287*sqrt(3);
			set_cell_mechanical_props(i,Ks,Kb,Kv,C,Ca_i);			
		}
	} 
}



// --------------------------------------------------------
// Calculate cell mechanical properties based 
// on a given distribution:
// --------------------------------------------------------

void class_capsule_ibm3D::rescale_cell_radii(float a, float stddevA, std::string cellSizeDist)
{
	// UNIFORM distribution in cell sizes:
	if (cellSizeDist == "uniform") {	
		set_cells_radii(a);
	}	
	
	// NORMAL distribution in cell sizes:
	else if (cellSizeDist == "normal") {		
		// output file:
		ofstream outfile;
		std::stringstream filenamecombine;
		filenamecombine << "vtkoutput/" << "distribution_Radii.dat";
		string filename = filenamecombine.str();
		outfile.open(filename.c_str(), ios::out | ios::app);
		// random number generator:
		std::default_random_engine generator;
		std::normal_distribution<double> distribution(a,stddevA);
		// loop over cells:
		for (int i=0; i<nCells; i++) {
			float a_i = distribution(generator);
			float scale = a_i/a;
			set_cell_radius(i,a_i);
			resize_cell_radius(i,scale);			
			outfile << fixed << setprecision(5) << a_i << endl;
		}
		outfile.close();
	}	
	
	// BIMODAL distribution in cell sizes:
	else if (cellSizeDist == "bimodal") {
		GetPot inputParams("input.dat");
		float a1 = inputParams("IBM/a1",6.0); 
		float a2 = inputParams("IBM/a2",6.0);
		float xa1 = inputParams("IBM/xa1",0.5);
		srand(time(NULL));
		for (int i=0; i<nCells; i++) {
			float a_i = a2;
			float R = ((float) rand()/(RAND_MAX));
			if (R <= xa1) a_i = a1;
			float scale = a_i/a;
			set_cell_radius(i,a_i);
			resize_cell_radius(i,scale);
		}
	}
}



// --------------------------------------------------------
// Assign the reference node to every cell.  The reference
// node is arbitrary (here we use the first node), but it
// is necessary for handling PBC's.
// --------------------------------------------------------

void class_capsule_ibm3D::assign_refNode_to_cells()
{
	for (int c=0; c<nCells; c++) {
		cellsH[c].refNode = c*nNodesPerCell;
	}
}	



// --------------------------------------------------------
// Assign the cell ID to every node:
// --------------------------------------------------------

void class_capsule_ibm3D::assign_cellIDs_to_nodes()
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

void class_capsule_ibm3D::duplicate_cells()
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
// Line up cells in a single-file line in the middle of
// the channel:
// --------------------------------------------------------

void class_capsule_ibm3D::single_file_cells(int Nx, int Ny, int Nz, float cellSpacingX, float offsetY)
{
	// copy node positions from device to host:
	cudaMemcpy(rH, r, sizeof(float3)*nNodes, cudaMemcpyDeviceToHost);
	
	// determine cell spacing:
	float evenSpacingX = float(Nx)/nCells;
	if (cellSpacingX > evenSpacingX) cellSpacingX = evenSpacingX;
	cout << "Initial cell spacing = " << cellSpacingX << endl;
	
	// line up cells in the x-direction:
	int cnt = 1;
	for (int c=0; c<nCells; c++) {
		float3 shift = make_float3(0.0);
		shift.x = cellSpacingX/2.0 + c*cellSpacingX;
		shift.y = float(Ny-1)/2.0  + float(cnt)*offsetY;
		shift.z = float(Nz-1)/2.0;
		cout << c << " " << shift.y << endl;
		rotate_and_shift_node_positions(c,shift.x,shift.y,shift.z);
		cnt = -cnt;
	}
	
	// last, copy node positions from host to device:
	cudaMemcpy(r, rH, sizeof(float3)*nNodes, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// With the Host, shrink cells and randomly shift them 
// within the box:
// --------------------------------------------------------

void class_capsule_ibm3D::shrink_and_randomize_cells(float shrinkFactor, float sepMin, float sepWall)
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
			shift.z = sepWall + (float)rand()/RAND_MAX*(Box.z-2.0*sepWall);
			// check with other cells
			for (int d=0; d<c; d++) {
				float sep = calc_separation_pbc(shift,cellCOM[d]);
				sep -= cellsH[c].rad + cellsH[d].rad;
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
// With the Host, randomly place cells within
// the box above a certain z-plane:
// --------------------------------------------------------

void class_capsule_ibm3D::randomize_cells_above_plane(float shrinkFactor, float sepMin, float sepWall, float zmin)
{
	// copy node positions from device to host:
	cudaMemcpy(rH, r, sizeof(float3)*nNodes, cudaMemcpyDeviceToHost);
		
	// randomly shift cells, without overlapping previous cells:
	float zdepth = Box.z - zmin;
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
			shift.y = sepWall + (float)rand()/RAND_MAX*(Box.y -2.0*sepWall);
			shift.z = sepWall + (float)rand()/RAND_MAX*(zdepth-2.0*sepWall) + zmin;
			// check with other cells
			for (int d=0; d<c; d++) {
				float sep = calc_separation_pbc(shift,cellCOM[d]);
				sep -= cellsH[c].rad + cellsH[d].rad;
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

float class_capsule_ibm3D::calc_separation_pbc(float3 r1, float3 r2)
{
	float3 dr = r1 - r2;
	dr -= roundf(dr/Box)*Box;
	return length(dr);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_capsule_ibm3D::shift_node_positions(int cellID, float xsh, float ysh, float zsh)
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

void class_capsule_ibm3D::rotate_and_shift_node_positions(int cellID, float xsh, float ysh, float zsh)
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

void class_capsule_ibm3D::write_output(std::string tagname, int tagnum)
{
	//write_vtk_immersed_boundary_3D(tagname,tagnum,
	//nNodes,nFaces,rH,facesH);
	write_vtk_immersed_boundary_3D_cellID(tagname,tagnum,
	nNodes,nFaces,rH,facesH,cellsH);
}



// --------------------------------------------------------
// Write IBM output to file, including more information
// (edge angles):
// --------------------------------------------------------

void class_capsule_ibm3D::write_output_long(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_normals_3D(tagname,tagnum,
	nNodes,nFaces,nEdges,rH,facesH,edgesH);
}



// --------------------------------------------------------
// Calculate rest geometries (Spring model):
// --------------------------------------------------------

void class_capsule_ibm3D::rest_geometries(int nBlocks, int nThreads)
{
	// zero the cell reference volume & global area:
	zero_reference_vol_area_IBM3D
	<<<nBlocks,nThreads>>> (cells,nCells);
	
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

void class_capsule_ibm3D::rest_geometries_skalak(int nBlocks, int nThreads)
{
	// zero the cell reference volume & global area:
	zero_reference_vol_area_IBM3D
	<<<nBlocks,nThreads>>> (cells,nCells);
	
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

void class_capsule_ibm3D::relax_node_positions(int nIts, float scale, float M, int nBlocks, int nThreads) 
{
	// per iteraction scale factor:
	float power = 1.0/float(nIts);
	float scalePerIter = powf(scale,power);
	
	// make sure node coordinates are wrapped for 
	// PBC's prior to building bin-lists the first time:
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,pbcFlag,nNodes);	
	
	// iterate to relax node positions while scaling equilibirum
	// cell size:
	for (int i=0; i<nIts; i++) {
		if (i%10000 == 0) cout << "relax step " << i << endl;		
		scale_equilibrium_cell_size(scalePerIter,nBlocks,nThreads);		
		reset_bin_lists(nBlocks,nThreads);		
		build_bin_lists(nBlocks,nThreads);		
		compute_node_forces(nBlocks,nThreads);		
		nonbonded_node_interactions(nBlocks,nThreads);		
		//wall_forces_ydir(nBlocks,nThreads);		
		wall_forces_ydir_zdir(nBlocks,nThreads);
		update_node_positions_vacuum(M,nBlocks,nThreads);		
		cudaDeviceSynchronize();
	}	
}



// --------------------------------------------------------
// For a certain number of iterations, relax the 
// the node positions (for example, after cells are shrunk 
// to allow them to readjust to their regular volume):
// --------------------------------------------------------

void class_capsule_ibm3D::relax_node_positions_skalak(int nIts, float scale, float M, int nBlocks, int nThreads) 
{
	// per iteraction scale factor:
	float power = 1.0/float(nIts);
	float scalePerIter = powf(scale,power);
	
	// make sure node coordinates are wrapped for 
	// PBC's prior to building bin-lists the first time:
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,pbcFlag,nNodes);	
	
	// iterate to relax node positions while scaling equilibirum
	// cell size:
	for (int i=0; i<nIts; i++) {
		if (i%10000 == 0) cout << "relax step " << i << endl;
		scale_equilibrium_cell_size(scalePerIter,nBlocks,nThreads);		
		reset_bin_lists(nBlocks,nThreads);		
		build_bin_lists(nBlocks,nThreads);		
		compute_node_forces_skalak(nBlocks,nThreads);		
		nonbonded_node_interactions(nBlocks,nThreads);		
		//wall_forces_ydir(nBlocks,nThreads);
		wall_forces_ydir_zdir(nBlocks,nThreads);
		update_node_positions_vacuum(M,nBlocks,nThreads);		
		cudaDeviceSynchronize();
	}	
}



// --------------------------------------------------------
// Take step forward with both IBM and LBM:
// --------------------------------------------------------

void class_capsule_ibm3D::stepIBM(class_scsp_D3Q19& lbm, float bodyForx, int nBlocks, int nThreads) 
{
	
	// the traditional IBM update, except here
	// the forces on the IBM nodes are included to calculate the
	// new node positions (see 'update_node_positions_verlet_1')
	
	if (ibmUpdate == "ibm") {
		
		// zero fluid forces:
		lbm.zero_forces(nBlocks,nThreads);
	
		// re-build bin lists for IBM nodes:
		reset_bin_lists(nBlocks,nThreads);
		build_bin_lists(nBlocks,nThreads);
			
		// update IBM:
		compute_node_forces_skalak(nBlocks,nThreads);
		nonbonded_node_interactions(nBlocks,nThreads);
		wall_forces_zdir(nBlocks,nThreads);
		lbm.interpolate_velocity_to_IBM(nBlocks,nThreads,r,v,nNodes);
		update_node_positions_verlet_1(nBlocks,nThreads);   // include forces in position update (more accurate)
		//update_node_positions(nBlocks,nThreads);          // standard IBM approach, only including velocities (less accurate)
			
		// update fluid:
		lbm.extrapolate_forces_from_IBM(nBlocks,nThreads,r,f,nNodes);
		lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
		lbm.stream_collide_save_forcing(nBlocks,nThreads);	
		
		// CUDA sync
		cudaDeviceSynchronize();
		
	} 
	
	//  here, the velocity-Verlet algorithm is used to update the 
	//  node positions - using a viscous drag force proportional
	//  to the difference between the node velocities and the 
	//  fluid velocities
	
	else if (ibmUpdate == "verlet") {
	
		// zero fluid forces:
		lbm.zero_forces(nBlocks,nThreads);
	
		// first step of IBM velocity verlet:
		update_node_positions_verlet_1(nBlocks,nThreads);
	
		// re-build bin lists for IBM nodes:
		reset_bin_lists(nBlocks,nThreads);
		build_bin_lists(nBlocks,nThreads);
			
		// update IBM:
		compute_node_forces_skalak(nBlocks,nThreads);
		nonbonded_node_interactions(nBlocks,nThreads);
		wall_forces_zdir(nBlocks,nThreads);
		lbm.viscous_force_IBM_LBM(nBlocks,nThreads,gam,r,v,f,nNodes);
		update_node_positions_verlet_2(nBlocks,nThreads);
			
		// update fluid:		
		lbm.add_body_force(bodyForx,0.0,0.0,nBlocks,nThreads);
		lbm.stream_collide_save_forcing(nBlocks,nThreads);
			
		// CUDA sync		
		cudaDeviceSynchronize();
		
	}
		
}









// **********************************************************************************************
// Calls to CUDA kernels for main calculations
// **********************************************************************************************











// --------------------------------------------------------
// Call to "update_node_position_vacuum_IBM3D" kernel:
// --------------------------------------------------------

void class_capsule_ibm3D::update_node_positions_vacuum(float M, int nBlocks, int nThreads)
{
	update_node_position_vacuum_IBM3D
	<<<nBlocks,nThreads>>> (r,f,M,nNodes);
	
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,pbcFlag,nNodes);	
}



// --------------------------------------------------------
// Call to "update_node_position_IBM3D" kernel:
// --------------------------------------------------------

void class_capsule_ibm3D::update_node_positions(int nBlocks, int nThreads)
{
	update_node_position_IBM3D
	<<<nBlocks,nThreads>>> (r,v,nNodes);
	
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,pbcFlag,nNodes);	
}



// --------------------------------------------------------
// Call to "update_node_position_dt_IBM3D" kernel:
// --------------------------------------------------------

void class_capsule_ibm3D::update_node_positions_dt(int nBlocks, int nThreads)
{
	update_node_position_dt_IBM3D
	<<<nBlocks,nThreads>>> (r,v,dt,nNodes);
	
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,pbcFlag,nNodes);	
}



// --------------------------------------------------------
// Call to "update_node_position_verlet_1_IBM3D" kernel:
// --------------------------------------------------------

void class_capsule_ibm3D::update_node_positions_verlet_1(int nBlocks, int nThreads)
{
	update_node_position_verlet_1_IBM3D
	<<<nBlocks,nThreads>>> (r,v,f,dt,1.0,nNodes);
	
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,pbcFlag,nNodes);	
}



// --------------------------------------------------------
// Call to "update_node_position_verlet_2_IBM3D" kernel:
// --------------------------------------------------------

void class_capsule_ibm3D::update_node_positions_verlet_2(int nBlocks, int nThreads)
{
	update_node_position_verlet_2_IBM3D
	<<<nBlocks,nThreads>>> (v,f,dt,1.0,nNodes);
}



// --------------------------------------------------------
// Call to "zero_velocities_forces_IBM3D" kernel:
// --------------------------------------------------------

void class_capsule_ibm3D::zero_velocities_forces(int nBlocks, int nThreads)
{
	zero_velocities_forces_IBM3D
	<<<nBlocks,nThreads>>> (v,f,nNodes);
}



// --------------------------------------------------------
// Call to "add_xdir_force_IBM3D" kernel:
// --------------------------------------------------------

void class_capsule_ibm3D::add_xdir_force_to_nodes(int nBlocks, int nThreads, float fx)
{
	add_xdir_force_IBM3D
	<<<nBlocks,nThreads>>> (f,fx,nNodes);
}



// --------------------------------------------------------
// Call to "interpolate_velocity_IBM3D" kernel:
// --------------------------------------------------------

void class_capsule_ibm3D::interpolate_velocity(float* uLBM, float* vLBM, 
	float* wLBM, int nBlocks, int nThreads)
{
	interpolate_velocity_IBM3D
	<<<nBlocks,nThreads>>> (r,v,uLBM,vLBM,wLBM,N.x,N.y,N.z,nNodes);	
}



// --------------------------------------------------------
// Call to "extrapolate_force_IBM3D" kernel:
// --------------------------------------------------------

void class_capsule_ibm3D::extrapolate_force(float* fxLBM, float* fyLBM, 
	float* fzLBM, int nBlocks, int nThreads)
{
	extrapolate_force_IBM3D
	<<<nBlocks,nThreads>>> (r,v,fxLBM,fyLBM,fzLBM,N.x,N.y,N.z,nNodes);	
}



// --------------------------------------------------------
// Call to kernel that builds the binMap array:
// --------------------------------------------------------

void class_capsule_ibm3D::build_binMap(int nBlocks, int nThreads)
{
	if (nCells > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;	
		cout << "nnbins = " << nnbins << endl;	
		build_binMap_IBM3D
		<<<nBlocks,nThreads>>> (binMap,numBins,nnbins,nBins);
	}	
}



// --------------------------------------------------------
// Call to kernel that resets bin lists:
// --------------------------------------------------------

void class_capsule_ibm3D::reset_bin_lists(int nBlocks, int nThreads)
{
	if (nCells > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;
		reset_bin_lists_IBM3D
		<<<nBlocks,nThreads>>> (binOccupancy,binMembers,binMax,nBins);
	}	
}



// --------------------------------------------------------
// Call to kernel that builds bin lists:
// --------------------------------------------------------

void class_capsule_ibm3D::build_bin_lists(int nBlocks, int nThreads)
{
	if (nCells > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;
		build_bin_lists_IBM3D
		<<<nBlocks,nThreads>>> (r,binOccupancy,binMembers,numBins,sizeBins,nNodes,binMax);
	}	
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_capsule_ibm3D::nonbonded_node_interactions(int nBlocks, int nThreads)
{
	if (nCells > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;
		nonbonded_node_interactions_IBM3D
		<<<nBlocks,nThreads>>> (r,f,binOccupancy,binMembers,binMap,cellIDs,numBins,sizeBins,
		                        repA,repD,repFmax,nNodes,binMax,nnbins,Box,pbcFlag);
	}	
}



// --------------------------------------------------------
// Calls to kernels that compute forces on nodes based 
// on the membrane mechanics model (Spring model):
// --------------------------------------------------------

void class_capsule_ibm3D::compute_node_forces(int nBlocks, int nThreads)
{
	// First, zero the node forces and the cell volumes:
	zero_node_forces_IBM3D
	<<<nBlocks,nThreads>>> (f,nNodes);
			
	zero_cell_volumes_IBM3D
	<<<nBlocks,nThreads>>> (cells,nCells);
	
	// Second, unwrap node coordinates:
	unwrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,cells,cellIDs,Box,pbcFlag,nNodes);	
					
	// Third, compute the area dilation force for each face:
	compute_node_force_membrane_area_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,cells,ka,nFaces);	
		
	// Forth, compute the edge extension and bending force for each edge:
	compute_node_force_membrane_edge_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,edges,ks,nEdges);
	
	compute_node_force_membrane_bending_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,edges,cells,nEdges);
		
	// Fifth, compute the volume conservation force for each face:
	compute_node_force_membrane_volume_IBM3D
	<<<nBlocks,nThreads>>> (faces,f,cells,nFaces);
	
	// Sixth, compute the global area conservation force for each face:
	compute_node_force_membrane_globalarea_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,cells,kag,nFaces);
		
	// Seventh, re-wrap node coordinates:
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,pbcFlag,nNodes);
			
}



// --------------------------------------------------------
// Calls to kernels that compute forces on nodes based 
// on the membrane mechanics model (Skalak model):
// --------------------------------------------------------

void class_capsule_ibm3D::compute_node_forces_skalak(int nBlocks, int nThreads)
{
	// First, zero the node forces and the cell volumes:
	zero_node_forces_IBM3D
	<<<nBlocks,nThreads>>> (f,nNodes);
			
	zero_cell_volumes_IBM3D
	<<<nBlocks,nThreads>>> (cells,nCells);
	
	// Second, unwrap node coordinates:
	unwrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,cells,cellIDs,Box,pbcFlag,nNodes);	
					
	// Third, compute the Skalak forces for each face:
	compute_node_force_membrane_skalak_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,cells,nFaces);
	
	// Fourth, compute the bending force for each edge:		
	compute_node_force_membrane_bending_IBM3D
	<<<nBlocks,nThreads>>> (faces,r,f,edges,cells,nEdges);
		
	// Fifth, compute the volume conservation force for each face:
	compute_node_force_membrane_volume_IBM3D
	<<<nBlocks,nThreads>>> (faces,f,cells,nFaces);
			
	// Sixth, re-wrap node coordinates:
	wrap_node_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (r,Box,pbcFlag,nNodes);
			
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir:
// --------------------------------------------------------

void class_capsule_ibm3D::wall_forces_ydir(int nBlocks, int nThreads)
{
	wall_forces_ydir_IBM3D
	<<<nBlocks,nThreads>>> (r,f,Box,nNodes);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in z-dir:
// --------------------------------------------------------

void class_capsule_ibm3D::wall_forces_zdir(int nBlocks, int nThreads)
{
	wall_forces_zdir_IBM3D
	<<<nBlocks,nThreads>>> (r,f,Box,repA,repD,nNodes);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir
// and z-dir:
// --------------------------------------------------------

void class_capsule_ibm3D::wall_forces_ydir_zdir(int nBlocks, int nThreads)
{
	wall_forces_ydir_zdir_IBM3D
	<<<nBlocks,nThreads>>> (r,f,Box,repA,repD,nNodes);
}



// --------------------------------------------------------
// Call to kernel that changes the default cell volume:
// --------------------------------------------------------

void class_capsule_ibm3D::change_cell_volume(float change, int nBlocks, int nThreads)
{
	change_cell_volumes_IBM3D
	<<<nBlocks,nThreads>>> (cells,change,nCells);
}



// --------------------------------------------------------
// Call to kernel that scales the default cell geometry:
// --------------------------------------------------------

void class_capsule_ibm3D::scale_equilibrium_cell_size(float scale, int nBlocks, int nThreads)
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

void class_capsule_ibm3D::scale_edge_lengths(float scale, int nBlocks, int nThreads)
{
	scale_edge_lengths_IBM3D
	<<<nBlocks,nThreads>>> (edges,scale,nEdges);
}











// **********************************************************************************************
// Analysis and Geometry calculations done by the host (CPU)
// **********************************************************************************************











// --------------------------------------------------------
// Unwrap node coordinates based on difference between node
// position and the cell's reference node position:
// --------------------------------------------------------

void class_capsule_ibm3D::unwrap_node_coordinates()
{
	for (int i=0; i<nNodes; i++) {
		int c = cellIDsH[i];
		int j = cellsH[c].refNode;
		float3 rij = rH[j] - rH[i];
		rH[i] = rH[i] + roundf(rij/Box)*Box*pbcFlag; // PBC's		
	}	
}



// --------------------------------------------------------
// Calculate various geometry properties of capsules,
// including center-of-mass, Taylor deformation index, etc.
// --------------------------------------------------------

void class_capsule_ibm3D::membrane_geometry_analysis(std::string tagname, int tagnum)
{
	
	// Define the file location and name:
	/*
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".dat";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	outfile << nCells << endl;
	*/
	
	// -----------------------------------------
	// Define the file location and name:
	// -----------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << "capsule_data.dat";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	ofstream outfile2;
	std::stringstream filenamecombine2;
	filenamecombine2 << "vtkoutput/" << "cell_free_thickness.dat";
	string filename2 = filenamecombine2.str();
	outfile2.open(filename2.c_str(), ios::out | ios::app);
	
	ofstream outfile3;
	std::stringstream filenamecombine3;
	filenamecombine3 << "vtkoutput/" << "averaged_max_T1.dat";
	string filename3 = filenamecombine3.str();
	outfile3.open(filename3.c_str(), ios::out | ios::app);
	
	// -----------------------------------------
	// Loop over the capsules, calculate center-of-mass
	// and Taylor deformation parameter.  Here, I'm using
	// the method described in: Eberly D, Polyhedral Mass
	// Properties (Revisited), Geometric Tools, Redmond WA	
	// -----------------------------------------
	
	float yCFL = float(N.y);
	float zCFL = float(N.z);
	float maxT1Aver = 0.0; // average maximum tension
		
	for (int c=0; c<nCells; c++) {
		
		float D = 0.0;
		float3 com = make_float3(0.0,0.0,0.0);
		float mult[10] = {1.0/6.0,1.0/24.0,1.0/24.0,1.0/24.0,1.0/60.0,1.0/60.0,1.0/60.0,1.0/120.0,1.0/120.0,1.0/120.0};
		float intg[10] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
		float maxT1 = -100.0;  // maximum principle tension of capsule
						
		for (int f=0; f<nFacesPerCell; f++) {
			
			// -----------------------------------------
			// get vertices of triangle i:
			// -----------------------------------------
			
			int fID = f + c*nFacesPerCell;
			int v0 = facesH[fID].v0;
			int v1 = facesH[fID].v1;
			int v2 = facesH[fID].v2;
			float x0 = rH[v0].x;
			float y0 = rH[v0].y;
			float z0 = rH[v0].z;
			float x1 = rH[v1].x;
			float y1 = rH[v1].y;
			float z1 = rH[v1].z;
			float x2 = rH[v2].x;
			float y2 = rH[v2].y;
			float z2 = rH[v2].z;
			
			// -----------------------------------------
			// get edges and cross product of edges:
			// -----------------------------------------
			
			float a1 = x1-x0;
			float b1 = y1-y0;
			float c1 = z1-z0;
			float a2 = x2-x0;
			float b2 = y2-y0;
			float c2 = z2-z0;
			float d0 = b1*c2-b2*c1;
			float d1 = a2*c1-a1*c2;
			float d2 = a1*b2-a2*b1;
			
			// -----------------------------------------
			// compute integral terms:
			// -----------------------------------------
			
			float f1x,f2x,f3x,g0x,g1x,g2x;
			float f1y,f2y,f3y,g0y,g1y,g2y;
			float f1z,f2z,f3z,g0z,g1z,g2z;
			subexpressions(x0,x1,x2,f1x,f2x,f3x,g0x,g1x,g2x);
			subexpressions(y0,y1,y2,f1y,f2y,f3y,g0y,g1y,g2y);
			subexpressions(z0,z1,z2,f1z,f2z,f3z,g0z,g1z,g2z);
			
			// -----------------------------------------
			// update integrals:
			// -----------------------------------------
			
			intg[0] += d0*f1x;
			intg[1] += d0*f2x;
			intg[2] += d1*f2y;
			intg[3] += d2*f2z;
			intg[4] += d0*f3x;
			intg[5] += d1*f3y;
			intg[6] += d2*f3z;
			intg[7] += d0*(y0*g0x + y1*g1x + y2*g2x);
			intg[8] += d1*(z0*g0y + z1*g1y + z2*g2y);
			intg[9] += d2*(x0*g0z + x1*g1z + x2*g2z);
			
			// -----------------------------------------
			// check cell-free layer value.  Here, we
			// assume half-way bounceback conditions, so
			// wall is 0.5dx away from outer grid pos:
			// -----------------------------------------
			
			float ypos = (y0+y1+y2)/3.0;
			float zpos = (z0+z1+z2)/3.0;
			float ywallsep = std::fmin(ypos-(0.0-0.5),float(N.y-1+0.5)-ypos);
			float zwallsep = std::fmin(zpos-(0.0-0.5),float(N.z-1+0.5)-zpos);
			if (ywallsep < yCFL) yCFL = ywallsep;
			if (zwallsep < zCFL) zCFL = zwallsep;
			
			// -----------------------------------------
			// update maximum tension:
			// -----------------------------------------
			
			if (facesH[fID].T1 > maxT1) maxT1 = facesH[fID].T1;
		}
		
		for (int i=0; i<10; i++) intg[i] *= mult[i];
		
		// -----------------------------------------
		// center of mass:
		// -----------------------------------------
		
		float mass = intg[0];
		float vol = mass;   // assume density = 1
		com.x = intg[1]/mass;
		com.y = intg[2]/mass;
		com.z = intg[3]/mass;
		cellsH[c].com.x = com.x;
		cellsH[c].com.y = com.y;
		cellsH[c].com.z = com.z;		
		
		// -----------------------------------------
		// sum maxT1 for average calculation:
		// -----------------------------------------
		
		maxT1Aver += maxT1;
		
		// -----------------------------------------
		// inertia tensor relative to center of mass:
		// -----------------------------------------
		
		float Ixx = intg[5] + intg[6] - mass*(com.y*com.y + com.z*com.z);
		float Iyy = intg[4] + intg[6] - mass*(com.z*com.z + com.x*com.x);
		float Izz = intg[4] + intg[5] - mass*(com.x*com.x + com.y*com.y);
		float Ixy = -(intg[7] - mass*com.x*com.y);
		float Iyz = -(intg[8] - mass*com.y*com.z);
		float Ixz = -(intg[9] - mass*com.x*com.z);
		float I[3][3] = {{Ixx,Ixy,Ixz}, {Ixy,Iyy,Iyz}, {Ixz,Iyz,Izz}};
		
		// calculate longest and shortest axes of capsule:
		// S = sqrt((5/2/vol)*(Ixx + Iyy - sqrt((Ixx-Iyy)^2 + 4*Ixy^2))/2);
		// L = sqrt((5/2/vol)*(Ixx + Iyy + sqrt((Ixx-Iyy)^2 + 4*Ixy^2))/2);
		// Dsl = (L-S)/(L+S)
		
		// -----------------------------------------
		// calculate eigenvalues of inertia tensor:
		// -----------------------------------------
		
		float eigvals[3] = {0.0,0.0,0.0}; 
		float eigvecs[3][3] = {{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
		eigen_decomposition(I,eigvecs,eigvals);
		float L1 = sqrt(5/2/vol*(eigvals[1] + eigvals[2] - eigvals[0]));
		float L2 = sqrt(5/2/vol*(eigvals[0] + eigvals[2] - eigvals[1]));
		float L3 = sqrt(5/2/vol*(eigvals[0] + eigvals[1] - eigvals[2]));
		
		// -----------------------------------------
		// calculate Taylor deformation parameters:
		// -----------------------------------------
		
		float Lmax = std::max({L1,L2,L3});
		float Lmin = std::min({L1,L2,L3});
		D = (Lmax-Lmin)/(Lmax+Lmin);
		
		// -----------------------------------------		
		// calculate the inclination angle:
		// -----------------------------------------
		
		//phi = 0.5*atan(2*Ixy/(Ixx-Iyy));
		//phi = phi/pi;
		
		// -----------------------------------------
		// add cell to the y-z bucket:
		// -----------------------------------------
		
		if (bucketsFlag) {
			int bucketID = int(floor(com.z/sizeBuckets.z))*numBuckets.y + 
			               int(floor(com.y/sizeBuckets.y));
			bucketCnt[bucketID]++;
			totalBucketCnt++;
		}	
		
		// -----------------------------------------
		// print data:
		// -----------------------------------------
		
		outfile << fixed << setprecision(4) << vol << "  " << com.x << "  " << com.y << "  " << com.z << "  "
		        << D << "  " << maxT1 << endl;						
	}
	
	// -----------------------------------------
	// print the cell-free layer thickness in the y-dir and z-dir:
	// -----------------------------------------
	
	outfile2 << fixed << setprecision(4) << tagnum << "  " << yCFL << "  " << zCFL << endl;
	
	// -----------------------------------------
	// print the average maximum T1:
	// -----------------------------------------
	
	outfile3 << fixed << setprecision(5) << tagnum << "  " << maxT1Aver/float(nCells) << endl;
	
	// -----------------------------------------	
	// close file
	// -----------------------------------------
	
	outfile.close();
	outfile2.close();
	outfile3.close();
	
}



void class_capsule_ibm3D::subexpressions(
	const float w0,
	const float w1,
	const float w2,
	float& f1,
	float& f2,
	float& f3,
	float& g0,
	float& g1,
	float& g2)
{
    float temp0 = w0 + w1;
    float temp1 = w0*w0;
    float temp2 = temp1 + w1*temp0;
    f1 = temp0 + w2;
	f2 = temp2 + w2*f1;
    f3 = w0*temp1 + w1*temp2 + w2*f2;
    g0 = f2 + w0*(f1 + w0); 
    g1 = f2 + w1*(f1 + w1);
    g2 = f2 + w2*(f1 + w2);	
}



// --------------------------------------------------------
// Print the cell distributions in the y-z plane:
// --------------------------------------------------------

void class_capsule_ibm3D::print_cell_distributions_yz_plane(std::string tagname, int tagnum)
{
	if (bucketsFlag) {
		// Define the file location and name:
		ofstream outfile;
		std::stringstream filenamecombine;
		filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".dat";
		string filename = filenamecombine.str();
		outfile.open(filename.c_str(), ios::out | ios::app);
		// print number of buckets in y-dir and z-dir:
		outfile << numBuckets.y << " " << numBuckets.z << endl;
		// print distributions:
		int numBucketsTot = numBuckets.y*numBuckets.z;
		for (int i=0; i<numBucketsTot; i++) {
			outfile << float(bucketCnt[i])/float(totalBucketCnt) << endl;
		}	
		// close file
		outfile.close();
	}	
}



// --------------------------------------------------------
// Calculate an order parameter for the fraction of 
// cells in a train-like structure (oriented in the x-dir)
// --------------------------------------------------------

void class_capsule_ibm3D::capsule_train_fraction(float rcut, float thetacut, int step)
{
		
	// -----------------------------------------
	// Define the file location and name:
	// -----------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << "capsule_allignment.dat";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
		
	// -----------------------------------------
	// Initialize variables:
	// -----------------------------------------
	
	float rcut2 = rcut*rcut;
	
	int numNabors[nCells];
	for (int c=0; c<nCells; c++) {
		numNabors[c] = 0;
	}
			
	for (int c=0; c<nCells; c++) {
		cellsH[c].intrain = false;
	}
	
	// -----------------------------------------
	// Loop over the capsule pairs to determine
	// how many nabors a capsule has in the 
	// specified alignment (x-dir) zone.  
	// -----------------------------------------
	
	for (int c=0; c<nCells; c++) {
		
		float ix = cellsH[c].com.x;
		float iy = cellsH[c].com.y;
		float iz = cellsH[c].com.z;
		
		for (int d=0; d<nCells; d++) {
			
			if (d==c) continue;
			
			float jx = cellsH[d].com.x;
			float jy = cellsH[d].com.y;
			float jz = cellsH[d].com.z;
			
			float dx = ix - jx;
			float dy = iy - jy;
			float dz = iz - jz;
			dx -= roundf(dx/Box.x)*Box.x;
			dy -= roundf(dy/Box.y)*Box.y;
			float r2 = dx*dx + dy*dy + dz*dz;
			
			if (r2 < rcut2) {
				float theta = atan2(dy,dx)*180.0/M_PI; 
				// check if 'c' is in front of 'd':
				if (theta < thetacut and theta > -thetacut) {
					numNabors[c]++;
					//cellsH[c].intrain = true;
					//cellsH[d].intrain = true;
				}
				// check if 'd' is in front of 'c':
				if (theta > (180.0-thetacut) or theta < (-180+thetacut)) { 
					numNabors[c]++;
					//cellsH[c].intrain = true;
					//cellsH[d].intrain = true;
				}					
			}			
		}		
	}
	
	// -----------------------------------------
	// Loop over the capsules to see which are 
	// in trains.  
	// -----------------------------------------
	
	for (int c=0; c<nCells; c++) {
		
		// if a capsule has 2 nabors (front & back),
		// it is in a train:
		if (numNabors[c] >= 2) {
			cellsH[c].intrain = true;
			continue;
		}
		
		// if a capsule has 1 nabor, check if that
		// nabor has 2 nabors, if so then include
		// it in the train:
		if (numNabors[c] == 1) {
			
			float ix = cellsH[c].com.x;
			float iy = cellsH[c].com.y;
			float iz = cellsH[c].com.z;
		
			for (int d=0; d<nCells; d++) {
			
				if (d==c) continue;
			
				float jx = cellsH[d].com.x;
				float jy = cellsH[d].com.y;
				float jz = cellsH[d].com.z;
			
				float dx = ix - jx;
				float dy = iy - jy;
				float dz = iz - jz;
				dx -= roundf(dx/Box.x)*Box.x;
				dy -= roundf(dy/Box.y)*Box.y;
				float r2 = dx*dx + dy*dy + dz*dz;
			
				if (r2 < rcut2) {
					float theta = atan2(dy,dx)*180.0/M_PI; 
					// check if 'c' is in front of 'd':
					if (theta < thetacut and theta > -thetacut) {
						if (numNabors[d] >= 2) cellsH[c].intrain = true;
					}
					// check if 'd' is in front of 'c':
					if (theta > (180.0-thetacut) or theta < (-180+thetacut)) { 
						if (numNabors[d] >= 2) cellsH[c].intrain = true;
					}					
				}			
			}				
		}		
	}
	
	// -----------------------------------------
	// Find fraction of cells in trains:
	// -----------------------------------------
	
	int nCellsTrain = 0;
	for (int c=0; c<nCells; c++) {
		if (cellsH[c].intrain == true) nCellsTrain++;
	}
	float fracTrain = float(nCellsTrain)/float(nCells);
	
	// -----------------------------------------
	// Print results:
	// -----------------------------------------
	
	outfile << fixed << setprecision(4) << step << "  " << fracTrain << endl;
	outfile.close();
	
	// -----------------------------------------
	// Define the file location and name:
	// -----------------------------------------
	
	ofstream outfile2;
	std::stringstream filenamecombine2;
	filenamecombine2 << "vtkoutput/" << "capsule_dynamics.dat";
	string filename2 = filenamecombine2.str();
	outfile2.open(filename2.c_str(), ios::out | ios::app);
	
	// -----------------------------------------
	// calculate capsule velocities & print:
	// -----------------------------------------
	
	for (int i=0; i<nCells; i++) {
		float3 cellv = make_float3(0.0,0.0,0.0);
		for (int n=0; n<nNodesPerCell; n++) {
			int indx = n + i*nNodesPerCell;
			cellv += vH[indx];
		}
		cellv /= nNodesPerCell;
		outfile2 << fixed << setprecision(6) << cellsH[i].rad << "  " << cellsH[i].Ca << "  "
			                                 << cellv.x << "  " << cellv.y << "  " << cellv.z << "  " 
										     << cellsH[i].intrain << endl;
	}
	outfile2.close();
		
}



