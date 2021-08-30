
# ifndef CLASS_MEMBRANE_IBM3D_H
# define CLASS_MEMBRANE_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "kernels_ibm3D.cuh"
# include "kernels_membrane_ibm3D.cuh"
# include "kernels_nonbonded_ibm3D.cuh"
# include "membrane_data.h"
# include <cuda.h>
# include <string>


class class_membrane_ibm3D {
	
	public:  // treat like a struct
	
	// data:
	int nNodes;
	int nFaces; 
	int nEdges;
	int nCells;
	int nNodesPerCell;
	int nFacesPerCell; 
	int nEdgesPerCell;
	int binMax;
	int nnbins;
	int nBins;
	int3 N;
	int3 numBins;
	float sizeBins;
	float ks,kb,ka,kag,kv;
	float dt;
	float3 Box;
	bool binsFlag;
			
	// host arrays:
	float3* rH;	
	triangle* facesH;
	edge* edgesH;
	cell* cellsH;
	int* cellIDsH;
		
	// device arrays:
	float3* r;
	float3* v;
	float3* f;
	triangle* faces;
	edge* edges;
	cell* cells;
	int* binMembers;
	int* binOccupancy;
	int* binMap;
	int* cellIDs;
	
	// methods:
	class_membrane_ibm3D();
	~class_membrane_ibm3D();
	void allocate();
	void deallocate();	
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void read_ibm_information(std::string);
	void assign_refNode_to_cells();
	void assign_cellIDs_to_nodes();
	void duplicate_cells();	
	void shift_node_positions(int,float,float,float);
	void rest_geometries(int,int);
	void rest_geometries_skalak(int,int);
	void shrink_and_randomize_cells(float,float,float);
	float calc_separation_pbc(float3,float3);
	void write_output(std::string,int);
	void write_output_long(std::string,int);
	void update_node_positions(int,int);
	void update_node_positions_dt(int,int);
	void relax_node_positions(int,float,float,int,int);
	void update_node_positions_vacuum(float,int,int);
	void interpolate_velocity(float*,float*,float*,int,int);
	void extrapolate_force(float*,float*,float*,int,int);	
	void build_binMap(int,int);
	void reset_bin_lists(int,int);
	void build_bin_lists(int,int);
	void nonbonded_node_interactions(int,int);	
	void compute_node_forces(int,int);
	void compute_node_forces_skalak(int,int);
	void wall_forces_ydir(int,int);
	void change_cell_volume(float,int,int);
	void scale_equilibrium_cell_size(float,int,int); 
	void scale_edge_lengths(float,int,int);
	void unwrap_node_coordinates();
	
};

# endif  // CLASS_MEMBRANE_IBM3D_H