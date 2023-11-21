
# ifndef CLASS_CAPSULES_IBM3D_H
# define CLASS_CAPSULES_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "../../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "kernels_ibm3D.cuh"
# include "kernels_capsules_ibm3D.cuh"
# include "kernels_nonbonded_ibm3D.cuh"
# include "data_structs/membrane_data.h"
# include "data_structs/neighbor_bins_data.h"
# include <cuda.h>
# include <string>


class class_capsules_ibm3D {
	
	public:  // treat like a struct
	
	// scalars:
	int nNodes;
	int nFaces; 
	int nEdges;
	int nCells;
	int nNodesPerCell;
	int nFacesPerCell; 
	int nEdgesPerCell;	
	int3 N;
	float ks,kb,ka,kag,kv;
	float C;
	float dt;
	float repA;
	float repD;
	float repFmax;
	float nodeFmax;
	float gam;
	float3 Box;
	int3 pbcFlag;
	bool binsFlag;
	std::string ibmUpdate;
	std::string membraneModel;
	bindata bins;
			
	// host arrays:
	node* nodesH;
	triangle* facesH;
	edge* edgesH;
	cell* cellsH;
		
	// device arrays:
	node* nodes;
	triangle* faces;
	edge* edges;
	cell* cells;	
	
	// methods:
	class_capsules_ibm3D();
	~class_capsules_ibm3D();
	void allocate();
	void allocate_bin_arrays();
	void deallocate();	
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void read_ibm_information(std::string);
	void initialize_bins();
	void set_pbcFlag(int,int,int);
	void set_ks(float);
	void set_ka(float);
	void set_kb(float);
	void set_kv(float);
	void set_kag(float);
	void set_C(float);
	void set_cells_mechanical_props(float,float,float,float,float);
	void set_cell_mechanical_props(int,float,float,float,float,float);
	void set_cells_radii(float);
	void set_cell_radius(int,float);
	void set_cells_types(int);
	void set_cell_type(int,int);
	int get_max_array_size();
	void calculate_cell_membrane_props(float,float,float,float,float,float,
		                               float,float,float,std::string); 
	void rescale_cell_radii(float,float,std::string);
	void resize_cell_radius(int,float);
	void assign_refNode_to_cells();
	void assign_cellIDs_to_nodes();
	void duplicate_cells();	
	void single_file_cells(int,int,int,float,float);
	void shift_node_positions(int,float,float,float);
	void rotate_and_shift_node_positions(int,float,float,float);
	void rest_geometries(int,int);
	void rest_geometries_spring(int,int);
	void rest_geometries_skalak(int,int);
	void rest_geometries_FENE(int,int);
	void randomize_cells(float);
	void shrink_and_randomize_cells(float,float,float);
	void randomize_cells_above_plane(float,float,float,float);
	float calc_separation_pbc(float3,float3);
	void write_output(std::string,int);
	void write_output_long(std::string,int);
	void update_node_positions(int,int);
	void update_node_positions_dt(int,int);
	void update_node_positions_verlet_1(int,int);
	void update_node_positions_verlet_2(int,int);
	void update_node_positions_verlet_1_drag(int,int);
	void update_node_positions_verlet_2_drag(int,int);
	void zero_velocities_forces(int,int);
	void enforce_max_node_force(int,int);
	void add_drag_force_to_nodes(float,int,int);
	void add_xdir_force_to_nodes(int,int,float);
	void relax_node_positions_spring(int,float,float,int,int);
	void relax_node_positions_skalak(int,float,float,int,int);
	void compute_wall_forces(int,int);
	void stepIBM(class_scsp_D3Q19&,int,int);
	void stepIBM_no_fluid(int,bool,int,int);
	void update_node_positions_vacuum(float,int,int);
	void interpolate_velocity(float*,float*,float*,int,int);
	void extrapolate_force(float*,float*,float*,int,int);	
	void build_binMap(int,int);
	void reset_bin_lists(int,int);
	void build_bin_lists(int,int);
	void nonbonded_node_interactions(int,int);
	void nonbonded_node_bead_interactions(bead*,bindata,int,int);	
	void compute_node_forces(int,int);
	void compute_node_forces_spring(int,int);
	void compute_node_forces_skalak(int,int);
	void compute_node_forces_FENE(float,int,int);
	void wall_forces_ydir(int,int);
	void wall_forces_zdir(int,int);
	void wall_forces_ydir_zdir(int,int);
	void change_cell_volume(float,int,int);
	void scale_equilibrium_cell_size(float,int,int); 
	void scale_edge_lengths(float,int,int);
	void unwrap_node_coordinates();
	void set_edge_rest_angles(float,int,int);
	void output_capsule_data();
	void capsule_geometry_analysis(int);
	void subexpressions(const float,const float,const float,float&,float&,float&,float&,float&,float&);
	void capsule_train_fraction(float,float,int);
	
};

# endif  // CLASS_CAPSULES_IBM3D_H