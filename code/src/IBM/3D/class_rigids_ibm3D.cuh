
# ifndef CLASS_RIGIDS_IBM3D_H
# define CLASS_RIGIDS_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "../../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "kernels_rigids_ibm3D.cuh"
# include "kernels_nonbonded_ibm3D.cuh"
# include "data_structs/rigid_data.h"
# include "data_structs/neighbor_bins_data.h"
# include <cuda.h>
# include <string>


class class_rigids_ibm3D {
	
	public:  // treat like a struct
	
	// scalars:
	int nNodes;
	int nBodies;
	int nNodesPerBody;
	int3 N;
	float dt;
	float repA;
	float repD;
	float repFmax;
	float bodyFmax;
	float bodyTmax;
	float chRad;
	float3 Box;
	int3 pbcFlag;
	bool binsFlag;
	std::string channelShape;
	bindata bins;
			
	// host arrays:
	rigidnode* nodesH;
	rigid* bodiesH;
		
	// device arrays:
	rigidnode* nodes;
	rigid* bodies;
	
	// methods:
	class_rigids_ibm3D();
	~class_rigids_ibm3D();
	void allocate();
	void allocate_bin_arrays();
	void deallocate();	
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void read_ibm_information(std::string);
	void initialize_bins();
	void set_pbcFlag(int,int,int);
	void set_cells_types(int);
	void set_cell_type(int,int);
	int get_max_array_size();
	void assign_refNode_to_cells();
	void assign_cellIDs_to_nodes();
	void duplicate_cells();
	void relative_node_position_versus_com();
	void cell_mass_moment_of_inertia_cylinder(float,float);
	void shift_node_positions(int,float,float,float);
	void rotate_and_shift_node_positions(int,float,float,float);
	void rotate_and_shift_node_positions(int,float,float,float,float,float,float);
	void randomize_cells(float);
	void randomize_capsules_xdir_alligned_cylinder(float,float,float,float);
	void semi_randomize_capsules_xdir_alligned_cylinder(float,float,float,float);
	float calc_separation_pbc(float3,float3);
	void write_output(std::string,int);
	void write_output_cylinders(std::string,int);
	void write_output_long(std::string,int);
	void update_node_positions(int,int);
	void compute_wall_forces(int,int);
	void stepIBM(class_scsp_D3Q19&,int,int);
	void zero_node_forces(int,int);
	void zero_rigid_body_forces_torques(int,int);
	void enforce_rigid_body_max_forces_torques(int,int);
	void update_node_positions_velocities(int,int);
	void update_rigid_body(int,int);
	void sum_rigid_forces_torques(int,int);
	void unwrap_node_coordinates(int,int);
	void wrap_node_coordinates(int,int);
	void wrap_rigid_body_coordinates(int,int);	
	void build_binMap(int,int);
	void reset_bin_lists(int,int);
	void build_bin_lists(int,int);
	void nonbonded_node_interactions(int,int);
	void nonbonded_node_lubrication_interactions(float,float,int,int);
	void wall_forces_ydir(int,int);
	void wall_forces_zdir(int,int);
	void wall_forces_ydir_zdir(int,int);
	void wall_forces_cylinder(float,int,int);
	void wall_lubrication_forces_cylinder(float,float,float,int,int);
	void unwrap_node_coordinates();
	void output_capsule_data();
	void capsule_orientation_cylinders(int,int);
	bool cylinder_overlap(float3,float3,float,float,float);
	
};

# endif  // CLASS_RIGIDS_IBM3D_H