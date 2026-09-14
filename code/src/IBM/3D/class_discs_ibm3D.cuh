
# ifndef CLASS_DISCS_IBM3D_H
# define CLASS_DISCS_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "../../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "kernels_discs_ibm3D.cuh"
# include "data_structs/disc_data.h"
# include "data_structs/tensor.h"
# include "data_structs/neighbor_bins_data.h"
# include "data_structs/radix_sort_data.h"
# include <cuda.h>
# include <curand.h>
# include <curand_kernel.h>
# include <string>


class class_discs_ibm3D {
	
	public:  // treat like a struct
	
	// data:
	int nBeads;
	int nDiscs;
	int nBeadsPerDisc;
	int3 N;
	float dt;
	float repA;
	float repD;
	float lubforceMax;
	float repWall;
	float fricWall;
	float beadFmax;
	float discFmax;
	float discTmax;
	float gam;
	float fricT;
	float fricR;
	float chRad;
	float3 Box;
	int3 pbcFlag;
	bool binsFlag;
	bindata bins;
	radixdata radix;	
				
	// host arrays:
	beaddisc* beadsH;
	disc* discsH;
		
	// device arrays:
	beaddisc* beads;
	disc* discs;
	curandState* states;
	
	// methods:
	class_discs_ibm3D();
	~class_discs_ibm3D();
	void allocate();
	void deallocate();	
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void create_first_disc();
	void set_pbcFlag(int,int,int);	
	void set_discs_radii(float);
	void set_disc_radius(int,float);
	void set_discs_half_thickness(float);
	void set_disc_half_thickness(int,float);
	void set_discs_types(int);
	void set_disc_type(int,int);
	void set_aspect_ratio(float);
	void set_mobility_coefficients(float,float,float);
	int get_max_array_size();
	void assign_discIDs_to_beads();
	void duplicate_discs();	
	void shift_bead_positions(int,float,float,float);
	void rotate_and_shift_bead_positions(int,float,float,float);
	void rotate_and_shift_bead_positions(int,float,float,float,float,float,float);
	void rotate_and_shift_bead_positions_using_orientation_vector(int);
	void randomize_discs(float);
	void randomize_discs_cylinder(float,float);
	void randomize_discs_duct();
	void randomize_discs_nozzle(float,float,float,float);
	void randomize_discs_nozzle_backfill(float,float,float,float);
	void randomize_rods_xdir_alligned_cylinder(float,float,float,float);
	void semi_randomize_rods_xdir_alligned_cylinder(float,float,float,float);
	float calc_separation_pbc(float3,float3);
	void stepIBM_Euler(class_scsp_D3Q19&,int,int);
	void stepIBM_Euler_cylindrical_channel(class_scsp_D3Q19&,float,int,int);
	void stepIBM_Euler_cylindrical_channel_radix(class_scsp_D3Q19&,float,int,int);	
	void init_rand_kernel(int,int);
	void zero_disc_forces_torques_moments(int,int);
	void set_disc_position_orientation(int,int);
	void update_bead_position_discs(int,int);
	void update_bead_velocity_discs(int,int);
	void update_disc_position_orientation_fluid(int,int);
	void update_disc_position_orientation_no_fluid(int,int);
	void assign_velocity_to_backfill_discs(float,int,int);
	void move_disc_back_to_inlet_random(float,float,float,int,int);
	void zero_bead_forces(int,int);
	void enforce_max_bead_force(int,int);
	void enforce_max_disc_force_torque(int,int);
	void sum_disc_forces_torques_moments(int,int);
	void add_gravity_force_to_beads(float,int,int);
	void unwrap_bead_coordinates(int,int);
	void wrap_bead_coordinates(int,int);	
	void add_xdir_force_to_beads(int,int,float);
	void compute_wall_forces(int,int);
	void build_binMap(int,int);
	void build_cellMap_radix(int,int);
	void reset_bin_lists(int,int);
	void build_bin_lists(int,int);
	void nonbonded_bead_interactions(int,int);
	void nonbonded_bead_interactions_with_friction(int,int);
	void radix_nonbonded_bead_interactions_with_friction(int,int);
	void radix_reorder_beads(int,int);
	void wall_forces_ydir(int,int);
	void wall_forces_zdir(int,int);
	void wall_forces_ydir_zdir(int,int);
	void compute_wall_forces_cylinder(float,int,int);
	void compute_wall_forces_nozzle(float,float,float,int,int);
	void push_beads_inside_sphere(float,float,float,float,int,int);
	void push_discs_inside_cylinder(float,int,int);
	void push_discs_inside_duct(int,int);
	void push_discs_inside_slit(int,int);
	void push_discs_inside_nozzle(float,float,float,int,int);
	void write_output(std::string,int);
	void unwrap_bead_coordinates();
	void orientation_in_cylindrical_channel(int);
	void write_vtk_discs_beads();
	
};

# endif  // CLASS_DISCS_IBM3D_H