
# ifndef CLASS_RODS_IBM3D_H
# define CLASS_RODS_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "../../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "kernels_filaments_ibm3D.cuh"
# include "kernels_rods_ibm3D.cuh"
# include "class_capsules_ibm3D.cuh"
# include "data_structs/rod_data.h"
# include "data_structs/neighbor_bins_data.h"
# include <cuda.h>
# include <curand.h>
# include <curand_kernel.h>
# include <string>


class class_rods_ibm3D {
	
	public:  // treat like a struct
	
	// data:
	int nBeads;
	int nRods;
	int nBeadsPerRod;
	int3 N;
	float dt;
	float repA;
	float repD;
	float repA_bn;
	float repD_bn;
	float beadFmax;
	float rodFmax;
	float rodTmax;
	float gam;
	float fricT;
	float fricR;
	float L0;
	float kT;
	float noisekT;
	float noisekTforce;
	float noisekTtorque;
	float3 Box;
	int3 pbcFlag;
	bool binsFlag;
	bindata bins;
			
	// host arrays:
	beadrod* beadsH;
	rod* rodsH;
		
	// device arrays:
	beadrod* beads;
	rod* rods;
	curandState* rngState;
	
	// methods:
	class_rods_ibm3D();
	~class_rods_ibm3D();
	void allocate();
	void deallocate();	
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void create_first_rod();
	void set_pbcFlag(int,int,int);	
	void set_fp(float);
	void set_up(float);
	void set_rods_radii(float);
	void set_rod_radius(int,float);
	void set_rods_types(int);
	void set_rod_type(int,int);
	void set_friction_coefficient_translational(float);
	void set_friction_coefficient_rotational(float);
	void set_noise_strength_translational(float);
	void set_noise_strength_rotational(float);
	int get_max_array_size();
	void assign_rodIDs_to_beads();
	void duplicate_rods();	
	void shift_bead_positions(int,float,float,float);
	void rotate_and_shift_bead_positions(int,float,float,float);
	void randomize_rods(float);
	void randomize_rods_inside_sphere(float,float,float,float,float);
	float calc_separation_pbc(float3,float3);
	void stepIBM_Euler_no_fluid(int,int);
	void stepIBM_Euler(class_scsp_D3Q19&,int,int);
	void stepIBM_capsules_rods_no_fluid(class_capsules_ibm3D&,int,int);
	void stepIBM_capsules_rods(class_capsules_ibm3D&,class_scsp_D3Q19&,int,int);
	void stepIBM_push_into_sphere(int,float,float,float,float,int,int);
	void initialize_cuRand(int,int);
	void zero_rod_forces_torques_moments(int,int);
	void set_rod_position_orientation(int,int);
	void update_bead_position_rods(int,int);
	void update_rod_position_orientation(int,int);
	void update_rod_position_orientation_fluid(int,int);
	void zero_bead_forces(int,int);
	void enforce_max_bead_force(int,int);
	void enforce_max_rod_force_torque(int,int);
	void compute_bead_thermal_force(int,int);
	void compute_thermal_force_torque_rod(int,int);
	void compute_rod_propulsion_force(int,int);
	void sum_rod_forces_torques_moments(int,int);
	void unwrap_bead_coordinates(int,int);
	void wrap_bead_coordinates(int,int);	
	void add_xdir_force_to_beads(int,int,float);
	void compute_wall_forces(int,int);
	void build_binMap(int,int);
	void reset_bin_lists(int,int);
	void build_bin_lists(int,int);
	void nonbonded_bead_interactions(int,int);
	void nonbonded_bead_node_interactions(class_capsules_ibm3D&,int,int);
	void wall_forces_ydir(int,int);
	void wall_forces_zdir(int,int);
	void wall_forces_ydir_zdir(int,int);
	void push_beads_inside_sphere(float,float,float,float,int,int);
	void write_output(std::string,int);
	void unwrap_bead_coordinates();
	
};

# endif  // CLASS_RODS_IBM3D_H