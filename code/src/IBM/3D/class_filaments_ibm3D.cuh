
# ifndef CLASS_FILAMENTS_IBM3D_H
# define CLASS_FILAMENTS_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "../../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "kernels_filaments_ibm3D.cuh"
# include "filament_data.h"
# include <cuda.h>
# include <string>


class class_filaments_ibm3D {
	
	public:  // treat like a struct
	
	// data:
	int nBeads;
	int nEdges;
	int nFilams;
	int nBeadsPerFilam;
	int nEdgesPerFilam;
	int binMax;
	int nnbins;
	int nBins;
	int3 N;
	int3 numBins;
	float sizeBins;
	float ks,kb;
	float dt;
	float repA;
	float repD;
	float beadFmax;
	float gam;
	float3 Box;
	int3 pbcFlag;
	bool binsFlag;
	std::string ibmUpdate;
			
	// host arrays:
	bead* beadsH;
	edgefilam* edgesH;
	filament* filamsH;
		
	// device arrays:
	bead* beads;
	edgefilam* edges;
	filament* filams;
	int* binMembers;
	int* binOccupancy;
	int* binMap;
	int* cellIDs;
	
	// methods:
	class_filaments_ibm3D();
	~class_filaments_ibm3D();
	void allocate();
	void deallocate();	
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void read_ibm_information(std::string);
	void set_pbcFlag(int,int,int);
	void set_ks(float);
	void set_kb(float);
	void set_filams_mechanical_props(float,float);
	void set_filam_mechanical_props(int,float,float);
	void set_filams_radii(float);
	void set_filam_radius(int,float);
	void set_filams_types(int);
	void set_filam_type(int,int);
	int get_max_array_size();
	void assign_filamIDs_to_beads();
	void duplicate_filaments();	
	void shift_bead_positions(int,float,float,float);
	void rotate_and_shift_bead_positions(int,float,float,float);
	void randomize_filaments(float);
	float calc_separation_pbc(float3,float3);
	void write_output(std::string,int);
	void write_output_long(std::string,int);
	void update_bead_positions_verlet_1(int,int);
	void update_bead_positions_verlet_2(int,int);
	void zero_bead_velocities_forces(int,int);
	void enforce_max_bead_force(int,int);
	void add_drag_force_to_beads(float,int,int);
	void add_xdir_force_to_beads(int,int,float);
	void compute_wall_forces(int,int);
	void stepIBM(class_scsp_D3Q19&,int,int);
	void stepIBM_no_fluid(int,bool,int,int);
	void build_binMap(int,int);
	void reset_bin_lists(int,int);
	void build_bin_lists(int,int);
	void nonbonded_bead_interactions(int,int);	
	void compute_bead_forces(int,int);
	void wall_forces_ydir(int,int);
	void wall_forces_zdir(int,int);
	void wall_forces_ydir_zdir(int,int);
	void unwrap_bead_coordinates();
	void output_filament_data();
	
};

# endif  // CLASS_FILAMENTS_IBM3D_H