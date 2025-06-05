
# ifndef CLASS_FIBERS_IBM3D_H
# define CLASS_FIBERS_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "../../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "kernels_fibers_ibm3D.cuh"
# include "data_structs/fiber_data.h"
# include "data_structs/neighbor_bins_data.h"
# include <cuda.h>
# include <curand.h>
# include <curand_kernel.h>
# include <string>


class class_fibers_ibm3D {
	
	public:  // treat like a struct
	
	// data:
	int nBeads;
	int nEdges;
	int nFibers;
	int nBeadsPerFiber;
	int nEdgesPerFiber;	
	int3 N;
	float gam;
	float dS;
	float dt;
	float repA;
	float repD;
	float beadFmax;
	float fricBead;
	float3 Box;
	int3 pbcFlag;
	bool binsFlag;
	bindata bins;
			
	// host arrays:
	beadfiber* beadsH;
	edgefiber* edgesH;
	fiber* fibersH;
		
	// device arrays:
	beadfiber* beads;
	edgefiber* edges;
	fiber* fibers;
	curandState* rngState;
	
	// methods:
	class_fibers_ibm3D();
	~class_fibers_ibm3D();
	void allocate();
	void deallocate();	
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void create_first_fiber();
	void set_pbcFlag(int,int,int);
	void set_gamma(float);
	void set_fibers_radii(float);
	int get_max_array_size();
	void assign_fiberIDs_to_beads();
	void duplicate_fibers();	
	void shift_bead_positions(int,float,float,float);
	void rotate_and_shift_bead_positions(int,float,float,float);
	void randomize_fibers(float);
	void randomize_fibers_xdir_alligned(float);
	float calc_separation_pbc(float3,float3);
	void compute_wall_forces(int,int);
	void zero_bead_velocities_forces(int,int);
	void zero_bead_forces(int,int);
	void enforce_max_bead_force(int,int);
	void unwrap_bead_coordinates(int,int);
	void wrap_bead_coordinates(int,int);
	void build_binMap(int,int);
	void reset_bin_lists(int,int);
	void build_bin_lists(int,int);
	void nonbonded_bead_interactions(int,int);
	void wall_forces_ydir(int,int);
	void wall_forces_zdir(int,int);
	void wall_forces_ydir_zdir(int,int);
	void write_output(std::string,int);
	void unwrap_bead_coordinates();
	
};

# endif  // CLASS_FIBERS_IBM3D_H