
# ifndef CLASS_SCSP_D3Q19_H
# define CLASS_SCSP_D3Q19_H

# include "../init/lattice_builders_D3Q19.cuh"
# include "../init/bounding_box_nList_construct_D3Q19.cuh"
# include "../init/stream_index_builder_D3Q19.cuh"
# include "../iolets/boundary_condition_iolet.cuh"
# include "../inout/inside_hemisphere_D3Q19.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../IO/read_lattice_geometry.cuh"
# include "../../IBM/3D/kernels_ibm3D.cuh"
# include "kernels_scsp_D3Q19.cuh"
# include <cuda.h>
# include <string>

class class_scsp_D3Q19 {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int Nx,Ny,Nz;
	int numIolets;
	int xvelAveCnt;
	float nu;
	float dt;
	float relViscAve;
	bool forceFlag;
	bool velIBFlag;
	bool inoutFlag;
	bool solidFlag;
	bool xyzFlag;
		
	// host arrays:
	float* uH;
	float* vH;
	float* wH;
	float* rH;
	int* xH;
	int* yH;
	int* zH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	int* solidH;
	int* inoutH;
	iolet* ioletsH;
	float* xvelAve;
	
	// device arrays:
	float* u;
	float* v;
	float* w;
	float* r;
	float* f1;
	float* f2;
	float* Fx;
	float* Fy;
	float* Fz;
	float* uIBvox;
	float* vIBvox;
	float* wIBvox;
	float* weights;
	int* x;
	int* y;
	int* z;
	int* voxelType;
	int* streamIndex;
	int* solid;
	int* inout;
	iolet* iolets;
	
public:

	class_scsp_D3Q19();
	~class_scsp_D3Q19();
	void allocate();
	void deallocate();
	void allocate_forces();
	void allocate_IB_velocities();
	void allocate_voxel_positions();
	void allocate_inout();
	void allocate_solid();
	void memcopy_host_to_device();
	void memcopy_host_to_device_iolets();
	void memcopy_host_to_device_solid();
	void memcopy_device_to_host();
	void memcopy_device_to_host_inout();	
	void create_lattice_box();
	void create_lattice_box_periodic();
	void create_lattice_box_shear();
	void create_lattice_box_slit();
	void create_lattice_box_channel();
	void create_lattice_box_periodic_solid_walls();
	void bounding_box_nList_construct();
	void stream_index_push();
	void stream_index_pull();
	void read_iolet_info(int,const char*);
	void setNu(float);
	void setU(int,float);
	void setV(int,float);
	void setW(int,float);
	void setR(int,float);
	void setS(int,int);
	void setVoxelType(int,int);
	void setIoletU(int,float);
	void setIoletV(int,float);
	void setIoletW(int,float);
	void setIoletR(int,float);
	void setIoletType(int,int);
	float getU(int);
	float getV(int);
	float getW(int);
	float getR(int);
	int getX(int);
	int getY(int);
	int getZ(int);
	int getNList(int);
	void initial_equilibrium(int,int);
	void stream_collide_save(int,int,bool);	
	void stream_collide_save_forcing(int,int);
	void stream_collide_save_forcing_solid(int,int);
	void stream_collide_save_IBforcing(int,int);
	void set_boundary_shear_velocity(float,float,int,int);
	void set_boundary_slit_velocity(float,int,int);
	void set_channel_wall_velocity(float,int,int);
	void set_boundary_slit_density(int,int);
	void set_boundary_duct_density(int,int);
	void zero_forces(int,int);
	void zero_forces_with_IBM(int,int);	
	void add_body_force(float,float,float,int,int);
	void add_body_force_with_solid(float,float,float,int,int);	
	void extrapolate_velocity_from_IBM(int,int,float3*,float3*,int);
	void interpolate_velocity_to_IBM(int,int,float3*,float3*,int);
	void extrapolate_forces_from_IBM(int,int,float3*,float3*,int);
	void viscous_force_IBM_LBM(int,int,float,float3*,float3*,float3*,int);
	void inside_hemisphere(int,int);
	void calculate_flow_rate_xdir(std::string,int);
	void calculate_relative_viscosity(std::string,float,int);
	void print_flow_rate_xdir(std::string,int);
	void read_lattice_geometry(int);
	void vtk_structured_output_ruvw(std::string,int,int,int,int);
	void vtk_structured_output_iuvw_inout(std::string,int,int,int,int);
	void vtk_structured_output_iuvw_vtype(std::string,int,int,int,int);
	void vtk_polydata_output_ruvw(std::string,int);

};

# endif  // CLASS_SCSP_D3Q19_H