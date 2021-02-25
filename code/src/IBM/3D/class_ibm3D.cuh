
# ifndef CLASS_IBM3D_H
# define CLASS_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "kernels_ibm3D.cuh"
# include <cuda.h>
# include <string>
# include "../../Utils/helper_math.h"

class class_ibm3D {
	
	public:   // treat this like a struct
	
	// data:
	int nNodes;
	int nFaces; 
	float k_stiff;	
	bool facesFlag;
		
	// host arrays:
	float3* rH;	
	float3* rH_start;
	float3* rH_end;
	int* faceV1;
	int* faceV2;
	int* faceV3;
		
	// device arrays:
	float3* r;
	float3* r_start;
	float3* r_end;
	float3* v;
	float3* f;		
	
	// methods:
	class_ibm3D();
	~class_ibm3D();
	void allocate();
	void deallocate();	
	void allocate_faces();
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void read_ibm_start_positions(std::string);
	void read_ibm_end_positions(std::string);
	void initialize_positions_to_start();
	void shift_start_positions(float,float,float);
	void shift_end_positions(float,float,float);
	void write_output(std::string,int);
	void update_node_positions(int,int,int,int);	

};

# endif  // CLASS_IBM3D_H