
# ifndef CLASS_MEMBRANE_IBM3D_H
# define CLASS_MEMBRANE_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "kernels_ibm3D.cuh"
# include "kernels_membrane_ibm3D.cuh"
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
	float ks,kb,ka,kv;	
			
	// host arrays:
	float3* rH;	
	triangle* facesH;
	edge* edgesH;
	cell* cellsH;
		
	// device arrays:
	float3* r;
	float3* v;
	float3* f;
	triangle* faces;
	edge* edges;
	cell* cells;	
	
	// methods:
	class_membrane_ibm3D();
	~class_membrane_ibm3D();
	void allocate();
	void deallocate();	
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void read_ibm_information(std::string);
	void shift_node_positions(float,float,float);
	void write_output(std::string,int);
	void update_node_positions(int,int);
	void interpolate_velocity(float*,float*,float*,int,int,int,int);
	void extrapolate_force(float*,float*,float*,int,int,int,int);
	void compute_node_forces(int,int);


};

# endif  // CLASS_MEMBRANE_IBM3D_H