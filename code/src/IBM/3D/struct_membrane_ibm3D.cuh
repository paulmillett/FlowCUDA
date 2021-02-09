
# ifndef STRUCT_MEMBRANE_IBM3D_H
# define STRUCT_MEMBRANE_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "kernels_ibm3D.cuh"
# include <cuda.h>
# include <string>
# include "data.h"
# include "../../Utils/helper_math.h"

struct struct_membrane_ibm3D {
	
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
	struct_membrane_ibm3D();
	~struct_membrane_ibm3D();
	void allocate();
	void deallocate();	
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void read_ibm_information(std::string);
	void shift_node_positions(float,float,float);
	void write_output(std::string,int);
	void update_node_positions(int,int);	

};

# endif  // STRUCT_MEMBRANE_IBM3D_H