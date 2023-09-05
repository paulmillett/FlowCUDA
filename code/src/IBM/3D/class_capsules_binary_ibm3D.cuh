
# ifndef CLASS_CAPSULES_BINARY_IBM3D_H
# define CLASS_CAPSULES_BINARY_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "../../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "class_capsule_ibm3D.cuh"
# include "kernels_ibm3D.cuh"
# include "kernels_capsule_ibm3D.cuh"
# include "kernels_nonbonded_ibm3D.cuh"
# include "membrane_data.h"
# include <cuda.h>
# include <string>


// ---------------------------------------------------------------------------
// Note: this class inherits from "class_capsule_ibm3D" and should
//       be used for binary suspensions (i.e. two different types of capsules)
// ---------------------------------------------------------------------------


class class_capsules_binary_ibm3D : public class_capsule_ibm3D {
	
	public:  // treat like a struct
	
	// data:
	int nCells1;
	int nCells2;
	int nNodesPerCell1;
	int nFacesPerCell1;	
	int nEdgesPerCell1;
	int nNodesPerCell2;
	int nFacesPerCell2;	
	int nEdgesPerCell2;
	float a1;
	float a2;
	
	// methods:
	class_capsules_binary_ibm3D();
	~class_capsules_binary_ibm3D();
	void read_ibm_information(std::string,std::string);
	void read_ibm_file_long(std::string,int,int,int,int,int,int,int);
	//void assign_refNode_to_cells();
	//void assign_cellIDs_to_nodes();
	void duplicate_cells();	
	void set_cells_radii_binary();
	void set_cells_types_binary();
	//void shrink_and_randomize_cells(float,float,float);
	//void shift_node_positions(int,float,float,float);
	//void rotate_and_shift_node_positions(int,float,float,float);
	
};

# endif  // CLASS_CAPSULES_BINARY_IBM3D_H