# ifndef READ_IBM_INFORMATION_H
# define READ_IBM_INFORMATION_H
# include <string>
# include "../IBM/3D/data_structs/cell_data.h"
# include "../IBM/3D/data_structs/rigid_data.h"

void read_ibm_information(std::string,int,int,float3*,int*,int*,int*);

void read_ibm_information_long(std::string,int,int,int,node*,triangle*,edge*);

void read_ibm_information_just_nodes(std::string,int,rigidnode*);

# endif  // READ_IBM_INFORMATION_H