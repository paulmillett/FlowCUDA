# ifndef READ_IBM_INFORMATION_H
# define READ_IBM_INFORMATION_H
# include <string>
# include "../IBM/3D/data_structs/membrane_data.h"

void read_ibm_information(std::string,int,int,float3*,int*,int*,int*);

void read_ibm_information_long(std::string,int,int,int,node*,triangle*,edge*);

# endif  // READ_IBM_INFORMATION_H