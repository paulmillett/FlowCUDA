# ifndef LATTICE_BUILDERS_D3Q19_H
# define LATTICE_BUILDERS_D3Q19_H

void build_box_lattice_D3Q19(int,int,int,int,int*,int*);
void build_box_lattice_shear_D3Q19(int,int,int,int,int*,int*);
void build_box_lattice_slit_D3Q19(int,int,int,int,int*,int*);
void build_box_lattice_channel_D3Q19(int,int,int,int,int*,int*);
void build_box_lattice_D3Q19(int,int,int,int,int,int,int,int,int,int,int,int*,int*);
void build_box_lattice_solid_walls_D3Q19(int,int,int,int,int*,int*,int*);
int voxel_index(int,int,int,int,int,int);
int voxel_index_solid(int,int,int,int,int,int,int*);

# endif  // LATTICE_BUILDERS_D3Q19_H