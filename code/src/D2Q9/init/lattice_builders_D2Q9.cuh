# ifndef LATTICE_BUILDERS_D2Q9_H
# define LATTICE_BUILDERS_D2Q9_H

void build_box_lattice_D2Q9(int,int,int,int*,int*);
void build_box_lattice_shear_D2Q9(int,int,int,int*,int*);
void build_box_lattice_enclosed_D2Q9(int,int,int,int*,int*);
void build_box_lattice_D2Q9(int,int,int,int,int,int,int,int,int*,int*);
int voxel_index(int,int,int,int);

# endif  // LATTICE_BUILDERS_D2Q9_H