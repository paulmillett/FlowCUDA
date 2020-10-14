# ifndef BOUNDARY_CONDITION_IOLET_H
# define BOUNDARY_CONDITION_IOLET_H

// --------------------------------------------------------
// This struct contains information needed for each
// in inlet/outlet boundary:
// --------------------------------------------------------

struct iolet {
	int type;
	float uBC;
	float vBC;
	float wBC;
	float rBC;
	float pBC;
};

# endif  // BOUNDARY_CONDITION_IOLET_H