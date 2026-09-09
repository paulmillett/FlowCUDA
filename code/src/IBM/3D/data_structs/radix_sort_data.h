# ifndef RADIX_SORT_DATA_H
# define RADIX_SORT_DATA_H



// --------------------------------------------------------
// struct that defines radix-sort data for fast neighbor
// searching:
// --------------------------------------------------------

struct radixdata {
	int nCells;
	int nncells;
	int cellMax;
	int3 numCells;
	float sizeCells;
	int* particleCellIDs;
	int* particleIndices;
	int* cellStart;
	int* cellEnd;
	int* cellMap;
	int* newParticleIndices;
};


# endif  // RADIX_SORT_DATA_H