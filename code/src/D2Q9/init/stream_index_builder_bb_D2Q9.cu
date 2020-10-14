
# include "stream_index_builder_bb_D2Q9.cuh"



// --------------------------------------------------------
// Build the streamIndex[] array, which stores the indices
// of the streaming populations of neighboring voxels:
// Note: this is for "PUSH" streaming!
// --------------------------------------------------------

void stream_index_push_bb_D2Q9(int nVoxels, int* nListDir, int* s, int* streamIndex)
{		
	int Q = 9;
	for (int i=0; i<nVoxels; i++) {
		int offst = i*Q;
		
		// dir. 0 (no streaming)
		streamIndex[offst+0] = offst;
		
		// dir. 1 streamed to nbr1
		streamIndex[offst+1] = Q*nListDir[offst + 1] + 1;
		if (s[nListDir[offst+1]] == 1) {
			streamIndex[offst+1] = offst + 3;  // bounce-back
		}
		
		// dir. 2 streamed to nbr2
		streamIndex[offst+2] = Q*nListDir[offst + 2] + 2;
		if (s[nListDir[offst+2]] == 1) {
			streamIndex[offst+2] = offst + 4;  // bounce-back
		}
		
		// dir. 3 streamed to nbr3
		streamIndex[offst+3] = Q*nListDir[offst + 3] + 3;
		if (s[nListDir[offst+3]] == 1) {
			streamIndex[offst+3] = offst + 1;  // bounce-back
		}
		
		// dir. 4 streamed to nbr4
		streamIndex[offst+4] = Q*nListDir[offst + 4] + 4;
		if (s[nListDir[offst+4]] == 1) {
			streamIndex[offst+4] = offst + 2;  // bounce-back
		}
		
		// dir. 5 streamed to nbr5
		streamIndex[offst+5] = Q*nListDir[offst + 5] + 5;
		if (s[nListDir[offst+5]] == 1) {
			streamIndex[offst+5] = offst + 7;  // bounce-back
		}
		
		// dir. 6 streamed to nbr6
		streamIndex[offst+6] = Q*nListDir[offst + 6] + 6;
		if (s[nListDir[offst+6]] == 1) {
			streamIndex[offst+6] = offst + 8;  // bounce-back
		}
		
		// dir. 7 streamed to nbr7
		streamIndex[offst+7] = Q*nListDir[offst + 7] + 7;
		if (s[nListDir[offst+7]] == 1) {
			streamIndex[offst+7] = offst + 5;  // bounce-back
		}
		
		// dir. 8 streamed to nbr8
		streamIndex[offst+8] = Q*nListDir[offst + 8] + 8;
		if (s[nListDir[offst+8]] == 1) {
			streamIndex[offst+8] = offst + 6;  // bounce-back
		}	
				
	}				
}