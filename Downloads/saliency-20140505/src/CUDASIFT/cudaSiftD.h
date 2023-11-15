#ifndef CUDASIFTD_H
#define CUDASIFTD_H
#define WARP_SIZE     16

#define ROW_TILE_W    160
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48

#define MINMAX_SIZE   128
#define POSBLK_SIZE   32
#define LOWPASS5_DX 160
#define LOWPASS5_DY 16

__device__ __constant__ float d_Kernel[17]; // NOTE: Maximum radius

#endif
