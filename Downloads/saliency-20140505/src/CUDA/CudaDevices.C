#include "CudaDevices.H"

// Class CudaDevices static initializations
map<int,Dims> CudaDevices::deviceTileSizes  = map<int,Dims>();
map<int,int> CudaDevices::deviceSharedMemorySizes  = map<int,int>();
int CudaDevices::currentDevice = -1;
