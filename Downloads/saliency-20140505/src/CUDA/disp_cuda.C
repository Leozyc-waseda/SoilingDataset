#include <cuda.h>
#include <cuda_runtime.h>
#include "CudaDevices.H"

int main(int argc, char **argv)
{
  unsigned int free = 0, total = 0;
  int dev;
  if(argc < 2)
    dev = 0;
  else
    dev=atoi(argv[1]);
  CudaDevices::displayProperties(dev);
  
  // looks like this function does not exist anymore?
  //cuMemGetInfo(&free, &total);
  printf("Sorry, cuMemGetInfo problem...\n");

  printf("^^^^ Device: %d free %u, total %u\n",dev,free,total);
  return 0;
}
