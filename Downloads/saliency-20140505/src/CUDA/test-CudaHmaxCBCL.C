#include "CUDA/CudaHmaxCBCL.H"
#include "Image/Image.H"
#include "Raster/Raster.H"
#include "Util/log.H"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#include <dirent.h>

int main(int argc,char* argv[])
{
  Image<float> img;
  std::string c0patchFile = std::string("c0Patches.txt");
  std::string c1patchFile = std::string("c1Patches.txt");
  std::string c2outFile = std::string("c2.out");

  if(argc != 2)
    {
      fprintf(stderr,"testprog <image_file>\n");
      exit(1);
    }
  CudaHmaxCBCL hmax = CudaHmaxCBCL(c0patchFile,c1patchFile);
  // Load image
  img = Raster::ReadGrayNTSC(argv[1]);
  // Get C2
  hmax.getC2(img.getArrayPtr(),img.getWidth(),img.getHeight());
  // Write out C2
  hmax.writeOutFeatures(c2outFile,1,true);
  // Clear C2 memory
  hmax.clearC2();
}
