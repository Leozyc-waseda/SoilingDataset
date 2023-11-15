#include "CUDA/CudaHmaxCBCL.H"
#include "CUDA/CudaDevices.H"
#include "Component/ModelManager.H"
#include "Util/log.H"

int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;
  ModelManager *mgr = new ModelManager("Extract Patches for Hmax with Feature Learning");

  mgr->exportOptions(MC_RECURSE);

  // required arguments
  // <c1patchesDir> <trainPosDir>

  if (mgr->parseCommandLine(
                            (const int)argc, (const char**)argv, "<cudadev> <c0patchesFile> <c1patchesFile> <trainPosDir>", 4, 4) == false)
    return 1;
  int dev = strtol(mgr->getExtraArg(0).c_str(),NULL,0);
  CudaDevices::setCurrentDevice(dev);

  std::string c0PatchesFile;
  std::string c1PatchesFile;
  std::string trainPosName; // Directory where positive images are
  c0PatchesFile = mgr->getExtraArg(1);
  c1PatchesFile = mgr->getExtraArg(2);
  trainPosName = mgr->getExtraArg(3);

  CudaHmaxCBCL hmax;
  hmax.loadC0(c0PatchesFile);

  // Extract random patches from a set of images in a positive training directory
  std::vector<std::string> trainPos = hmax.readDir(trainPosName);

  hmax.extractC1Patches(trainPos,250,4,4);
  hmax.writeOutC1Patches(c1PatchesFile);
  return 0;


}
