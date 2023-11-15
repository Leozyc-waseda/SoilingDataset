#include "Component/ModelManager.H"
#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Media/FrameSeries.H"
#include <Image/Image.H>
#include <Image/Pixels.H>
#include "Raster/GenericFrame.H"
#include "GUI/XWindow.H"
#include "Raster/Raster.H"

#include "Ice/IceImageCompressor.H"
#include "Ice/IceImageDecompressor.H"

#include "Ice/IceImageUtils.H"
#include <Ice/Ice.h>
#include "Ice/ImageIce.ice.H"
#include <fstream>

using namespace std;
using namespace ImageIceMod;

int main(const int argc, const char **argv) {

  IceImageCompressor   comp;
  IceImageDecompressor decomp;

  ModelManager *mgr = new ModelManager("EchoImageClientManager");
  mgr->debugMode();
  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
  mgr->addSubComponent(ifs);
  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);

  mgr->parseCommandLine(argc, argv, "", 0, 0);
  mgr->start();
  Image< PixRGB<byte> > inputImg;
  ifs->updateNext();
  GenericFrame input = ifs->readFrame();
  inputImg = input.asRgb();

  int status = 0;
//  Ice::CommunicatorPtr ic;
//  try {
//    int dummy=0;
//    ic = Ice::initialize(dummy,0);
//    Ice::ObjectPrx base = ic->stringToProxy(
//        "SimpleImageShuttle:default -p 10000");
//    ImageShuttlePrx imageshuttle = ImageShuttlePrx::checkedCast(base);
//    if(!imageshuttle)
//      throw "Invalid proxy";


    while(true)
    {

      //inputImg = Image<PixRGB<byte> >(2,2,ZEROS);
      std::vector<unsigned char> compressedImage = comp.CompressImage(inputImg);
      std::cout << "Finished Compression! Final Array is " << compressedImage.size() << " bytes" << std::endl;
//      imageshuttle->transferImage(Image2Ice(inputImg));

      ofstream outfile ("COMPRESSEDOUTPUT.jpg", ios::out | ios::binary);
      //outfile.write(&compressedImage[0], compressedImage.size());
      copy(compressedImage.begin(), compressedImage.end(), ostreambuf_iterator<char>(outfile));
      outfile.close();


      Image<PixRGB<byte> > uncompressedImage = decomp.DecompressImage(compressedImage);
      ofs->writeRGB(uncompressedImage, "FRAME");

      Raster::waitForKey();


      ifs->updateNext();
      GenericFrame input = ifs->readFrame();
      inputImg = input.asRgb();

      if(!inputImg.initialized()) break;
    }


  //}
  //catch (const Ice::Exception& ex) {
  //  cerr << ex << endl;
  //  status = 1;
  //}
  //catch(const char* msg) {
  //  cerr << msg << endl;
  //  status = 1;
  //}
  //if (ic)
  //  ic->destroy();
  return status;
}

