#include "Image/Image.H"
#include "GUI/XWindow.H"
#include "Raster/Raster.H"
#include "Image/Pixels.H"

#include <Ice/Ice.h>
#include <IceUtil/Thread.h>
#include "Ice/ImageIce.ice.H"
#include "Ice/IceImageUtils.H"
#include "Component/ModelManager.H"
#include "Media/FrameSeries.H"
#include "Image/JPEGUtil.H"




class ImageShuttleI: public ImageShuttle, public IceUtil::Thread, public ModelComponent {
  public:
    ImageShuttleI(OptionManager &mgr,
        const std::string& descrName = "Image Shuttle",
        const std::string& tagName = "ImageShuttle") :
      ModelComponent(mgr,descrName, tagName),
      itsOfs(new OutputFrameSeries(mgr))
  {
    addSubComponent(itsOfs);
  }

    void start2()
    {
      IceUtil::ThreadPtr thread = this;
      thread->start();
    }

    void transferImage(const ImageIce& s, const Ice::Current&)
    {
      std::vector<unsigned char> bytes = s.data;
      itsImage = itsDecompressor.DecompressImage(bytes);
    }

    void run()
    {
      while(1)
      {
        itsImageMutex.lock();
        {
          if(itsImage.initialized())
            itsOfs->writeRGB(itsImage, "Input");
        }
        itsImageMutex.unlock();
      }

    }

  private:
    Image<PixRGB<byte> > itsImage;
    nub::ref<OutputFrameSeries> itsOfs;
    IceUtil::Mutex itsImageMutex;
    JPEGDecompressor itsDecompressor;
};




int main(int argc, char** argv) {
  int status = 0;

  ModelManager manager("EchoImageServerManager");
  nub::ref<ImageShuttleI> imageShuttle(new ImageShuttleI(manager));
  manager.addSubComponent(imageShuttle);
   // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
        "Usage:: EchoImageServerIce <port> --out=display\n"
        "Use the number of your board to determine which port to put here,\n"
        "eg. if you have g1 use port 10001. If you have g2, use port 10002,\n"
        "and so on...\n", 1, 1)
      == false) return(1);

  manager.start();
  char buffer[64];
  sprintf(buffer,"default -p %s", manager.getExtraArg(0).c_str());


  Ice::CommunicatorPtr ic;
  try {
    ic = Ice::initialize(argc, argv);
      Ice::ObjectAdapterPtr adapter =
        ic->createObjectAdapterWithEndpoints(
            "SimpleImageShuttleAdapter", buffer);
      Ice::ObjectPtr object = imageShuttle.get();
      adapter->add(object, ic->stringToIdentity("SimpleImageShuttle"));
      adapter->activate();
      ic->waitForShutdown();
    } catch (const Ice::Exception& e) {
        cerr << e << endl;
        status = 1;
    } catch (const char* msg) {
        cerr << msg << endl;
        status = 1;
    }
    if (ic) {
        try {
          ic->destroy();
} catch (const Ice::Exception& e) {
            cerr << e << endl;
            status = 1;
        }
    }
    return status;



  return 1;
}


