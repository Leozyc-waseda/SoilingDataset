
#include "CUDA/CudaCutPaste.H"
#include "CUDA/CudaSaliency.H"
#include "CUDA/CudaMathOps.H"
#include "CUDA/CudaFramework.H"
#include "CUDA/CudaImageDisplayGL.H"
#include "Component/ModelManager.H"
#include "INVT/neovision2cuda.H"
#include "Util/StringConversions.H"
#include "Transport/FrameInfo.H"
#include "Util/log.H"
#include "Util/StringUtil.H"
#include <string> 
#include <vector>
#define  MAX_INPUT_WIDTH 1920
#define  MAX_INPUT_HEIGHT 1080
#define  CHARACTER_SIZE 10
#define  NFRAMESAVG 3 // Number of frames to average over to get Frames/Second

//Nv2LabelReader *reader = NULL;


neovision2cuda::neovision2cuda(nub::ref<InputFrameSeries> ifs_in, nub::ref<OutputFrameSeries> ofs_in, nub::ref<CudaSaliency> csm_in, std::vector<rutz::shared_ptr<Nv2LabelReader> > readers_in) : CudaImageDisplayGL(), ifs(ifs_in), ofs(ofs_in), csm(csm_in), readers(readers_in)
{
  frameTimes.clear();
}

// ######################################################################
void neovision2cuda::idleFunction()
{
  if(getShutdown())
    {
      printf("Shutdown\n");
      exit(0);
      return;
    }
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  pwin->update(); // handle pending preference window events

  //Getting 1st frame image
  //printf("Handling frame\n");
  const FrameState is = ifs->updateNext();
  if (is == FRAME_COMPLETE)
    {
      printf("\n\n************************************** SHUTTING DOWN ******************************************\n\n");
      setShutdown(true);
      return;
    }
  GenericFrame input = ifs->readFrame();
  if (!input.initialized())
    {
      printf("\n\n************************************** SHUTTING DOWN (Empty Frame) ******************************************\n\n");
      setShutdown(true);
      return;
    }
    
  static int FrameNumber = -1;
  FrameNumber++;

  const Image<PixRGB<byte> > inbyte = input.asRgb();

  Image<PixRGB<float> > infloat = inbyte;
  //printf("Copying CPU image to CUDA\n");
  CudaImage<PixRGB<float> > cfloat = CudaImage<PixRGB<float> >(infloat,framework.getMP(),framework.getDev());
  //printf("Updating Canvas With Raw Image\n");
     
   // Need to shrink input if larger than 1920x1080    
  //CudaImage<PixRGB<float> > cfloat_resize = cudaRescale(cfloat,int(600*x_scale),int(400*y_scale));
  //Updating
  framework.updateCanvas(1,cfloat);

  //Getting 2nd frame image
  //printf("Calculating Saliency\n");
  csm->doCudaInput(cfloat);

  //Different Maps
  CudaImage<float> out = csm->getCudaOutput();

  CudaImage<PixRGB<float> > cimap = cudaRescale(cudaToRGB(csm->getIMap()),w_cm,h_cm);
  CudaImage<PixRGB<float> > ccmap = cudaRescale(cudaToRGB(csm->getCMap()),w_cm,h_cm);
  CudaImage<PixRGB<float> > comap = cudaRescale(cudaToRGB(csm->getOMap()),w_cm,h_cm);
  CudaImage<PixRGB<float> > cfmap = cudaRescale(cudaToRGB(csm->getFMap()),w_cm,h_cm); 
  CudaImage<PixRGB<float> > cmmap = cudaRescale(cudaToRGB(csm->getMMap()),w_cm,h_cm);
  CudaImage<PixRGB<float> > cinhibitionmap = cudaRescale(cudaToRGB(csm->getInhibitionMap()),w_sm,h_sm);

  Point2D<int> salmap_peak;
  Point2D<int> rawmap_peak;
  Point2D<int> inputmap_peak; // Modified from raw map peak to fit on image, given a rectangle size


  std::vector<Point2D<int> > maxsaliency = csm->getSalMaxLoc();
  std::vector<float > max_val = csm->getSalMax();

  //Scaling max_point
  CudaImage<PixRGB<float> > csal = cudaRescale(cudaToRGB(out),w_sm,h_sm);


  //Updating
  framework.updateCanvas(2,csal);
  framework.updateCanvas(3,cimap);
  framework.updateCanvas(4,ccmap);
  framework.updateCanvas(5,comap);
  framework.updateCanvas(6,cfmap);
  framework.updateCanvas(7,cmmap);
  framework.updateCanvas(8,cinhibitionmap);

  char buffer[25];
  // Push a frame time on
  frameTimes.push_back(Timer());
  if(frameTimes.size() > NFRAMESAVG)
    {
      Timer t = frameTimes[0];
      frameTimes.pop_front();
      sprintf(buffer,"Frame No. %d, FPS %.2f",ifs->frame(),NFRAMESAVG/t.getSecs());
    }
  else
    sprintf(buffer,"Frame No. %d",ifs->frame());
  
  int rectW = csm->getPatchSize();
  int rectH = csm->getPatchSize();



  //Text update functions
  // framework.initTextLayer(CHARACTER_SIZE*40,20);
  // framework.setText(buffer,Point2D<int>(w_in,2*h_sm),PixRGB<float>(255.0f,0.0f,0.0f),PixRGB<float>(0.0f,0.0f,0.0f),SimpleFont::FIXED(10),false);
  // framework.initTextLayer(CHARACTER_SIZE*4,20);
  // framework.setText("IMAP",imapFramePoint,PixRGB<float>(255.0f,0.0f,0.0f),PixRGB<float>(0.0f,0.0f,0.0f),SimpleFont::FIXED(10),false);
  // framework.setText("CMAP",cmapFramePoint,PixRGB<float>(255.0f,0.0f,0.0f),PixRGB<float>(0.0f,0.0f,0.0f),SimpleFont::FIXED(10),false);
  // framework.setText("OMAP",omapFramePoint,PixRGB<float>(255.0f,0.0f,0.0f),PixRGB<float>(0.0f,0.0f,0.0f),SimpleFont::FIXED(10),false);
  // framework.setText("FMAP",fmapFramePoint,PixRGB<float>(255.0f,0.0f,0.0f),PixRGB<float>(0.0f,0.0f,0.0f),SimpleFont::FIXED(10),false);
  // framework.setText("MMAP",mmapFramePoint,PixRGB<float>(255.0f,0.0f,0.0f),PixRGB<float>(0.0f,0.0f,0.0f),SimpleFont::FIXED(10),false);
  // framework.initTextLayer(CHARACTER_SIZE*8,20);
  // framework.setText("InhibMAP",inhibitionMapFramePoint,PixRGB<float>(255.0f,0.0f,0.0f),PixRGB<float>(0.0f,0.0f,0.0f),SimpleFont::FIXED(10),false);
  // framework.initTextLayer(CHARACTER_SIZE*11,20);
  // framework.setText("SaliencyMAP",saliencyFramePoint,PixRGB<float>(255.0f,0.0f,0.0f),PixRGB<float>(0.0f,0.0f,0.0f),SimpleFont::FIXED(10),false);

 // Draw rectangles to frame border of mapsup
  framework.drawRectangle_topleftpoint(imapFramePoint,PixRGB<float>(255.0f,255.0f,255.0f),w_cm,h_cm,1);
  framework.drawRectangle_topleftpoint(cmapFramePoint,PixRGB<float>(255.0f,255.0f,255.0f),w_cm,h_cm,1);
  framework.drawRectangle_topleftpoint(omapFramePoint,PixRGB<float>(255.0f,255.0f,255.0f),w_cm,h_cm,1);
  framework.drawRectangle_topleftpoint(fmapFramePoint,PixRGB<float>(255.0f,255.0f,255.0f),w_cm,h_cm,1);
  framework.drawRectangle_topleftpoint(mmapFramePoint,PixRGB<float>(255.0f,255.0f,255.0f),w_cm,h_cm,1);
  framework.drawRectangle_topleftpoint(saliencyFramePoint,PixRGB<float>(255.0f,255.0f,255.0f),w_sm,h_sm,1);
  framework.drawRectangle_topleftpoint(inhibitionMapFramePoint,PixRGB<float>(255.0f,255.0f,255.0f),w_sm,h_sm,1);


  for(unsigned int p=0;p<maxsaliency.size();p++)
    {
      rawmap_peak.i = maxsaliency[p].i * cfloat.getWidth() / double(out.getWidth());
      rawmap_peak.j = maxsaliency[p].j * cfloat.getHeight() / double(out.getHeight());
      inputmap_peak = rawmap_peak;


      // Make sure bounding box does not exceed size of image
      rectW=std::min((cfloat.getWidth()-rawmap_peak.i-1)*2,rectW);
      rectH=std::min((cfloat.getHeight()-rawmap_peak.j-1)*2,rectH);
  
      // If we are running over the left edge
      if(inputmap_peak.i-rectW/2.0 < 0)
	{
	  inputmap_peak.i=rectW/2.0;
	}
      // If we are running over the right edge
      if(inputmap_peak.i+rectW/2.0 >= cfloat.getWidth())
	{
	  inputmap_peak.i=cfloat.getWidth()-1-rectW/2.0;
	}
      // If we are running over the top edge
      if(inputmap_peak.j-rectH/2.0 < 0)
	{
	  inputmap_peak.j=rectH/2.0;      
	}
      // If we are running over the bottom edge
      if(inputmap_peak.j+rectH/2.0 >= cfloat.getHeight())
	{
	  inputmap_peak.j=cfloat.getHeight()-1-rectH/2.0;
	}

      // Given the known good rectangle sitting at inputmap_peak, calculate the equivalent for the saliency map
      salmap_peak.i = saliencyFramePoint.i + inputmap_peak.i * csal.getWidth() / cfloat.getWidth();
      salmap_peak.j = inputmap_peak.j * csal.getHeight() / cfloat.getHeight();
      int salmapRectW = rectW*csal.getWidth()/double(cfloat.getWidth());
      int salmapRectH = rectH*csal.getHeight()/double(cfloat.getHeight());


      // Draw rectangles around peak activation
      framework.drawRectangle_centrepoint(inputmap_peak,PixRGB<float>(255.0f,0.0f,0.0f),rectW,rectH,1);
      framework.drawRectangle_centrepoint(salmap_peak,PixRGB<float>(255.0f,0.0f,0.0f),salmapRectW,salmapRectH,1);


      // Send the patch to any listening patch readers
      int fi = std::max(rawmap_peak.i-rectW/2,0);
      int fj = std::max(rawmap_peak.j-rectH/2,0);
      Rectangle foa = Rectangle(Point2D<int>(fi,fj),Dims(rectW,rectH));
      int patch_id = ifs->frame()*maxsaliency.size()+p;
      rutz::time patchElapsedTime = rutz::time::wall_clock_now();  
  
      bool isTraining = false;
      std::string label = std::string("FindLabel");
      std::string remoteCommand = std::string("");
      bool ignoreNonMatch = false;
      int textLength = 80;
      Image<PixRGB<float> > img = cfloat.exportToImage();
      Image<PixRGB<float> > patch = crop(img,foa);
      printf("Sending patch to readers\n");
      for(unsigned int i=0;i<readers.size();i++)
	readers[i]->sendPatch(patch_id,img,foa,patch,patchElapsedTime,isTraining,label,remoteCommand,rawmap_peak);
      //printf("Reading patches\n");


      //std::vector<Nv2LabelReader::LabeledImage> result;

      for(unsigned int i=0;i<readers.size();i++)
	{
	  Nv2LabelReader::LabeledImage li = readers[i]->getNextLabeledImage(ignoreNonMatch,textLength,FrameNumber);
	  if (li.label != std::string(""))
	    {
	      printf("*************** GOT LABEL [%s] *****************\n",li.label.c_str());
	      if(li.img.initialized())
		{
		  ofs->writeRGB(li.img, li.ident, FrameInfo("object-labeled image", SRC_POS));
		}
	    }
	}
    }
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);


  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


// ######################################################################
void neovision2cuda::runDisplay(int w_originput, int h_originput, MemoryPolicy mp, int dev)
{
  
  n_cm = 5;
  n_sm = 2;
  aspect_ratio = w_originput/double(h_originput);
  if(w_originput > MAX_INPUT_WIDTH || h_originput > MAX_INPUT_HEIGHT)
    { 
      if(aspect_ratio > MAX_INPUT_WIDTH/double(MAX_INPUT_HEIGHT))
	{
	  // Width is violating max principle
	  w_in = MAX_INPUT_WIDTH;
	  h_in = w_in/aspect_ratio;
	}
      else
	{
	  // Height is violating max principle
	  h_in = MAX_INPUT_HEIGHT;
	  w_in = h_in*aspect_ratio;
	}
    }
  else
    {
      // Input dimensions are OK
      w_in = w_originput;
      h_in = h_originput;
    }

  int w = 1.25 * w_originput;
  w_cm = w/double(n_cm);
  h_cm = w_cm/aspect_ratio;
  int h = h_originput + h_cm;
  w_sm = 0.2 * w;
  h_sm = w_sm/aspect_ratio;

  // Setup preferences
  pwin = new PrefsWindow("control panel", SimpleFont::FIXED(8));
  pwin->setValueNumChars(16);
  pwin->addPrefsForComponent(csm.get());
  
  //Setting up frame starting points (top left coordinates)
  //Large Maps
  rawFramePoint = Point2D<int>(0,0);
  saliencyFramePoint = Point2D<int>( w_in,0);
  inhibitionMapFramePoint = Point2D<int>(w_in,h_sm);
  //Small maps
  int mrg_sm = 0;  // Margin between small maps
  imapFramePoint = Point2D<int>(int(w_cm+mrg_sm)*0,h_in);
  cmapFramePoint = Point2D<int>(int(w_cm+mrg_sm)*1,h_in);
  omapFramePoint = Point2D<int>(int(w_cm+mrg_sm)*2,h_in);
  fmapFramePoint = Point2D<int>(int(w_cm+mrg_sm)*3,h_in);
  mmapFramePoint = Point2D<int>(int(w_cm+mrg_sm)*4,h_in);
  //Associating coordinates with numeric frame symbols like 1,2,3...
  framework.setPoint(1,rawFramePoint);
  framework.setPoint(2,saliencyFramePoint);
  framework.setPoint(3,imapFramePoint);
  framework.setPoint(4,cmapFramePoint);
  framework.setPoint(5,omapFramePoint);
  framework.setPoint(6,fmapFramePoint);
  framework.setPoint(7,mmapFramePoint);
  framework.setPoint(8,inhibitionMapFramePoint);

  //Initialise text layer of size 250 X 20
  framework.initTextLayer(w_in,20);

  //Start framework
  framework.startFramework(w,h,dev,mp);
  createDisplay(w,h);

  // Register this class' idle function
  glutIdleFunc(neovision2cuda::idleWrapper);
  glutMainLoop();

  cuda_invt_allocation_show_stats(1, "final",0);

  //cutilExit(argc, argv);

}

// ######################################################################
int main(int argc, char **argv)
{
  MYLOGVERB = LOG_INFO;
  // instantiate a model manager (for camera input):
  ModelManager *mgr = new ModelManager("CudaSaliency Tester");
  // NOTE: make sure you register your OutputFrameSeries with the
  // manager before you do your InputFrameSeries, to ensure that
  // outputs for the current frame get saved before the next input
  // frame is loaded.
  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
  nub::ref<CudaSaliency> csm(new CudaSaliency(*mgr));

  const PixRGB<byte> colors[4] =
    {
      PixRGB<byte>(128,255,0),
      PixRGB<byte>(0,255,128),
      PixRGB<byte>(0,0,255),
      PixRGB<byte>(255,0,255)
    };

  // Change destination of ofs

  //reader = new Nv2LabelReader(col,localPort,serverStr);
  mgr->addSubComponent(ofs);
  mgr->addSubComponent(ifs);
  mgr->addSubComponent(csm);

  mgr->exportOptions(MC_RECURSE);
  // Parse command-line:
  if (mgr->parseCommandLine(argc, argv, "<Cuda Dev> <ip:port,ip:port,...>", 2, 2) == false) return -1;
  std::string devStr = mgr->getExtraArg(0);
  int dev = strtol(devStr.c_str(),NULL,0);
  std::string addrlist = mgr->getExtraArg(1);
  MemoryPolicy mp = GLOBAL_DEVICE_MEMORY;
  printf("Using CUDA Device %d\n",dev);
  // Set device before model manager start
  csm->setDevice(mp,dev);

  // if (addrlist.empty())
  //   {
  //     int localPort = 9931;
  //     std::string serverStr = std::string("127.0.0.1:9930");
  //   }

  std::vector<rutz::shared_ptr<Nv2LabelReader> > readers;
  std::vector<std::string> addrs;
  split(addrlist, ",", std::back_inserter(addrs));

  for (size_t i = 0; i < addrs.size(); ++i)
    {
      std::vector<std::string> serverAddress;
      split(addrs[i],":",std::back_inserter(serverAddress));
      ASSERT(serverAddress.size() == 2);
      int localPort = strtol(serverAddress[1].c_str(),NULL,0)+1;
      readers.push_back
        (rutz::make_shared
         (new Nv2LabelReader(colors[i%4],
                             localPort,
                             addrs[i])));
    }



  mgr->start();
  ifs->startStream();
  Dims inDims = ifs->peekDims();

  // Check if it is hd or not . 
  // If hd then fit everything in max window size else resize the window size accordigly. 
  // Preserving the aspect ration and layout percentages to rescale.
  // int w = 2406; //Default
  // int h = 1376; //Default
     
      
 /*
    }
   
   }
  */
  neovision2cuda *n = neovision2cuda::createCudaDisplay(ifs,ofs,csm,readers);
  n->runDisplay(inDims.w(),inDims.h(),mp,dev);
  //delete reader;
  return 0;
}
