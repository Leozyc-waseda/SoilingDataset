/*! @file INVT/StereoVision.C [put description here] */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/StereoVision.C $
// $Id: StereoVision.C 10827 2009-02-11 09:40:02Z itti $

#include "Channels/StereoChannel.H"
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Media/FrameSeries.H"
#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/ShapeOps.H"
#include "Media/MediaSimEvents.H"
#include "Neuro/StdBrain.H"
#include "Neuro/NeuroSimEvents.H"
#include "Neuro/VisualCortex.H"
#include "Raster/Raster.H"
#include "Simulation/SimEventQueueConfigurator.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

void printRegion(Image<float> img,int sX,int eX,int dX, int sY,int eY, int dY);

int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Stereo Vision Model");

  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Instantiate our various ModelComponents:
  nub::soft_ref<StdBrain> brain(new StdBrain(manager));
  manager.addSubComponent(brain);

  nub::soft_ref<StereoChannel> stc(new StereoChannel(manager));
  manager.addSubComponent(stc);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<image> <image>", 2, 2) == false)
    return(1);

  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  CloseButtonListener wList;
  //int w = 256;  int h = 256;
  Image<float> imgL//;imgL.resize(w,h,true);
  = Raster::ReadGray(manager.getExtraArg(0));
  Image<float> imgR//;imgR.resize(w,h,true);
  = Raster::ReadGray(manager.getExtraArg(1));
  int w = imgL.getWidth();
  int h = imgR.getHeight();

  /*
  for (int i = 0; i < 4; i++)
    for (int j = 0; j <4; j++)
    {
      if(!(i == 2 & j == 1))
      {
        drawPatch(imgR, *(new Point2D<int>(32+64*i  ,32+64*j)),2,(float)255.0);
        drawPatch(imgL, *(new Point2D<int>(32+64*i+2,32+64*j)),2,(float)255.0);
      }
      else
      {
        drawPatch(imgR, *(new Point2D<int>(32+64*i+2,32+64*j)),2,(float)255.0);
        drawPatch(imgL, *(new Point2D<int>(32+64*i  ,32+64*j)),2,(float)255.0);
      }
    }
  //drawPatch(imgR, *(new Point2D<int>(84,80)),2,(float)205.0);
  //drawPatch(imgL, *(new Point2D<int>(82,80)),2,(float)205.0);

  inplaceAddBGnoise2(imgL, 2.0);
  inplaceAddBGnoise2(imgR, 2.0);
  */

  //imgL = zoomXY(imgL, 4,-1);
  //imgR = zoomXY(imgR, 4,-1);

  XWinManaged L(Dims(w,h),1000,000,"imgL");
  L.drawImage(zoomXY(imgL, 1,-1),0,0); wList.add(L);
  XWinManaged R(Dims(w,h),1000,300,"imgR");
  R.drawImage(zoomXY(imgR, 1,-1),0,0); wList.add(R);

  Image<byte> bimgL = imgL;
  Image<byte> bimgR = imgR;
  Image <byte> zerro(bimgL.getWidth(),bimgL.getHeight(),ZEROS);
  XWinManaged LR(Dims(w,h),1000,800,"imgL+imgR");
  LR.drawImage(zoomXY(makeRGB(bimgL,bimgR,zerro), 1,-1),0,0); wList.add(LR);

  // let's get all our ModelComponent instances started:
  printf("\nSTART\n\n");
  manager.start();

  printf("adding Stereo Channel");

  LFATAL("fixme");
  /*
  brain->getVC()->addSubChan(stc);

  // set the weight of channels besides stereo to 0;
  brain->getVC()->setSubchanTotalWeight("color", 0.0);
  brain->getVC()->setSubchanTotalWeight("flicker", 0.0);
  brain->getVC()->setSubchanTotalWeight("intensity", 0.0);
  brain->getVC()->setSubchanTotalWeight("orientation", 0.0);
  brain->getVC()->setSubchanTotalWeight("motion", 0.0);
  */
  XWinManaged final(Dims(w,h),1000,800,"FINAL");
  wList.add(final);

  // main loop:
  printf("\nMAIN_LOOP\n\n");

  while(1)
  {
    // read new image in?
    const FrameState is = ifs->update(seq->now());
    if (is == FRAME_COMPLETE) break; // done
    if (is == FRAME_NEXT || is == FRAME_FINAL) // new frame
    {
      stc->setSecondImage(&imgR);
      rutz::shared_ptr<SimEventInputFrame>
        e(new SimEventInputFrame(brain.get(), GenericFrame(bimgL), 0));
      seq->post(e); // post the image to the brain
    }

    // evolve brain:
    (void) seq->evolve();

    // write outputs or quit?
    bool gotcovert = false;
    if (seq->check<SimEventWTAwinner>(0)) gotcovert = true;
    const FrameState os = ofs->update(seq->now(), gotcovert);

    if (os == FRAME_NEXT || os == FRAME_FINAL)
      brain->save(SimModuleSaveInfo(ofs, *seq));

    if (os == FRAME_FINAL) break;             // done

    // if we displayed a bunch of images, let's pause:
    if (ifs->shouldWait() || ofs->shouldWait())
      Raster::waitForKey();

    /*
  Image<float> ***fImgL, ***fL;
  Image<float> ***fImgR, ***fR;

  ImageSet<float> dispMap[12];

  stc->getRawFilteredImages(&fImgL,&fImgR);
  stc->dispChan(0,0).getRawFilteredImages(&fL,&fR);

  for(int i = 0; i<3; i++)
    stc->dispChan(0,i).getDispMap(&dispMap[i]);

  int chan = 0;
  Image<float> temp0 =  stc->dispChan(0,chan).getOutput();
  XWinManaged subchan0(Dims(w,h),1000,500,"Conspicuous0");
  subchan0.drawImage(zoomXY(temp0, 16,-1),0,0); wList.add(subchan0);

  chan = 1;
  Image<float> temp1 =  stc->dispChan(0,chan).getOutput();
  XWinManaged subchan1(Dims(w,h),1000,600,"Conspicuous1");
  subchan1.drawImage(zoomXY(temp1, 16,-1),0,0); wList.add(subchan1);

  chan = 2;
  Image<float> temp2 =  stc->dispChan(0,chan).getOutput();
  XWinManaged subchan2(Dims(w,h),1000,700,"Conspicuous2");
  subchan2.drawImage(zoomXY(temp2, 16,-1),0,0); wList.add(subchan2);

  Image<float> temp =  stc->getOutput();
  final.drawImage(zoomXY(temp, 16,-1),0,0);

  XWinManaged *SM[6];
  for(int i = 0; i< 6; i++)
  {
    temp =  stc->dispChan(0,chan).getSubmap(i);
    temp.setVal(0,0,0);
    temp.setVal(0,1,1);

    SM[i] = new XWinManaged(Dims(w,h),750,i*150,"SUBMAP");
    SM[i]->drawImage(zoomXY(temp,16,-1),0,0);
    wList.add(*SM[i]);

  }

  //int de = 0;
  XWinManaged *DM[3][7];
  //float min, max;
  for(int i = 0;i<3;i++)
    for(int de = 0; de< 7; de++)
    {
      //dispMap[i][de].getMinMax(min,max);
      //printf("MIN: %f, MAX: %f\n\n",min,max);

      //DM[i][de] = new XWinManaged(Dims(w,h),i*250,de*200,"imgL");
      //DM[i][de]->drawImage(zoomXY(dispMap[i][de],(int)pow(2.0,de),-1),0,0);
      //wList.add(*DM[i][de]);
    }
    */
  }
  while(!(wList.pressedAnyCloseButton()))sleep(1);
  // stop all our ModelComponents
  manager.stop();

  // all done!
  printf("All done\n");
  return 0;
}


void printRegion(Image<float> img,int sX,int eX,int dX, int sY,int eY, int dY)
{
  for(int j = sY; j<=eY; j+=dY)
  {
    for(int i = sX; i<=eX; i+=dX)
      printf("%8.3f ", img.getVal(i,j));
    printf(" \n");
  }
  printf("\n");
}

