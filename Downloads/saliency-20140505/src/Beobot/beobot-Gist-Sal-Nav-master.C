/*!@file Beobot/beobot-Gist-Sal-Nav-master.C Robot navigation using a
  combination saliency and gist.
  Run beobot-Gist-Sal-Nav-master at A to do vision
  Run beobot-Gist-Sal-Nav        at B to move                           */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-Gist-Sal-Nav-master.C $
// $Id: beobot-Gist-Sal-Nav-master.C 10982 2009-03-05 05:11:22Z itti $
//
////////////////////////////////////////////////////////
// beobot-Gist-Sal-Nav-Master.C <input_train.txt>
//
// a version of Gist/test-Gist-Sal-Nav.C to be run on the Beobot
//

#include "Channels/ChannelOpts.H"
#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWinManaged.H"
#include "Gist/FFN.H"
#include "Gist/trainUtils.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/ImageCache.H"
#include "Image/MathOps.H"
#include "Image/MatrixOps.H"
#include "Image/Pixels.H"
#include "Image/Transforms.H"
#include "Media/MPEGStream.H"
#include "Media/MediaOpts.H"
#include "Media/MediaSimEvents.H"
#include "Neuro/GistEstimatorStd.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/NeuroSimEvents.H"
#include "Neuro/ShapeEstimator.H"
#include "Neuro/ShapeEstimatorModes.H"
#include "Neuro/SpatialMetrics.H"
#include "Neuro/StdBrain.H"
#include "Neuro/gistParams.H"
#include "Raster/Raster.H"
#include "SIFT/Histogram.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Beobot/GridMap.H"
//#include "Beobot/TopologicalMap.H"
#include "Simulation/SimEventQueueConfigurator.H"


#define W_ASPECT_RATIO  320 // ideal minimum width for display
#define H_ASPECT_RATIO  240 // ideal minimum height for display

CloseButtonListener wList;
XWinManaged *inputWin;
XWinManaged *salWin;

XWinManaged *dispWin;
int wDisp, hDisp, sDisp, scaleDisp;
int wDispWin,  hDispWin;

// gist display
int pcaW = 16, pcaH = 5;
int winBarW = 5, winBarH = 25;

// ######################################################################
void setupDispWin(int w, int h);

Image< PixRGB<byte> > getDispImg
(Image< PixRGB<byte> > img, Image<float> gistImg,
 Image<float> gistPcaImg, Image<float> outHistImg);

// ######################################################################
// Main function
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Beobot: Navigation Model");

  // we cannot use saveResults() on our various ModelComponent objects
  // here, so let's not export the related command-line options.
  manager.allowOptions(OPTEXP_ALL & (~OPTEXP_SAVE));

  // Instantiate our various ModelComponents:
  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<StdBrain> brain(new StdBrain(manager));
  manager.addSubComponent(brain);

  nub::ref<SpatialMetrics> metrics(new SpatialMetrics(manager));
  manager.addSubComponent(metrics);

  manager.exportOptions(MC_RECURSE);
  metrics->setFOAradius(30); // FIXME
  metrics->setFoveaRadius(30); // FIXME
  manager.setOptionValString(&OPT_MaxNormType, "FancyOne");
  manager.setOptionValString(&OPT_UseRandom, "false");
  //  manager.setOptionValString("ShapeEstimatorMode","SaliencyMap");
  //  manager.setOptionValString(&OPT_ShapeEstimatorMode,"ConspicuityMap");
  manager.setOptionValString(&OPT_ShapeEstimatorMode, "FeatureMap");
  manager.setOptionValString(&OPT_ShapeEstimatorSmoothMethod, "Chamfer");
  //manager.setOptionValString(&OPT_ShapeEstimatorSmoothMethod, "Gaussian");
  manager.setOptionValString(&OPT_RawVisualCortexChans,"OIC");
  //manager.setOptionValString(&OPT_IORtype, "ShapeEstFM");
  manager.setOptionValString(&OPT_IORtype, "Disc");

  // setting up the GIST ESTIMATOR
  manager.setOptionValString(&OPT_GistEstimatorType,"Std");

  // Request a bunch of option aliases (shortcuts to lists of options):
  REQUEST_OPTIONALIAS_NEURO(manager);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<input_train.txt>",
                               1, 1) == false)
    return(1);

  // do post-command-line configs
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful -- ABORT");
  int w = gb->getWidth(), h = gb->getHeight();
  std::string dims = convertToString(Dims(w, h));
  manager.setOptionValString(&OPT_InputFrameDims, dims);
  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  // frame delay in seconds
  double fdelay = 33.3667/1000.0; // real time

  // let's get all our ModelComponent instances started:
  manager.start();

  // main loop:
  SimTime prevstime = SimTime::ZERO();
  int fNum = 0;
  Image< PixRGB<byte> > inputImg;
  Image< PixRGB<byte> > dispImg;
  Image< PixRGB<byte> > tImg;

  // instantiate a 3-layer feed-forward network
  // initialize with the provided parameters
  rutz::shared_ptr<FeedForwardNetwork> ffn_place(new FeedForwardNetwork());
  FFNtrainInfo pcInfo(manager.getExtraArg(0).c_str());
  ffn_place->init3L(pcInfo.h1Name, pcInfo.h2Name, pcInfo.oName,
                    pcInfo.redFeatSize, pcInfo.h1size, pcInfo.h2size,
                    pcInfo.nOutput, 0.0, 0.0);

  // setup the PCA eigenvector
  Image<double> pcaVec =
    setupPcaIcaMatrix(pcInfo.trainFolder+pcInfo.evecFname,
                      pcInfo.oriFeatSize, pcInfo.redFeatSize);

  // get the frame grabber to start streaming:
  gb->startStream();

  // MAIN LOOP
  while(1)
  {
    // has the time come for a new frame?
    // LATER ON GIST WILL DECIDE IF WE WANT TO SLOW THINGS DOWN
    if (fNum == 0 ||
        ((seq->now() - 0.5 * (prevstime - seq->now())).secs() - fNum * fdelay
         > fdelay))
      {
        // NEED CONDITION TO END THE LOOP
        //if (??) break;

        // grab a frame
        inputImg = gb->readRGB();
        tImg = inputImg;

        // setup  display  at the start of stream
        // NOTE: wDisp, hDisp, and sDisp are modified here
        if (fNum == 0) setupDispWin(w, h);
        //inputWin->drawImage(inputImg,0,0);

        // pass input to brain:
        seq->post(rutz::make_shared(new SimEventInputFrame(brain.get(), GenericFrame(inputImg), 0)));
        LINFO("\nnew frame :%d",fNum);

        // get the gist feature vector
        Image<double> cgist;
        if (SeC<SimEventGistOutput> ee = seq->check<SimEventGistOutput>(brain.get())) cgist = ee->gv();
        else LFATAL("No gist output in the queue");

        // reduce feature dimension (if available)
        Image<double> in;
        if(pcInfo.isPCA) in = matrixMult(pcaVec, cgist);
        else in = cgist;

        // recognize the place
        ffn_place->run3L(in);
        rutz::shared_ptr<Histogram> resHist(new Histogram(pcInfo.nOutput));

        for(uint i = 0; i < pcInfo.nOutput; i++)
          {
            LINFO("pl[%3d]: %.4f",i,ffn_place->getOutput().getVal(i));
            resHist->addValue(i,ffn_place->getOutput().getVal(i));
          }

        // display or save the visuals
        LFATAL("FIXME SimEventGistOutput does not contain an image and should be updated");
        /*
        dispImg = getDispImg
          (tImg,
           ge->getGistImage(sDisp),
           getPcaIcaFeatImage(in, pcaW, pcaH, sDisp*2),
           resHist->getHistogramImage(wDisp,sDisp*2 *pcaH, 0.0, 1.0));
        */

        // have to press a key to continue to the next frame
        dispWin->drawImage(dispImg,0,0);
        Raster::waitForKey();

        // if we got an image, save it:
//         if (inputImg.initialized())
//           {
//             std::string base = "../data/CameraCalib/beobotCF_";
//             Raster::WriteRGB
//               (inputImg, PNM, sformat("%s%03d",base.c_str(),fNum));
//             LINFO("saving %s%03d", base.c_str(), fNum);
//           }

       // increment frame count
        fNum++;
      }

    // evolve brain:
    prevstime = seq->now(); // time before current step
    const SimStatus status = seq->evolve();

    // process if salient location is selected
    if (SeC<SimEventWTAwinner> e = seq->check<SimEventWTAwinner>(0))
    {
      const Point2D<int> winner = e->winner().p;

      // use Shape estimator to focus on the attended region
      Image<float> fmask; std::string label;
      if (SeC<SimEventShapeEstimatorOutput>
          e = seq->check<SimEventShapeEstimatorOutput>(0))
        { fmask = e->smoothMask(); label = e->winningLabel(); }

      Image<float> roiImg;
      if (fmask.initialized())
        roiImg = fmask * luminance(inputImg);
      else
        roiImg = luminance(inputImg);

      drawCircle(roiImg, winner, 10, 0.0f, 1);
      LINFO("\nFrame: %d, winner: (%d,%d) in %s\n\n",
            fNum, winner.i, winner.j, label.c_str());
      //salWin->drawImage(roiImg,0,0);
      //Raster::waitForKey();
      //doSomeSalStuff(img, brain, winner);
    }

    if (SIM_BREAK == status) // Brain decided it's time to quit
      break;
  }

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
// setup display window for visualization purposes
void setupDispWin(int w, int h)
{
  // figure out the best display w, h, and scale

  // check if both dimensions of the image
  // are much smaller than the desired resolution
  scaleDisp = 1;
  while (w*scaleDisp < W_ASPECT_RATIO*.75 && h*scaleDisp < H_ASPECT_RATIO*.75)
    scaleDisp++;

  // check if the height is longer aspect-ratio-wise
  // this is because the whole display is setup wrt/ to it
  wDisp = w*scaleDisp; hDisp = h*scaleDisp;
  if(wDisp/(0.0 + W_ASPECT_RATIO) > hDisp/(0.0 + H_ASPECT_RATIO))
    hDisp = (int)(wDisp / (0.0 + W_ASPECT_RATIO) * H_ASPECT_RATIO)+1;
  else
    wDisp = (int)(hDisp / (0.0 + H_ASPECT_RATIO) * W_ASPECT_RATIO)+1;

  // add slack so that the gist feature entry is square
  sDisp = (hDisp/NUM_GIST_FEAT + 1);
  hDisp =  sDisp * NUM_GIST_FEAT;

  // add space for all the visuals
  wDispWin = wDisp + sDisp * NUM_GIST_COL;
  hDispWin = hDisp + sDisp * pcaH * 2;

  dispWin  = new XWinManaged(Dims(wDispWin, hDispWin), 0, 0, "dispImg");
  wList.add(dispWin);

  inputWin = new XWinManaged(Dims(w, h), w, 0, "input" );
  wList.add(inputWin);

  salWin   = new XWinManaged(Dims(w, h), 2*w, 0, "Sal" );
  wList.add(salWin);
}

// ######################################################################
// get display image for visualization purposes
Image< PixRGB<byte> > getDispImg (Image< PixRGB<byte> > img,
                                  Image<float> gistImg,
                                  Image<float> gistPcaImg,
                                  Image<float> outHistImg)
{
  Image< PixRGB<byte> > dispImg(wDispWin, hDispWin, ZEROS);
  int w = img.getWidth(); int h = img.getHeight();

  // grid the displayed input image
  drawGrid(img, w/4,h/4,1,1,PixRGB<byte>(255,255,255));
  inplacePaste(dispImg, img,        Point2D<int>(0, 0));

  // display the gist features
  inplaceNormalize(gistImg, 0.0f, 255.0f);
  inplacePaste(dispImg, Image<PixRGB<byte> >(gistImg),    Point2D<int>(wDisp, 0));

  // display the PCA gist features
  inplaceNormalize(gistPcaImg, 0.0f, 255.0f);
  inplacePaste(dispImg, Image<PixRGB<byte> >(gistPcaImg), Point2D<int>(wDisp, hDisp));

  // display the classifier output histogram
  inplaceNormalize(outHistImg, 0.0f, 255.0f);
  inplacePaste(dispImg, Image<PixRGB<byte> >(outHistImg), Point2D<int>(0, hDisp));

  // draw lines delineating the information
  drawLine(dispImg, Point2D<int>(0,hDisp),
           Point2D<int>(wDispWin,hDisp),
           PixRGB<byte>(255,255,255),1);
  drawLine(dispImg, Point2D<int>(wDisp-1,0),
           Point2D<int>(wDisp-1,hDispWin-1),
           PixRGB<byte>(255,255,255),1);
  return dispImg;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
