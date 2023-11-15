/*! @file Beobot/app-egomotion.C application to demonstrate egomotion from
  visual data - input is MPEG or list (in gistlist format) of image files
  (assume .ppm files)
  Note: reading the velocity:
        right hand rule, x-positive to the left, y-positive to the north,
        z-positive forward                                              */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/app-egomotion.C $
// $Id: app-egomotion.C 15495 2014-01-23 02:32:14Z itti $
//

#include "Beobot/Landmark.H"
#include "Channels/ChannelOpts.H"
#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Component/OptionManager.H"
#include "GUI/XWinManaged.H"
#include "Gist/FFN.H"
#include "Gist/trainUtils.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Media/MPEGStream.H"
#include "Media/MediaOpts.H"
#include "Media/MediaSimEvents.H"
#include "Neuro/GistEstimator.H"
#include "Neuro/InferoTemporal.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/NeuroSimEvents.H"
#include "Neuro/Retina.H"
#include "Neuro/ShapeEstimator.H"
#include "Neuro/ShapeEstimatorModes.H"
#include "Neuro/SpatialMetrics.H"
#include "Neuro/StdBrain.H"
#include "Neuro/gistParams.H"
#include "Neuro/VisualCortex.H"
#include "Raster/Raster.H"
#include "SIFT/CameraIntrinsicParam.H"
#include "SIFT/Histogram.H"
#include "SIFT/Keypoint.H"
#include "SIFT/VisualObject.H"
#include "SIFT/VisualObjectDB.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Util/Timer.H"

#include <iostream>
#include <fstream>


#include "SIFT/SIFTegomotion.H"

#define DB_NAME "out_database"

#define W_ASPECT_RATIO  320 // ideal minimum width for display
#define H_ASPECT_RATIO  240 // ideal minimum height for display

FeedForwardNetwork *ffn_place;
double **gistW   = NULL;

CloseButtonListener wList;
rutz::shared_ptr<XWinManaged> salWin;
rutz::shared_ptr<XWinManaged> objWin;
rutz::shared_ptr<XWinManaged> trajWin;

int wDisp, hDisp, sDisp, scaleDisp;
int wDispWin,  hDispWin;

// gist display
int pcaW = 16, pcaH = 5;
int winBarW = 5, winBarH = 25;

// number of landmarks produced
int numObj = 0;

// ######################################################################
void                  setupDispWin     (int w, int h);
Image< PixRGB<byte> > getSalDispImg    (Image< PixRGB<byte> > img, Image<float> roiImg,
                                        Image< PixRGB<byte> > objImg, Image< PixRGB<byte> > objImg2);
void                  processSalCue    (Image<PixRGB<byte> > inputImg,
                                        nub::soft_ref<StdBrain> brain, Point2D<int> winner, int fNum,
                                        std::vector< rutz::shared_ptr<Landmark> >& landmarks,
                                        const Image<float>& semask, const std::string& selabel);
void                  getGistFileList  (std::string fName,  std::vector<std::string>& tag,
                                        std::vector<int>& start, std::vector<int>& num);
Image< PixRGB<byte> > getTrajImg       (std::vector<Image <double> > traj, int w, int h);

// ######################################################################

// Main function
/*! Load a database, enrich it with new VisualObject entities
  extracted from the given images, and save it back. */
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Egomotion Model");

  // we cannot use saveResults() on our various ModelComponent objects
  // here, so let's not export the related command-line options.
  manager.allowOptions(OPTEXP_ALL & (~OPTEXP_SAVE));

  // Instantiate our various ModelComponents:
  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::soft_ref<InputMPEGStream>
    ims(new InputMPEGStream(manager, "Input MPEG Stream", "InputMPEGStream"));
  manager.addSubComponent(ims);

  nub::soft_ref<StdBrain> brain(new StdBrain(manager));
  manager.addSubComponent(brain);

  nub::ref<SpatialMetrics> metrics(new SpatialMetrics(manager));
  manager.addSubComponent(metrics);

  manager.exportOptions(true);
  metrics->setFOAradius(30); // FIXME
  metrics->setFoveaRadius(30); // FIXME
  manager.setOptionValString(&OPT_MaxNormType, "FancyOne");
  manager.setOptionValString(&OPT_UseRandom, "false");

  manager.setOptionValString(&OPT_IORtype, "Disc");
  manager.setOptionValString(&OPT_RawVisualCortexChans,"OIC");

  // customize the region considered part of the "object"
  //  manager.setOptionValString("ShapeEstimatorMode","SaliencyMap");
  //  manager.setOptionValString(&OPT_ShapeEstimatorMode,"ConspicuityMap");
  manager.setOptionValString(&OPT_ShapeEstimatorMode, "FeatureMap");
  manager.setOptionValString(&OPT_ShapeEstimatorSmoothMethod, "Chamfer");
  //manager.setOptionValString(&OPT_ShapeEstimatorSmoothMethod, "Gaussian");

  // DO NOT set up the INFEROTEMPORAL
  //manager.setOptionValString(&OPT_InferoTemporalType,"Std");
  //manager.setOptionValString(&OPT_AttentionObjRecog,"yes");
  //manager.setOptionValString(&OPT_MatchObjects,"false");

  // Request a bunch of option aliases (shortcuts to lists of options):
  REQUEST_OPTIONALIAS_NEURO(manager);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<*.mpg or *_gistList.txt>",
                               1, 1) == false)
    return(1);

  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  // if the file passed ends with _gistList.txt
  // we have a different protocol
  bool isGistListInput = false;
  int ifLen = manager.getExtraArg(0).length();
  if(ifLen > 13 &&
     manager.getExtraArg(0).find("_gistList.txt",ifLen - 13) != std::string::npos)
    isGistListInput = true;

  // NOTE: this could now be controlled by a command-line option
  // --preload-mpeg=true
  manager.setOptionValString(&OPT_InputMPEGStreamPreload, "true");

  // do post-command-line configs:
  std::vector<std::string> tag;
  std::vector<int> start;
  std::vector<int> num;
  if(isGistListInput)
    {
      LINFO("we have a gistList input");
      getGistFileList(manager.getExtraArg(0).c_str(), tag, start, num);
    }
  else
    {
      LINFO("we have an mpeg input");
      ims->setFileName(manager.getExtraArg(0));
      manager.setOptionValString(&OPT_InputFrameDims,
                                 convertToString(ims->peekDims()));
    }

  // frame delay in seconds
  double rtdelay = 33.3667/1000.0;    // real time
  double fdelay  = rtdelay * 3;           // 3 times slower than real time
  (void)fdelay;

  // let's get all our ModelComponent instances started:
  manager.start();

  // create a landmark covering the whole scene
  rutz::shared_ptr<Landmark> scene(new Landmark());
  scene->setMatchWin(objWin);

  // we HARD CODE the camera intrinsic parameter FOR BEOBOT
  rutz::shared_ptr<CameraIntrinsicParam>
    cip(new CameraIntrinsicParam(435.806712867904707, 438.523234664943345,
                                 153.585228257964644,  83.663180940275609, 0.0));
  scene->setCameraIntrinsicParam(cip);

  // main loop:
  SimTime prevstime = SimTime::ZERO(); uint fNum = 0;
  Image< PixRGB<byte> > inputImg;
  Image< PixRGB<byte> > dispImg;
  int w = 0;  // 320 or iDims.w() - 50 + 1;
  int h = 0;  // 240 or iDims.h();
  unsigned int cLine = 0;
  int cIndex = start[0];
  std::string folder =  "";
  std::string::size_type sPos = manager.getExtraArg(0).rfind("/",ifLen);
  if(sPos != std::string::npos)
    folder = manager.getExtraArg(0).substr(0,sPos+1);
  std::vector<Image<double> > traj;
  while(1)
  {
     // has the time come for a new frame?
     // If we want to SLOW THINGS DOWN change fdelay
     if (fNum == 0 ||
        (seq->now() - 0.5 * (prevstime - seq->now())).secs() - fNum * fdelay > fdelay)
       {
         // load new frame
         std::string fName;
         if(isGistListInput)
           {
             if (cLine >= tag.size()) break;  // end of input list

             // open the current file
             char tNumStr[100]; sprintf(tNumStr,"%06d",cIndex);
             fName = folder + tag[cLine] + std::string(tNumStr) + ".ppm";

             inputImg = Raster::ReadRGB(fName);
             cIndex++;

             if(cIndex >= start[cLine] + num[cLine])
               {
                 cLine++;
                 if (cLine < tag.size()) cIndex = start[cLine];
               }

             // reformat the file name to a gist name
             int fNameLen = fName.length();
             unsigned int uPos = fName.rfind("_",fNameLen);
             fName = fName.substr(0,uPos)+ ".ppm";
           }
         else
           {
             fName = manager.getExtraArg(0);
             inputImg = ims->readRGB();
             if (inputImg.initialized() == false) break;  // end of input stream
             // format new frame
             inputImg = crop(inputImg,
                             Rectangle::tlbrI(0,25,inputImg.getHeight()-1, inputImg.getWidth()-25));
             cIndex = fNum+1;
           }

         // setup  display  at the start of stream
         // NOTE: wDisp, hDisp, and sDisp are modified
         if (fNum == 0)
           {
             w = inputImg.getWidth(); h = inputImg.getHeight();
             setupDispWin(w, h); LINFO("w: %d, h: %d",w, h);
           }

         dispImg = inputImg;
         salWin->drawImage(dispImg,0,0);
         LINFO("\nnew frame :%d",fNum);

         // take out frame borders NOTE: ONLY FOR SONY CAMCORDER
         //inputImg = crop(inputImg, Rectangle::tlbrI(0, 25, h-1, 25 + w - 1));

         // pass input to brain:
         rutz::shared_ptr<SimEventInputFrame>
           e(new SimEventInputFrame(brain.get(), GenericFrame(inputImg), 0));
         seq->post(e); //post the image to the brain

         // track the view
         std::string viewName(sformat("view%07d", fNum));
         rutz::shared_ptr<VisualObject>
           cv(new VisualObject(viewName, "", inputImg));
         rutz::shared_ptr<VisualObjectMatch> cmatch = scene->build(cv, fNum);

         // CHECK EGOMOTION
         if(fNum != 0)
           {
             rutz::shared_ptr<SIFTegomotion>
               egm(new SIFTegomotion(cmatch, cip, objWin));

             // reading the velocity:
             // right hand rule, x-positive to the left,
             // y-positive to the north, z-positive forward
             traj.push_back(egm->getItsVel());
             egm->print(traj[traj.size() -1] ,"final velocity");
             trajWin->drawImage(getTrajImg(traj, 5*w, 2*h),0,0);
             //Raster::waitForKey();
           }

         // increment frame count
         fNum++;
       }

    // evolve brain:
    prevstime = seq->now(); // time before current step
    const SimStatus status = seq->evolve();

    // process SALIENT location found
    if (SeC<SimEventWTAwinner> e = seq->check<SimEventWTAwinner>(0))
      {
        const Point2D<int> winner = e->winner().p;
        LINFO("Frame: %d, winner: (%d,%d)", fNum, winner.i, winner.j);

        Image<float> semask; std::string selabel;
        if (SeC<SimEventShapeEstimatorOutput>
            e = seq->check<SimEventShapeEstimatorOutput>(0))
          { semask = e->smoothMask(); selabel = e->winningLabel(); }

        //processSalCue(inputImg, brain, winner, fNum, landmarks,
        // semask, selabel);

        if (SIM_BREAK == status) // Brain decided it's time to quit
          break;
      }
  }

  // save the resulting database:
  //if(vdb->numObjects() != 0)
  //  vdb->saveTo(DB_NAME);

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
// process salient cues
void processSalCue(const Image<PixRGB<byte> > inputImg,
                   nub::soft_ref<StdBrain> brain, Point2D<int> winner, int fNum,
                   std::vector< rutz::shared_ptr<Landmark> >& landmarks,
                   const Image<float>& semask, const std::string& selabel)
{
  // segment out the object -> maybe port to infero-temporal later
  // ----------------------------------------------

  // use Shape estimator to focus on the attended region
  Image<float> roiImg;
  Image<PixRGB<byte> > objImgSE;
  Point2D<int> objOffsetSE;
  if (semask.initialized())
    {
      roiImg = semask * luminance(inputImg);

      float mn, mx; getMinMax(semask,mn,mx);
      Rectangle r = findBoundingRect(semask, mx*.05f);
      Image<PixRGB<byte> > objImgSE = crop(inputImg, r);
      objOffsetSE = Point2D<int>(r.left(),r.top());
    }
  else
    {
      LINFO("SE Smooth Mask not yet initialized");
      roiImg = luminance(inputImg);
      objImgSE = inputImg;
      objOffsetSE = Point2D<int>(0,0);
    }

  // ----------------------------------------------
  // or with pre-set window
  Rectangle roi =
    Rectangle::tlbrI(winner.j - 50, winner.i - 50,
                    winner.j + 50, winner.i + 50);
  roi = roi.getOverlap(inputImg.getBounds());
  LINFO("[%d,%d,%d,%d]",roi.top(),roi.left(),roi.bottomI(),roi.rightI());
  Image<PixRGB<byte> > objImgWIN =  crop(inputImg, roi);
  Point2D<int> objOffsetWIN(roi.left(),roi.top());

  // ----------------------------------------------

  LINFO("TOP LEFT at: SE:(%d,%d) WIN:(%d,%d)",
        objOffsetSE.i  , objOffsetSE.j,
        objOffsetWIN.i , objOffsetWIN.j);

  // draw the results
  drawCircle(roiImg, winner, 10, 0.0f, 1);
  drawPoint(roiImg, winner.i, winner.j, 0.0f);
  LINFO("Frame: %d, winner: (%d,%d) in %s", fNum, winner.i, winner.j,
        selabel.c_str());
  salWin->drawImage(getSalDispImg(inputImg,roiImg,objImgWIN,objImgSE),0,0);
  //salWin->drawImage(Image<PixRGB<byte> >(inputImg.getDims() * 2, ZEROS) ,0,0);
  Raster::waitForKey();

  // WE CHOOSE: SE
  Image<PixRGB<byte> > objImg(objImgSE);
  Point2D<int> objOffset(objOffsetSE);

  // need a Visual Cortex to obtain the feature vector
  LFATAL("fixme using a SimReq");
  //////////nub::soft_ref<VisualCortex> vc = brain->getVC();
  std::vector<float> fvec; ////////vc->getFeatures(winner, fvec);

  // create a new VisualObject (a set of SIFT keypoints)
  // with the top-left coordinate of the window
  rutz::shared_ptr<VisualObject>
    obj(new VisualObject("NewObject", "NewObject",
                         objImg, winner - objOffset, fvec));

  std::string objName(sformat("obj%07d", numObj));
  obj->setName(objName);
  obj->setImageFname(objName + ".png");
  numObj++;

  // check with the salient regions DB before adding
  int trackAccepted = 0;
  LINFO("we have: %" ZU " landmarks to match", landmarks.size());
  for(uint i = 0; i < landmarks.size(); i++)
    {
       LINFO("tracking landmark number: %d",i);
       rutz::shared_ptr<VisualObjectMatch> cmatch =
         landmarks[i]->build(obj, objOffset, fNum);
       if(cmatch.is_valid() && cmatch->getScore() > 3.0)
         trackAccepted++;
    }

  // if it's not used by any of the existing landmarks entry
  if(trackAccepted == 0)
    {
      // create a new one
      LINFO("create a new Landmark number %" ZU ,landmarks.size());
      std::string lmName(sformat("landmark%07" ZU , landmarks.size()));
      rutz::shared_ptr<Landmark>
        newlm(new Landmark(obj, objOffset, fNum, lmName));
      newlm->setMatchWin(objWin);
      landmarks.push_back(newlm);
      Raster::waitForKey();
    }
  else if(trackAccepted > 1)
    {
       LINFO("May have: %d objects jumbled together", trackAccepted);
    }
}

// ######################################################################
void getGistFileList(std::string fName,  std::vector<std::string>& tag,
                     std::vector<int>& start, std::vector<int>& num)
{
  char comment[200]; FILE *fp;  char inLine[1000];

  // open the file
  if((fp = fopen(fName.c_str(),"rb")) == NULL)
    LFATAL("samples file: %s not found",fName.c_str());
  LINFO("fName: %s",fName.c_str());

  // get number of samples
  int nSamples; if (fgets(inLine, 1000, fp) == NULL) LFATAL("fgets failed"); 
  sscanf(inLine, "%d %s", &nSamples, comment);

  // the number of categories
  int tNcat; if (fgets(inLine, 1000, fp) == NULL) LFATAL("fgets failed"); 
  sscanf(inLine, "%d %s", &tNcat, comment);

  // get the type of ground truth
  char gtOpt[100]; if (fgets(inLine, 1000, fp) == NULL) LFATAL("fgets failed"); 
  sscanf(inLine, "%s %s", gtOpt, comment);

  // skip column headers
  if (fgets(inLine, 1000, fp) == NULL) LFATAL("fgets failed");

  char cName[100]; char ext[100];  int cStart, cNum; int gTruth;
  while(fgets(inLine, 1000, fp) != NULL)
  {
    // get the files in this category and ground truth
    sscanf(inLine, "%s %d %d %d %s", cName, &cStart, &cNum,  &gTruth, ext);
    LINFO("    sName: %s %d %d %d %s",cName, cStart, cNum, gTruth, ext);

    tag.push_back(cName);
    start.push_back(cStart);
    num.push_back(cNum);
  }
  fclose(fp);
}

// ######################################################################
// setup display window for visualization purposes
void setupDispWin(int w, int h)
{
  salWin.reset(new XWinManaged(Dims(2*w, 2*h), 2*w, 0, "Saliency Related" ));
  wList.add(*salWin);

  objWin.reset(new XWinManaged(Dims(2*w, 2*h), 0, 0, "Object Match" ));
  wList.add(*objWin);

  trajWin.reset(new XWinManaged(Dims(5*w, 2*h), 0, 0, "Trajectory" ));
  wList.add(*objWin);
}

// ######################################################################
// get saliency display image for visualization purposes
Image< PixRGB<byte> > getSalDispImg   (Image< PixRGB<byte> > img,
                                       Image<float> roiImg,
                                       Image< PixRGB<byte> > objImg,
                                       Image< PixRGB<byte> > objImg2)
{
  int w = img.getWidth(), h = img.getHeight();
  Image< PixRGB<byte> > salDispImg(2*w,2*h,ZEROS);

  inplacePaste(salDispImg, img,        Point2D<int>(0, 0));
  Image< PixRGB<byte> > t = makeRGB(roiImg,roiImg,roiImg);
  inplacePaste(salDispImg, t,          Point2D<int>(0, h));
  inplacePaste(salDispImg, objImg,     Point2D<int>(w, 0));
  inplacePaste(salDispImg, objImg2,     Point2D<int>(w, h));

  return salDispImg;
}

// ######################################################################
// get trajectory image for visualization purposes
Image< PixRGB<byte> > getTrajImg (std::vector<Image <double> > traj, int w, int h)
{
  Image< PixRGB<byte> > trajImg(w, h, ZEROS);

  int sX = 10; int sY = h/2;

  // velocity is relative in the range of -1.0 to 1.0
  double scale = 5.0;
  double locX = double(sX);
  double locY = double(sY);

  // draw each trajectory in the history
  for(uint i = 0;  i < traj.size(); i++)
    {
      double dX =  traj[i].getVal(0,2)*scale;
      double dY = -traj[i].getVal(0,0)*scale;
      LINFO("%d. %f,%f -> dx: %f, dy: %f ", i, traj[i].getVal(0,2), traj[i].getVal(0,0),
            dX,dY);

      drawDisk(trajImg, Point2D<int>(int(locX),int(locY)), 2, PixRGB<byte>(255,0,0));
      drawLine (trajImg,
                Point2D<int>(int(locX),int(locY)),
                Point2D<int>(int(locX + dX),int(locY + dY)),
                PixRGB<byte>(255,255,255),1);
      locX = locX + dX;
      locY = locY + dY;
    }

  return trajImg;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
