/*! @file Beobot/app-build-salObjDB.C Build a database of salient VisualObject
    from a stream input */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/app-build-salObjDB.C $
// $Id: app-build-salObjDB.C 15495 2014-01-23 02:32:14Z itti $
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
#include "SIFT/Histogram.H"
#include "SIFT/Keypoint.H"
#include "SIFT/VisualObject.H"
#include "SIFT/VisualObjectDB.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Util/Timer.H"


#define DB_NAME "out_database"

#define W_ASPECT_RATIO  320 // ideal minimum width for display
#define H_ASPECT_RATIO  240 // ideal minimum height for display

FeedForwardNetwork *ffn_place;
double **gistW   = NULL;

CloseButtonListener wList;
XWinManaged *salWin;
XWinManaged *gistWin;
rutz::shared_ptr<XWinManaged> objWin;

int wDisp, hDisp, sDisp, scaleDisp;
int wDispWin,  hDispWin;

// gist display
int pcaW = 16, pcaH = 5;
int winBarW = 5, winBarH = 25;

// number of landmarks produced
int numObj = 0;

// clip list
uint nCat = 0;
std::vector<std::string>* clipList;

// ######################################################################
void                  setupDispWin     (int w, int h);
Image< PixRGB<byte> > getGistDispImg   (Image< PixRGB<byte> > img,
                                        Image<float> gistImg,
                                        Image<float> gistPcaImg,
                                        Image<float> outHistImg);
Image< PixRGB<byte> > getSalDispImg    (Image< PixRGB<byte> > img,
                                        Image<float> roiImg,
                                        Image< PixRGB<byte> > objImg,
                                        Point2D<int> winner, int fNum);
void                  processSalCue    (Image<PixRGB<byte> > inputImg,
                                        nub::soft_ref<StdBrain> brain,
                                        Point2D<int> winner, int fNum,
                                        std::vector< rutz::shared_ptr<Landmark> >&
                                        landmarks,
                                        const Image<float>& semask, const std::string& selabel);
void                  setupCases       (const char* fname);
// ######################################################################

// Main function
/*! Load a database, enrich it with new VisualObject entities
  extracted from the given images, and save it back. */
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Salient objects DB Builder Model");

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

  manager.exportOptions(MC_RECURSE);
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

  // set up the GIST ESTIMATOR
  //manager.setOptionValString(&OPT_GistEstimatorType,"Std");

  // DO NOT set up the INFEROTEMPORAL
  //manager.setOptionValString(&OPT_InferoTemporalType,"Std");
  //manager.setOptionValString(&OPT_AttentionObjRecog,"yes");
  //manager.setOptionValString(&OPT_MatchObjects,"false");

  // Request a bunch of option aliases (shortcuts to lists of options):
  REQUEST_OPTIONALIAS_NEURO(manager);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<input_gistList.txt>",
                               1, 1) == false)
    return(1);

  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  // NOTE: this could now be controlled by a command-line option
  // --preload-mpeg=true
  manager.setOptionValString(&OPT_InputMPEGStreamPreload, "true");

  setupCases(manager.getExtraArg(0).c_str());

  // frame delay in seconds
  double rtdelay = 33.3667/1000.0;        // real time
  double fdelay  = rtdelay * 3;           // NOTE: 3 times slower than real time

  Image< PixRGB<byte> > inputImg;
  Image< PixRGB<byte> > gistDispImg;
  int w = 0, h = 0;

  SimTime prevstime = SimTime::ZERO(); uint fNum = 0;
  fNum = 0;

  // let's get all our ModelComponent instances started:
  manager.start();

  // FIX: WE NEED TO START UTILIZING THIS
  // load the database: REPLACE BY LANDMARK for tracking purposes
  //  rutz::shared_ptr<VisualObjectDB> vdb(new VisualObjectDB());
  //if (vdb->loadFrom(DB_NAME))
  //  LINFO("Starting with empty VisualObjectDB.");

  // SIFT visual object related
  std::vector< rutz::shared_ptr<Landmark> >** landmarks
    = new std::vector< rutz::shared_ptr<Landmark> >*[nCat];

  // for each category in the list
  int fTotal = 0;
  for(uint i = 0; i < nCat; i++)
    {
      landmarks[i] = new std::vector< rutz::shared_ptr<Landmark> >
        [clipList[i].size()];
      // FIX: index is bigger than itsObject.size()
      // seems that a value is not reset when we are changing clips

      // for each movie in that category
      for(uint j = 0; j < clipList[i].size(); j++)
        {
          // do post-command-line configs:
          ims->setFileName(clipList[i][j]);
          LINFO("Loading[%d][%d]: %s",i,j,clipList[i][j].c_str());
          Raster::waitForKey();

          if(i ==0 && j == 0)
            {
              Dims iDims = ims->peekDims();
              manager.setOptionValString(&OPT_InputFrameDims,
                                         convertToString(ims->peekDims()));
              w = iDims.w() - 50 + 1; h = iDims.h();
              LINFO("w: %d, h: %d",w, h);

              // setup  display  at the start of stream
              // NOTE: wDisp, hDisp, and sDisp are modified here
              setupDispWin(w, h);
            }

          bool eoClip = false;
          fNum = 0;

          // process until end of clip
          while(!eoClip)
            {
              // has the time come for a new frame?
              // If we want to SLOW THINGS DOWN change fdelay
              if (fNum == 0 ||
                  (seq->now() - 0.5 * (prevstime - seq->now())).secs() - fTotal * fdelay > fdelay)
                {
                  // load new frame: // FIX THE SECOND CONDITION LATER
                  inputImg = ims->readRGB();
                  if (inputImg.initialized() == false || (fNum == 5))
                    eoClip = true;  // end of input stream
                  else
                    {
                      // take out frame borders NOTE: ONLY FOR SONY CAMCORDER
                      inputImg = crop(inputImg, Rectangle::tlbrI(0, 25, h-1, 25 + w - 1));

                      // pass input to brain:
                      LINFO("new frame Number: %d",fNum);
                      rutz::shared_ptr<SimEventInputFrame>
                        e(new SimEventInputFrame(brain.get(), GenericFrame(inputImg), 0));
                      seq->post(e); // post the image to the brain

                      // if we are tracking objects
                      LINFO("Currently we have: %" ZU " objects in DB[%d][%d]",
                            landmarks[i][j].size(),i,j);
                      std::string imgName(sformat("image%07d", fNum));

                      // FIX: is this redundant w/ IT
                      rutz::shared_ptr<VisualObject>
                        newVO(new VisualObject(imgName, "", inputImg));
                      for(uint k = 0; k < landmarks[i][j].size(); k++)
                        {
                          landmarks[i][j][k]->build(newVO, fNum);

                          // print the current location and velocity
                          Point2D<int> pos = landmarks[i][j][k]->getPosition();
                          //Point2D<int> vel = landmarks[i][j][k]->getVelocity();
                          LINFO("landmark[%d][%d][%d]: %s is at %d,%d", i, j, k,
                                landmarks[i][j][k]->getName().c_str(), pos.i, pos.j);
                          // FIX NOTE: maybe need to put the position
                          // (and thus the motion) in the name (for servoing)
                        }

                      // increment frame count
                      fNum++;fTotal++;
                    }
                }

              // evolve brain:
              prevstime = seq->now(); // time before current step
              const SimStatus status = seq->evolve();

              // process if SALIENT location is found
              if (SeC<SimEventWTAwinner>
                  e = seq->check<SimEventWTAwinner>(0))
                {
                  // segment out salient location
                  // check against the database
                  const Point2D<int> winner = e->winner().p;
                  //if(landmarks[i][j].size() == 0) // <------CHANGE THIS LATER

                  Image<float> semask; std::string selabel;
                  if (SeC<SimEventShapeEstimatorOutput>
                      e = seq->check<SimEventShapeEstimatorOutput>(0))
                    { semask = e->smoothMask(); selabel = e->winningLabel(); }

                  processSalCue(inputImg, brain, winner, fNum-1, landmarks[i][j], semask, selabel);
                }

              if (SIM_BREAK == status) // Brain decided it's time to quit
                eoClip = true;

            } // END while(!eoClip)

          // display the current resulting database:
          LINFO("there are %" ZU " landmarks recovered in DB[%d][%d]",
                landmarks[i][j].size(),i,j);
          for(uint k = 0; k < landmarks[i][j].size(); k++)
            {
              LINFO("  %d: %s", k, landmarks[i][j][k]->getName().c_str());
              rutz::shared_ptr<VisualObjectDB> voDB =
                landmarks[i][j][k]->getVisualObjectDB();

              // check the number of evidence for each landmark
              for(uint l = 0; l < voDB->numObjects(); l++)
                {
                  LINFO("    %d: %s", l, voDB->getObject(l)->getName().c_str());
                  Image< PixRGB<byte> > tImg(2*w,2*h,ZEROS);
                  inplacePaste(tImg,  voDB->getObject(l)->getImage(), Point2D<int>(0, 0));
                  objWin->drawImage(tImg,0,0);
                  Raster::waitForKey();
                }
            }
        }

      // we can now combine the salient objects across lighting condition
      // FIX: ADD

      // take out moving things by discarding objects that are only exist in 1 clip.

      // keep objects with a lot of salient hits

      // order object with the starting frame number

      // watch out for overlapping objects
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
  const int w = inputImg.getWidth();
  const int h = inputImg.getHeight();

  // segment out the object -> maybe port to infero-temporal later
  // ----------------------------------------------
  Image<float> roiImg;
  Image<PixRGB<byte> > objImg; Point2D<int> objOffset;

  bool useSE = true;

  // use Shape estimator to focus on the attended region when available
  if (semask.initialized())
    {
      roiImg = semask * luminance(inputImg);
      float mn, mx; getMinMax(semask, mn, mx);
      Rectangle r = findBoundingRect(semask, mx*.05f);
      objImg = crop(inputImg, r);
      objOffset = Point2D<int>(r.left(),r.top());

      // and size is not too big (below 50% input image)
      int wSE = objImg.getWidth(), hSE = objImg.getHeight();
      if(wSE * hSE > .5 * w * h)
        {
          LINFO("SE Smooth Mask is too big: %d > %d", wSE*hSE, int(.5*w*h));
          useSE = false;
        }
      else
        LINFO("SE Smooth Mask is used %d <= %d", wSE*hSE, int(.5*w*h));
    }
  else
    {
      roiImg = luminance(inputImg);
      objImg = inputImg;
      objOffset = Point2D<int>(0,0);
      useSE = false;
      LINFO("SE Smooth Mask not yet initialized");
    }

  // otherwise use pre-set 100x100window
  if(!useSE)
    {
      Rectangle roi =
        Rectangle::tlbrI(winner.j - 50, winner.i - 50,
                        winner.j + 50, winner.i + 50);
      roi = roi.getOverlap(inputImg.getBounds());

      // keep the roiImg
      objImg = crop(inputImg, roi);
      objOffset =  Point2D<int>(roi.left(),roi.top());

      LINFO("SE not ready");
      Raster::waitForKey();
    }

  LINFO("TOP LEFT at: (%d,%d)", objOffset.i, objOffset.j);

  // draw the results
  salWin->drawImage(getSalDispImg(inputImg,roiImg,objImg, winner, fNum),0,0);
  LINFO("Frame: %d, winner: (%d,%d) in %s", fNum, winner.i, winner.j,
        selabel.c_str());
  if(fNum > 50)
    Raster::waitForKey();

  // need a Visual Cortex to obtain the feature vector
  LFATAL("fixme using a SimReq");
  ////////nub::soft_ref<VisualCortex> vc = brain->getVC();
  std::vector<float> fvec; /////////vc->getFeatures(winner, fvec);

  // create a new VisualObject (a set of SIFT keypoints)
  // with the top-left coordinate of the window
  rutz::shared_ptr<VisualObject>
    obj(new VisualObject("NewObject", "NewObject", objImg,
                         winner - objOffset, fvec));

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
      rutz::shared_ptr<Landmark> newlm(new Landmark(obj, objOffset, fNum, lmName));
      newlm->setMatchWin(objWin);
      landmarks.push_back(newlm);
      if(fNum > 50)
        Raster::waitForKey();
    }
  else if(trackAccepted > 1)
    {
       LINFO("May have: %d objects jumbled together", trackAccepted);
    }
}

// ######################################################################
// setup display window for visualization purposes
void setupDispWin(int w, int h)
{

  //====================================================================
  /*
  // figure out the best display w, h, and scale for gist

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

  gistWin  = new XWinManaged(Dims(wDispWin, hDispWin), 0, 0, "Gist Related");
  wList.add(gistWin);
  */
  //====================================================================

  salWin   = new XWinManaged(Dims(2*w, 2*h), 2*w, 0, "Saliency Related" );
  wList.add(salWin);

  objWin.reset(new XWinManaged(Dims(2*w, 2*h), 0, 0, "Object Match" ));
  wList.add(*objWin);

}

// ######################################################################
// open the *_gistList.txt file containing all the list of .mpg files
void setupCases(const char* fname)
{
  char comment[200]; char folder[200];
  FILE *fp;  char inLine[1000];

  // get the folder, 47 is a slash '/'
  const char* tp = strrchr(fname,47);
  strncpy(folder,fname,tp-fname+1); folder[tp-fname+1] = '\0';
  LINFO("Folder %s -> %s", fname, folder);

  // open a file that lists the sample with ground truth
  if((fp = fopen(fname,"rb")) == NULL)
    LFATAL("gistList file: %s not found",fname);

  // skip number of samples
  if (fgets(inLine, 1000, fp) == NULL) LFATAL("fgets failed");

  // get the number of categories
  if (fgets(inLine, 1000, fp) == NULL) LFATAL("fgets failed"); sscanf(inLine, "%d %s", &nCat, comment);
  clipList = new std::vector<std::string>[nCat];

  // skip the type of ground truth and column headers
  if (fgets(inLine, 1000, fp) == NULL) LFATAL("fgets failed");
  if (fgets(inLine, 1000, fp) == NULL) LFATAL("fgets failed");

  char fileName[200];
  char cName[100]; char sName[100]; char ext[100];
  int cStart, cNum; int gTruth;

  while(fgets(inLine, 1000, fp) != NULL)
  {
    // get the files in this category and ground truth
    sscanf(inLine, "%s %d %d %d %s", cName, &cStart, &cNum,  &gTruth, ext);
    char* cname = strrchr(cName,95); // 95 is underscore '_'
    strncpy(sName,cName,cname-cName); sName[cname-cName] = '\0';
    sprintf(fileName,"%s%s.mpg", folder,sName);
    clipList[gTruth].push_back(fileName);
    //LINFO("    sName: %s -:- %d", fileName, gTruth);
  }

//   //for display
//   for(uint i = 0; i < nCat; i++)
//     {
//       for(uint j = 0; j < clipList[i].size(); j++)
//         {
//           LINFO("%d %d: %s",i,j,clipList[i][j].c_str());
//         }
//       LINFO(" ");
//     }

  fclose(fp);
}

// ######################################################################
// get saliency display image for visualization purposes
Image< PixRGB<byte> > getSalDispImg   (Image< PixRGB<byte> > img,
                                       Image<float> roiImg,
                                       Image< PixRGB<byte> > objImg,
                                       Point2D<int> winner,
                                       int fNum)
{
  int w = img.getWidth(), h = img.getHeight();
  Image< PixRGB<byte> > salDispImg(2*w,2*h,ZEROS);

  inplacePaste(salDispImg, img,        Point2D<int>(0, 0));
  Image<float> rRoiImg = roiImg;
  float min,max;
  getMinMax(roiImg,min,max);
  drawCircle( roiImg, winner, 10, 0.0f, 1);
  drawPoint ( roiImg, winner.i, winner.j, 0.0f);
  drawCircle(rRoiImg, winner, 10, 255.0f, 1);
  drawPoint (rRoiImg, winner.i, winner.j, 255.0f);
  Image< PixRGB<byte> > t = makeRGB(rRoiImg,roiImg,roiImg);
  inplacePaste(salDispImg, t,         Point2D<int>(0, h));
  inplacePaste(salDispImg, objImg,    Point2D<int>(w, h));

  writeText(salDispImg, Point2D<int>(w,0), sformat("%d",fNum).c_str(),
            PixRGB<byte>(0,0,0), PixRGB<byte>(255,255,255));
  return salDispImg;
}

// ######################################################################
// get gist display image for visualization purposes
Image< PixRGB<byte> > getGistDispImg (Image< PixRGB<byte> > img,
                                      Image<float> gistImg,
                                      Image<float> gistPcaImg,
                                      Image<float> outHistImg)
{
  Image< PixRGB<byte> > gistDispImg(wDispWin, hDispWin, ZEROS);
  int w = img.getWidth(); int h = img.getHeight();

  // grid the displayed input image
  drawGrid(img, w/4,h/4,1,1,PixRGB<byte>(255,255,255));
  inplacePaste(gistDispImg, img,        Point2D<int>(0, 0));

  // display the gist features
  inplaceNormalize(gistImg, 0.0f, 255.0f);
  inplacePaste(gistDispImg, Image<PixRGB<byte> >(gistImg),    Point2D<int>(wDisp, 0));

  // display the PCA gist features
  inplaceNormalize(gistPcaImg, 0.0f, 255.0f);
  inplacePaste(gistDispImg, Image<PixRGB<byte> >(gistPcaImg), Point2D<int>(wDisp, hDisp));

  // display the classifier output histogram
  inplaceNormalize(outHistImg, 0.0f, 255.0f);
  inplacePaste(gistDispImg, Image<PixRGB<byte> >(outHistImg), Point2D<int>(0, hDisp));

  // draw lines delineating the information
  drawLine(gistDispImg, Point2D<int>(0,hDisp),
           Point2D<int>(wDispWin,hDisp),
           PixRGB<byte>(255,255,255),1);
  drawLine(gistDispImg, Point2D<int>(wDisp-1,0),
           Point2D<int>(wDisp-1,hDispWin-1),
           PixRGB<byte>(255,255,255),1);
  return gistDispImg;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
