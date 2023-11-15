/*!@file Beobot/BeobotBrainMT.C efficient implementation of the
  feature pyramids, Saliency, Gist, Shape Estimator  for Beobot         */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotBrainMT.C $
// $Id: BeobotBrainMT.C 15426 2012-11-02 21:44:22Z siagian $
//

// ######################################################################

#include "Beobot/BeobotBrainMT.H"
#include "Neuro/gistParams.H"
#include "Neuro/GistEstimatorStd.H"

#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Image/FilterOps.H"
#include "Image/PyrBuilder.H"
#include "Image/ColorOps.H"
#include "Util/Timer.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/MorphOps.H"     // for openImg()

// ######################################################################
void *BeobotBrainMT_threadCompute(void *c)
{
  BeobotBrainMT *d = (BeobotBrainMT *)c;
  d->threadCompute();
  return NULL;
}

// ######################################################################
BeobotBrainMT::BeobotBrainMT(OptionManager& mgr,
                             const std::string& descrName,
                             const std::string& tagName)
  :
  ModelComponent(mgr, descrName, tagName)
{
  numWorkers = 0U;

  // the number of channels we are going to compute
  itsNumChannels = NUM_CHANNELS;

  // initialize the channel weights
  itsChanWeight.resize(itsNumChannels);
  itsChanWeight[REDGREEN  ] = CWEIGHT;
  itsChanWeight[BLUEYELLOW] = CWEIGHT;
  itsChanWeight[INTENSITY ] = IWEIGHT;
  itsChanWeight[ORI0      ] = OWEIGHT;
  itsChanWeight[ORI45     ] = OWEIGHT;
  itsChanWeight[ORI90     ] = OWEIGHT;
  itsChanWeight[ORI135    ] = OWEIGHT;

  // conspicuity maps
  itsCMaps.resize(itsNumChannels);

  // initialize CSmaps, raw CS maps
  uint cscount =
    (level_max - level_min + 1) * (delta_max - delta_min + 1);
  itsCSMaps.resize(itsNumChannels);
  itsRawCSMaps.resize(itsNumChannels);
  for(uint i = 0; i < itsNumChannels; i++)
    {
      itsCSMaps[i].resize(cscount);
      itsRawCSMaps[i].resize(cscount);
    }

  // feature maps storage
  itsImgPyrs.resize(itsNumChannels);

  //create the structuring element for Chamfer smoothing
  const int ss = 8;
  structEl = Image<byte>(ss+ss,ss+ss,ZEROS);
  drawDisk(structEl, Point2D<int>(ss,ss),ss,(byte)1);

  // initializing gist vector
  itsGistFeatSize = NUM_GIST_FEAT_SIZE;
  itsGistVector = Image<double>(1, itsGistFeatSize, NO_INIT);
  LINFO("initialized");

  // window to debug: default 320 x 240
  //itsWin =
  //  new XWinManaged(Dims(2*IMAGE_WIDTH,2*IMAGE_HEIGHT), 0, 0, "bbmtWin");

  itsTimer.reset(new Timer(1000000));
  itsProcessTime = -1.0F;
}

// ######################################################################
void BeobotBrainMT::start1()
{
  // start threads. They should go to sleep on the condition since no
  // jobs have ben queued up yet:
  pthread_mutex_init(&jobLock, NULL);
  pthread_mutex_init(&mapLock, NULL);
  pthread_cond_init(&jobCond, NULL);

  LINFO("Starting with %d threads...", numBBMTthreads);

  // get our processing threads started:
  worker = new pthread_t[numBBMTthreads];
  for (uint i = 0; i < numBBMTthreads; i ++)
    {
      pthread_create(&worker[i], NULL, BeobotBrainMT_threadCompute,
                     (void *)this);

      // all threads should go and lock against our job condition.
      // Sleep a bit to make sure this really happens:
      usleep(100000);
    }
}

// ######################################################################
void BeobotBrainMT::stop2()
{
  // should cleanup the threads, mutexes, etc...
  pthread_cond_destroy(&jobCond);

  //for (uint i = 0; i < numBBMTthreads; i ++)
  //  pthread_delete(&worker[i].....

  delete [] worker;
}

// ######################################################################
BeobotBrainMT::~BeobotBrainMT()
{ }

// ######################################################################
bool BeobotBrainMT::outputReady()
{
  bool ret = false;

  pthread_mutex_lock(&jobLock);
  if (jobsTodo == 0U) ret = true;
  //LINFO("jobs left: %d", jobsTodo);

  // for the initial condition
  pthread_mutex_lock(&mapLock);
  if(!itsCurrImg.initialized()) ret = false;
  pthread_mutex_unlock(&mapLock);

  pthread_mutex_unlock(&jobLock);

  return ret;
}

// ######################################################################
void BeobotBrainMT::input(Image< PixRGB<byte> > img)
{
  itsTimer->reset();
  computeCIOpyr(img);
}

// ######################################################################
void BeobotBrainMT::computeCIOpyr(Image< PixRGB<byte> > img)
{
  pthread_mutex_lock(&mapLock);

  //LINFO("new input.....");
  // store current color image:
  itsCurrImg = img;
  itsCurrImgWidth  = img.getWidth();
  itsCurrImgHeight = img.getHeight();
  //itsWin->setDims(Dims(2*itsCurrImgWidth, 2*itsCurrImgHeight));

  // also kill any old output and internals:
  itsSalmap.freeMem();
  gotLum = false; gotRGBY = false;
  pthread_mutex_unlock(&mapLock);

  // setup job queue:
  pthread_mutex_lock(&jobLock);
  jobQueue.clear();

  // color channel
  jobQueue.push_back(jobData(REDGREEN, Gaussian5, CWEIGHT, 0.0F));
  jobQueue.push_back(jobData(BLUEYELLOW, Gaussian5, CWEIGHT, 0.0F));

  // intensity channel
  jobQueue.push_back(jobData(INTENSITY, Gaussian5, IWEIGHT, 0.0F));

  // orientation channel
  jobQueue.push_back(jobData(ORI0, Oriented5, OWEIGHT, 0.0F));
  jobQueue.push_back(jobData(ORI45, Oriented5, OWEIGHT, 45.0F));
  jobQueue.push_back(jobData(ORI90, Oriented5, OWEIGHT, 90.0F));
  jobQueue.push_back(jobData(ORI135, Oriented5, OWEIGHT, 135.0F));

  jobsTodo = jobQueue.size();
  pthread_mutex_unlock(&jobLock);

  // broadcast on job queue condition to wake up worker threads:
  pthread_cond_broadcast(&jobCond);
  //LINFO("new input ok.....");
}

// ######################################################################
Image<float> BeobotBrainMT::getSalMap()
{
  Image<float> ret;

  pthread_mutex_lock(&mapLock);
  ret = itsSalmap;
  pthread_mutex_unlock(&mapLock);

  return ret;
}

// ######################################################################
Image<double> BeobotBrainMT::getGist()
{
  Image<double> ret;
  pthread_mutex_lock(&gistLock);
  ret = itsGistVector;
  pthread_mutex_unlock(&gistLock);

  return ret;
}

// ######################################################################
// The threaded function
void BeobotBrainMT::threadCompute()
{
  pthread_mutex_lock(&mapLock);
  uint myNum = numWorkers ++;
  pthread_mutex_unlock(&mapLock);
  LINFO("  ... worker %u ready.", myNum);

  while(true)
    {
      // wait until there are jobs in the queue that we can process:
      pthread_mutex_lock(&jobLock);
      jobData current(0, Gaussian5, 0.0F, 0.0F); bool nojobs = true;
      if (jobQueue.empty() == false)
        {
          current = jobQueue.front();
          jobQueue.pop_front();
          nojobs = false;
        }
      else
        pthread_cond_wait(&jobCond, &jobLock);
      pthread_mutex_unlock(&jobLock);

      // if we don't have a job to do, just wait more:
      if (nojobs) continue;
      //LINFO("[%u] GOT: job type %d", myNum, int(current.jobType));

      // read next entry in job queue and perform desired action on
      // current image and record result in output image
      // (accumulative)
      Image<float> curImage;

      // The case statement on this end parses the desired action from
      // the job queue and performs the needed image pre-processing
      pthread_mutex_lock(&mapLock);
      switch(current.jobType)
        {
          // While shared resources are used here, they are only read,
          // so they should not need to be protected by mutexers

          // ##################################################
        case REDGREEN:
          if (gotRGBY == false)
            {
              getRGBY(itsCurrImg, rgimg, byimg, 25.5f);
              gotRGBY = true;
            }
            curImage = rgimg;
          break;

          // ##################################################
        case BLUEYELLOW:
          if (gotRGBY == false)
            {
              getRGBY(itsCurrImg, rgimg, byimg, 25.5f);
              gotRGBY = true;
            }
          curImage = byimg;
          break;

          // ##################################################
        case ORI0:
        case ORI45:
        case ORI90:
        case ORI135:
        case INTENSITY:
          if (gotLum == false)
            {
              itsCurrLumImg =
                Image<float>(luminance(Image<PixRGB<float> >(itsCurrImg)));
              gotLum = true;
            }
          curImage = itsCurrLumImg;
          break;

          // ##################################################
        default:
          LERROR("invalid job type");
          curImage = itsCurrLumImg;
        }
      pthread_mutex_unlock(&mapLock);

      ImageSet<float> pyr;

      // alloc conspicuity map and clear it:
      int cW = itsCurrImgWidth /(int)(pow(2,sml));
      int cH = itsCurrImgHeight/(int)(pow(2,sml));
      Image<float> cmap(cW,cH, ZEROS);

      // get the proper conspicuity map and get its corresponding gist vector
      // The case statement on this end parses the desired action from
      // the job queue and performs the needed image pre-processing
      int count = 0; uint cscount = 0;
      int s_index = 0; int offset = 0;
      switch(current.jobType)
        {
          // ##################################################
        case INTENSITY:  s_index +=  6;
        case BLUEYELLOW: s_index +=  6;
        case REDGREEN:   s_index += 16;

          // compute & store oriented pyramid:
          pyr = buildPyrGaussian(curImage, 0, maxdepth, 5);
          itsImgPyrs[current.jobType] = pyr;

          // intensities is the max-normalized weighted sum of IntensCS:
          // we also add the gist extraction line
          for (int lev = level_min; lev <= level_max; lev ++)
            for (int delta = delta_min; delta <= delta_max; delta ++)
              {
                Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);

                // extract and store the gist
                LDEBUG("%d_%d_%d",current.jobType,lev,lev+delta);
                pthread_mutex_lock(&gistLock);
                inplacePaste(itsGistVector, getSubSum(tmp),
                             Point2D<int>(0, (s_index + count)*NUM_GIST_FEAT));
                pthread_mutex_unlock(&gistLock);
                count++;

                // accumulate the CMaps
                tmp = rescale(tmp, cmap.getWidth(), cmap.getHeight());
                //tmp = downSize(tmp, cmap.getWidth(), cmap.getHeight());
                float minv, maxv; getMinMax(tmp, minv, maxv);
                itsRawCSMaps[current.jobType][cscount] = tmp;
                tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
                cmap += tmp;
                itsCSMaps[current.jobType][cscount] = tmp; cscount++;
              }
          // store the center surround maps
          break;

          // ##################################################

          // ##################################################
        case ORI135: offset += 1;
        case ORI90:  offset += 1;
        case ORI45:  offset += 1;
        case ORI0:

          // compute & store oriented pyramid:
          pyr = buildPyrOriented(curImage, 0, maxdepth, 9,
                                 offset*45.0f, 10.0f);
          itsImgPyrs[current.jobType] = pyr;

          // extract the gist for the orientation channel
          for(int i = 0; i < NUM_GIST_LEV; i++)
            {
              LDEBUG("or_%f_%d",offset*45.0f,i);
              pthread_mutex_lock(&gistLock);
              inplacePaste
                (itsGistVector, getSubSum(pyr[i]),
                 Point2D<int>(0, (i * NUM_GIST_LEV + offset) * NUM_GIST_FEAT));
              pthread_mutex_unlock(&gistLock);
             }

          // intensities is the max-normalized weighted sum of IntensCS:
          for (int lev = level_min; lev <= level_max; lev ++)
            for (int delta = delta_min; delta <= delta_max; delta ++)
              {
                Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
                //tmp = downSize(tmp, cmap.getWidth(), cmap.getHeight());
                tmp = rescale(tmp, cmap.getWidth(), cmap.getHeight());
                itsRawCSMaps[current.jobType][cscount] = tmp;
                tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
                cmap += tmp;

                itsCSMaps[current.jobType][cscount] = tmp; cscount++;
              }
          // store the center surround maps

          break;
          // ##################################################
        default:
          LERROR("invalid job type");
        }

      // store the Conspicuity maps
      itsCMaps[current.jobType] = cmap;

      // add noise to the saliency map
      inplaceAddBGnoise(cmap, 25.0F);

      if (normtyp == VCXNORM_MAXNORM)
        cmap = maxNormalize(cmap, MAXNORMMIN, MAXNORMMAX, normtyp);
      else
        cmap = maxNormalize(cmap, 0.0f, 0.0f, normtyp);

      // multiply by conspicuity coefficient:
      if (current.weight != 1.0F) cmap *= current.weight;

      // Add to saliency map:
      pthread_mutex_lock(&mapLock);
      if (itsSalmap.initialized()) itsSalmap += cmap;
      else itsSalmap = cmap;
      pthread_mutex_unlock(&mapLock);

      pthread_mutex_lock(&jobLock);
      jobsTodo--;

      //LINFO("done with job type %d, %u todo...",
      //      int(current.jobType),jobsTodo);
      if(jobsTodo == 0)
        {
          // get the objects found
          findObjects();

          // compute the salient features of the objects
          computeSalientFeatures();

          itsProcessTime = itsTimer->get()/1000.0F;
        }
      pthread_mutex_unlock(&jobLock);
    }
}

// ######################################################################
void BeobotBrainMT::findObjects()
{
  // reset everything
  itsWinner.clear();
  itsObjRect.clear();
  itsWinningChan.clear();
  itsWinningSubmapNum.clear();
  itsWinningMap.clear();
  itsObjectMask.clear();

  int scale = (int)(pow(2,sml));
  Image<float> currSalMap = itsSalmap;
  Image<byte> accMask(currSalMap.getDims(),ZEROS);
  int area = 0;

  // find most salient location
  float maxval; Point2D<int> currwin; findMax(currSalMap, currwin, maxval);
  float cmval = maxval;
  int maxArea = currSalMap.getSize();
  LDEBUG("max val: %f", maxval);
  // criteria of keep searching for objects
  //   space to segment < 50%
  //   number of objects < 5
  //   strength of value is higher than 5%
  uint i = 0;
  while((.05 *maxval < cmval) && (i < 5) && (float(area)/maxArea < .5))
    {
      // get the next object
      Point2D<int> cwins = currwin*scale;
      itsWinner.push_back(cwins);
      setWinningMap(cwins);
      Image<float> cobjmask = getCurrObjectMask();
      itsObjectMask.push_back(cobjmask);
      Rectangle crect = getSEBoundingBox();
      itsObjRect.push_back(crect);

      // suppress the previous salient region
      Image<float> temp(cobjmask);
      temp = convGauss<float>(temp,4,4,2);
      inplaceNormalize(temp, 0.0F,3.0F);
      inplaceClamp(temp, 0.0F, 1.0F);
      Image<float> supMask(temp);

      //Image<float> supMask(cobjmask);
      inplaceNormalize(supMask, 0.0F, 1.0F);
      supMask = (supMask * -1.0F) + 1.0F;
      currSalMap *= supMask;

      // get the next salient location
      findMax(currSalMap, currwin, cmval);

      // get amount of space segmented out
      accMask += cobjmask;
      area = int(sum(cobjmask)/255.0);
      LDEBUG("accMask: %d cmval: %f", area, cmval);

      // check for rectangle overlap
      float maxovl = 0.0;
      for(uint j = 0; j < itsObjRect.size()-1; j++)
        {
          Rectangle covl = crect.getOverlap(itsObjRect[j]);
          float pcnt = 0.0;
          if(covl.isValid())
            {
              pcnt = (float(covl.area())/crect.area() +
                      float(covl.area())/itsObjRect[j].area())/2.0;
              if(maxovl < pcnt) maxovl = pcnt;
              //LINFO("area: %d pcnt[%d]: %f", covl.area(), j, pcnt);
            }
          //else LINFO("area: 0 pcnt[%d]: 0.0",j);
        }

      // too much overlap
      if(maxovl > .66)
        {
          // pop back the last inserted object
          itsWinner.pop_back();
          itsObjRect.pop_back();
          itsWinningChan.pop_back();
          itsWinningSubmapNum.pop_back();
          itsWinningMap.pop_back();
          itsObjectMask.pop_back();
          LDEBUG("object overlap with previous ones > .66");
        }
      else
        {
          LDEBUG("sal pt %u done",i); i++;
        }
      //Raster::waitForKey();
    }
}

// ######################################################################
bool BeobotBrainMT::setWinningMap(Point2D<int> winner)
{
  Point2D<int> scwin = downScaleCoords(winner, itsCMaps[0].getDims());

  // loop over the channels
  uint wchan; float mx = -1.0F;
  for (uint i = 0; i < itsNumChannels; ++i)
    {
      Image<float> output = itsCMaps[i] * itsChanWeight[i];
      float curr_val = output.getVal(scwin);
      if (curr_val >= mx){mx = curr_val; wchan = i;}
    }
  //if (mx <= 0.0F) return false;

  // loop over the csmaps
  uint wschan; mx = -1.0F;
  for (uint j = 0; j < itsCSMaps[wchan].size(); j++)
    {
      Image<float> csmap = itsCSMaps[wchan][j];
      float curr_val = csmap.getVal(scwin);
      if (curr_val >= mx) { mx = curr_val; wschan = j; }

      //imean  = float(mean(csmap)); istdev = float(stdev(csmap));
      //LINFO("    csmap[%d][%d]: val: %f [%f,%f] -> %f Min: %f, Max: %f",
      //      i,j, curr_val, imean, istdev,
      //      (curr_val - imean)/istdev, min, max);
    }

//  // loop over the submaps of the winning channel
//   mx = -1.0F;
//   for (uint i = 0; i < itsImgPyrs[wchan].size(); ++i)
//     {
//       Image<float> submap = (itsImgPyrs[wchan])[i];
//       float curr_val =
//         submap.getVal(downScaleCoords(winner,submap.getDims()));
//       LINFO("submap = %i -> val = %f", i, curr_val);
//       if (curr_val >= mx) { mx = curr_val; wschan = i; }
//     }
//   if (mx <= 0.0F) return false;

  // store the results
  itsWinningChan.push_back(wchan);
  itsWinningSubmapNum.push_back(wschan);
  itsWinningMap.push_back(itsCSMaps[wchan][wschan]);

  LDEBUG("[%d,%d] winner: chan(%d), sub-chan(%d)",
         winner.i, winner.j, wchan, wschan);
  return true;
}

// ######################################################################
Point2D<int> BeobotBrainMT::getWinningChannelIndex(Point2D<int> winner)
{
  if(itsCMaps.size() == 0) return Point2D<int>(-1,-1);
  
  Point2D<int> scwin = downScaleCoords(winner, itsCMaps[0].getDims());

  // loop over the channels
  int wchan = -1; float mx = -1.0F;
  for (uint i = 0; i < itsNumChannels; ++i)
    {
      Image<float> output = itsCMaps[i] * itsChanWeight[i];
      float curr_val = output.getVal(scwin);
      if (curr_val >= mx){mx = curr_val; wchan = int(i);}
    }
  //if (mx <= 0.0F) return false;

  // loop over the csmaps
  int wschan = -1; mx = -1.0F;
  for (uint j = 0; j < itsCSMaps[wchan].size(); j++)
    {
      Image<float> csmap = itsCSMaps[wchan][j];
      float curr_val = csmap.getVal(scwin);
      if (curr_val >= mx) { mx = curr_val; wschan = int(j); }
    }

  LDEBUG("[%d,%d] winner: chan(%d), sub-chan(%d)",
         winner.i, winner.j, wchan, wschan);
  return Point2D<int>(wchan,wschan);
}

// ######################################################################
Point2D<int> BeobotBrainMT::downScaleCoords(Point2D<int> winner, Dims targetDims)
{

  int i = (winner.i * targetDims.w()) / itsCurrImgWidth;
  int j = (winner.j * targetDims.h()) / itsCurrImgHeight;
  return Point2D<int>(i,j);
}

// ######################################################################
Image<byte> BeobotBrainMT::getCurrObjectMask()
{
  int index = itsWinningMap.size() - 1;
  Image<float> winMapNormalized = itsWinningMap[index];
  inplaceNormalize(winMapNormalized, 0.0F, 1.0F);

  Point2D<int> wp = downScaleCoords(itsWinner[index],
                               itsWinningMap[index].getDims());

  // the winner usually falls at the junction of four pixels in the
  // saliency/conspicuity/feature map - have a look at all of the pixels
  std::vector<Point2D<int> > surr(4);
  surr[0] = Point2D<int>(0,0); surr[1] = Point2D<int>(-1,0);
  surr[2] = Point2D<int>(0,1); surr[3] = Point2D<int>(-1,1);

  bool haveValue = false;
  for (uint i = 0; i < surr.size(); ++i)
    {
      Point2D<int> wp2 = wp + surr[i];
      wp2.clampToDims(itsWinningMap[index].getDims());
      if (itsWinningMap[index].getVal(wp2) > 0.0F)
        {
          wp = wp2; haveValue = true; break;
        }
    }

  Image<byte> objMask;
  if (haveValue) objMask = segmentObjectClean(winMapNormalized, wp, 4);
  return objMask;
}

// ######################################################################
Rectangle BeobotBrainMT::getSEBoundingBox()
{
  int scale = (int)(pow(2,sml));

  // get bounding box and add 1 pixel padding
  int index = itsObjectMask.size() - 1;
  Rectangle r = findBoundingRect(itsObjectMask[index], byte(255));
  int tr = r.top();    if(tr > 0) tr--;
  int lr = r.left();   if(lr > 0) lr--;
  int br = r.bottomO(); if(br < itsCurrImgHeight/scale) br++;
  int rr = r.rightO();  if(rr < itsCurrImgWidth/scale)  rr++;
  Rectangle rsc =
    Rectangle::tlbrO(tr*scale, lr*scale, br*scale, rr*scale);

  // correct the salient region to optimize SIFT recognition
  int w = itsCMaps[0].getWidth()*scale;
  int h = itsCMaps[0].getHeight()*scale;
  Rectangle rscc = correctBB(rsc, itsWinner[index], w, h);

  return rscc;
}

// ######################################################################
Rectangle BeobotBrainMT::correctBB(Rectangle r, Point2D<int> p, int w, int h)
{
  float wlow = .35;        float whigh = .5;
  float hlow = .35;        float hhigh = .5;

  // pixel slack
  int pslack = 5;

  int tr = r.top();       int lr = r.left();
  int br = r.bottomI();    int rr = r.rightI();
  int wr = r.width();     int hr = r.height();
  LDEBUG("SE bb [%3d,%3d,%3d,%d](%dx%d): p:[%d,%d]",
         tr, lr, br, rr, wr, hr, p.i, p.j);

  // before we start make sure that
  // the salient point is within the rectangle
  if(((lr + pslack) > p.i) || ((rr - pslack) < p.i) ||
     ((tr + pslack) > p.j) || ((br - pslack) < p.j)   )
    {
      LDEBUG("point is out of or near border of box");

      // expand window to get the points
      // to be more in the center
      if((lr + pslack) > p.i)
        { lr = p.i - pslack; if(lr < 0  ) lr = 0;  LDEBUG("left  "); }
      if((rr - pslack) < p.i)
        { rr = p.i + pslack; if(rr > w-1) rr = w-1;LDEBUG("right "); }
      if((tr + pslack) > p.j)
        { tr = p.j - pslack; if(tr < 0  ) tr = 0;  LDEBUG("top   "); }
      if((br - pslack) < p.j)
        { br = p.j + pslack; if(br > h-1) br = h-1;LDEBUG("bottom"); }

      wr = rr - lr;
      hr = br - tr;

      LDEBUG("CNT SE bb [%3d,%3d,%3d,%d](%dx%d) p: [%d,%d]",
             tr, lr, br, rr, wr, hr, p.i, p.j);
    }

  // correct for size

  // correct the width: make sure it's not too small
  if(wr < wlow * w)
    {
      int wdiff = int(wlow * w) - wr;
      float pnt = float(wdiff)/(wr + 0.0f);
      LDEBUG("enlarge width (%d -> %d) by: %d (%f)",
             wr, int(wlow * w), wdiff, pnt);

      int ls = 0, rs = 0;
      if((p.i - lr)/(wr+0.0) > .7)
        {
          ls = int(.7 * wr); rs = int(.3 * wr); LDEBUG("left skew");
        }
      else if((rr - p.i)/(wr+0.0) > .7)
        {
          ls = int(.3 * wr); rs = int(.7 * wr); LDEBUG("right skew");
        }
      else
        {
          ls = p.i - lr; rs = rr - p.i; LDEBUG("horz centered");
        }
      int ladd = int(ls * pnt);
      int radd = int(rs * pnt);
      LDEBUG("add: l: %d, r: %d", ladd, radd);

      if(lr < ladd)
        {
          rr += radd + ladd - lr; lr = 0;
          LDEBUG("hit left border: lr: %d, rr:%d", lr, rr);
        }
      else if(w - 1 < (rr + radd))
        {
          ladd += rr + radd - w + 1;
          lr = lr - ladd;
          rr = w - 1;
          LDEBUG("hit right border: lr: %d, rr:%d", lr, rr);
        }
      else
        {
          lr = lr - ladd; rr = rr + radd;
          LDEBUG("centered: lr: %d, rr:%d", lr, rr);
        }
    }

  // or too large
  else if(wr > whigh * w)
    {
      int wdiff = wr - int(whigh * w);
      float pnt = float(wdiff)/(wr + 0.0f);
      LDEBUG("decrease width (%d -> %d) by: %d (%f)",
            wr, int(whigh * w), wdiff, pnt);
      int mid = (lr + rr)/2;
      LDEBUG("mid horz: %d", mid);

      // if the sal pt is to the left of mid point
      if(mid > p.i)
        {
          // check sal point slack before cutting
          int ldiff = p.i - lr;
          LDEBUG("Point more left %d, right %d", ldiff, rr - p.i);
          if(ldiff >= pslack)
            {
              int cut = int (ldiff * pnt);
              lr = lr + cut;
              wdiff-= cut;
              LDEBUG("cut left slack %d, lr %d, wdiff now: %d",
                     cut, lr, wdiff);
            }
          else LDEBUG("left is not cut");

          rr-= wdiff; LDEBUG("rr: %d", rr);
        }
      // if it's to the right
      else
        {
          // check sal point slack before cutting
          int rdiff = rr - p.i;
          LDEBUG("Point more right %d left: %d", rdiff, p.i - lr);
          if(rdiff >= pslack)
            {
              int cut = int (rdiff * pnt);
              rr = rr - cut;
              wdiff-= cut;
              LDEBUG("cut right slack %d, rr %d, wdiff now: %d",
                     cut, rr, wdiff);
            }
          else LDEBUG("right is not cut");

          lr+= wdiff; LDEBUG("lr: %d", lr);
        }
    }

  // correct the height: make sure it's not too small
  if(hr < hlow * h)
    {
      int hdiff = int(hlow * h) - hr;
      float pnt = float(hdiff)/(hr + 0.0f);
      LDEBUG("increase the height (%d -> %d) by: %d (%f)",
             hr, int(hlow * h), hdiff, pnt);

      int ts = 0, bs = 0;
      if((p.j - tr)/(hr+0.0) > .7)
        {
          ts = int(.7 * hr); bs = int(.3 * hr); LDEBUG("top skew");}
      else if((br - p.j)/(hr+0.0) > .7)
        {
          ts = int(.3 * hr); bs = int(.7 * hr); LDEBUG("bottom skew");
        }
      else
        {
          ts = p.j - tr; bs = br - p.j; LDEBUG("vert centered");
        }

      int tadd = int(ts * pnt);
      int badd = int(bs * pnt);
      LDEBUG("adding: t: %d, b: %d", tadd, badd);

      if(tr < tadd)
        {
          br += badd + tadd - tr; tr = 0;
          LDEBUG("hit top border: tr: %d, br:%d", tr, br);
        }
      else if(h - 1 < (br + badd))
        {
          tadd += br + badd - h + 1;
          tr = tr - tadd;
          br = h - 1;
          LDEBUG("hit bottom border: tr: %d, br:%d", tr, br);
        }
      else
        {
          tr = tr - tadd; br = br + badd;
          LDEBUG("centered vertically: lr: %d, rr:%d", tr, br);
        }
    }
  // or too large
  else if(hr > hhigh * h)
    {
      int hdiff = hr - int(hhigh * h);
      float pnt = float(hdiff)/(hr + 0.0f);
      LDEBUG("enlarge the heigth (%d -> %d) by: %d (%f)",
             hr, int(hhigh * h), hdiff, pnt);
      int mid = (tr + br)/2;
      LDEBUG("mid vert: %d", mid);

      // if the sal pt is on top of midpoint
      if(mid > p.j)
        {
          // check sal point slack before cutting
          int tdiff = p.j - tr;
          LDEBUG("Point more top %d, bottom %d", tdiff, br - p.j);
          if(tdiff >= pslack)
            {
              int cut = int (tdiff * pnt);
              tr = tr + cut;
              hdiff-= cut;
              LDEBUG("cut top slack %d, tr %d, hdiff now: %d",
                     cut, tr, hdiff);
            }
          else LDEBUG("top not cut");

          br-= hdiff; LDEBUG("br: %d", br);
        }
      // if it's lower than the midpoint
      else
        {
          // check sal point slack before cutting
          int bdiff = br - p.j;
          LDEBUG("Point more bottom %d top: %d", bdiff, p.j - tr);
          if(bdiff >= pslack)
            {
              int cut = int (bdiff * pnt);
              br = br - cut;
              hdiff-= cut;
              LDEBUG("cut bottom slack %d, br %d, wdiff now: %d",
                     cut, br, hdiff);
            }
          else LDEBUG("bottom not cut");

          tr+= hdiff; LDEBUG("tr: %d", tr);
        }
    }

  // also, after we are finished make sure that
  // the salient point is within the rectangle
  if(((lr + pslack) > p.i) || ((rr - pslack) < p.i) ||
     ((tr + pslack) > p.j) || ((br - pslack) < p.j)   )
    {
      LDEBUG("point is out of or near border of box");
      if((lr + pslack) > p.i)
        { lr = p.i - pslack; if(lr < 0  ) lr = 0;   LDEBUG("left  "); }
      if((rr - pslack) < p.i)
        { rr = p.i + pslack; if(rr > w-1) rr = w-1; LDEBUG("right "); }
      if((tr + pslack) > p.j)
        { tr = p.j - pslack; if(tr < 0  ) tr = 0;   LDEBUG("top   "); }
      if((br - pslack) < p.j)
        { br = p.j + pslack; if(br > h-1) br = h-1; LDEBUG("bottom"); }
    }

  Rectangle newR = Rectangle::tlbrI(tr, lr, br, rr);
  LDEBUG("SE bb [%3d,%3d,%3d,%d](%dx%d): p:[%d,%d]",
         tr, lr, br, rr, newR.width(), newR.height(), p.i, p.j);
  return newR;
}

// ######################################################################
void BeobotBrainMT::computeSalientFeatures()
{
  itsSalientFeatures.resize(itsWinner.size());
  for(uint i = 0; i < itsWinner.size(); i++)
    {
      // The coordinates we receive are at the scale of the original
      // image, and we will need to rescale them to the size of the
      // various submaps we read from. The first image in our first
      // pyramid has the dims of the input:
      //float iw = float(itsImgPyrs[0][0].getWidth());
      //float ih = float(itsImgPyrs[0][0].getHeight());
      itsSalientFeatures[i].clear();
      Point2D<int> locn = itsWinner[i];
      for(uint c = 0; c < itsNumChannels; c++)
        {
          for(int di = -2; di < 3; di++)
            for(int dj = -2; dj < 3; dj++)
              {
                uint fidx = 0;
                for (int lev = level_min; lev <= level_max; lev ++)
                  for (int delta = delta_min; delta <= delta_max; delta ++)
                    {
                      // CS maps implementations
                      // Features are all normalized [ 0.0 ... 1.0 ]
                      // but if range is < 1.0 (no dominant peak) val = 0.0
                      int csi = locn.i/4, csj = locn.j/4;
                      float rval = 0.0;
                      if(itsCSMaps[c][fidx].coordsOk(csi + di, csj + dj))
                        rval = fabs(itsCSMaps[c][fidx].getVal(csi, csj));
                      float minv, maxv;
                      getMinMax(itsCSMaps[c][fidx], minv, maxv);
                      float val;
                      if((maxv - minv) < 1.0) val = 0.0F;
                      else val = (rval - minv)/(maxv - minv);
                      //LINFO("(%d,%d:%d,%d):[%f %f]  %f -> %f",
                      //      csi, csj ,clev, slev, minv, maxv, rval, val);

                      itsSalientFeatures[i].push_back(double(val));
                      fidx++;
                    }
              }
        }
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
