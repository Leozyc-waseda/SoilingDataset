/*!@file CMapDemo/SaliencyCMapMT.C A class for quick-and-dirty
  saliency mapping integrated with cmap CORBA object                    */
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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CMapDemo/SaliencyCMapMT.C $
// $Id: SaliencyCMapMT.C 14376 2011-01-11 02:44:34Z pez $
//

#include "CMapDemo/SaliencyCMapMT.H"
#include "Demo/DemoOpts.H"
#include "Corba/CorbaUtil.H"
#include "Image/PyrBuilder.H"
#include "Image/OpenCVUtil.H"

#include <sys/time.h>
#include <errno.h>

#ifdef HAVE_OPENCV_CV_H
#include <opencv/cv.h>
#endif

//are we using a local linked cmap server or a remote
//set this if we are running on only one computer
#ifdef LocalCMapServer
//#include "Corba/Objects/CMapServer.H"
#endif
// ######################################################################
// ##### Global options:
// ######################################################################


// relative feature weights:
#define IWEIGHT 1.0
#define CWEIGHT 1.0
#define OWEIGHT 1.0
#define FWEIGHT 4.0
#define SWEIGHT 0.7

// image size vars
#define IMAGEWIDTH 160
#define IMAGEHEIGHT 120


#define numthreads 1

// ######################################################################
void *SaliencyMT_CMAP(void *c)
{
  SaliencyMT *d = (SaliencyMT *)c;
  d->computeCMAP();
  return NULL;
}

// ######################################################################
SaliencyMT::SaliencyMT(OptionManager& mgr,CORBA::ORB_ptr orb, short saliencyMapLevel,
                       const std::string& descrName,
                       const std::string& tagName):
  ModelComponent(mgr, descrName, tagName), nCmapObj(0),
  itsNumThreads(&OPT_SMTnumThreads, this), // see Demo/DemoOpts.{H,C}
  SMBias(ImageSet<float>(14))

{
  numWorkers = 0U;

  if (!getMultiObjectRef(orb, "saliency.CMapServers", CMap_ref, nCmapObj)){
    LFATAL("Can not find any object to bind with");
  }
  for(int i=0; i<nCmapObj; i++)
    CMap::_narrow(CMap_ref[i])->setSaliencyMapLevel(saliencyMapLevel);
  biasSM = false;
}

void SaliencyMT::setBiasSM(bool val){
  biasSM = val;
}

// ######################################################################
void SaliencyMT::setSaliencyMapLevel(const short saliencyMapLevel){
  //set the saliency map level to return
  for(int i=0; i<nCmapObj; i++)
    CMap::_narrow(CMap_ref[i])->setSaliencyMapLevel(saliencyMapLevel);
}


// ######################################################################
//!The current cmap object to send the request to
CMap_ptr SaliencyMT::getCmapRef(){
  static int current_obj = 0;

  //just do a round robin
  current_obj  = (current_obj+1)%nCmapObj;

  LDEBUG("Using cmap object number %i\n", current_obj);

  return CMap::_narrow(CMap_ref[current_obj]);
}

// ######################################################################
void SaliencyMT::start1()
{
  // start threads. They should go to sleep on the condition since no
  // jobs have ben queued up yet:
  pthread_mutex_init(&jobLock, NULL);
  pthread_mutex_init(&mapLock, NULL);
  pthread_mutex_init(&jobStatusLock, NULL);
  pthread_cond_init(&jobCond, NULL);
  pthread_cond_init(&jobDone, NULL);

  LINFO("Starting with %u threads...", itsNumThreads.getVal());

  // get our processing threads started:
  worker = new pthread_t[itsNumThreads.getVal()];
  for (uint i = 0; i < itsNumThreads.getVal(); i ++)
    {
      pthread_create(&worker[i], NULL, SaliencyMT_CMAP, (void *)this);

      // all threads should go and lock against our job condition. Sleep a
      // bit to make sure this really happens:
      usleep(100000);
    }
}

// ######################################################################
void SaliencyMT::stop2()
{
  // should cleanup the threads, mutexes, etc...
  pthread_cond_destroy(&jobCond);

  //for (uint i = 0; i < numthreads; i ++)
  //  pthread_delete(&worker[i].....

  delete [] worker;
}

// ######################################################################
SaliencyMT::~SaliencyMT(){ }

// ######################################################################
void SaliencyMT::newInput(Image< PixRGB<byte> > img)
{
  //LINFO("new input.....");
  // store current color image:
  pthread_mutex_lock(&mapLock);
  colima = img;

  // also kill any old output and internals:
  outmap.freeMem();
  gotLum = false; gotRGBY = false; gotSkin = false;
  pthread_mutex_unlock(&mapLock);

  // setup job queue:
  pthread_mutex_lock(&jobLock);
  jobQueue.clear();

  jobQueue.push_back(jobData(INTENSITY, Gaussian5, IWEIGHT, 0.0F));

  jobQueue.push_back(jobData(REDGREEN, Gaussian5, CWEIGHT, 0.0F));
  jobQueue.push_back(jobData(BLUEYELLOW, Gaussian5, CWEIGHT, 0.0F));

  ////jobQueue.push_back(jobData(SKINHUE, Gaussian5, SWEIGHT, 0.0F));

  // jobQueue.push_back(jobData(ORI0, Oriented5, OWEIGHT, 0.0F));
  // jobQueue.push_back(jobData(ORI45, Oriented5, OWEIGHT, 45.0F));
  // jobQueue.push_back(jobData(ORI90, Oriented5, OWEIGHT, 90.0F));
  // jobQueue.push_back(jobData(ORI135, Oriented5, OWEIGHT, 135.0F));

  //jobQueue.push_back(jobData(FLICKER, Gaussian5, FWEIGHT, 0.0F));

  jobsTodo = jobQueue.size();
  pthread_mutex_unlock(&jobLock);

  // broadcast on job queue condition to wake up worker threads:
  pthread_cond_broadcast(&jobCond);
  //LINFO("new input ok.....");
}

// ######################################################################
bool SaliencyMT::outputReady()
{
  bool ret = false;

  pthread_mutex_lock(&jobLock);
  if (jobsTodo == 0U) ret = true;
  pthread_mutex_unlock(&jobLock);

  return ret;
}

// ######################################################################
Image<float> SaliencyMT::getOutput()
{
  Image<float> ret;

  pthread_mutex_lock(&mapLock);
  ret = outmap;
  pthread_mutex_unlock(&mapLock);

  return ret;
}

// ######################################################################
Image<float> SaliencyMT::getSMap(Image< PixRGB<byte> > img)
{
  Image<float> ret;

  newInput(img);

  LINFO("Getting smap");
  //wait for done signal
  pthread_mutex_lock(&jobStatusLock);

  LINFO("Waiting for smap");
  struct timeval abstime_tv;
  gettimeofday(&abstime_tv, NULL);

  struct timespec abstime;
  abstime.tv_sec = abstime_tv.tv_sec;
  abstime.tv_sec += 3; //wait 3 seconds for condition
  abstime.tv_nsec = 0;

  //pthread_cond_wait(&jobDone, &jobStatusLock);
  if (pthread_cond_timedwait(&jobDone, &jobStatusLock, &abstime) == ETIMEDOUT){
    LINFO("TIme out");
  }
  pthread_mutex_unlock(&jobStatusLock);

  LINFO("Getting smap ");

  pthread_mutex_lock(&mapLock);
  ret = outmap;
  pthread_mutex_unlock(&mapLock);

  return ret;
}

// ######################################################################
//Bias the main saliency map
void SaliencyMT::setSMBias(ImageSet<float> &bias){

  for(unsigned int i=0; i<bias.size(); i++){
    if (bias[i].initialized())
      SMBias[i] = bias[i];
  }
}

// ######################################################################
//Bias the CMap
void SaliencyMT::setBias(int type, std::vector<float> &bias)
{

  CMap::BiasSeq *curBias = NULL;
  switch (type) {
  case REDGREEN:
    curBias = &cmapBias.redgreen;
    break;
  case BLUEYELLOW:
    curBias = &cmapBias.blueyellow;
    break;
  case SKINHUE:
    curBias = &cmapBias.skinhue;
    break;
  case ORI0:
    curBias = &cmapBias.ori0;
    break;
  case ORI45:
    curBias = &cmapBias.ori45;
    break;
  case ORI90:
    curBias = &cmapBias.ori90;
    break;
  case ORI135:
    curBias = &cmapBias.ori135;
    break;
  case INTENSITY:
    curBias = &cmapBias.intensity;
    break;
  case FLICKER:
    curBias = &cmapBias.flicker;
    break;
  default:
    LINFO("Unknown type");
  }


  //assign the bias
  if (curBias != NULL){
    curBias->length(bias.size());
    for(unsigned int i=0; i<bias.size(); i++)
      (*curBias)[i] = bias[i];
  }

}

// ######################################################################
// Get the CMap
void SaliencyMT::getBias(Image< PixRGB<byte> > &ima,
                         std::vector<float> &bias, int type, Point2D<int> &loc)
{
  PyramidType ptype = Gaussian5;
  float weight = 0;
  float ori = 0;
  Image<byte> curImage;

  Image<float> local_lum;        //curent luminance image
  Image<byte> local_r, local_g, local_b, local_y;  //curent RGBY images
  Image<float> local_skinima;    //skin hue map

  switch (type) {
  case REDGREEN:
    ptype = Gaussian5;
    ori = 0.0F;
    getRGBY(ima, local_r, local_g, local_b, local_y, byte(25));
    curImage = local_r - local_g;
    break;

    // ##################################################
  case BLUEYELLOW:
    ptype = Gaussian5;
    ori = 0.0F;
    getRGBY(ima, local_r, local_g, local_b, local_y, byte(25));
    curImage = local_b - local_y;
    break;

    // ##################################################
  case SKINHUE:
    ptype = Gaussian5;
    ori = 0.0F;
    local_skinima = hueDistance(ima, COL_SKIN_MUR, COL_SKIN_MUG,
                                COL_SKIN_SIGR, COL_SKIN_SIGG,
                                COL_SKIN_RHO);
    curImage = local_skinima;
    break;

    // ##################################################
  case ORI0:
    ptype = Oriented5;
    ori = 0.0F;

    curImage = Image<byte>(luminance(ima));
    break;
  case ORI45:
    ptype = Oriented5;
    ori = 45.0F;
    curImage = Image<byte>(luminance(ima));
    break;
  case ORI90:
    ptype = Oriented5;
    ori = 90.0F;
    curImage = Image<byte>(luminance(ima));
    break;
  case ORI135:
    ptype = Oriented5;
    ori = 135.0F;
    curImage = Image<byte>(luminance(ima));
    break;
  case INTENSITY:
    ptype = Gaussian5;
    ori = 0.0F;
    curImage = Image<byte>(luminance(ima));
    break;

    // ##################################################
  case FLICKER:
    ptype = Gaussian5;
    ori = 0.0F;
    curImage = Image<byte>(luminance(ima));
    break;

    // ##################################################
  default:
    LERROR("What is going on around here?");

  }

  CMap_ptr CMap = getCmapRef();
  Point2DOrb locOrb;
  locOrb.i = loc.i; locOrb.j = loc.j;

  ImageOrb *curImageOrb = image2Orb(curImage);
  CMap::BiasSeq *curBias = CMap->getBiasCMAP(*curImageOrb, ptype, ori, weight, locOrb);
  delete curImageOrb;

  //assign the bias
  if (curBias != NULL){
    curBias->length(bias.size());
    for(unsigned int i=0; i<bias.size(); i++)
      bias[i] = (*curBias)[i];
  }
  delete curBias;

}

// ######################################################################
// the threaded function
void SaliencyMT::computeCMAP()
{
#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed to use this function");
#else
  pthread_mutex_lock(&mapLock);
  uint myNum = numWorkers ++;
  pthread_mutex_unlock(&mapLock);
  LINFO("  ... worker %u ready.", myNum);

  while(true)
    {
      // wait until there are jobs in the queue that we can process:
      pthread_mutex_lock(&jobLock);
      jobData current; bool nojobs = true;
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
      LDEBUG("[%u] GOT: job %d", myNum, int(current.jobType));

      // read next entry in job queue and perform desired action on
      // current image and record result in output image
      // (accumulative)
      Image<byte> curImage;

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
            { getRGBY(colima, r, g, b, y, byte(25)); gotRGBY = true; }
          curImage = r - g;
          break;

          // ##################################################
        case BLUEYELLOW:
          if (gotRGBY == false)
            { getRGBY(colima, r, g, b, y, byte(25)); gotRGBY = true; }
          curImage = b - y;
          break;

          // ##################################################
        case SKINHUE:
          if (gotSkin == false)
            {
              skinima = hueDistance(colima, COL_SKIN_MUR, COL_SKIN_MUG,
                                    COL_SKIN_SIGR, COL_SKIN_SIGG,
                                    COL_SKIN_RHO);
              gotSkin = true;
            }
          curImage = skinima;
          break;

          // ##################################################
        case ORI0:
          if (gotLum == false)
            { lum = Image<byte>(luminance(colima)); gotLum = true; }
          curImage = lum;
          break;
        case ORI45:
          if (gotLum == false)
            { lum = Image<byte>(luminance(colima)); gotLum = true; }
          curImage = lum;
          break;
        case ORI90:
          if (gotLum == false)
            { lum = Image<byte>(luminance(colima)); gotLum = true; }
          curImage = lum;
          break;
        case ORI135:
          if (gotLum == false)
            { lum = Image<byte>(luminance(colima)); gotLum = true; }
          curImage = lum;
          break;
        case INTENSITY:
          if (gotLum == false)
            { lum = Image<byte>(luminance(colima)); gotLum = true; }
          curImage = lum;
          break;

          // ##################################################
        case FLICKER:
          if (gotLum == false)
            { lum = Image<byte>(luminance(colima)); gotLum = true; }
          // compute flicker consp map and send to collector:
          if (prev.initialized() == false)
            {
              prev = lum;
              curImage.resize(lum.getDims(), true); // clear
            }
          else
            {
              curImage = lum - prev;
              prev = lum;
            }
          break;

          // ##################################################
        default:
          LERROR("What is going on around here?");
          curImage = lum;
        }
      pthread_mutex_unlock(&mapLock);

      /*
        CMap_var CMapObj = getCmapRef();        //get the object to send to

        ImageOrb *imgOrb;
        ImageOrb *curImageOrb = image2Orb(curImage);

        if (biasSM && curBias != NULL && curBias->length()){
        imgOrb = CMapObj->computeBiasCMAP(*curImageOrb,
        current.ptyp, current.orientation, current.weight, *curBias);
        } else {
        imgOrb = CMapObj->computeCMAP(*curImageOrb,
        current.ptyp, current.orientation, current.weight);
        }
        delete curImageOrb;

        Image<float> cmap;
        orb2Image(*imgOrb, cmap);
        delete imgOrb;
      */

      Image<float> cmap = curImage;

      //Image<float> biasedCMap;
      Image<float> biasedCMap(cmap.getWidth()-SMBias[current.jobType].getWidth()+1,
                              cmap.getHeight()-SMBias[current.jobType].getHeight()+1,
                              NO_INIT);

      if (biasSM){
        if (SMBias[current.jobType].initialized())
          cvMatchTemplate(img2ipl(cmap),
                          img2ipl(SMBias[current.jobType]),
                          img2ipl(biasedCMap),
                          //CV_TM_CCOEFF);
                          CV_TM_SQDIFF);

        // biasedCMap = correlation(cmap, SMBias[current.jobType]);
      }


      // Add to saliency map:
      pthread_mutex_lock(&mapLock);
      cmaps[current.jobType] = cmap;        //save the cmap

      if (biasSM){
        if (outmap.initialized()) outmap += biasedCMap;
        else outmap = biasedCMap;
      } else {
        if (outmap.initialized()) outmap += cmap;
        else outmap = cmap;
      }

      pthread_mutex_unlock(&mapLock);

      pthread_mutex_lock(&jobLock);
      -- jobsTodo;
      LDEBUG("done with job %d, %u todo...", int(current.jobType),jobsTodo);

      if (jobsTodo == 0U)         //Last job, let know that we are done for block calls
        pthread_cond_signal(&jobDone);


      pthread_mutex_unlock(&jobLock);
    }
#endif
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
