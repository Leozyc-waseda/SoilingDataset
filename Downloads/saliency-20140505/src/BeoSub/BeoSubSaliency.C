/*!@file BeoSub/BeoSubSaliency.C A class for quick-and-dirty saliency mapping */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubSaliency.C $
// $Id: BeoSubSaliency.C 12074 2009-11-24 07:51:51Z itti $
//

#include "BeoSub/BeoSubSaliency.H"
#include "Demo/DemoOpts.H" //Is this worth it? FIX?

// ######################################################################
void *BeoSubSaliency_CMAP(void *c)
{
  BeoSubSaliency *d = (BeoSubSaliency *)c;
  d->computeCMAP();
  return NULL;
}

// ######################################################################
BeoSubSaliency::BeoSubSaliency(OptionManager& mgr,
           const std::string& descrName,
           const std::string& tagName):
  ModelComponent(mgr, descrName, tagName),
  itsNumThreads(&OPT_SMTnumThreads, this)
{
  MYLOGVERB = LOG_INFO;

  win = Point2D<int>(IMAGEWIDTH/2, IMAGEHEIGHT/2); // coordinates of attended location
  debugmode = false;
  hasRun = false;
  //start threads. They should go to sleep on the condition since no jobs have ben queued up yet
  pthread_mutex_init(&jobLock, NULL);
  pthread_mutex_init(&condLock, NULL);
  pthread_mutex_init(&mapLock, NULL);
  pthread_cond_init(&jobCond, NULL);

  // get our processing threads started:
  worker = new pthread_t[itsNumThreads.getVal()];
  for (uint i = 0; i < itsNumThreads.getVal(); i ++){
    pthread_create(&worker[i], NULL, BeoSubSaliency_CMAP, (void *)this);

    // all threads should go and lock against our job condition. Sleep a
    // bit to make sure this really happens: HACKY
    usleep(100000);
  }

}

// ######################################################################
BeoSubSaliency::~BeoSubSaliency(){

  pthread_cond_destroy(&jobCond);
  delete [] worker;
}

Point2D<int> BeoSubSaliency::run(Image< PixRGB<byte> > img, bool debug){

  debugmode = debug;
  dataReady = false;
  totalJobs = 0;
  jobsDone = 0;
  colima = img;

  if(debugmode && !hasRun){
    hasRun = true;
    wini.reset( new XWindow(img.getDims(), -1, -1, "input window") );
    wini->setPosition(0, 0);
    wino.reset( new XWindow(img.getDims(), -1, -1, "output window") );
    wino->setPosition(370, 0);
  }

  gotLum = false; gotRGBY = false;

   //a queue builder for the threads' job queue
  pthread_mutex_lock(&jobLock);
  jobQueue.clear();

  jobQueue.push_back(jobData(INTENSITY, Gaussian5, IWEIGHT, 0.0F));

  jobQueue.push_back(jobData(REDGREEN, Gaussian5, CWEIGHT, 0.0F));

  jobQueue.push_back(jobData(SKINHUE, Gaussian5, SWEIGHT, 0.0F));

  //jobQueue.push_back(jobData(ORI0, Oriented5, OWEIGHT, 0.0F));
  //jobQueue.push_back(jobData(ORI45, Oriented5, OWEIGHT, 45.0F));
  //jobQueue.push_back(jobData(ORI90, Oriented5, OWEIGHT, 90.0F));
  //jobQueue.push_back(jobData(ORI135, Oriented5, OWEIGHT, 135.0F));

  jobQueue.push_back(jobData(FLICKER, Gaussian5, FWEIGHT, 0.0F));

  jobQueue.push_back(jobData(BLUEYELLOW, Gaussian5, CWEIGHT, 0.0F));

  totalJobs = jobQueue.size();

  pthread_mutex_unlock(&jobLock);

  //broadcast on job queue condition to wake up worker threads
  pthread_cond_broadcast(&jobCond);

  //Make setup() busy-wait until a bool is set by the finishing thread NOTE: may be better to run this in a thread that sleeps on a condition
  while(!dataReady){//NEEDS TO BE FIXED SO THAT THIS CAN BE A WAIT ON A CONDITION (MAKE THIS A THREAD)!
    usleep(100);
  }

  //find the point of highest saliency
  findMax(outmap, winsm, maxval);

  float minOut, maxOut;

  //get values for display
  getMinMax(outmap, minOut, maxOut);
  LINFO("Min: %f Max: %f\n", minOut, maxOut);

  // rescale winner coordinates according to PRESCALE:
  win.i = winsm.i << sml;
  win.i += int(((1<<(sml-1)) * float(rand()))/RAND_MAX);
  win.j = winsm.j << sml;
  win.j += int(((1<<(sml-1)) * float(rand()))/RAND_MAX);

  //return position
  if(debugmode){
    Image<float> tmp = quickInterpolate(outmap, (1<<sml));
    inplaceNormalize(tmp, 0.0F, 255.0F);
    Image< PixRGB<byte> > temp = tmp;
    drawDisk(temp, win, 4, PixRGB<byte>(225, 225, 20));
    drawDisk(img, win, 4, PixRGB<byte>(225, 225, 20));
    wini->drawImage(img);
    wino->drawImage(temp);
  }

  LINFO("The point of highest saliency is: %d, %d\n", win.i, win.j);
  outmap.clear();
  return win;
}

// ######################################################################
//The threaded function
void BeoSubSaliency::computeCMAP()
{

  while(true){
    jobData current(0, Gaussian5, 0.0F, 0.0F);
    bool jobsEmpty = true;

    //check for another job in the queue. If there is one, process it
    pthread_mutex_lock(&jobLock);
    if(!jobQueue.empty()){
      current = jobQueue.front();
      jobQueue.pop_front();
      jobsEmpty = false;
    }
    else{
      jobsEmpty = true;
    }

    if(!jobsEmpty){
      //read next entry in job queue and perform desired action on current image and record result in output image (accumulative)
      Image<float> curImage;

      //The case statement on this end parses the desired action from the job queue and performs the needed image pre-processing
      pthread_mutex_lock(&mapLock);
      switch(current.jobType)
        {
          //While shared resources are used here, they are only read, so they should not need to be protected by mutexers
        case REDGREEN:        // ##############################
          {
            if (gotRGBY == false){
              getRGBY(colima, r, g, b, y, byte(25)); gotRGBY = true;
            }
            curImage = r-g;
          }
          break;
        case BLUEYELLOW:        // ##############################
          {
            if (gotRGBY == false){
              getRGBY(colima, r, g, b, y, byte(25)); gotRGBY = true;
            }
            curImage = b-y;
          }
          break;
        case SKINHUE:       // ##############################
          {
            skinima = hueDistance(colima, COL_SKIN_MUR, COL_SKIN_MUG,
                                  COL_SKIN_SIGR, COL_SKIN_SIGG,
                                  COL_SKIN_RHO);

            curImage = skinima;
            break;
          }
        case ORI0:        // ##############################
          {
          }
        case ORI45:        // ##############################
          {
          }
        case ORI90:        // ##############################
          {
          }
        case ORI135:        // ##############################
          {
          }
        case FLICKER:        // ##############################
          {
            if (gotLum == false){
              lum = Image<float>(luminance(colima)); gotLum = true;
            }
            // compute flicker consp map and send to collector:
            if (!previma.initialized()){
              previma = lum;
              curImage.resize(lum.getDims(), true); // clear
            }
            else{
              curImage = lum - previma;
              previma = lum;
            }
          }
          break;
        case INTENSITY:      // #############################
          {
            if (gotLum == false){
              lum = Image<float>(luminance(colima)); gotLum = true;
            }
            curImage = lum;
          }
          break;
        default:           // #############################
          {
            //should never get here
            LERROR("Attempt to pass an invalid jobtype DENIED");
            curImage = lum;
          }
          break;
          //add additional cases as more channels are added

        }
      pthread_mutex_unlock(&mapLock);

      // compute pyramid:
      ImageSet<float> pyr = buildPyrGeneric(curImage, 0, maxdepth,
                                            current.ptyp,
                                            current.orientation);

      // alloc conspicuity map and clear it:
      Image<float> cmap(pyr[sml].getDims(), ZEROS);

      // intensities is the max-normalized weighted sum of IntensCS:
      for (int delta = delta_min; delta <= delta_max; delta ++)
        for (int lev = level_min; lev <= level_max; lev ++)
          {
            Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
            tmp = downSize(tmp, cmap.getWidth(), cmap.getHeight());
            inplaceAddBGnoise(tmp, 255.0);
            tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
            cmap += tmp;
          }

      inplaceAddBGnoise(cmap, 25.0F);

      if (normtyp == VCXNORM_MAXNORM)
        cmap = maxNormalize(cmap, MAXNORMMIN, MAXNORMMAX, normtyp);
      else
        cmap = maxNormalize(cmap, 0.0f, 0.0f, normtyp);

      // multiply by conspicuity coefficient:
      if (current.weight != 1.0F) cmap *= current.weight;

      // inject newly received saliency map:
      pthread_mutex_lock(&mapLock);//lock outmap access

      /*Uncomment if statement to use exclusivity cmap*/

      //if (current.jobType == FLICKER)
      //  {

      if (outmap.initialized()){
        outmap += cmap;
      }
      else{
        outmap = cmap;
      }

      //  }

      jobsDone++;
      pthread_mutex_unlock(&mapLock);//unlock outmap access

     }
    else{
      //Set counter var used by run. Hacky since it needs to be changed every time a job is added.
      //pthread_mutex_lock(&condLock);
      if(jobsDone >= totalJobs){
        dataReady = true;
      }

    pthread_mutex_unlock(&jobLock);

    pthread_cond_wait(&jobCond, &condLock);

    pthread_mutex_unlock(&condLock);

    pthread_mutex_lock(&jobLock);

    }
    pthread_mutex_unlock(&jobLock);
  }
  return;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
