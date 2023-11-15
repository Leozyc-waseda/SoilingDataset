/*!@file RCBot/SceneRec.C  Recognize scenes                             */
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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/SceneRec.C $
// $Id: SceneRec.C 14125 2010-10-12 06:29:08Z itti $
//

#include "RCBot/SceneRec.H"
#include "Component/OptionManager.H"
#include "Image/CutPaste.H"
#include <cstdio> // for sprintf

#define DEBUG
#ifdef DEBUG

#include "GUI/XWinManaged.H"
#include "Image/DrawOps.H"
XWinManaged xwin(Dims(320,240), -1, -1, "SceneRec");

#endif

// ######################################################################
// The working Thread
void *SceneRecWorkTh(void *c)
{
  SceneRec *d = (SceneRec*)c;
  d->computeLocation();
  return NULL;
}

// ######################################################################
SceneRec::SceneRec(OptionManager& mgr, const std::string& descrName,
                   const std::string& tagName):
  ModelComponent(mgr, descrName, tagName)
{ }

// ######################################################################
void SceneRec::start1()
{
  pthread_mutex_init(&jobLock, NULL);
  pthread_mutex_init(&locLock, NULL);
  pthread_mutex_init(&vdbLock, NULL);
  pthread_cond_init(&jobCond, NULL);

  currentLeg = 0;
  jobDone = true;

  currentLegVdb = NULL;
  nextLegVdb = NULL;
  loadVisualDB(currentLeg);

  // start the working thread
  worker = new pthread_t[1];
  pthread_create(&worker[0], NULL, SceneRecWorkTh, (void *)this);
  usleep(100000);
}

// ######################################################################
void SceneRec::stop2()
{
  delete [] worker;
}

// ######################################################################
SceneRec::~SceneRec()
{  }

// ######################################################################
void SceneRec::newInput(const Image< PixRGB<byte> > img)
{
  pthread_mutex_lock(&jobLock);
  workImg = img;
  jobDone = false;
  pthread_mutex_unlock(&jobLock);

  // let the thread know we are ready for processing
  pthread_cond_broadcast(&jobCond);
}

// ######################################################################
bool SceneRec::outputReady()
{
  bool ret = false;

  pthread_mutex_lock(&jobLock);
  if (jobDone) ret = true;
  pthread_mutex_unlock(&jobLock);

  return ret;
}

// ######################################################################
short SceneRec::getLandmarkLoc(Point2D<int> &loc)
{
  pthread_mutex_lock(&locLock);
  loc = landmarkLoc;
  pthread_mutex_unlock(&locLock);

  return currentLeg;
}

// ######################################################################
void SceneRec::trainFeature(Image<PixRGB<byte> > &img, Point2D<int> loc,
                            Dims window, short leg)
{
  int width = img.getWidth();
  int height = img.getHeight();
  winsize = window;

  try {
    LDEBUG("Training loc (%i,%i) Win(%i,%i) leg=%i currentLeg=%i",
           loc.i, loc.j, window.w(), window.h(), leg, currentLeg);

    // if we are on a different leg, the load the correct vdb
    if (currentLeg != leg){
      pthread_mutex_lock(&vdbLock);
      loadVisualDB(leg);
      pthread_mutex_unlock(&vdbLock);
    }

    char name[255]; char fileName[255];
    sprintf(name, "Loc%i_%i", objNum, currentLeg);
    sprintf(fileName, "Loc%i_%i.png", objNum, currentLeg);

    // the location is at the center, move it to the top left
    Point2D<int> topLeft(loc.i-((window.w()-1)/2), loc.j-((window.h()-1)/2));

    // fix the window so that it does not go outside the image
    if (topLeft.i < 0 ) topLeft.i = 0;
    if (topLeft.i > width - window.w()) topLeft.i = width - window.w();

    if (topLeft.j < 0 ) topLeft.j = 0;
    if (topLeft.j > height - window.h()) topLeft.j = height - window.h();

    Image<PixRGB<byte> > templLoc = crop(img,topLeft, window);
    rutz::shared_ptr<VisualObject> voTempl(new VisualObject(name,fileName, templLoc));

    LINFO("Saving %s to %s", name, fileName);
    pthread_mutex_lock(&vdbLock);
    currentLegVdb->addObject(voTempl);
    sprintf(fileName, "path%i.vdb", currentLeg);
    currentLegVdb->saveTo(fileName);
    pthread_mutex_unlock(&vdbLock);
    objNum++;

  } catch(...) {
    LDEBUG("Error in training");
    pthread_mutex_unlock(&vdbLock);
  }
}

// ######################################################################
void SceneRec::computeLocation()
{
  while (true){
    // wait until a job comes
    pthread_mutex_lock(&jobLock);
    pthread_cond_wait(&jobCond, &jobLock);

    // got a job
    Image<PixRGB<byte> > img = workImg;
    pthread_mutex_unlock(&jobLock);

    LDEBUG("Got a location to recognize, currentLeg = %i", currentLeg);
    pthread_mutex_lock(&locLock);
    landmarkLoc.i = -1; landmarkLoc.j = -1;
    pthread_mutex_unlock(&locLock);

    try {
      std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
      unsigned int nmatch = 0;
      if (img.initialized()){
        rutz::shared_ptr<VisualObject> voimg(new VisualObject("PIC", "PIC", img));

        float scale; Point2D<int> loc; bool landmarkFound = false;

        // search the next location, if we do not look at the current one
        if (nextLegVdb->numObjects()) { // do we have any objects
          pthread_mutex_lock(&vdbLock);
          nmatch = nextLegVdb->getObjectMatches(voimg, matches,
                                                VOMA_KDTREEBBF);
          pthread_mutex_unlock(&vdbLock);

          getLandmarkInfo(scale, loc, nmatch, matches, img);
          LINFO("Next DB: Found landmark at (%i,%i), scale = %f", loc.i, loc.j, scale);
          if (nmatch > 0 && scale > 0.75) { //We found the next location
            loadVisualDB(currentLeg+1); //load the next vdbs
            landmarkFound = true;
          }
        }

        // the next Landmark is not found, search the current db
        if (!landmarkFound){
          if (currentLegVdb->numObjects()) { // do we have any objects
            pthread_mutex_lock(&vdbLock);
            nmatch = currentLegVdb->getObjectMatches(voimg, matches,
                                                     VOMA_KDTREEBBF);
            pthread_mutex_unlock(&vdbLock);

            getLandmarkInfo(scale, loc, nmatch, matches, img);
            LINFO("Curr DB: Found landmark at (%i,%i), scale = %f", loc.i, loc.j, scale);
            if (nmatch > 0){ //match at any scale
              landmarkFound = true;
            }
          }
        }

        if (landmarkFound) { // if we found the landmark then store the position
          pthread_mutex_lock(&locLock);
          landmarkLoc = loc;
          pthread_mutex_unlock(&locLock);
        }
      }

    } catch (...){
      LDEBUG("Got an error " );
      pthread_mutex_unlock(&vdbLock);
      pthread_mutex_unlock(&locLock);
    }

    // sleep(1);
    pthread_mutex_lock(&jobLock);
    jobDone = true;
    pthread_mutex_unlock(&jobLock);
  }
}

// ######################################################################
void SceneRec::getLandmarkInfo(float &scale, Point2D<int> &loc,
                               unsigned int nmatch,
                               std::vector< rutz::shared_ptr<VisualObjectMatch> > &matches,
                               Image<PixRGB<byte> > &img)
{
  LDEBUG("Found %i currentLeg %i\n", nmatch, currentLeg);
  if (nmatch > 0 )
  {
    // look at the first match
    for(unsigned int i = 0; i < 1; i++)
    {
      rutz::shared_ptr<VisualObjectMatch> vom = matches[i];
      rutz::shared_ptr<VisualObject> obj = vom->getVoTest();

      // get the number of the object
      const char *nameNum = obj->getName().c_str();
      int objNum = atoi(nameNum+3); // skip over Loc

      LDEBUG("### Object match with '%s' score=%f Number %i",
             obj->getName().c_str(), vom->getScore(), objNum);
      Point2D<int> tl, tr, br, bl;
      vom->getTransfTestOutline(tl, tr, br, bl);

      // find the center:
      Point2D<int> landmarkCenter( tl.i + ((br.i - tl.i)/2),
                              tl.j + ((br.j - tl.j)/2));
      loc = landmarkCenter;

#ifdef DEBUG
      drawLine(img, tl, tr, PixRGB<byte>(255, 0, 0));
      drawLine(img, tr, br, PixRGB<byte>(255, 0, 0));
      drawLine(img, br, bl, PixRGB<byte>(255, 0, 0));
      drawLine(img, bl, tl, PixRGB<byte>(255, 0, 0));
      drawDisk(img, Point2D<int>(landmarkCenter),
               4, PixRGB<byte>(255, 0, 0));
      xwin.drawImage(img);
#endif

      float size =  (br.i - tl.i)*(br.j - tl.j);
      LINFO("Size: %i %i %f", br.i - tl.i, br.j - tl.j, size);

      scale = size / (winsize.w()*winsize.h());
    }
  } else {
    scale = 0;
    loc.i = -1; loc.j = -1;
  }
}

// ######################################################################
void SceneRec::loadVisualDB(short leg)
{
  char fileName[255];

  //save the old dbs
  if (currentLegVdb) {
    if (currentLegVdb->numObjects()){
      sprintf(fileName, "path%i.vdb", currentLeg);
      currentLegVdb->saveTo(fileName);
    }
    delete currentLegVdb;
  }

  if (nextLegVdb){
    if (nextLegVdb->numObjects()){
      sprintf(fileName, "path%i.vdb", currentLeg+1);
      nextLegVdb->saveTo(fileName);
    }
    delete nextLegVdb;
  }

  // load dbs
  sprintf(fileName, "path%i.vdb", leg);
  currentLegVdb = new VisualObjectDB();
  currentLegVdb->loadFrom(fileName);
  objNum = currentLegVdb->numObjects();
  LINFO("Load current=%i(%i) next=%i", leg, objNum, leg+1);

  sprintf(fileName, "path%i.vdb", leg+1);
  nextLegVdb = new VisualObjectDB();
  nextLegVdb->loadFrom(fileName);

  currentLeg = leg;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
