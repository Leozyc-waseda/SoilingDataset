/*!@file BeoSub/BeoSub.C An autonomous submarine */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSub.C $
// $Id: BeoSub.C 14376 2011-01-11 02:44:34Z pez $
//

#include "BeoSub/BeoSub.H"

#include "BeoSub/BeoSub-defs.H"
#include "BeoSub/BeoSubOpts.H"
#include "BeoSub/CannyModel.H"
#include "Devices/FrameGrabberFactory.H"
#include "Channels/RawVisualCortex.H"
#include "SIFT/VisualObjectDB.H"
#include "SIFT/VisualObjectMatch.H"

#define DEBUG

// ######################################################################
BeoSub::BeoSub(OptionManager& mgr, const std::string& descrName,
               const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),

  itsFrontVODBfname(&OPT_FrontVODBfname, this),
  itsFrontVODB(new VisualObjectDB()),
  itsDownVODBfname(&OPT_DownVODBfname, this),
  itsDownVODB(new VisualObjectDB()),
  itsUpVODBfname(&OPT_UpVODBfname, this),
  itsUpVODB(new VisualObjectDB()),
  itsMasterClock(1000000), itsCkPt(0), itsCurrentAttitude(),
  itsGlobalHeading(0),
  itsTargetAttitude(),
  itsVisualCortex(new RawVisualCortex(mgr)),
  itsFrontDBfname("/home/tmp/u/beosub/front.db"),
  itsFrontDB(new BeoSubDB()),
  itsDownDBfname("/home/tmp/u/beosub/down.db"),
  itsDownDB(new BeoSubDB()),
  itsUpDBfname("/home/tmp/u/beosub/up.db"),
  itsUpDB(new BeoSubDB()),
  itsShapeDetector(new BeoSubCanny(mgr)),
  itsTaskDecoder(new BeoSubTaskDecoder(mgr)),
  itsColorTracker(new ColorTracker(mgr)),
  itsColorTracker2(new ColorTracker(mgr)),
  decoderIsRed(true),
  taskAdone(false),
  taskBdone(false),
  taskCdone(false)
  //itscurrentblabla
{
  pthread_mutex_init(&itsLock, NULL);
  addSubComponent(itsVisualCortex);
  addSubComponent(itsShapeDetector);

  //set up default tasks order. Change if decoding is successful!
  itsTasks.push_back('A');
  itsTasks.push_back('B');
  itsTasks.push_back('C');
  itsTasksIter = itsTasks.begin();

  taskAposition.x = 15; taskAposition.y = 20;
  taskBposition.x = 5; taskBposition.y = 50;
  taskCposition.x = 10; taskCposition.y = 100;
  itsGlobalPosition.x = -19; itsGlobalPosition.y = -8;
  itsGlobalHeading = 60;
  // mgr.loadConfig("camconfig.pmap");
}

// ######################################################################
BeoSub::~BeoSub()
{
  pthread_mutex_destroy(&itsLock);
}

// ######################################################################
void BeoSub::start1()
{

  // load our visual object databases:
  itsFrontVODB->loadFrom(itsFrontVODBfname.getVal());
  itsDownVODB->loadFrom(itsDownVODBfname.getVal());
  itsUpVODB->loadFrom(itsUpVODBfname.getVal());

  // Pre-build the KDTrees on our visual object databases so that we
  // don't waste time on that later:
  if (itsFrontVODB->numObjects()) itsFrontVODB->buildKDTree();
  if (itsDownVODB->numObjects()) itsDownVODB->buildKDTree();
  if (itsUpVODB->numObjects()) itsUpVODB->buildKDTree();

  // load our position and sensor databases (assumes already initialized)
  itsFrontDB->loadDatabase(itsFrontDBfname);
  itsDownDB->loadDatabase(itsDownDBfname);
  itsUpDB->loadDatabase(itsUpDBfname);

  // get a hold of our special objects:
  itsVOtaskAdown = itsDownVODB->getObject("taskAdown");
  itsVOtaskAfront = itsFrontVODB->getObject("taskAfront");
  itsVOtaskBdown = itsDownVODB->getObject("taskBdown");
  itsVOtaskBfront = itsFrontVODB->getObject("taskBfront");
  itsVOtaskCdown = itsDownVODB->getObject("taskCdown");
  itsVOtaskCfront = itsFrontVODB->getObject("taskCfront");
  itsVOtaskCup = itsUpVODB->getObject("taskCtop");
  itsVOtaskGdown = itsDownVODB->getObject("taskGdown");
  itsVOtaskGfront = itsFrontVODB->getObject("taskGfront");
  itsVOtaskGup = itsUpVODB->getObject("taskGtop");
  itsVOtaskDfront = itsFrontVODB->getObject("taskDfront");

}

// ######################################################################
bool BeoSub::targetReached(const float tol) const
{
  // NOTE: we ignore pitch and roll
  bool ret = true;
  pthread_mutex_lock(const_cast<pthread_mutex_t *>(&itsLock));

  if (fabsf(itsCurrentAttitude.depth - itsTargetAttitude.depth) > tol * 0.2F)
    ret = false;

  Angle diff = itsCurrentAttitude.heading - itsTargetAttitude.heading;
  if (diff.getVal() < - tol * 10.0F || diff.getVal() > tol * 10.0F)
    ret = false;

  pthread_mutex_unlock(const_cast<pthread_mutex_t *>(&itsLock));

  return ret;
}

// ######################################################################
void BeoSub::turnAbs(const Angle finalHeading, const bool blocking)
{
  pthread_mutex_lock(&itsLock);
  itsTargetAttitude.heading = finalHeading;
  const Angle diff = finalHeading - itsCurrentAttitude.heading;
  pthread_mutex_unlock(&itsLock);
  if (blocking) waitMove(fabs(diff.getVal() * 0.5) + 30.0);
}

// ######################################################################
void BeoSub::pitchAbs(const Angle finalPitch, const bool blocking)
{
  pthread_mutex_lock(&itsLock);
  itsTargetAttitude.pitch = finalPitch;
  const Angle diff = finalPitch - itsCurrentAttitude.pitch;
  pthread_mutex_unlock(&itsLock);
  if (blocking) waitMove(fabs(diff.getVal() * 0.5) + 30.0);
}

// ######################################################################
void BeoSub::turnRel(const Angle relHeading, const bool blocking)
{
  /*
  pthread_mutex_lock(&itsLock);
  itsTargetAttitude.heading += relHeading;
  pthread_mutex_unlock(&itsLock);
  if (blocking) waitMove(fabs(relHeading.getVal() * 0.5) + 30.0);
  */
  //ANDRE -- NOTE: GlobalHeading value
  itsGlobalHeading = itsGlobalHeading + relHeading;
  turnOpen(relHeading, blocking);
}

// ######################################################################
void BeoSub::diveAbs(const float finalDepth, const bool blocking)
{
  pthread_mutex_lock(&itsLock);
  itsTargetAttitude.depth = finalDepth;
  pthread_mutex_unlock(&itsLock);
  if (blocking)
    waitMove(fabs(finalDepth - itsCurrentAttitude.depth) * 5.0 + 30.0);
}

// ######################################################################
void BeoSub::diveRel(const float relDepth, const bool blocking)
{
  pthread_mutex_lock(&itsLock);
  itsTargetAttitude.depth += relDepth;
  pthread_mutex_unlock(&itsLock);
  if (blocking) waitMove(fabs(relDepth) * 5.0 + 30.0);
}

// ######################################################################
//NOTE: This will work with open turn as an open strafe. FIX?
void BeoSub::strafeRel(const float relDist)
{
  // assume we are immobile. First, turn by some angle, then go
  // backwards, then turn back, then forward. The first turn (only)
  // should have rst true. All calls should have blocking to true.
  double alpha = relDist * 30.0; // turn by 30deg if 1m strafe
  if (alpha > 80.0) alpha = 80.0; else if (alpha < -80.0) alpha = -80.0;
  if (fabs(alpha) < 1.0)
    { LERROR("Strafe too small -- IGNORED"); return; }

  // more precisely, if we are going to turn by alpha and want to
  // strafe by reldist, we should turn by alpha then go backwards by
  // relDist/sin(alpha), then turn back by -alpha, finally advance by
  // relDist/tan(alpha). Let's do it:
  turnRel(Angle(alpha), true);
  advanceRel(-fabs(relDist / sin(alpha * M_PI / 180.0)));
  turnRel(Angle(alpha), true);
  advanceRel(fabs(relDist / tan(alpha * M_PI / 180.0)));
}

// ######################################################################
void BeoSub::waitMove(const double timeout)
{
  // Wait until the desired attitude has been reached. Note: in the
  // base class, this will never happen since we have no actuation. In
  // derived classes there should be some function running in a thread
  // that will activate the various actuators so as to try to reduce
  // the difference between itsTargetAttitude and itsCurrentAttitude:

  double startt = itsMasterClock.getSecs();
  while(targetReached() == false &&
        itsMasterClock.getSecs() - startt < timeout) usleep(200000);

  if (targetReached() == false)
    LERROR("Timeout occurred on move -- IGNORED");
}

// ######################################################################
double BeoSub::getTime() const
{ return itsMasterClock.getSecs(); }

// ######################################################################
Attitude BeoSub::getCurrentAttitude() const
{
  pthread_mutex_lock(const_cast<pthread_mutex_t *>(&itsLock));
  Attitude att = itsCurrentAttitude;
  pthread_mutex_unlock(const_cast<pthread_mutex_t *>(&itsLock));
  return att;
}

// ######################################################################
Attitude BeoSub::getTargetAttitude() const
{
  pthread_mutex_lock(const_cast<pthread_mutex_t *>(&itsLock));
  Attitude att = itsTargetAttitude;
  pthread_mutex_unlock(const_cast<pthread_mutex_t *>(&itsLock));
  return att;
}

// ######################################################################
Angle BeoSub::getHeading() const
{ return getCurrentAttitude().heading; }

// ######################################################################
Angle BeoSub::getPitch() const
{ return getCurrentAttitude().pitch; }

// ######################################################################
Angle BeoSub::getRoll() const
{ return getCurrentAttitude().roll; }

// ######################################################################
void BeoSub::getCompass(Angle& heading, Angle& pitch, Angle& roll) const
{
  Attitude att = getCurrentAttitude();
  heading = att.heading; pitch = att.pitch; roll = att.roll;
}

// ######################################################################
float BeoSub::getDepth() const
{ return getCurrentAttitude().depth; }

// ######################################################################
const char* beoSubCameraName(const BeoSubCamera cam)
{
  if (cam == BEOSUBCAMFRONT) return "Front";
  if (cam == BEOSUBCAMDOWN) return "Down";
  if (cam == BEOSUBCAMUP) return "Up";
  LERROR("Unknown BeoSubCamera value %d", int(cam));
  return "Unknown";
}

// ######################################################################
Image<float> BeoSub::getSaliencyMap(const enum BeoSubCamera cam) const
{
  Image< PixRGB<byte> > img = grabImage(cam);
  itsVisualCortex->input(InputFrame::fromRgb
                         (&img, itsMasterClock.getSimTime()));
  return itsVisualCortex->getOutput();
}

// ######################################################################
bool BeoSub::recognizeSIFT(const enum BeoSubCamera cam,
                           MappingData& data, Angle& myHeading) const
{
  // grab an image from the specified camera:
  Image< PixRGB<byte> > im = grabImage(cam);

  // create visual object and extract keypoints:
  rutz::shared_ptr<VisualObject> vo(new VisualObject("Captured Image", "", im));

  // get the matches:
  std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
  switch(cam)
    {
    case BEOSUBCAMFRONT:
      itsFrontVODB->getObjectMatches(vo, matches, VOMA_KDTREEBBF);
      break;
    case BEOSUBCAMDOWN:
      itsDownVODB->getObjectMatches(vo, matches, VOMA_KDTREEBBF);
      break;
    case BEOSUBCAMUP:
      itsUpVODB->getObjectMatches(vo, matches, VOMA_KDTREEBBF);
      break;
    default:
      LERROR("Wrong camera -- IGNORED");
    }

  // if we got nothing, stop here:
  if (matches.size() == 0U) return false;

  // check that we have a good affine:
  if (matches[0]->checkSIFTaffine() == false) return false;

  // compute our current heading based on the heading associated with
  // the matching picture and the rotation derived from the affine:
  SIFTaffine aff = matches[0]->getSIFTaffine();
  float theta, sx, sy, str;
  aff.decompose(theta, sx, sy, str);

  // our best match is the first in the list of matches. Get its
  // associated image name so that we can look it up in the
  // appropriate BeoSubDB:
  std::string name = matches[0]->getVoTest()->getImageFname();

  // now this has a .png extension while the BeoSubDB uses .txt; let's
  // change that:
  name.replace(name.length()-3, 3, "txt");

  // query the appropriate metadata database:
  switch(cam)
    {
    case BEOSUBCAMFRONT:
      data = itsFrontDB->getMappingData(name);
      break;
    case BEOSUBCAMDOWN:
      data = itsDownDB->getMappingData(name);
      break;
    case BEOSUBCAMUP:
      data = itsUpDB->getMappingData(name);
      break;
    default:
      LERROR("Wrong camera -- IGNORED");
    }

  // get our heading from the mapping data and the sift affine:
  myHeading = data.itsHeading;
  myHeading += theta;

  return true;
}

// ######################################################################
bool BeoSub::matchSIFT(const enum BeoSubCamera cam,
                       const rutz::shared_ptr<VisualObject>& obj) const
{
  // grab an image from the specified camera:
  Image< PixRGB<byte> > im = grabImage(cam);

  // create visual object and extract keypoints:
  rutz::shared_ptr<VisualObject> vo(new VisualObject("Captured Image", "", im));

  // get the matches:
  VisualObjectMatch vom(obj, vo, VOMA_SIMPLE);
  vom.prune();
  if (vom.checkSIFTaffine() == false) return false;

  if (vom.getScore() >= 0.5F) return true;
  return false;
}

// ######################################################################
bool BeoSub::affineSIFT(const enum BeoSubCamera cam, rutz::shared_ptr<VisualObject> goal)
{
  float x = 160.0F, y = 120.0F;            // center of ref image
  const float cu = 160.0F, cv = 120.0F;  // center of test image
  const float factor = (5.0F - getDepth()) * 0.05F; //NOTE: rough estimate
  int counter = 0;
  while(++counter < 10)
    {
      rutz::shared_ptr<VisualObject>
        current(new VisualObject("current", "", grabImage(cam)));
      VisualObjectMatch match(goal, current, VOMA_KDTREEBBF);
      match.prune();
      if (match.size() == 0U)
        {
          LINFO("No matches found... giving up");
          return false;
        }
    LINFO("Found %u matches", match.size());
    SIFTaffine aff = match.getSIFTaffine();
    float u, v; aff.transform(x, y, u, v);
    if (fabsf(u - cu) < 10.0F)
      {
        LINFO("Good enough! -- DONE");
        return true;
      }

    if (cam == BEOSUBCAMDOWN)
      {
        // this camera is mounted rotated 90deg
        if (v < cv) { LINFO("turning left"); turnRel(-10.0F, true); }
        else { LINFO("turning right"); turnRel(10.0F, true); }
        advanceRel((u - cu) * factor);
      }
    else
      {
        if (u < cu) { LINFO("turning left"); turnRel(-10.0F, true); }
        else { LINFO("turning right"); turnRel(10.0F, true); }
      }
    }
  // counter timeout:
  return false;
}

// ######################################################################

bool BeoSub::findShape(rutz::shared_ptr<ShapeModel>& shapeArg, const char* colorArg, const enum BeoSubCamera camArg) const
{

  Image< PixRGB<byte> > im = grabImage(camArg);
  int numDims = shapeArg->getNumDims();
  double* dims = (double*)calloc(numDims+1, sizeof(double));
  dims = shapeArg->getDimensions();

  itsShapeDetector->setupCanny(colorArg, im, false);//declare detector

  bool shapeFound = false;

  //give shape detector 5 chances to work, using different starting dimensions each time

  //Middle
  dims[1] = 150.0;
  dims[2] = 120.0;
  shapeArg->setDimensions(dims);
  shapeFound = itsShapeDetector->runCanny(shapeArg);
  if(!shapeFound){ //Upper left
    dims[1] = 60.0; //Xcenter
    dims[2] = 180.0; //Ycenter
    shapeArg->setDimensions(dims);
    shapeFound = itsShapeDetector->runCanny(shapeArg);
  }
  if(!shapeFound){ //Upper right
    dims[1] = 260.0; //Xcenter
    dims[2] = 180.0; //Ycenter
    shapeArg->setDimensions(dims);
    shapeFound = itsShapeDetector->runCanny(shapeArg);
  }
  if(!shapeFound){ //Lower left
    dims[1] = 60.0; //Xcenter
    dims[2] = 60.0; //Ycenter
    shapeArg->setDimensions(dims);
    shapeFound = itsShapeDetector->runCanny(shapeArg);
  }
  if(!shapeFound){ //Lower right
    dims[1] = 260.0; //Xcenter
    dims[2] = 60.0; //Ycenter
    shapeArg->setDimensions(dims);
    shapeFound = itsShapeDetector->runCanny(shapeArg);
  }

  if(!shapeFound){
    return false;
  }
  return true;
}

// ######################################################################
bool BeoSub::centerColor(const char* colorArg, const enum BeoSubCamera camArg, float& thresholdMass){
  Image< PixRGB<byte> > im;
  float x = 0.0, y = 0.0, mass = 0.0;
  float threshold = thresholdMass;
  float delta = 0;
  int lossCount = 0;
  bool checking = false;
  float xL = (160.0 - (threshold));
  float xR = (160.0 + (threshold));
  float yU = (120.0 - (threshold));
  float yD = (120.0 + (threshold));
  bool xOff = true, yOff = true;

  if(camArg == BEOSUBCAMFRONT){
    while(xOff){
      im = grabImage(camArg);
      itsColorTracker->setupTracker(colorArg, im, false);
      checking = itsColorTracker->runTracker(threshold, x, y, mass);
      if( checking && (x >= xL && x <= xR)){
        //Light centered in x plane. Now center in y
        xOff = false;
        lossCount = 0;
      }
      else if(checking){
        printf("Fcam turn\n");
        delta = x - 160.0; //delta = x_found - x_center
        //TURN left or right, in greater or lesser amounts depending on position
        turnRel(delta/2, true); //delta/2 based off of 160degree POV and 320pxl x axis
        lossCount = 0;
      }
      else{
        //Color lost. Use counter to allow for a few noisy losses, then fail
        lossCount++;
        if(lossCount >4){
          return false;
        }
      }
    }
    /*    while(yOff){
      im = grabImage(camArg);
      itsColorTracker->setupTracker(colorArg, im, false);
      checking = itsColorTracker->runTracker(threshold, x, y, mass);
      if( checking && (y >= yU && y <= yD)){
        //Light centered in y plane. Stop.
        yOff = false;
        lossCount = 0;
      }
      else if(checking){
        delta = y - 120.0; //delta = x_found - x_center
        //DIVE or surface, in greater or lesser amounts depending on th position of the light
        printf("Dcam turn\n");
        //NOTE: a constant like 2.07 would need testing, since with TASK A,
        //there is no cloear way to distinguish between pixel distance and
        //difference of depth between the SUB and the Object. FIX????
        float dZ = (30/(mass/10)) * tan(delta/2.07); //amt based off of 116degree POV and 240pxl y axis, as well as dist = (30/(mass/10))
        diveRel(dZ, true);
        lossCount = 0;
      }
      else{
        //Color lost. Use counter to allow for a few noisy losses, then fail
        lossCount++;
        if(lossCount >3){
          return false;
        }
      }
      }*/
    thresholdMass = mass;
    //STOP
    return true;
  }

  //NOTE!: Down camera may need a change in axes if we do not reorient it! FIX!!
  else if(camArg == BEOSUBCAMDOWN || camArg == BEOSUBCAMUP){
    while(xOff){
      im = grabImage(camArg);
      itsColorTracker->setupTracker(colorArg, im, false);
      checking = itsColorTracker->runTracker(threshold, x, y, mass);
      if( checking && ( ((x >= xL && x <= xR) && camArg == BEOSUBCAMUP) || ((y >= yU && y <= yD) && camArg == BEOSUBCAMDOWN) ) ){
        //Light centered in x plane. Now center in y
        xOff = false;
        lossCount = 0;
      }
      else if(checking){
        //TURN left or right, in greater or lesser amounts depending on position
        if(camArg == BEOSUBCAMDOWN){
          turnRel(((y-120)/2.07), true); //delta/2 based off of 160degree POV and 320pxl x axis

        }
        else{
          turnRel(((x-160)/2), true);
        }
        lossCount = 0;
      }
      else{
        //Color lost. Use counter to allow for a few noisy losses, then fail
        lossCount++;
        if(lossCount >3){
          return false;
        }
      }
    }
    while(yOff){
      im = grabImage(camArg);
      itsColorTracker->setupTracker(colorArg, im, false);
      checking = itsColorTracker->runTracker(threshold, x, y, mass);
      if( checking && (((y >= yU && y <= yD) && camArg == BEOSUBCAMUP) || ((x >= xR && x <= xL) && camArg == BEOSUBCAMDOWN))){
        //Light centered in y plane. Stop.
        yOff = false;
        lossCount = 0;
      }
      else if(checking){
        //ADVANCE, in greater or lesser amounts depending on the position
        if(camArg == BEOSUBCAMUP){
          float dY = (30/(mass/10)) * tan((y-120)/2.07);
          advanceRel(dY);
        }
        else{
          float dY = (30/(mass/10)) * tan((x-160)/2);
          advanceRel(dY);
        }
        lossCount = 0;
      }
      else{
        //Color lost. Use counter to allow for a few noisy losses, then fail
        lossCount++;
        if(lossCount >3){
          return false;
        }
      }
    }

    return true;
  }
  return false; //to make compiler happy
}

// ######################################################################
bool BeoSub::approachArea(std::string name, const enum BeoSubCamera cam, float stepDist){
  switch(cam){
  case(BEOSUBCAMUP):{
    MappingData goal = itsUpDB->getMappingData(name);
  }
  case(BEOSUBCAMFRONT):{
    MappingData goal = itsFrontDB->getMappingData(name);
  }
  case(BEOSUBCAMDOWN):{
    MappingData goal = itsDownDB->getMappingData(name);
  }

  }
  //return approachArea(itsCurrentArea, goal, stepDist);

  return true;

}

// ######################################################################
bool BeoSub::approachArea(MappingData goalArea, MappingData currentArea, float stepDist){
  Attitude targetAtt;
  float targetDist = 0.0, expectedDist = 0.0;
  MappingData contData;
  itsDownDB->getDirections(currentArea, goalArea, targetAtt, targetDist);
  expectedDist = targetDist;
  while(targetDist >= stepDist){
    turnRel(targetAtt.heading, true);
    diveAbs(targetAtt.depth, true);//NOTE: should these both be true?
    //approach in steps the size of stepDist
    advanceRel(stepDist);
    expectedDist -= stepDist;
    if(expectedDist <= -15.0){//NOTE: needs calibrating! FIX!
      //fail
      return false;
    }
    Angle mh;
    //get new currentArea
    if(recognizeSIFT(BEOSUBCAMDOWN, contData, mh)){
      itsDownDB->getDirections(contData, goalArea, targetAtt, targetDist);
    }
    else if(recognizeSIFT(BEOSUBCAMFRONT, contData, mh)){
      itsFrontDB->getDirections(contData, goalArea, targetAtt, targetDist);
    }
    else{//continue in expected path
      targetDist -= stepDist;//NOTE: may be bad! FIX?
    }
  }
  //finish off approach
  turnRel(targetAtt.heading, true);
  diveAbs(targetAtt.depth, true);
  advanceRel(targetDist);
  //double-check where we are using mapping, and if OK, return true


  Image< PixRGB<byte> > im = grabImage(BEOSUBCAMDOWN);
  /* rotate the image grabbed from down cameraby 90 counterclockwise*/


  rutz::shared_ptr<VisualObject> vo(new VisualObject("Captured Image", "", im));
  std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
  itsDownVODB->getObjectMatches(vo, matches, VOMA_KDTREEBBF);
  if (matches.size() == 0U){
    itsFrontVODB->getObjectMatches(vo, matches, VOMA_KDTREEBBF);
    if(matches.size() == 0U) return false;
  }//If we cannot recognize where we are, return that we are not sure. NOTE: this false should likely be different than the false above, sinc eit is a better result. FIX? NOTE: perhaps we should keep track of the # of times we estimated a step?
  for(uint i = 0; i <= matches.size(); i++){
    if(!strcmp((matches[0]->getVoTest()->getImageFname()).c_str(), goalArea.itsImgFilename.c_str())){//if we have found a match...
      return true; //success!
    }
  }
  return false; //again, we may want to distinguish different "failures"
}

// ######################################################################
bool BeoSub::Decode(){
        //Decode from the order of tasks to the order of the bins!!!
  ImageSet< PixRGB<byte> > inStream;
  Image< PixRGB<byte> > img;
  int NAVG = 20; Timer tim; uint64 t[NAVG]; int frame = 0; //NOTE: what is this NAVG?
  float avg2 = 0.0;
  //Grab a series of images
  for(int i = 0; i < 100; i++){
    tim.reset();

    img = grabImage(BEOSUBCAMFRONT);
    inStream.push_back(img);

    uint64 t0 = tim.get();  // to measure display time
    t[frame % NAVG] = tim.get();
    t0 = t[frame % NAVG] - t0;
    // compute framerate over the last NAVG frames:
    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
        avg2 = 1000.0F / float(avg) * float(NAVG);
      }
    frame ++;
  }
  bool foundRed = true;
  itsTaskDecoder->setupDecoder("Red", false);
  itsTaskDecoder->runDecoder(inStream, avg2);
  float hertz = itsTaskDecoder->calculateHz();
  if(hertz <= .5){
    //red not detected. Try green
    foundRed = false;
    itsTaskDecoder->setupDecoder("Green", false);
    itsTaskDecoder->runDecoder(inStream, avg2);
    hertz = itsTaskDecoder->calculateHz();
    if(hertz <= .5){//fail
      return false;
    }
  }
  itsTasks.clear();
  if(foundRed){//RED
    decoderIsRed = true;
    if(hertz <= 2.8){//AT 2Hz
      itsTasks.push_back('B');
      itsTasks.push_back('A');
      itsTasks.push_back('C');
    }
    else{//AT 5Hz
      itsTasks.push_back('A');
      itsTasks.push_back('C');
      itsTasks.push_back('B');
    }
  }
  else{//GREEN
    decoderIsRed = false;
    if(hertz <= 2.8){//AT 2Hz
      itsTasks.push_back('C');
      itsTasks.push_back('A');
      itsTasks.push_back('B');
    }
    else{//AT 5Hz
      itsTasks.push_back('B');
      itsTasks.push_back('C');
      itsTasks.push_back('A');
    }
  }
  itsTasksIter = itsTasks.begin();
  return true;
}

// ######################################################################
bool BeoSub::TaskGate(){
  //  int counter = 0;
  //turn towards gate (assumes starting on launch platform)
  //NOTE: LAUNCHTOGATE will have to be found onsite! FIX!
  //turnAbs(LAUNCHTOGATE);
  //NOTE: perhaps try to recognize gate here to confirm? may also want to recognize light (which may be easier)
  //NOTE: may need to approach here
  /*
  while(!Decode() && (counter < 10)){
    //while the decode light is not found, strafe left until seen. NOTE that this may or may not work
    strafeRel(-0.5);
    counter++;
  }
  if(counter >= 10){
    return false;
  }
  float thresh = 20.0;
  //center decoder in camera
  if(decoderIsRed){
    centerColor("Red", BEOSUBCAMFRONT, thresh);//perhaps should use thresh?
  }
  else{
    centerColor("Green", BEOSUBCAMFRONT, thresh);
  }
  advanceRel(20.0);//NOTE: this value should be far enough to completely clear the gate
  return true;//somewhat bunk
  */
  int counter = 0;
  while(!matchSIFT(BEOSUBCAMFRONT, itsVOtaskDfront) && counter <= 10){
    turnOpen(10.0);
    counter++;
  }
  if(counter > 10){
    return false;
  }
  diveAbs(2.0);
  return true;
}

// ######################################################################


bool BeoSub::TaskScheduler(int TaskOrder)   //Schedule the order of the three tasks
{

//pre-task A
        //Assmue we get thru the gate, and recognize the light box
        //looking for red light

        //approaching red light

//task A
        if(TaskA()) //if Task A is done, then go to task B normally
        {
        //looking for yellow pipeline

        //approaching yellow pipeline

        }// end of if
        else                //if task A fails, then use emergancy to find task B
        {

        }//end of else


//task B
        if(TaskB())        //emergancy for task B
        {
                //well done, go to task C

        }//end of if
        else
        {
                //not done, do task B again? or go to task C

        }//end of else


//task C
        if(TaskC())
        {

        }
        else
        {

        }

//post-task C


        //done
        return true;

}//end of TaskScheduler

//=======================Begins TaskA==============================
bool BeoSub::TaskA(){//NOTE: assumes that sub has already approached task A and dove to the correct depth, as well as turned to a predicted heading
    //LOOK FOR THE LIGHT
  /*    float x = 0.0;
    float y = 0.0;
    float mass = 0;
    float close = 30.0;
    bool notDone = true;
    int state = 0;
  */
    /*
      while sift not found, spiral search slowly
      once found, center using sift affine
      dive

    */
    if(!affineSIFT(BEOSUBCAMDOWN, itsVOtaskAdown)){//approach
      if(!affineSIFT(BEOSUBCAMFRONT, itsVOtaskAfront)){
        float dist = sqrt(((taskAposition.x - itsGlobalPosition.x)*(taskAposition.x - itsGlobalPosition.x)) + ((taskAposition.y - itsGlobalPosition.y)*(taskAposition.y - itsGlobalPosition.y)));
        Angle turn = acos(taskAposition.y/dist);
        Angle down = 180;
        turn = down - turn;
        turn = turn - itsGlobalHeading;
        turnOpen(turn);
        advanceRel(dist);
      }
    }
    /*
    while(notDone){
        switch(state){
        case 0: //LOOK FOR LIGHT
          {
            //while we don't see the light...
            turnRel(10.0, true);//NOTE: 10 perhaps too big. FIX!
            itsColorTracker->setupTracker("Red", grabImage(BEOSUBCAMFRONT), false); //reset using most recent image
            if(itsColorTracker->runTracker(5.0, x, y, mass)){
              if(mass >= close){
                state = 3;
              }
              else{
                state = 1;
              }
              break;
            }
          }
        case 1: //CENTER LIGHT IN CAMERA
          {
            if(centerColor("Red", BEOSUBCAMFRONT, mass)){
              if(mass >= close){
                //NOTE: check for correct depth vs. decoder depth?
                state = 3;
              }
              else{
                state = 2;
              }
            }
            else{
              state = 0;
            }
          }
        case 2: //APPROACH THE LIGHT (really ghetto right now)
            {
              advanceRel(1.0);//NOTE: should be some number based off of the mass! FIX!
            }
        case 3: //RAM THE LIGHT!
            //go forward at full speed, but allow color tracker to continue running
            itsColorTracker->setupTracker("Red", grabImage(BEOSUBCAMFRONT), false);
            if(itsColorTracker->runTracker(40.0, x, y, mass)){
              advanceRel(12.0, false);
              advanceRel(-1.0, false); //to stop
            }
            else{
                //light is assumd to have ben hit. stop ramming and continue to next stage
                notDone = false;
                break;
            }
        }
    }
    */
    taskAdone = true;
    return true;
}

//Look around for the red light
bool BeoSub::LookForRedLight()
{
  //front camera
  //dive close to the pipe

  ModelManager camManager("ColorTracker Tester");

  nub::soft_ref<FrameIstream>
    gb(makeIEEE1394grabber(camManager, "colorcam", "cocam"));

  camManager.addSubComponent(gb);

  camManager.start();

  // instantiate a model manager for the color tracker module:
  ModelManager manager("ColorTracker Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<ColorTracker> test(new ColorTracker(manager));
  manager.addSubComponent(test);

  manager.start();

  Image< PixRGB<byte> > Img;
  float x = 320/2, y=240/2;
  float mass;
  bool found;

  //  float width;

  rutz::shared_ptr<XWindow> wini;

  //turn around until I find orange
  //#################control#################

#ifndef DEBUG
  turnRel(Angle(15.0)); //turn
#else
  printf("DEBUG!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf("i'm doing the first turing!\n");
#endif

  //Note: should check if it turned 360 degrees without finding anything
  //now just use a counter for test

  int turnCounter=0;

#ifndef DEBUG
  Angle initHeading = getHeading();
  Angle currHeading;
  Angle epsHeading = Angle(5.0);
#else
#endif

  while(1)
  {
#ifndef DEBUG
    turnRel(5);
#else
    printf("turning right to find the red light!\n");
#endif

    Img = gb->readRGB();

#ifndef DEBUG
    test->setupTracker("Red", grabImage(BEOSUBCAMFRONT), true);
#else
    test->setupTracker("Red", Img /*grabImage(BEOSUBCAMFRONT)*/, true);
#endif

    found = test->runTracker(25.0, x, y, mass);
    if (found)
    {
#ifdef DEBUG
      turnRel(0.01); //stop turning
#else
      printf("i see the red light!");
#endif
      break;
    }

    //test if we have turned for 360 degrees
    //if so, go randomly or go according to the mapping info
    //now we're just using a counter
#ifndef DEBUG
    currHeading = getHeading();
    if(fabs(currHeading.getVal() - initHeading.getVal()) <= - (epsHeading.getVal()))
      {
        //if we are trying too many times, just give up
        if(turnCounter++ > 10)
          {
            return false;
          }
        //make a random turn and go (-180, 180)
        else
          {
            int rndTurn = rand()*180;
            Angle rndHeading = Angle(360-rndTurn);
            turnRel(rndHeading);
            advanceRel(10.0);
          }
      }
#else
    if(turnCounter++ > 1000)
      {
        printf("never see the red light, i give up!");
        return false;
      }
#endif

  }//end of while

  return true;
}

//Center the red light and ready to go
bool BeoSub::CenterRedLight()
{
  return false;
}

//Approach the red light until close enough
bool BeoSub::ApproachRedLight()
{
  ModelManager camManager("ColorTracker Tester");

  nub::soft_ref<FrameIstream>
    gb(makeIEEE1394grabber(camManager, "colorcam", "cocam"));

  camManager.addSubComponent(gb);

  camManager.start();

  // instantiate a model manager for the color tracker module:
  ModelManager manager("ColorTracker Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<ColorTracker> test(new ColorTracker(manager));
  manager.addSubComponent(test);

  manager.start();

  Image< PixRGB<byte> > Img;
  float x = 320/2, y=240/2;

  //arbitary area of mass, need to calibrate!!!!
  float mass=10.0;
  bool found;

  float width;

  rutz::shared_ptr<XWindow> wini;
  //approach the orange
  bool redInFront = false; //can I see the pipe in front of me

  //we're trying to see the red light
  //in both the bottom and front cameras (this is kinda idealistic)
  while(1)
    {
      Img = gb->readRGB();
      //chase after red in the front cam
#ifndef DEBUG
      test->setupTracker("Red", grabImage(BEOSUBCAMFRONT), true);
#else
      test->setupTracker("Red", Img, true);
#endif
      redInFront = test->runTracker(25.0, x, y, mass);
#ifndef DEBUG
      test->setupTracker("Red", grabImage(BEOSUBCAMDOWN), true);
#else
      test->setupTracker("Red", Img, true);
#endif
      test->runTracker(25.0, x, y, mass);

#ifndef DEBUG
      advanceRel(1.5);
#else
      printf("i'm going forward!\n");
#endif

      width = Img.getWidth();
      //make sure that positive is up-right
      x = x - width/2;

      if (redInFront)
        { //if seen in the front camera
          printf("i see the red light in the front!\n");
          if (fabs(x) < 10)
            { //turn left or right
#ifndef DEBUG
              advanceRel(1.5);
#else
              printf("now going forward to the pipe!\n");
#endif
            }
          else
            {
              if(x>0)
                {

#ifndef DEBUG
                  turnRel(Angle(5));
#else
                  printf("turning right to the red light!\n");
#endif
                }
              else
                {
#ifndef DEBUG
                  turnRel(Angle(-5));
#else
                  printf("turning left to the red light!\n");
#endif
                }
            }//end of if abs of x
        }//end of if red in front
       else
         { //dive until you see red in front again

           printf("now i lost the red light, diving to get it!\n");
           while(1)
             {
               Img = gb->readRGB();
#ifndef DEBUG
               test->setupTracker("Red", grabImage(BEOSUBCAMFRONT), true);
#else
               test->setupTracker("Red", Img /*grabImage(BEOSUBCAMFRONT)*/, true);
#endif
               redInFront = test->runTracker(25.0, x, y, mass);

               //need a timer/counter, if the red never shows in the front,
               //we probably are lost, or hit the bottom of the pool!
               if (redInFront)
                 {
                   printf("got it after dive, now going!\n");
                   break;
                 }
#ifndef DEBUG
               diveRel(.3);
#else
               printf("i'm diving for the red light!!!\n");
#endif

             }//end of while

         }//end of else red light in front

      //when a significant amount of red is in the bottom camera
      //we are now above the red light, and ready to push!
#ifndef DEBUG
       test->setupTracker("Red", grabImage(BEOSUBCAMDOWN), true);
#else
       test->setupTracker("Red", Img /*grabImage(BEOSUBCAMDOWN)*/, true);
#endif
       found = test->runTracker(25.0, x, y, mass);

       printf("testing if the red light shows in my bottom camera\n");
       if (found == true)
         {
           //see orange in the bottom camera
           printf("ok, finally i'm above the red light!\n");
           return true;
         }
       printf("never see the red light under me, keep going!\n");

    }//end of the while(1) loop

  return false;
}

//Push the red light to finish task A
bool BeoSub::PushRedLight()
{
  //l: length of the light bar
  //alpha: the min angle to push
  //pushDist = l * sin(alpha)
  //diveDist = pushDist * tan(alpha)
  float minAngle = 30; //degrees
  float barLength = 10.0;
  float pushDist = barLength * sin(minAngle);
  //float diveDist = pushDist * tan(minAngle);

  //total advanced distant counter
  float advDist = 0.0;
  //total dived distant counter
  float divDist = 0.0;
  ModelManager camManager("ColorTracker Tester");

  nub::soft_ref<FrameIstream>
    gb(makeIEEE1394grabber(camManager, "colorcam", "cocam"));

  camManager.addSubComponent(gb);

  camManager.start();

  // instantiate a model manager for the color tracker module:
  ModelManager manager("ColorTracker Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<ColorTracker> test(new ColorTracker(manager));
  manager.addSubComponent(test);

  manager.start();

  Image< PixRGB<byte> > Img;
  float x = 320/2, y=240/2;

  //arbitary area of mass, need to calibrate!!!!
  float mass = 10.0;
  float width;

  rutz::shared_ptr<XWindow> wini;
  //approach the red light
  bool redInFront = false; //can I see the pipe in front of me
  bool redInBottom = false;

  //we're trying to see orange in both the bottom and front cameras (this is kinda idealistic)
  while(1)
    {
      Img = gb->readRGB();
      //chase after orange in the front cam
#ifndef DEBUG
      test->setupTracker("Red", grabImage(BEOSUBCAMFRONT), true);
#else
      test->setupTracker("Red", Img, true);
#endif
      redInFront = test->runTracker(25.0, x, y, mass);
#ifndef DEBUG
      test->setupTracker("Red", grabImage(BEOSUBCAMDOWN), true);
#else
      test->setupTracker("Red", Img, true);
#endif
      redInBottom = test->runTracker(25.0, x, y, mass);

      width = Img.getWidth();
      //make sure that positive is up-right
      x = x - width/2;

      if (redInBottom)
        {
          //if seen in the bottom camera
          //dive until never see in the bottom camera
          //but still sees it in the front camera
#ifndef DEBUG
          diveRel(.3);
#else
          printf("now diving to push the red light!\n");
#endif
        }
      else if(redInFront)
        {
          //now we are at the right pos to push
          //GO!
          printf("i see the red light in the front!\n");
          if (fabs(x) < 10)
            { //turn left or right
#ifndef DEBUG
              advanceRel(1.5);
              advDist += 1.5;
              diveRel(1.5 * tan(minAngle));
              divDist += 1.5 * tan(minAngle);
#else
              printf("now going forward to the pipe!\n");
#endif
            }
          else
            {
              if(x>0)
                {
#ifndef DEBUG
                  turnRel(Angle(5));
#else
                  printf("turning right to the red light!\n");
#endif
                }
              else
                {
#ifndef DEBUG
                  turnRel(Angle(-5));
#else
                  printf("turning left to the red light!\n");
#endif
                }
            }//end of if abs of x

          if(advDist > pushDist)
            {
              //we are done!
              //now go up and leave the red light
#ifndef DEBUG
              //go up and release the red
              //we are both free!
              diveRel(- divDist - 5.0);
#else
              printf("Job done! Going up to leave the red light!\n");
              printf("up dist: %f", divDist);
#endif
              return true;
            }

        }//end of if red in front
       else
         {
           //we lost the red light
           //should we do it again???
           printf("No, lost the red light\n");

           return false;
         }
    }//end of while(1)

 return false;
}

//====================End TaskA=========================

//====================Begins TaskB==========================
bool BeoSub::TaskB(){
  //NOTE: The bottom camera axes are not correct! Will NEED to FIX the indiscrepancy in code below!
  //ADD: Orange line flollwing state. AffineXfrm correction,  marker dropping

  //int stage = 0;
  //int counter = 0;
  bool TaskBDone = false; //has it dropped the marker into the target bin yet?
  //bool foundDown = false; bool foundFront = false;
  //float x, y, mass, x2, y2, mass2;
  //float usedX, usedY, usedMass;
  Image< PixRGB<byte> > image;

  /***********************new code ********************************/
  //Assume the order is A to B to C
  //So we get from A to B

  //  const int HatchNull = 0;
  // const int HatchNo = 1;
  //  const int HatchShort = 2;
  // const int Hatch45 = 3;
  //const int HatchLong = 4;

  unsigned int BinCounter = 0;
   unsigned int ApproachCounter = 0;

  //get random order of the bins
   //  int TaskBBin = HatchNo;

  //  int CurrentPos = 0;
  /*
  ApproachPipeLine();
  while(BinCounter<1) {
    bool recbin = FollowPipeLine();
    if (recbin) {
      CenterBin();
      DropMarker();
      BinCounter++;
    }

  FollowPipeLine();
  TaskBDone = true;
  }
  */
   //   bool recognize = false;

   const unsigned int binNum = 1;

   bool testBin=false;

   while((BinCounter < binNum) && (!TaskBDone))
     {
       //!!!maybe need to change: looking for pipe and approaching pipe
       if(ApproachPipeLine())
         {
           ApproachCounter = 0;

           //TRUE if found bin at the end of follow pipeline
           if( (testBin = FollowPipeLine()) )
            {
              //recognize the 4 bins
              //need to change to really recognize
              printf("TaskB: found a bin at the end of the pipeline! Now checking if the bin is the one we want\n");

              //if(RecognizeBin()==tastBin)
              if(testBin == true)        //if the bin is the recognized one, drop the marke, else just pass it
                {
                  printf("TaskB: found the bin we want! Now going to center the bin\n");

                  if(CenterBin())
                    {
                      printf("TaskB: we have centered the bin! Now we're going to drop the marker!\n");

                      if(DropMarker())
                        {
                          printf("TaskB: good, we dropped the marker!\n");
                          TaskBDone = true;
                          BinCounter++;
                        }//end ofif drop marker
                      else
                        {
                          //failed to drop the marker
                          //drop again?
                          printf("TaskB: failed to drop the marker!!!\n");
                          return false;
                        }//end of drop marker

                    }//end of if center bin
                  else
                    {
                      printf("TaskB: well, we failed to center the bin!\n");
                      //fail to center the bin
                      //center again?
                      return false;
                    }//end of center bin

                }//end of if test bin
              else
                {
                  printf("TaskB: it's not the bin we want! Just go thru it!\n");

                  if(PassBin())
                    {
                      printf("TaskB: Ok, we passed the bin! Now follow the pipe to the next bin!\n");
                    }//end of if pass bin
                  else
                    {
                      printf("TaskB: well, we failed the pass the bin!");
                      //failed to pass the bin
                      //find the pipeline directly?
                      return false;

                    }//end of pass bin

                  BinCounter++;

                }//end of test bin

            }//end of if follow
          else if(TaskBDone)
            {
              printf("TaskB: Well, we lost track of the pipe, but the task is done!\n");

              //if(BinCounter<4)
              if(BinCounter < 1)        //we miss some bins, but we are done
                {
                  return true;
                }
              else                        //We are done, go out of the task B zone
                {
                  return true;
                }

            }//end of else if

          else        //there is something wrong that we are out of the task B zone
            {
              printf("TaskB: We lost track of the pipe!\n");
              //go back and check again
              return false;
            }//end of else follow

         }//end of if approach

        else        //failed to approach, try again?
          {
            printf("TaskB: Failed to approach to the pipe at try %i\n", ApproachCounter);
            if(++ApproachCounter>5) //we give up trying
              {
                printf("TaskB: Never find the pipe! Give up!\n");
                return false;
              }

          }//end else approach

     }//end of while

   printf("TaskB: i'm not suppose to come here, but task is done!\n");
   return true;

   /**************************************old code****************************************/
  //while(notDone){
  //  switch(stage){
  //  case 0: //FINE-TUNE MAPPING MATCH. Accepts image filename and fine-tunes current result to param image
  //    {
  //      rutz::shared_ptr<VisualObject> goal = itsDownVODB->getObject("TaskB");
  //      if(affineSIFT(BEOSUBCAMDOWN, goal)){//approach
  //        stage = 3;
  //        break;
  //      }
  //      else{//look for pipes
  //        stage = 5;
  //        break;
  //      }
  //    }
  //  case 3: //MARKER DROPPER 1ST AND ONLY STAGE
  //    {
  //      for(int i = 0; i < 6; i++) {
  //        dropMarker();
  //      }
  //      return true;
  //      // call dropMarker()
  //    }
  //  case 5: //APPROACH ORANGE PIPES
  //    {

  //      if(taskCdone){//If we came from taskC direction...
  //        MappingData BtoC = itsDownDB->getMappingData("TaskBtoC.txt");
  //        MappingData current; Angle myHeading;
  //        if(!recognizeSIFT(BEOSUBCAMDOWN, current, myHeading)){//Try to recognize where we are (max 8 tries right now)
  //          counter++;
  //          //Ghetto circle search. FIX!
  //          advanceRel(-1.5);
  //          turnRel(20.0);
  //        }
  //        if(counter >= 8){//If we are lost, fail
  //          return false;
  //        }
  //        else{//approach task A to B pipes
  //          /*
  //          if(approachArea(BEOSUBCAMDOWN, BtoC, 3.0)){
  //            stage = 7;
  //            counter = 0;
  //            break;
  //          }
  //          */
  //          counter++;
  //        }
  //      }
  //      else if(taskAdone){//If we came from TaskA direction...
  //        MappingData BtoA = itsDownDB->getMappingData("TaskBtoA.txt");
  //        MappingData current; Angle myHeading;
  //        if(!recognizeSIFT(BEOSUBCAMDOWN, current, myHeading)){//Try to recognize where we are (max 8 tries right now)
  //          counter++;
  //          //Ghetto circle search. FIX!
  //          advanceRel(-3.5);
  //          turnRel(20.0);
  //        }
  //        if(counter >= 8){//If we are lost, fail
  //          return false;
  //        }
  //        else{//approach task A to B pipes
  //          /*
  //                      if(approachArea(BEOSUBCAMDOWN, BtoA, 3.0)){
  //                      stage = 7;
  //                      counter = 0;
  //                      break;
  //                      }
  //          */
  //          counter++;
  //        }
  //      }
  //    }
  //  case 7://PIPE FOLLOWING
  //    {
  //      float x = 0.0, y = 0.0, mass = 0.0;
  //      int counter = 0;
  //      itsColorTracker2->setupTracker("Orange", grabImage(BEOSUBCAMDOWN), false);

  //      while(itsColorTracker2->runTracker(10.0, x, y, mass) && counter < 12){
  //        ++counter;
  //        //NOTE: Down cam weirdness! FIX!!
  //        if(y < 120){//on left...
  //          turnRel(-10.0);
  //        }
  //        else{//on right...
  //          turnRel(10.0);
  //        }

  //        /****/ //FIX

  //        advanceRel(1.5);//follow

  //        itsColorTracker->setupTracker("White", grabImage(BEOSUBCAMDOWN), false);
  //        if(itsColorTracker->runTracker(500.0, x, y, mass)) {
  //          stage = 3;
  //          break;
  //        }

  //      }
  //           //itsColorTracker2->setupTracker("Orange", grabImage(BEOSUBCAMDOWN), false);
  //      stage = 3;
  //           //done following, try to find box
  //        // stage = 0; ///?????
  //      break;
  //    }
  //  }
  //}//end of while

  //return false;
}

bool BeoSub::ApproachPipeLine()
{
  //front camera
  //dive close to the pipe

  ModelManager camManager("ColorTracker Tester");

  nub::soft_ref<FrameIstream>
  gb(makeIEEE1394grabber(camManager, "colorcam", "cocam"));

  camManager.addSubComponent(gb);

  camManager.start();

  // instantiate a model manager for the color tracker module:
  ModelManager manager("ColorTracker Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<ColorTracker> test(new ColorTracker(manager));
  manager.addSubComponent(test);

  manager.start();

  Image< PixRGB<byte> > Img;
  float x = 320/2, y=240/2;
  //arbitary area of mass, need to calibrate!!!!
  float mass = 10.0;
  bool found;

  float width;

  rutz::shared_ptr<XWindow> wini;

  //turn around until I find orange
  //#################control#################

#ifndef DEBUG
  turnRel(Angle(15.0)); //turn
#else
  printf("DEBUG!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf("i'm doing the first turing!\n");
#endif

  //Note: should check if it turned 360 degrees without finding anything
  //now just use a counter for test

  int turnCounter=0;

#ifndef DEBUG
  Angle initHeading = getHeading();
  Angle currHeading;
  Angle epsHeading = Angle(5.0);
#else
#endif

  while(1)
  {
#ifndef DEBUG
    turnRel(5);
#else
    printf("turning right to find the pipe!\n");
#endif

    Img = gb->readRGB();

#ifndef DEBUG
    test->setupTracker("Orange", grabImage(BEOSUBCAMFRONT), true);
#else
    test->setupTracker("Orange", Img /*grabImage(BEOSUBCAMFRONT)*/, true);
#endif

    found = test->runTracker(25.0, x, y, mass);
    if (found)
    {
#ifdef DEBUG
      turnRel(0.01); //stop turning
#else
      printf("i see the pipe!");
#endif
      break;
    }

    //test if we have turned for 360 degrees
    //if so, go randomly or go according to the mapping info
    //now we're just using a counter
#ifndef DEBUG
    currHeading = getHeading();
    if(fabs(currHeading.getVal() - initHeading.getVal()) <= - (epsHeading.getVal()))
      {
        //if we are trying too many times, just give up
        if(turnCounter++ > 10)
          {
            return false;
          }
        //make a random turn and go (-180, 180)
        else
          {
            int rndTurn = rand()*180;
            Angle rndHeading = Angle(360-rndTurn);
            turnRel(rndHeading);
            advanceRel(10.0);
          }
      }
#else
    if(turnCounter++ > 1000)
      {
        printf("never see a pipe, i give up!");
        return false;
      }
#endif

  }//end of while

  //approach the orange
  bool pipeInFront = false; //can I see the pipe in front of me

  //we're trying to see orange in both the bottom and front cameras (this is kinda idealistic)
  while(1)
    {
      Img = gb->readRGB();
      //chase after orange in the front cam
#ifndef DEBUG
      test->setupTracker("Orange", grabImage(BEOSUBCAMFRONT), true);
#else
      test->setupTracker("Orange", Img, true);
#endif
      pipeInFront = test->runTracker(25.0, x, y, mass);
#ifndef DEBUG
      test->setupTracker("Orange", grabImage(BEOSUBCAMDOWN), true);
#else
      test->setupTracker("Orange", Img, true);
#endif
      test->runTracker(25.0, x, y, mass);

#ifndef DEBUG
      advanceRel(1.5);
#else
      printf("i'm going forward!\n");
#endif

      width = Img.getWidth();
      //make sure that positive is up-right
      x = x - width/2;

      if (pipeInFront)
        { //if seen in the front camera
          printf("i see the pipe in the front!\n");
          if (fabs(x) < 10)
            { //turn left or right
#ifndef DEBUG
              advanceRel(1.5);
#else
              printf("now going forward to the pipe!\n");
#endif
            }
          else
            {
              if(x>0)
                {

#ifndef DEBUG
                  turnRel(Angle(5));
#else
                  printf("turning right to the pipe!\n");
#endif
                }
              else
                {
#ifndef DEBUG
                  turnRel(Angle(-5));
#else
                  printf("turning left to the pipe!\n");
#endif
                }
            }//end of if abs of x
        }//end of if pipe in front
       else
         { //dive until you see orange in front again

           printf("now i lost the pipe, diving to get it!\n");
           while(1)
             {
               Img = gb->readRGB();
#ifndef DEBUG
               test->setupTracker("Orange", grabImage(BEOSUBCAMFRONT), true);
#else
               test->setupTracker("Orange", Img /*grabImage(BEOSUBCAMFRONT)*/, true);
#endif
               pipeInFront = test->runTracker(25.0, x, y, mass);

               //need a timer/counter, if the pipe never shows in the front,
               //we probably are lost, or hit the bottom of the pool!
               if (pipeInFront)
                 {
                   printf("got it after dive, now going!\n");
                   break;
                 }
#ifndef DEBUG
               diveRel(.3);
#else
               printf("i'm diving for the orange!!!\n");
#endif

             }//end of while

         }//end of else pipe in front

      //stop when a significant amount of orange is in the bottom camera
#ifndef DEBUG
       test->setupTracker("Orange", grabImage(BEOSUBCAMDOWN), true);
#else
       test->setupTracker("Orange", Img /*grabImage(BEOSUBCAMDOWN)*/, true);
#endif
       found = test->runTracker(25.0, x, y, mass);

       printf("testing if the pipe shows in my bottom camera\n");
       if (found == true)
         {
           //see orange in the bottom camera
           printf("ok, finally i'm above the pipe!\n");
           return true;
         }
       printf("never see the pipe under me, keep going!\n");

  }//end of while

  //never should come here!!!
  return false;

/********** Assume we start from A to B then to C
        if(taskCdone){//If we came from taskC direction...
          MappingData BtoC = itsDownDB->getMappingData("TaskBtoC.txt");
          MappingData current; Angle myHeading;
          if(!recognizeSIFT(BEOSUBCAMDOWN, current, myHeading)){//Try to recognize where we are (max 8 tries right now)
            counter++;
            //Ghetto circle search. FIX!
            advanceRel(-1.5);
            turnRel(20.0);
          }
          if(counter >= 8){//If we are lost, fail
            return false;
          }
          else{//approach task A to B pipes

          //  if(approachArea(BEOSUBCAMDOWN, BtoC, 3.0)){
          //    stage = 7;
          //    counter = 0;
           //   break;
          //  }

            //counter++;
          //}
        //}*/

    //printf("x:%f, y:%f, mass:%f\n",x,y, mass);



            // Should be data AtoB???
  //            MappingData BtoA = itsDownDB->getMappingData("TaskBtoA.txt");

  //          MappingData current;//=???
  //                Angle myHeading;//=???

      //Recognize my current position in the mapping, and then go toward the pipeline position

  //  if(!recognizeSIFT(BEOSUBCAMDOWN, current, myHeading)){//Try to recognize where we are (max 8 tries right now)
         //counter++;
         //Ghetto circle search. FIX!

         //need to calculate the 'distance' between current and goal (BtoA)
         //then drive the sub there
         //when to stop???

         //advanceRel(-3.5);
         //turnRel(20.0);
  //  }

          //if(counter >= 8){//If we are lost, fail
          //  return false;
          //}
          //else{//approach task A to B pipes
          //  /*
          //              if(approachArea(BEOSUBCAMDOWN, BtoA, 3.0)){
          //              stage = 7;
          //              counter = 0;
          //              break;
          //              }
          //  */
          // tesl counter++;
          //}


      return true;
}//end of ApproachPipeLine


bool BeoSub::FollowPipeLine()        //if the end is the bin, then return true, else, return false
{
  ModelManager camManager("ColorTracker Tester");

  nub::soft_ref<FrameIstream>
    gb(makeIEEE1394grabber(camManager, "colorcam", "cocam"));

  camManager.addSubComponent(gb);

  camManager.start();

  // instantiate a model manager for the color tracker module:
  ModelManager manager("ColorTracker Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<ColorTracker> test(new ColorTracker(manager));
  manager.addSubComponent(test);
  manager.start();

  Image< PixRGB<byte> > Img;
  float x = 320/2, y=240/2;
  //arbitary area of mass, need to calibrate!!!!
  float mass = 10.0;

  float width;
  float height;
  rutz::shared_ptr<XWindow> wini;

  while(1){
    //Get image to be matched
    //TO TEST FROM CAMERA
      Img = gb->readRGB();

      test->setupTracker("Orange", Img /*grabImage(BEOSUBCAMDOWN)*/, true);
      test->runTracker(25.0, x, y, mass);

      // wini->drawImage(Img);
      width = Img.getWidth();

      //make sure that positive is up-right
      x = x - width/2;

      height = Img.getHeight();
      y = height/2 - y;

#ifndef DEBUG
#else
       printf("X: %f, WIDTH: %f\n", x, width);
       printf("Y: %f, HEIGHT: %f\n", y, height);
#endif
      //test the area of the yellow pipe,
      //if it's small enough, it's the end;
      //otherwise, go straight

       if( mass > 1000 )
         {  //go straight
#ifndef DEBUG
           advanceRel(1.5);
#else
           printf("i'm going forward!\n");
#endif
         }
       else //it's either the end of the pipeline or there's a break for a box
         {
           printf("Break in pipeline\n");

           //test if the end is a bin or not
           //1 for black bin, other for white bin
           //bool rec = TestBin(1);
           bool rec=TestBin(gb,1);
           if (rec)
             {
               printf("FOUND BIN\n");
             }
           // return TestBin();
           return rec;
         }

       //PID control of the tracking,!!! now just x-tracking

       if(fabs(x)<=20)
         {//go straight
#ifndef DEBUG
           advanceRel(1.5);
#else
             printf("Go Straight %f\n", fabs(x));
#endif
         }
       else
         {
           if (y == 0.0)
             {
               y+=0.01;
             }  //prevent divide by zero error

           printf("turn: %f %f\n",atan(x/y)*180/M_PI, M_PI);

           if (atan(x/y)*180/M_PI > 0)
             {
               printf("Turn Right %f\n", fabs(x));
             }
           else
             {
               printf("Turn Left %f\n", fabs(x));
             }
#ifndef DEBUG
            turnRel(Angle(atan(x/y) * 180/M_PI));
#else
            printf("i'm turing to the center of the orange!\n");
#endif
         }

       //!!!need to check if overturned 180 degrees
  }//end of while

  //should never come here
  return false;

  /*//should x,y,mass be pointer?
          float x = 0.0, y = 0.0, mass = 0.0;
  //do we need a counter?
  int counter = 0;
  itsColorTracker2->setupTracker("Orange", grabImage(BEOSUBCAMDOWN), false);

                                                                                                                                                                        //referance???
  while(itsColorTracker2->runTracker(10.0, x, y, mass) && counter < 12){

     //if current pos is (0,0)
     //color tracker returned pos is (x,y)
     //should the sub turn to (x,y)

     //float eps=0.01;
     //if (x == 0.0){
     //          x += eps;
     //}
     //
     //if (y == 0.0){
     //          y += eps;
     //}
     //float Angle = tan(y/x);
     //float Dist = sqrt(x*x + y*y);

     //turn Angle and goto Dist???

     //++counter;
     //NOTE: Down cam weirdness! FIX!!
     //if(y < 120){//on left...
     // turnRel(-10.0);
     //}
     //else{//on right...
     //  turnRel(10.0);
     //}

      //FIX

     //advanceRel(1.5);//follow

                 //test if goes to the bins
     itsColorTracker->setupTracker("White", grabImage(BEOSUBCAMDOWN), false);
     if(itsColorTracker->runTracker(500.0, x, y, mass)) {
       //stage = 3;
       break;
     }//end of if itsColorTracker

   }//end of if itsColor Tracker2
             //itsColorTracker2->setupTracker("Orange", grabImage(BEOSUBCAMDOWN), false);
        //stage = 3;
             //done following, try to find box
          // stage = 0; ///?????
          */
        //break;
}//end of FollowPipeLine

int BeoSub::RecognizeBin()
{
  //recognize the pattern the bins
  //return the id of the bin

  return 0;
}


bool BeoSub::TestBin(nub::soft_ref<FrameIstream> gb, int testColor)
{  //test if the end is a bin or not

  const char* colorArg=NULL;

  if(testColor==0)
    {
      colorArg="White";
    }
  else
    {
      colorArg="Black";
    }

  int shapeCounter=0;//counter for the shape recognize loop

  // instantiate a model manager (for camera input):
  ModelManager manager("Canny Tester");
  // Instantiate our various ModelComponents:

  //######################################################
  //comment out in order to avoid camera setup problems
  // nub::soft_ref<FrameIstream>
  //  gb(makeIEEE1394grabber(manager, "cannycam", "cc"));

  //GRAB image from camera to be tested
  //manager.addSubComponent(gb);
  //#######################################################

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoSubCanny> test(new BeoSubCanny(manager));
  manager.addSubComponent(test);


   //Load in config file for camera FIX: put in a check whether config file exists!
  manager.loadConfig("camconfig.pmap");

  manager.start();

  //Test with a circle
  rutz::shared_ptr<ShapeModel> shape;

  Image< PixRGB<byte> > Img;

  double* p;

  //Set up shape to be matched
  //Recongnize the rectangle bin (or square???)

      //rectangle
  // p = (double*)calloc(6, sizeof(double));
  //p[1] = 150.0; //Xcenter
  // p[2] = 120.0; //Ycenter
  //  p[4] = 80.0f; // Width
  //  p[5] = 80.0f; // Height
  //  p[3] = (3.14159/4.0); //alpha
  //  shape.reset(new RectangleShape(120.0, p, true));


  //square
  p = (double*)calloc(5, sizeof(double));
  p[1] = 150.0; //Xcenter
  p[2] = 120.0; //Ycenter
  p[3] = 100.0; // Height
  p[4] = (3.14159/4.0); // alpha
  shape.reset(new SquareShape(100.0, p, true));

  while(1){

    //if not found anything in 5 loops, there is no bin

    if(shapeCounter++>5)
      {
        printf("Cannot find the bin, I give up!\n");
        return false;
      }

    //Get image to be matched
    //TO TEST FROM CAMERA
    Img = gb->readRGB();

    shape->setDimensions(p);

    //run the matching code, black bin
    test->setupCanny(colorArg, Img, true);

    //Middle
    //p[1] = 150.0;
    //p[2] = 120.0;
    //shape->setDimensions(p);
    bool shapeFound = test->runCanny(shape);

    if(!shapeFound){
      //stupid compiler, breaking on stupid warnings
    }

    //NOTE: Uncomment the following code to test using multiple starting points

    if(!shapeFound){ //Upper left
      p[1] = 60.0; //Xcenter
      p[2] = 180.0; //Ycenter
      shape->setDimensions(p);
      shapeFound = test->runCanny(shape);
    }
    if(!shapeFound){ //Upper right
      p[1] = 260.0; //Xcenter
      p[2] = 180.0; //Ycenter
      shape->setDimensions(p);
      shapeFound = test->runCanny(shape);
    }
    if(!shapeFound){ //Lower left
      p[1] = 60.0; //Xcenter
      p[2] = 60.0; //Ycenter
      shape->setDimensions(p);
      shapeFound = test->runCanny(shape);
    }
    if(!shapeFound){ //Lower right
      p[1] = 260.0; //Xcenter
      p[2] = 60.0; //Ycenter
      shape->setDimensions(p);
      shapeFound = test->runCanny(shape);
    }

    if(shapeFound)
      {//got the bin, go back!!!
        return true;
      }

    printf("shape not found in loop %i\n",shapeCounter-1);


  }//end of while loop

  //should never got here!!!
  return false;
}//end of TestBin

bool BeoSub::CenterBin()
{
//center the bin for marker dropping

//get the mapping of the connection of bin and the pipeline
//compare the mapping, and approach the bin

  ModelManager camManager("ColorTracker Tester");

  nub::soft_ref<FrameIstream>
    gb(makeIEEE1394grabber(camManager, "colorcam", "cocam"));

  camManager.addSubComponent(gb);

  camManager.start();

  // instantiate a model manager for the color tracker module:
  ModelManager manager("ColorTracker Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<ColorTracker> test(new ColorTracker(manager));
  manager.addSubComponent(test);
  manager.start();

  Image< PixRGB<byte> > Img;
  float x = 320/2, y=240/2;

  //arbitary area of mass, need to calibrate!!!!
  float mass = 10.0;

  float width;
  float height;
  rutz::shared_ptr<XWindow> wini;

  bool xdone=false;
  bool ydone=false;

  while(!(xdone && ydone)){
    //Get image to be matched
    //TO TEST FROM CAMERA
      Img = gb->readRGB();

      //track the center of the black bin
#ifndef DEBUG
      test->setupTracker("Black", grabImage(BEOSUBCAMDOWN), true);
#else
      test->setupTracker("Black", Img /*grabImage(BEOSUBCAMDOWN)*/, true);
#endif

      test->runTracker(25.0, x, y, mass);

      // wini->drawImage(Img);
      width = Img.getWidth();

      //make sure that positive is up-right
      x = x - width/2;

      height = Img.getHeight();
      y = height/2 - y;

      //centering part
      //need to change to PID control!!!
      //now just x,y tracking!

      //center x first
      printf("centering x...\n");
       if(fabs(x) > 20)
         {
           if (y == 0.0)
             {
               y+=0.01;
             }  //prevent divide by zero error

           printf("turn: %f %f\n",atan(x/y)*180/M_PI, M_PI);

#ifndef DEBUG
           turnRel(Angle(atan(x/y) * 180/M_PI));
#else
           if (atan(x/y)*180/M_PI > 0)
             {
               printf("Turn Right %f\n", fabs(x));
             }
           else
             {
               printf("Turn Left %f\n", fabs(x));
             }
#endif

         }//end of if x
       xdone=true;

       //then center y

       printf("x done!\ncentering y.\n");
       if(fabs(y) > 20)
         {
           if(y>0)
             {

#ifndef DEBUG
               advanceRel(1.5);
#else
               printf("going forward!\n");
#endif
             }
           else
             {
#ifndef DEBUG
               advanceRel(-1.5);
#else
               printf("going backward!\n");
#endif
             }
         }//end of if y
       printf("y done!\n");
       ydone=true;

       //!!!need to check if overturned 180 degrees
  }//end of while

  if(xdone && ydone)
    {
      printf("alright, center finished!\n");
      return true;
    }
  else
    {
      printf("woops, center failed!\n");
      return false;
    }

//        return true;
}//end of CenterBin

bool BeoSub::DropMarker()
{
        //can we see the marker???
  for(int i = 0; i < 6; i++)
    {
//############sub control part#####################
      //dropMarker();
    }
  printf("OK, marker dropped!\n");
  return true;

}//end of DropMarker


bool BeoSub::PassBin()        //Go Pass the bin
{
//get the mapping the connection of the bin and the pipeline
//compare the mapping, go through the bin to the next connection

  return true;
}//end of PassBin


bool BeoSub::TaskC(){

  int step = 0;
  bool notDone = true;
  int counter = 0, maxcounter = 15;
  //NOTE: should we try shape recognition or saliency here? FIX?
  while(notDone){
    switch(step){

    case 0: //LOOK FOR PINGER & OCTAGON AT SAME TIME USING MAPPING
      {
        if (counter > maxcounter) return false;

        //BOTTOM
        rutz::shared_ptr<VisualObject> down(new VisualObject("current", "", grabImage(BEOSUBCAMDOWN)));
        rutz::shared_ptr<VisualObject> pingerDown = itsDownVODB->getObject("TaskC");
        VisualObjectMatch matchDown(pingerDown, down, VOMA_KDTREEBBF);

        //FRONT
        advanceRel(-8.0);
        diveRel(.3);
        rutz::shared_ptr<VisualObject> front(new VisualObject("current", "", grabImage(BEOSUBCAMFRONT)));
        rutz::shared_ptr<VisualObject> pingerFront = itsFrontVODB->getObject("TaskC");
        VisualObjectMatch matchFront(pingerFront, front, VOMA_KDTREEBBF);
        diveRel(-.3);
        advanceRel(8.0);

        if(matchDown.checkSIFTaffine() || matchFront.checkSIFTaffine()){
          step = 1;
          break;
        }
        else if(counter < 10){
          //ghetto circle strafe
          advanceRel(-1.5);
          turnRel(20.0);
          counter++;
        }
        else{
          return false;
        }
      }
    case 1://FINE TUNE POSITION USING MAPPING
      {
        if (counter > maxcounter) return false;

        if(affineSIFT(BEOSUBCAMDOWN, itsVOtaskCdown)){//approach
          step = 2;
          break;
        }
        else{
          ++counter; step = 0;
        }
      }
    case 2: //SURFACE!
      {
        if (counter > maxcounter) return false;

        //set ballast to 0 and wait until depth says 0?
        //or diveAbs?
        //use if statement to say whether dive again or not
        diveAbs(-1.0, true); //surface
        //if(!last){
        //  diveAbs(4.0, true);
        // }
        notDone = false;
      }
    }
  }
  taskCdone = true;
  return true;
}
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
