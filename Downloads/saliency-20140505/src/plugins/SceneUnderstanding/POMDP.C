/*!@file SceneUnderstanding/POMDP.C  partially observable Markov decision processes */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/POMDP.C $
// $Id: POMDP.C 13551 2010-06-10 21:56:32Z itti $
//

#include "plugins/SceneUnderstanding/POMDP.H"

#include "Image/FilterOps.H"
#include "Image/PixelsTypes.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Raster/Raster.H"

#include <cstdio> // for printf()

// ######################################################################
POMDP::POMDP() :
  itsRetinaSize(640,480),
  itsAgentState(-1,-1),
  itsGoalState(-1,-1),
  itsLastDistance(10000),
  itsLastAction(-1)
{

}

// ######################################################################
POMDP::~POMDP()
{

}

// ######################################################################
bool POMDP::makeObservation(const Image<PixRGB<byte> > &img)
{
  const char* personObj = "/home/lior/saliency/etc/objects/person.ppm";
  const char* doorObj =   "/home/lior/saliency/etc/objects/door.ppm";
  const char* blockObj =  "/home/lior/saliency/etc/objects/wall.ppm";

  //The agent is the only one that moves
  itsAgentState = findObject(img, personObj);

  //build the state space
  //if (!itsStateSpace.initialized())
  {
    itsStateSpace = Image<byte>(img.getDims(), ZEROS);

    itsGoalState = findObject(img, doorObj);
    drawFilledRect(itsStateSpace,
        Rectangle(itsGoalState-(31/2), Dims(31,31)),
        (byte)GOAL);


    std::vector<Point2D<int> > wallState = findMultipleObjects(img, blockObj);
    for(uint i=0; i<wallState.size(); i++)
    {
      drawFilledRect(itsStateSpace,
          Rectangle(wallState[i]-(31/2), Dims(31,31)),
          (byte)WALL);
    }

    itsStateSpace = scaleBlock(itsStateSpace, itsStateSpace.getDims()/31);

    // itsStateSpace = Image<byte>(40,30,ZEROS);

    // //itsStateSpace.setVal(1,1,WALL);
    // drawLine(itsStateSpace, Point2D<int>(12,12), Point2D<int>(12,30), (byte)WALL);
    // itsStateSpace.setVal(37,25,GOAL);
    // itsStateSpace.setVal(10,15,HOLE);
    // itsStateSpace.setVal(22,16,HOLE);

    // itsStateSpace = Image<byte>(4,3,ZEROS);
    // itsStateSpace.setVal(1,1, WALL);
    // itsStateSpace.setVal(3,0, GOAL);
    // itsStateSpace.setVal(3,1, HOLE);

    itsCurrentPercep[0] = Image<float>(itsStateSpace.getDims(), ZEROS);
    itsCurrentPercep[0].setVal(itsAgentState/31, 1.0F);

    //itsStateTrans[78] = 124;
    return true;

  }

  return false;


}

void POMDP::init()
{
  Image<float> percep(itsRetinaSize, ZEROS);
  percep.clear(1.0e-5);
  itsCurrentPercep.push_back(percep);

  //Load the objects models

  const char* personObj = "/home/lior/saliency/etc/objects/person.ppm";
  Image<PixRGB<byte> > obj = Raster::ReadRGB(personObj);
  itsObjects.push_back(obj);

  const char* doorObj =   "/home/lior/saliency/etc/objects/door.ppm";
  obj = Raster::ReadRGB(doorObj);
  itsObjects.push_back(obj);

  const char* blockObj =  "/home/lior/saliency/etc/objects/wall.ppm";
  obj = Raster::ReadRGB(blockObj);
  itsObjects.push_back(obj);

  //itsStateSpace = Image<byte>(img.getDims(), ZEROS);

  //itsGoalState = findObject(img, doorObj);
  //drawFilledRect(itsStateSpace,
  //    Rectangle(itsGoalState-(31/2), Dims(31,31)),
  //    (byte)GOAL);


  //std::vector<Point2D<int> > wallState = findMultipleObjects(img, blockObj);
  //for(uint i=0; i<wallState.size(); i++)
  //{
  //  drawFilledRect(itsStateSpace,
  //      Rectangle(wallState[i]-(31/2), Dims(31,31)),
  //      (byte)WALL);
  //}
}


Image<float> POMDP::getPerception(const uint obj)
{
  if (obj < itsCurrentPercep.size())
    return itsCurrentPercep[obj];

  return Image<float>();
}

bool POMDP::goalReached()
{
  Point2D<int> loc; float maxVal;
  findMax(itsCurrentPercep[0], loc, maxVal);

  int currentState = loc.j*itsCurrentPercep[0].getWidth() + loc.i;
  LINFO("Current State %i\n", currentState);

  if (itsStateSpace.getVal(currentState) == GOAL)
    return true;


  return  false;
}

Image<float> POMDP::makePrediction(const ACTIONS action)
{
    //Find the current agent location with the most probability
    Point2D<int> loc; float maxVal;
    findMax(itsCurrentPercep[0], loc, maxVal);

    int currentState = loc.j*itsCurrentPercep[0].getWidth() + loc.i;
    int newState = doAction(currentState, action);

    Image<float> prediction(itsStateSpace.getDims(), ZEROS);
    if (newState != -1)
      prediction.setVal(newState, 1.0F);

    itsPrediction = prediction;
    return prediction;
}

float POMDP::updatePerception(const Image<PixRGB<byte> > &img)
{
  const char* personObj = "/home/lior/saliency/etc/objects/person.ppm";
  //The agent is the only one that moves
  itsAgentState = findObject(img, personObj);

  itsPreviousPercep = itsCurrentPercep[0];

  itsCurrentPercep[0].clear(0.0F);
  itsCurrentPercep[0].setVal(itsAgentState/31, 1.0F);

  //Run Bayes filter to update perception

  //Get the supprise which is the kl diffrance between
  //the posterior and prior

  float klDist = 0;
  for(int i=0; i<itsCurrentPercep[0].getSize(); i++)
  {
    float posterior = itsCurrentPercep[0].getVal(i);
    float prior = itsPrediction.getVal(i);

    if (prior == 0)  //avoid devide by zero
      prior = 1.0e-10;
    if (posterior == 0)  //avoid devide by zero
      posterior = 1.0e-10;

    klDist += posterior * log(posterior/prior);
  }

  return klDist;
}

void POMDP::updateStateTransitions(const ACTIONS action)
{

  Point2D<int> loc; float maxVal;

  //findMax(itsPreviousPercep, loc, maxVal);
  //int previousState = loc.j*itsPreviousPercep.getWidth() + loc.i;

  findMax(itsPrediction, loc, maxVal);
  int previousState = loc.j*itsPrediction.getWidth() + loc.i;

  findMax(itsCurrentPercep[0], loc, maxVal);
  int currentState = loc.j*itsCurrentPercep[0].getWidth() + loc.i;

  //if (previousState == 78)
    //itsStateTrans[previousState] = currentState;

  LINFO("Previous %i action %i current %i",
      previousState,
      action,
      currentState);

}

Image<byte> POMDP::getStateSpace()
{
  return itsStateSpace;
}

Point2D<int> POMDP::getAgentState()
{
  return itsAgentState;
}

Point2D<int> POMDP::getGoalState()
{
  return itsGoalState;
}

void POMDP::showTransitions()
{
  int nActions = 4;
  int nStates = itsStateSpace.getSize();

  for(int state=0; state<nStates; state++)
    for(int act=0; act<nActions; act++)
    {
      printf("State %i ", state);
      switch(act)
      {
        case NORTH: printf("Action=North "); break;
        case EAST:  printf("Action=East "); break;
        case WEST:  printf("Action=West "); break;
        case SOUTH: printf("Action=South "); break;
      }

      for(int newState=0; newState<nStates; newState++)
        if (itsTransitions[state][act][newState] > 0)
          printf("%i=%f ", newState, itsTransitions[state][act][newState]);
      printf("\n");
    }

}


int POMDP::doAction(const int state, const int act)
{

  ////if we are in any of the termination states or walls then we go to lala-land
  //switch(itsStateSpace.getVal(state))
  //{
  //  case GOAL:
  //  case HOLE:
  //  case WALL:
  //    return -1;
  //}

  int newState = state; //by defualt we dont move

  //std::map<int, int>::iterator it;
  //it = itsStateTrans.find(state);

  if (false) //it != itsStateTrans.end())  //did we learn about this
  {
    //newState = it->second; //the port hole
  } else {
    Point2D<int> currentPos;
    currentPos.j = state / itsStateSpace.getWidth();
    currentPos.i = state - (currentPos.j*itsStateSpace.getWidth());

    switch(act)
    {
      case NORTH: currentPos.j -= 1; break;
      case EAST: currentPos.i += 1; break;
      case WEST: currentPos.i -= 1; break;
      case SOUTH: currentPos.j += 1; break;
    }

    //check if the new state is valid
    if (itsStateSpace.coordsOk(currentPos) &&
        itsStateSpace.getVal(currentPos) != WALL)
    {
      newState = currentPos.j*itsStateSpace.getWidth() + currentPos.i;
    }
  }

  return newState;

}

void POMDP::valueIteration()
{
  int nActions = 4;
  int nStates = itsStateSpace.getSize();

  Image<float> newUtility(itsStateSpace.getDims(), ZEROS);

  //set initial values to the reward
  for(int state=0; state<newUtility.getSize(); state++)
    newUtility.setVal(state, getReward(state));

  float thresh = 0.1;
  float discount = 1;
  float lemda = 1;
  while(lemda > thresh*(1-discount)/discount + thresh)
  {

    itsUtility = newUtility;
    lemda = 0;

    for(int state=0; state<nStates; state++)
    {

      //find the maxmium action
      float maxActVal = -std::numeric_limits<float>::max();
      for(int act=0; act<nActions; act++)
      {
        float sum = 0;
        for(int newState=0; newState<nStates; newState++)
          sum += getTransProb(state,act,newState) * itsUtility.getVal(newState);

        if (sum > maxActVal)
          maxActVal = sum;
      }

      float u = getReward(state) + discount*maxActVal;
      newUtility.setVal(state,u);

      if (fabs(u - itsUtility.getVal(state)) > lemda)
        lemda = fabs(u - itsUtility.getVal(state));
    }
    LINFO("Lmeda %f\n", lemda);

  }
  ////show the new utility
  //for(int j=0; j<itsUtility.getHeight(); j++)
  //{
  //  for(int i=0; i<itsUtility.getWidth(); i++)
  //    printf("%0.3f ", itsUtility.getVal(i,j));
  //  printf("\n");
  //}
  //getchar();
 // inplaceNormalize(itsUtility, 0.0F, 255.0F);
//  SHOWIMG(scaleBlock(itsUtility, itsUtility.getDims()*5));

}

float POMDP::getTransProb(int state, int action, int newState)  // p(s'|s,u)
{

  float prob = 0;
 //do the actions and mark the state
  if (doAction(state,action) == newState)
    prob += 0.8;

  //with some probabilty the actions will go preperminculer
  //to the desired action
  switch(action)
  {
    case NORTH:
    case SOUTH:
      if (doAction(state,EAST) == newState)
        prob += 0.1;
      if (doAction(state,WEST) == newState)
        prob += 0.1;
      break;
    case EAST:
    case WEST:
      if (doAction(state,NORTH) == newState)
        prob += 0.1;
      if (doAction(state,SOUTH) == newState)
        prob += 0.1;
      break;
  }

  return prob;

}


float POMDP::getReward(int state)
{

  switch(itsStateSpace.getVal(state))
  {
    case GOAL:
      return 1.0F;
    case HOLE:
    case WALL:
      return -1.0F;
    default:
      return -0.01;
  }
  return 0; //Should never get here
}

void POMDP::doPolicy(const Point2D<int>& startPos)
{

  LINFO("Do policy from %ix%i\n", startPos.i, startPos.j);
  Image<byte> ssImg = itsStateSpace;
  inplaceNormalize(ssImg, (byte)0, (byte)255);

  Image<PixRGB<byte> > disp = ssImg;

  int state = (startPos.j/31)*itsStateSpace.getWidth() + (startPos.i/31);
  disp.setVal(state, PixRGB<byte>(255,0,0));

  Image<float> utilDisp = itsUtility;
  inplaceNormalize(utilDisp, 0.0F, 255.0F);
  SHOWIMG(scaleBlock(utilDisp, itsUtility.getDims()*5));
  for(int i=0; i<1000 && state != -1; i++)
  {
    int action = getAction(state);
    state = doAction(state, action);

    if (state != -1)
      disp.setVal(state, PixRGB<byte>(255,255,0));
  }
  if (state != -1)
  {
    LINFO("Can not solve, exploring");
    itsExploring =  true;
  } else {
    itsExploring = false;
  }

  SHOWIMG(scaleBlock(disp, disp.getDims()*5));



}

int POMDP::getPropAction()
{
  if (itsCurrentPercep.size() == 0)
    return -1;
  Point2D<int> loc; float maxVal;
  findMax(itsCurrentPercep[0], loc, maxVal);

  int currentState = loc.j*itsCurrentPercep[0].getWidth() + loc.i;

  return getAction(currentState);

}


POMDP::ACTIONS POMDP::getAction(int state)
{

  int nActions = 4;
  int nStates = itsStateSpace.getSize();

  float maxActVal = -std::numeric_limits<float>::max();
  int action = -1;

  for(int act=0; act<nActions; act++)
  {
    float sum = 0;
    for(int newState=0; newState<nStates; newState++)
    {
      if (newState < itsUtility.getSize())
        sum += getTransProb(state,act,newState) * itsUtility.getVal(newState);
    }

    if (sum > maxActVal)
    {
      maxActVal = sum;
      action = act;
    }
  }

  return (ACTIONS)action;
}

Image<float> POMDP::locateObject(const Image<float>& src, Image<float>& filter)
{

  Image<float> result(src.getDims(), ZEROS);
  const int src_w = src.getWidth();
  const int src_h = src.getHeight();

  Image<float>::const_iterator fptrBegin = filter.begin();
  const int fil_w = filter.getWidth();
  const int fil_h = filter.getHeight();

  Image<float>::const_iterator sptr = src.begin();

  const int srow_skip = src_w-fil_w;

  float maxDiff = 256*fil_w*fil_h;

  for (int dst_y = 0; dst_y < src_h-fil_h; dst_y++) {

    for (int dst_x = 0; dst_x < src_w-fil_w; dst_x++) {

      float dst_val = 0.0f;
      //find the object at this position
      Image<float>::const_iterator fptr = fptrBegin;
      Image<float>::const_iterator srow_ptr = sptr + (src_w*dst_y) + dst_x;
      for (int f_y = 0; f_y < fil_h; ++f_y)
      {
        for (int f_x = 0; f_x < fil_w; ++f_x){
          dst_val += fabs((*srow_ptr++) - (*fptr++));
        }
        srow_ptr += srow_skip;
      }
      float prob = 1-dst_val/(maxDiff * 0.25); //convert to a probability
      if (prob < 0) prob = 0;

      result.setVal(dst_x, dst_y, dst_val); //prob);
    }
  }

  return result;

}


// ######################################################################
Point2D<int> POMDP::findObject(const Image<PixRGB<byte> > &img, const char* filename)
{


  //template matching for object recognition
  Image<PixRGB<byte> > obj = Raster::ReadRGB(filename);

  Image<float> objLum = luminance(obj);
  Image<float> imgLum = luminance(img);

  Image<float> result = locateObject(imgLum, objLum);

  Point2D<int> loc; float maxVal;
  findMax(result, loc, maxVal);
  loc.i += (objLum.getWidth()/2);
  loc.j += (objLum.getHeight()/2);

  return loc;
}

// ######################################################################
std::vector<Point2D<int> > POMDP::findMultipleObjects(const Image<PixRGB<byte> > &img, const char* filename)
{

  std::vector<Point2D<int> > objectLocations;

  //template matching for object recognition
  Image<PixRGB<byte> > obj = Raster::ReadRGB(filename);

  Image<float> objLum = luminance(obj);
  Image<float> imgLum = luminance(img);

  Image<float> result = locateObject(imgLum, objLum);

  Point2D<int> loc; float maxVal;
  findMax(result, loc, maxVal);
  result.setVal(loc, 0.0F); //IOR
  loc.i += (objLum.getWidth()/2);
  loc.j += (objLum.getHeight()/2);
  objectLocations.push_back(loc);

  float objMaxVal = maxVal;
  //find any other objects
  while(1)
  {
    Point2D<int> loc; float maxVal;
    findMax(result, loc, maxVal);
    if (maxVal > objMaxVal*0.8)
    {
      result.setVal(loc, 0.0F); //IOR
      loc.i += (objLum.getWidth()/2);
      loc.j += (objLum.getHeight()/2);
      objectLocations.push_back(loc);
    } else {
      break;
    }
  }


  return objectLocations;
}

float POMDP::bayesFilter(const int action, const Image<PixRGB<byte> > &img)
{



  //SHOWIMG(scaleBlock(prevBelif, prevBelif.getDims()*100));

  //prediction
  LINFO("Making prediction");
  Image<float> prevBelif = itsCurrentPercep[0];
  float entropy = getEntropy(prevBelif);
  LINFO("Entorpy %f", entropy);
  SHOWIMG(prevBelif);

  Image<float> newBelif = prevBelif;


  if (entropy < 10)
  {
    //foreach posible state update the belif
    //This sould be sampled using ddmcmc
    //but for now check all posible states
    for(int state=0; state<prevBelif.getSize(); state++)
    {
      float sum=0;
      for(int i=0; i<prevBelif.getSize(); i++)
      {
        //p(X=state|action,i) * p(i_t-1)
        if (prevBelif[i] > 0) //only check states with some belif
          sum += getTransProb(i, action, state) * prevBelif[i];
      }
      newBelif[state] = sum;
    }
  }
  //nomalize
  LINFO("Done");
  //SHOWIMG(scaleBlock(newBelif, newBelif.getDims()*100));
  SHOWIMG(newBelif);


  int objID = 0; //the person object

  Image<float> objLum = luminance(itsObjects[objID]);
  Image<float> imgLum = luminance(img);

  Image<float> result = locateObject(imgLum, objLum);
  result /= sum(result);
  result = rescale(result, newBelif.getDims());


  itsCurrentPercep[0] = result;
  itsCurrentPercep[0] /= sum(itsCurrentPercep[0]);

  SHOWIMG(itsCurrentPercep[0]);
  entropy = getEntropy(itsCurrentPercep[0]);
  LINFO("new Entorpy %f", entropy);


  return 0;
}

float POMDP::particleFilter(const int action, const Image<PixRGB<byte> > &img)
{
  if (itsParticleStateSpace.size() == 0)
  {
    int nParticles = 10;
    //Initalize the particles
    for(int i=0; i< nParticles; i++)
    {
      State state(2*i, 2*i);
      itsParticleStateSpace.push_back(state);
    }
  }
  Image<float> objLum = luminance(itsObjects[0]);
  Image<float> objBlob = gaussianBlob<float>(
      objLum.getDims(),
      Point2D<int>(objLum.getWidth()/2, objLum.getHeight()/2),
      (float)objLum.getWidth(), (float)objLum.getHeight());
  objLum *= objBlob;

  Image<float> imgLum = luminance(img);

  objLum = rescale(objLum, objLum.getDims()/2);
  imgLum = rescale(imgLum, imgLum.getDims()/2);


  Image<float> result(imgLum.getWidth()-objLum.getWidth()+1,
      imgLum.getHeight()-objLum.getHeight()+1,
      NO_INIT);

  cvMatchTemplate(img2ipl(imgLum),
      img2ipl(objLum),
      img2ipl(result),
      //CV_TM_CCOEFF);
    CV_TM_CCOEFF_NORMED);

  result = abs(result);
  float entropy = getEntropy(result);
  LINFO("new Entorpy %f", entropy);

  SHOWIMG(result);


  return 0;

}

float POMDP::getEntropy(Image<float> &belif)
{
  float sum = 0;
  for(int i=0; i<belif.getSize(); i++)
    sum += belif[i] * log((belif[i] != 0) ? belif[i] : 1.0e-5);
  return -1.0*sum;
}


// Get the probabilty of the mesurment of the object given the state
// Obj should be a full bayes object
float POMDP::getObjProb(const Image<PixRGB<byte> > &img,
    const State state, const int objID)
{
  Image<float> objLum = luminance(itsObjects[objID]);
  Image<float> imgLum = luminance(img);

  Image<float>::const_iterator imgLumPtr = imgLum.begin();
  Image<float>::const_iterator fptrBegin = objLum.begin();
  const int fil_w = objLum.getWidth();
  const int fil_h = objLum.getHeight();
  const int srow_skip = imgLum.getWidth()-fil_w;

  float prob = 1.0f;

  //find the object at this position
  Image<float>::const_iterator fptr = fptrBegin;
  Image<float>::const_iterator srow_ptr = imgLumPtr + (imgLum.getWidth()*state.y) + state.x;
  for (int f_y = 0; f_y < fil_h; ++f_y)
  {
    for (int f_x = 0; f_x < fil_w; ++f_x){
      prob += fabs((*srow_ptr++) - (*fptr++));
    }
    srow_ptr += srow_skip;
  }

  return 1/prob;

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

