/*!@file SeaBee/AgentManagerB.C management class for agents on COM-B*/

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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/AgentManagerB.C $
// $Id: AgentManagerB.C 10794 2009-02-08 06:21:09Z itti $
//

#define NUM_STORED_FRAMES    20

#include "AgentManagerB.H"

// ######################################################################
// function for each of the separately threaded agents
// which calls their schedulers
void* runDownwardVisionAgent(void* a)
{

  AgentManagerB* am  = (AgentManagerB *)a;
  rutz::shared_ptr<DownwardVisionAgent> dv = am->getDownwardVisionAgent();

  dv->run();

  return NULL;
}

// ######################################################################
void* runSonarListenAgent(void* a)
{
  AgentManagerB* am  = (AgentManagerB *)a;
  rutz::shared_ptr<SonarListenAgent> pfc = am->getSonarListenAgent();

  pfc->run();
  return NULL;
}

// ######################################################################
AgentManagerB::AgentManagerB(OptionManager& mgr,
                             const std::string& descrName,
                             const std::string& tagName)
  :
  ModelComponent(mgr, descrName, tagName)
{
  rutz::shared_ptr<AgentManagerB> amb(this);
   // create the agents
  itsDownwardVisionAgent.reset(new DownwardVisionAgent("DownwardVisionAgent",amb));
  itsSonarListenAgent.reset(new SonarListenAgent("SonarListenAgent",amb));


  // store created agents in vector
  itsSubmarineAgents.push_back(itsDownwardVisionAgent);
  itsSubmarineAgents.push_back(itsSonarListenAgent);

  // create threads for the agents
  pthread_create
    (&itsDownwardVisionAgentThread, NULL, runDownwardVisionAgent,
     (void *)this);

  pthread_create
    (&itsSonarListenAgentThread, NULL, runSonarListenAgent,
     (void *)this);

  itsInputImageTimer.reset(new Timer(1000000));
  itsFrameDuration.resize(NUM_STORED_FRAMES);

  itsInputImageTimerB.reset(new Timer(1000000));
  itsFrameDurationB.resize(NUM_STORED_FRAMES);
}

// ######################################################################
AgentManagerB::~AgentManagerB()
{ }

// ######################################################################
void AgentManagerB::setCurrentImageF
(Image<PixRGB<byte> > image, uint fNum)
{
  // set the current image
  pthread_mutex_lock(&itsCurrentImageLock);
  itsCurrentImage = image;
  itsFrameNumber = fNum;
  pthread_mutex_unlock(&itsCurrentImageLock);

  itsDownwardVisionAgent->msgSensorUpdate();
//   // compute and show framerate over the last NAVG frames:
//   itsFrameDuration[fNum % NUM_STORED_FRAMES] = itsInputImageTimer->get();
//   itsInputImageTimer->reset();

//   uint nFrames = NUM_STORED_FRAMES;
//   if (nFrames < NUM_STORED_FRAMES) nFrames = fNum;
//   uint64 avg = 0ULL;
//   for(uint i = 0; i < nFrames; i ++) avg += itsFrameDuration[i];
//   float frate = 1000000.0F / float(avg) * float(NUM_STORED_FRAMES);

//   std::string ntext(sformat("%6d: %6.3f fps -> %8.3f ms/fr",
//                             fNum, frate, 1000.0/frate));
//   //  writeText(image, Point2D<int>(0,0), ntext.c_str());
//   if((fNum % 30)==0)
//     drawImage(image,Point2D<int>(0,0));
//   //  LINFO("%s",ntext.c_str());
}

// ######################################################################
void AgentManagerB::setCurrentImageB
(Image<PixRGB<byte> > image, uint fNum)
{
  // set the current image
  pthread_mutex_lock(&itsCurrentImageLockB);
  itsCurrentImageB = image;
  itsFrameNumberB = fNum;
  pthread_mutex_unlock(&itsCurrentImageLockB);

  // compute and show framerate over the last NAVG frames:
  itsFrameDurationB[fNum % NUM_STORED_FRAMES] = itsInputImageTimerB->get();
  itsInputImageTimerB->reset();

  uint nFrames = NUM_STORED_FRAMES;
  if (nFrames < NUM_STORED_FRAMES) nFrames = fNum;
  uint64 avg = 0ULL;
  for(uint i = 0; i < nFrames; i ++) avg += itsFrameDurationB[i];
  float frate = 1000000.0F / float(avg) * float(NUM_STORED_FRAMES);

  std::string ntext(sformat("%6d: %6.3f fps -> %8.3f ms/fr",
                            fNum, frate, 1000.0/frate));
  //  writeText(image, Point2D<int>(0,0), ntext.c_str());
  //  if((fNum % 30)==0)
    //    drawImageB(image, Point2D<int>(0,0));
  //  LINFO("%s",ntext.c_str());
}

// ######################################################################
void AgentManagerB::pushResult(SensorResult sensorResult)
{
  pthread_mutex_lock(&itsResultsLock);
  itsResults.push_back(sensorResult);
  pthread_mutex_unlock(&itsResultsLock);
}

// ######################################################################
uint AgentManagerB::getNumResults()
{
  return itsResults.size();
}

// ######################################################################
SensorResult AgentManagerB::popResult()
{
  pthread_mutex_lock(&itsResultsLock);
  SensorResult sr;

  if(getNumResults() > 0)
    {
      sr = itsResults.back();
      itsResults.pop_back();
    }
  pthread_mutex_unlock(&itsResultsLock);

  return sr;
}


// ######################################################################
// void AgentManagerB::drawImage(Image<PixRGB<byte> > ima,
//                               Point2D<int> point)
// {
//   pthread_mutex_lock(&itsDisplayLock);
//   inplacePaste(itsDisplayImage, ima, point);
//   itsWindow->drawImage(itsDisplayImage,0,0);
//   pthread_mutex_unlock(&itsDisplayLock);

// }

// ######################################################################
// void AgentManagerB::drawImageB(Image<PixRGB<byte> > ima, Point2D<int> point)
// {
//   pthread_mutex_lock(&itsDisplayLockB);
//   //  inplacePaste(itsDisplayImageB, ima, point);
//   //  itsWindowB->drawImage(itsDisplayImageB,0,0);
//   pthread_mutex_unlock(&itsDisplayLockB);
// }

// ######################################################################
void AgentManagerB::updateAgentsMission(Mission theMission)
{
  uint size = itsSubmarineAgents.size();

  // iterate through all agent manager's agents
  for(uint i = 0; i < size; i++)
    {
      // for each agent, update mission
      (itsSubmarineAgents.at(i))->msgUpdateMission(theMission);
    }
}

// ######################################################################
rutz::shared_ptr<DownwardVisionAgent> AgentManagerB::getDownwardVisionAgent()
{
  return itsDownwardVisionAgent;
}

// ######################################################################
rutz::shared_ptr<SonarListenAgent> AgentManagerB::getSonarListenAgent()
{
  return itsSonarListenAgent;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
