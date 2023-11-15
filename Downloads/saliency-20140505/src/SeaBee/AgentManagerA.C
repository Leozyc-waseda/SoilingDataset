/*!@file SeaBee/AgentManager.C
 for AUVSI2007 manage agents in COM_A                                   */
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
// /////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Micheal Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/AgentManagerA.C $
// $Id: AgentManagerA.C 10794 2009-02-08 06:21:09Z itti $
//
#define NUM_STORED_FRAMES    20

#include "AgentManager.H"

// ######################################################################
// function for each of the separately threaded agents
// which calls their schedulers
void* runDownwardVisionAgent(void* a)
{

  AgentManager* am  = (AgentManager *)a;
  rutz::shared_ptr<DownwardVisionAgent> dv = am->getDownwardVisionAgent();

  dv->run();

  return NULL;
}

// ######################################################################
void* runSonarListenAgent(void* a)
{
  AgentManager* am  = (AgentManager *)a;
  rutz::shared_ptr<SonarListenAgent> sl = am->getSonarListenAgent();

  sl->run();
  return NULL;
}

// ######################################################################
// function for each of the separately threaded agents
// which calls their schedulers
void* runForwardVisionAgent(void* a)
{
  AgentManager* am  = (AgentManager *)a;
  rutz::shared_ptr<ForwardVisionAgent> fv = am->getForwardVisionAgent();

  fv->run();

  return NULL;
}

// ######################################################################
void* runCaptainAgent(void* a)
{
  AgentManager* am  = (AgentManager *)a;
  rutz::shared_ptr<CaptainAgent> c = am->getCaptainAgent();

  c->run();
  return NULL;
}

// ######################################################################
void* runMovementAgent(void* a)
{
  AgentManager* am  = (AgentManager *)a;
  rutz::shared_ptr<MovementAgent> ma = am->getMovementAgent();

  ma->run();

  return NULL;
}

// ######################################################################
AgentManager::AgentManager(nub::soft_ref<SubController> subController,
                             nub::soft_ref<EnvVisualCortex> evc,
                             ModelManager& mgr,
                             const std::string& descrName,
                             const std::string& tagName)
  :
  ModelComponent(mgr, descrName, tagName)
{

//   nub::soft_ref<SubController> (new SubController(mgr,
//                                                               "Motor",
//                                                              "Primitive"));
  itsSubController = subController;
  itsEVC = evc;
  //  mgr.addSubComponent(itsSubController);

  rutz::shared_ptr<AgentManager> ama(this);

  //init itsSensorResults
  initSensorResults();

  // create the agents
  itsDownwardVisionAgent.reset(new DownwardVisionAgent("DownwardVisionAgent",ama));
  itsSonarListenAgent.reset(new SonarListenAgent("SonarListenAgent",ama));
  itsForwardVisionAgent.reset(new ForwardVisionAgent("ForwardVisionAgent",ama));
  itsCaptainAgent.reset(new CaptainAgent("CaptainAgen
t",ama));
  itsMovementAgent.reset(new MovementAgent(itsSubController,
                                           ama,
                                           "MovementAgent"));

  // store created agents in vector
  itsSubmarineAgents.push_back(itsDownwardVisionAgent);
  itsSubmarineAgents.push_back(itsSonarListenAgent);
  itsSubmarineAgents.push_back(itsForwardVisionAgent);
  itsSubmarineAgents.push_back(itsCaptainAgent);
  itsSubmarineAgents.push_back(itsMovementAgent);

  // connect the agents properly
  itsForwardVisionAgent->setCaptainAgent(itsCaptainAgent);
  itsForwardVisionAgent->setVisualCortex(itsEVC);
  itsCaptainAgent->setMovementAgent(itsMovementAgent);
  itsCaptainAgent->setForwardVisionAgent(itsForwardVisionAgent);
  itsMovementAgent->setCaptainAgent(itsCaptainAgent);


  pthread_mutex_init(&itsDisplayLock, NULL);
  pthread_mutex_init(&itsCurrentImageLock, NULL);
  pthread_mutex_init(&itsCommandsLock, NULL);
  pthread_mutex_init(&itsSensorResultsLock, NULL);


  itsTimer.reset(new Timer(1000000));
  itsFrameDuration.resize(NUM_STORED_FRAMES);

  // create threads for the agents
  pthread_create
    (&itsDownwardVisionAgentThread, NULL, runDownwardVisionAgent,
     (void *)this);

  pthread_create
    (&itsSonarListenAgentThread, NULL, runSonarListenAgent,
     (void *)this);

  pthread_create
    (&itsForwardVisionAgentThread, NULL, runForwardVisionAgent,
     (void *)this);

  pthread_create
    (&itsCaptainAgentThread, NULL, runCaptainAgent,
     (void *)this);

  pthread_create
    (&itsMovementAgentThread, NULL, runMovementAgent,
     (void *)this);
}


// ######################################################################
AgentManager::~AgentManager()
{ }

// ######################################################################
void AgentManager::startRun()
{
  // call prefrontal cortex to start
  itsCaptainAgent->start();
}

// ######################################################################
void AgentManager::setCurrentImage
(Image<PixRGB<byte> > image, uint fNum)
{
  // set the current image
  pthread_mutex_lock(&itsCurrentImageLock);
  itsCurrentImage = image;
  itsFrameNumber = fNum;
  pthread_mutex_unlock(&itsCurrentImageLock);

  // compute and show framerate over the last NAVG frames:
  itsFrameDuration[fNum % NUM_STORED_FRAMES] = itsTimer->get();
  itsTimer->reset();

  uint nFrames = NUM_STORED_FRAMES;
  if (nFrames < NUM_STORED_FRAMES) nFrames = fNum;
  uint64 avg = 0ULL;
  for(uint i = 0; i < nFrames; i ++) avg += itsFrameDuration[i];
  float frate = 1000000.0F / float(avg) * float(NUM_STORED_FRAMES);

  std::string ntext(sformat("%6d: %6.3f fps -> %8.3f ms/fr",
                            fNum,frate, 1000.0/frate));
  //  writeText(image, Point2D<int>(0,0), ntext.c_str());

  //if((fNum % 30) == 0)
  //  drawImage(image,Point2D<int>(0,0));
  //  LINFO("%s",ntext.c_str());
}

// ######################################################################
void AgentManager::pushCommand(CommandType cmdType,
                                rutz::shared_ptr<Mission> mission)
{
  rutz::shared_ptr<AgentManagerCommand> cmd(new AgentManagerCommand());
  cmd->itsCommandType = cmdType;
  cmd->itsMission = *mission;

  pthread_mutex_lock(&itsCommandsLock);
  itsCommands.push_back(cmd);
  pthread_mutex_unlock(&itsCommandsLock);
}

// ######################################################################
uint AgentManager::getNumCommands()
{
  return itsCommands.size();
}

// ######################################################################
rutz::shared_ptr<AgentManagerCommand> AgentManager::popCommand()
{
  rutz::shared_ptr<AgentManagerCommand> amc = itsCommands.front();
  itsCommands.pop_front();
  return amc;
}

// ######################################################################
void AgentManager::updateAgentsMission(Mission theMission)
{
  uint size = itsSubmarineAgents.size();

  // iterate through all agent manager's agents
  for(uint i = 0; i < size; i++)
    {
      if(itsSubmarineAgents.at(i) != itsCaptainAgent)
        {
          // for each non-Captain agent, update mission
          (itsSubmarineAgents.at(i))->msgUpdateMission(theMission);
        }
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
rutz::shared_ptr<ForwardVisionAgent>
AgentManager::getForwardVisionAgent()
{
  return itsForwardVisionAgent;
}

// ######################################################################
rutz::shared_ptr<CaptainAgent>
AgentManager::getCaptainAgent()
{
  return itsCaptainAgent;
}

// ######################################################################
rutz::shared_ptr<MovementAgent>
AgentManager::getMovementAgent()
{
  return itsMovementAgent;
}

// ######################################################################
// void AgentManager::drawImage(Image<PixRGB<byte> > ima, Point2D<int> point)
// {
//   pthread_mutex_lock(&itsDisplayLock);
//   inplacePaste(itsDisplayImage, ima, point);
//   //  itsWindow->drawImage(itsDisplayImage,0,0);
//   pthread_mutex_unlock(&itsDisplayLock);

// }

// ######################################################################
void AgentManager::initSensorResults()
{
  rutz::shared_ptr<SensorResult> buoy(new SensorResult(SensorResult::BUOY));
  rutz::shared_ptr<SensorResult> pipe(new SensorResult(SensorResult::PIPE));
  rutz::shared_ptr<SensorResult> bin(new SensorResult(SensorResult::BIN));
  rutz::shared_ptr<SensorResult> cross(new SensorResult(SensorResult::CROSS));
  rutz::shared_ptr<SensorResult> pinger(new SensorResult(SensorResult::PINGER));
  rutz::shared_ptr<SensorResult> saliency(new SensorResult(SensorResult::SALIENCY));
  rutz::shared_ptr<SensorResult> stereo(new SensorResult(SensorResult::STEREO));

  itsSensorResults.push_back(buoy);
  itsSensorResults.push_back(pipe);
  itsSensorResults.push_back(bin);
  itsSensorResults.push_back(cross);
  itsSensorResults.push_back(pinger);
  itsSensorResults.push_back(saliency);
  itsSensorResults.push_back(stereo);

  for(itsSensorResultsItr = itsSensorResults.begin();
      itsSensorResultsItr != itsSensorResults.end();
      itsSensorResultsItr++)
    {
      rutz::shared_ptr<SensorResult> r = *itsSensorResultsItr;
      r->startTimer();
    }

}

// ######################################################################
bool AgentManager::updateSensorResult
(rutz::shared_ptr<SensorResult> sensorResult)
{
  bool retVal = false;

  pthread_mutex_lock(&itsSensorResultsLock);
  SensorResult::SensorResultType type = sensorResult->getType();

  for(itsSensorResultsItr = itsSensorResults.begin();
      itsSensorResultsItr != itsSensorResults.end();
      itsSensorResultsItr++)
    {
      rutz::shared_ptr<SensorResult>
        currentResult = *(itsSensorResultsItr);

      if(currentResult->getType() == type)
        {
          currentResult->copySensorResult(*sensorResult);
          retVal = true;
        }
    }

  pthread_mutex_unlock(&itsSensorResultsLock);

  return retVal;
}

// ######################################################################
rutz::shared_ptr<SensorResult> AgentManager::getSensorResult
(SensorResult::SensorResultType type)
{
  pthread_mutex_lock(&itsSensorResultsLock);

  for(itsSensorResultsItr = itsSensorResults.begin();
      itsSensorResultsItr != itsSensorResults.end();
      itsSensorResultsItr++)
    {
      rutz::shared_ptr<SensorResult>
        currentResult = *(itsSensorResultsItr);

      if(currentResult->getType() == type)
        {
          pthread_mutex_unlock(&itsSensorResultsLock);
          return currentResult;
        }
    }

  LINFO("Requested SensorResult type not found");
  rutz::shared_ptr<SensorResult> notFound(new SensorResult());

  pthread_mutex_unlock(&itsSensorResultsLock);
  return notFound;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
