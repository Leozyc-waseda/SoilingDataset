/*!@file BeoSub/BeeBrain/AgentManagerA.C
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
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Micheal Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/AgentManagerA.C $
// $Id: AgentManagerA.C 9412 2008-03-10 23:10:15Z farhan $
//
#define NUM_STORED_FRAMES    20

#include "BeoSub/BeeBrain/AgentManagerA.H"

// ######################################################################
// function for each of the separately threaded agents
// which calls their schedulers
void* runForwardVisionAgent(void* a)
{
  AgentManagerA* am  = (AgentManagerA *)a;
  rutz::shared_ptr<ForwardVisionAgent> fv = am->getForwardVisionAgent();

  fv->run();

  return NULL;
}

// ######################################################################
void* runPreFrontalCortexAgent(void* a)
{
  AgentManagerA* am  = (AgentManagerA *)a;
  rutz::shared_ptr<PreFrontalCortexAgent> pfc = am->getPreFrontalCortexAgent();

  pfc->run();
  return NULL;
}

// ######################################################################
void* runPreMotorComplex(void* a)
{
  AgentManagerA* am  = (AgentManagerA *)a;
  rutz::shared_ptr<PreMotorComplex> pmc = am->getPreMotorComplex();

  //pmc->start();

  return NULL;
}

// ######################################################################
AgentManagerA::AgentManagerA(OptionManager& mgr,
                             const std::string& descrName,
                             const std::string& tagName)
  :
  ModelComponent(mgr, descrName, tagName)
{

  rutz::shared_ptr<AgentManagerA> ama(this);

   // create the agents
   itsForwardVisionAgent.reset(new ForwardVisionAgent());
   itsPreFrontalCortexAgent.reset(new PreFrontalCortexAgent("preFrontalCortexAgent",ama));
   itsPreMotorComplex.reset(new PreMotorComplex("preMotorComplexAgent"));

   // connect the agents properly
   itsForwardVisionAgent->setPreFrontalCortexAgent(itsPreFrontalCortexAgent);
   itsPreFrontalCortexAgent->setPreMotorComplex(itsPreMotorComplex);
   itsPreFrontalCortexAgent->setForwardVisionAgent(itsForwardVisionAgent);
   itsPreMotorComplex->setPreFrontalCortexAgent(itsPreFrontalCortexAgent);

   // create threads for the agents
   pthread_create
     (&itsForwardVisionAgentThread, NULL, runForwardVisionAgent,
      (void *)this);

   pthread_create
     (&itsPreFrontalCortexAgentThread, NULL, runPreFrontalCortexAgent,
      (void *)this);

   pthread_create
     (&itsPreMotorComplexThread, NULL, runPreMotorComplex,
      (void *)this);

  pthread_mutex_init(&itsDisplayLock, NULL);
  pthread_mutex_init(&itsCurrentImageLock, NULL);
  pthread_mutex_init(&itsCommandsLock, NULL);


  itsInputImageTimer.reset(new Timer(1000000));
  itsFrameDuration.resize(NUM_STORED_FRAMES);
 }

// ######################################################################
AgentManagerA::~AgentManagerA()
{ }

// ######################################################################
void AgentManagerA::startRun()
{
  // call prefrontal cortex to start
  itsPreFrontalCortexAgent->start();
}

// ######################################################################
void AgentManagerA::setCurrentImage
(Image<PixRGB<byte> > image, uint fNum)
{
  // set the current image
  pthread_mutex_lock(&itsCurrentImageLock);
  itsCurrentImage = image;
  itsFrameNumber = fNum;
  pthread_mutex_unlock(&itsCurrentImageLock);

  // compute and show framerate over the last NAVG frames:
  itsFrameDuration[fNum % NUM_STORED_FRAMES] = itsInputImageTimer->get();
  itsInputImageTimer->reset();

  uint nFrames = NUM_STORED_FRAMES;
  if (nFrames < NUM_STORED_FRAMES) nFrames = fNum;
  uint64 avg = 0ULL;
  for(uint i = 0; i < nFrames; i ++) avg += itsFrameDuration[i];
  float frate = 1000000.0F / float(avg) * float(NUM_STORED_FRAMES);

  std::string ntext(sformat("%6d: %6.3f fps -> %8.3f ms/fr",
                            fNum,frate, 1000.0/frate));
  writeText(image, Point2D<int>(0,0), ntext.c_str());

  if((fNum % 30) == 0)
    drawImage(image,Point2D<int>(0,0));
  //  LINFO("%s",ntext.c_str());
}

// ######################################################################
void AgentManagerA::pushCommand(CommandType cmdType,
                DataTypes dataType,
                rutz::shared_ptr<OceanObject> oceanObject)
{
  rutz::shared_ptr<AgentManagerCommand> cmd(new AgentManagerCommand());
  cmd->itsCommandType = cmdType;
  cmd->itsDataType = dataType;
  cmd->itsOceanObjectId = oceanObject->getId();
  cmd->itsOceanObjectType = oceanObject->getType();

  pthread_mutex_lock(&itsCommandsLock);
  itsCommands.push_back(cmd);
  pthread_mutex_unlock(&itsCommandsLock);
}

// ######################################################################
uint AgentManagerA::getNumCommands()
{
  return itsCommands.size();
}

// ######################################################################
rutz::shared_ptr<AgentManagerCommand> AgentManagerA::popCommand()
{
  rutz::shared_ptr<AgentManagerCommand> amc = itsCommands.front();
  itsCommands.pop_front();
  return amc;
}

// ######################################################################
rutz::shared_ptr<ForwardVisionAgent>
AgentManagerA::getForwardVisionAgent()
{
  return itsForwardVisionAgent;
}

// ######################################################################
rutz::shared_ptr<PreFrontalCortexAgent>
AgentManagerA::getPreFrontalCortexAgent()
{
  return itsPreFrontalCortexAgent;
}

// ######################################################################
rutz::shared_ptr<PreMotorComplex>
AgentManagerA::getPreMotorComplex()
{
  return itsPreMotorComplex;
}

// ######################################################################
void AgentManagerA::drawImage(Image<PixRGB<byte> > ima, Point2D<int> point)
{
  pthread_mutex_lock(&itsDisplayLock);
  inplacePaste(itsDisplayImage, ima, point);
  itsWindow->drawImage(itsDisplayImage,0,0);
  pthread_mutex_unlock(&itsDisplayLock);

}

// ######################################################################
bool AgentManagerA::updateOceanObject
(rutz::shared_ptr<OceanObject> oceanObject, DataTypes oceanObjectDataType)
{
  bool retVal = false;
  pthread_mutex_lock(&itsOceanObjectsLock);

  for(uint i = 0; i < itsOceanObjects.size(); i++)
    {
      if(itsOceanObjects[i]->getId() == oceanObject->getId())
        {
          // set the info
          switch(oceanObjectDataType)
            {
              case POSITION:
                {
                  itsOceanObjects[i]->setPosition(oceanObject->getPosition());
                  break;
                }

              case ORIENTATION:
                {
                  itsOceanObjects[i]->setOrientation(oceanObject->getOrientation());
                  break;
                }
              case FREQUENCY:
                {
                  itsOceanObjects[i]->setFrequency(oceanObject->getFrequency());
                  break;
                }
              case DISTANCE:
                {
                  itsOceanObjects[i]->setDistance(oceanObject->getDistance());
                  break;
                }
              case MASS:
                {
                  itsOceanObjects[i]->setMass(oceanObject->getMass());
                  break;
                }
            default: LFATAL("unknown ocean object data-type");
            }
          retVal = true;
          break;
        }
    }

  pthread_mutex_unlock(&itsOceanObjectsLock);

  return retVal;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
