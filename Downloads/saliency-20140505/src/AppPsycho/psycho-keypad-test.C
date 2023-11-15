/*!@file AppPsycho/psycho-keypad-test.C simple test for PsychoKeypad */

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
// Primary maintainer for this file: Nader Noori <nnoori@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-keypad-test.C $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Psycho/PsychoKeypad.H"

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Keypad Test");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  manager.setOptionValString(&OPT_EventLogFileName, "psycho-keypad.psy");
  
   if (manager.parseCommandLine(argc, argv,"", 0, -1)==false) return(1);
  d->setEventLog(el);
  manager.start();
  d->clearScreen();
  d->waitForKey();
  d->clearScreen();
  d->displayFixation();
  d->waitForKey();
  d->displayFixationBlink();
  vector<string> tokens1;
  tokens1.push_back("cat");
  tokens1.push_back("rat");
  tokens1.push_back("none");
  //now let's get the response from the subject up to 3 elements from list of tokens separated by space
  vector<string> ans1 = getKeypadResponse (  d,tokens1 , 1 ,1 ," " , "what is your favorit animal?");
  for(uint i = 0 ; i < ans1.size() ; i++)  std::cout<< ans1.at(i)<<endl;
  d->clearScreen();
  usleep(50000);
      
  d->clearScreen();
  d->displayFixation();
  d->waitForKey();
  d->displayFixationBlink();
  vector<string> tokens2;
  tokens2.push_back("chiken");
  tokens2.push_back("cat");
  tokens2.push_back("dog");
  tokens2.push_back("pig");
  tokens2.push_back("horse");
  //now let's get the response from the subject up to 3 elements from list of tokens separated by space
  vector<string> ans2 = getKeypadResponse (  d,tokens2 , (int)tokens2.size(), (int)tokens2.size() ,"," , "sort these animals");
  for(uint i = 0 ; i < ans2.size() ; i++)  std::cout<< ans2.at(i)<<endl;
  d->clearScreen();
  usleep(50000);
      
      
  d->clearScreen();
  d->displayFixation();
  d->waitForKey();
  d->displayFixationBlink();
  vector<string> tokens3;
  tokens3.push_back("yes");
  tokens3.push_back("no");
  //now let's get the response from the subject up to 3 elements from list of tokens separated by space
  vector<string> ans3 = getKeypadResponse (  d,tokens3 , 1,1 ," " , "Do you like this demo?");
  for(uint i = 0 ; i < ans3.size() ; i++)  std::cout<< ans3.at(i)<<endl;
  d->clearScreen();
  usleep(50000);
        
  
  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}
