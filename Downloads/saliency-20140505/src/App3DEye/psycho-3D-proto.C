/*!@file AppPsycho/psycho-still.C Psychophysics display of still images */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-still.C $
// $Id: psycho-still.C 10144 2008-08-27 01:16:43Z ilab19 $
//


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
#include "GameBoard/basic-graphics.H"
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include <SDL/SDL_mixer.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <time.h>
#include "Image/DrawOps.H"
#include "GameBoard/resize.h"
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <ctime>
#include "Util/LEDcontroller.H"

//////////////////////////////////////////////
// a function for stringigying things
//////////////////////////////////////////////

using namespace std;
template <class T> std::string stringify(T i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}


// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho 3D");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::ref<LEDcontroller> itsledcon(new LEDcontroller(manager));
  manager.addSubComponent(itsledcon);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  manager.setOptionValString(&OPT_EventLogFileName, "psycho3DET.psy");

  manager.setOptionValString(&OPT_EyeTrackerType, "EL");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               " ", 0, 0)==false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // let's get all our ModelComponent instances started:
  manager.start();

  // let's do an eye tracker calibration:
 // 
  int eFlag = 0 ;

  d->clearScreen();
  d->displayText("<SPACE> for Eye-Link calibration; other key for skipping");
  int c = d->waitForKey();
if (c == ' ') {et->calibrate(d);}

  d->clearScreen();
//  d->displayText("<SPACE> on-display calibration; other key for skipping");
//  c = d->waitForKey();
//  if (c == ' ') {d->displayEyeTrackerCalibration(3,5,1 , true);}

  d->displayText("<SPACE> for off screen eye-trakin; other key for skipping");
  c = d->waitForKey();
  if (c == ' ') {eFlag=1;}
  // setup array of movie indices:
  
  // main loop:
for (int j=0;j<1;j++){
d->pushEvent("round: "+ stringify(j));
d->clearScreen();
if(eFlag==1) d->displayText("get ready for a round of LED calibration ");

for(int i=1;i<5;i++){
	d->clearScreen();
	string s = stringify(i);
	d->displayText(s);
		itsledcon -> setLED(i,true);
		usleep(1000000);
        if(eFlag==1) {
          if(j==0){ 
            d->pushEvent("displayEvent start LEDCALIB" +s);
          }else{
             d->pushEvent("displayEvent start LEDVALID" +s+"_"+stringify(j));
          }
            et->track(true);
         
        }
        d->waitFrames(60);
        if(eFlag==1) {
          if(j==0){ 
            d->pushEvent("displayEvent stop LEDCALIB" +s);
          }else{
             d->pushEvent("displayEvent stop LEDVALID" +s+"_"+stringify(j));
          }
            et->track(false); 
        }
        itsledcon -> setLED(i,false);
		d->waitForKey();
	}
}

	d->clearScreen();
	d->displayText("get ready for first trial");
	d->waitForKey();
	d->clearScreen();
	//d->displayText("Start!");
	d->pushEvent("displayEvent start TRIAL1");
	et->track(true);
	d->waitForKey();
	d->pushEvent("displayEvent stop TRIAL1");
	et->track(false);
	d->clearScreen();
	d->displayText("get ready for second trial");
	d->waitForKey();
	d->clearScreen();
	//d->displayText("Start!");
	d->pushEvent("displayEvent start TRIAL2");
	et->track(true);
	d->waitForKey();
	d->pushEvent("displayEvent stop TRIAL2");
	et->track(false);
	d->clearScreen();
	d->displayText("get ready for third trial");
	d->waitForKey();
	d->clearScreen();
//d->displayText("Start!");
	d->pushEvent("displayEvent start TRIAL3");
	et->track(true);
	d->waitForKey();
	d->pushEvent("displayEvent stop TRIAL3");
	et->track(false);
	d->clearScreen();
	d->displayText("get ready for third trial");
	d->waitForKey();
	d->clearScreen();
//d->displayText("Start!");
	d->pushEvent("displayEvent start TRIAL4");
	et->track(true);
	d->waitForKey();
	d->pushEvent("displayEvent stop TRIAL4");
	et->track(false);

  d->clearScreen();
  d->displayText("Experiment is complete. Thank you!");
  d->waitForKey();
  

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

