/*!@file AppPsycho/psycho-vmc-nback-test.C  stimulus presentation program for parity test mental task*/ 

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Psycho/psycho-vmc-nback-test.C$

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "GameBoard/basic-graphics.H"
#include <vector>
#include <string>
#include <iostream>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "Psycho/PsychoKeypad.H"
#include "Image/Point3D.H"
#include "AppPsycho/psycho-vm-capacity.H"

#ifndef __PSYCHO_VM_CAPACITY_HV__
#define __PSYCHO_VM_CAPACITY_HV__


using namespace std;


std::string getUsageComment()
{

  string com = string ( "\nlist of arguments : \n" );

  com += "\nlogfile=[logfilename.psy] {default = psycho-vmc-nbt.psy}\n" ;
  com += "\nmemo=[a_string_without_white_space]\n";
  com += "\nstring-size=[>0](the size of counter string){default=4} \n";
  com += "\nsubject=[subject_name] \n" ;
  com += "\nnum-of-trials=[>=1] (number of trials ) {default=5}\n";
  com += "\nnum-of-signals=[>1](number of signals in each trial){default=5}\n" ;
 // com += "\nwait-time=[>0] (waiting for response in milliseconds right after starting the tone){default=1000000} \n" ;
  com += "\nsound-dir=[path to wav files directory]{default=..}\n";
  com += "\ntone1=[a valid file name](name of wav file without extension for tone 1){default=sin.wav}\n";
  com += "\ntone2=[a valid file name](name of wav file without extension for tone 2){default=square.wav}\n";
  com += "\nmode=[0,1 or 2](0 for just responding to the first tone, 1  one-back and 2 for two-back on two signals)\n";
  return com ;
}

extern "C" int main ( const int argc, char** argv )
  {
    MYLOGVERB = LOG_INFO;  // suppress debug messages
    ModelManager manager ( "Psycho-Spatial-Memory" );
    nub::soft_ref<PsychoDisplay> d ( new PsychoDisplay ( manager ) );
    //let's push the initial value for the parameters
    map<string,string> argMap ;
    argMap["experiment"]="vmc-parity-test";
    argMap["logfile"]="psycho-vmc-pt.psy" ;
    argMap["num-of-signals"]="5";
    argMap["mode"]="0";
    argMap["subject"]="";
    argMap["memo"]="";
    argMap["num-of-trials"] = "5";
    argMap["tone1"] = "sine.wav";
    argMap["tone2"] = "square.wav";
    argMap["sound-dir"] = ".." ;
    
    int totalMissed = 0 ;
    int totalFalseAlarm = 0 ;
    int totalAlarmNecessary=0;
    int totalNotAlarmNecessary = 0 ;
    manager.addSubComponent ( d );
    nub::soft_ref<EventLog> el ( new EventLog ( manager ) );
    manager.addSubComponent ( el );
    d->setEventLog ( el );
    // nub::soft_ref<EyeTrackerConfigurator>
    //               etc(new EyeTrackerConfigurator(manager));
    //   manager.addSubComponent(etc);
    
    if ( manager.parseCommandLine ( argc, argv,
                                    "at least one argument needed", 1, -1 ) ==false )
      {
        cout<<getUsageComment() <<endl;
        return ( 1 );
      }

    for ( uint i = 0 ; i < manager.numExtraArgs() ; i++ )
      {
        addArgument ( argMap, manager.getExtraArg ( i ),std::string ( "=" ) ) ;
      }
    //here we initialze some common global variables
    initialize_vmc();
    //we need to set up SDL mixer variables before running the executive memory task by calling this function
    initializeSounds(argMap["sound-dir"],argMap["tone1"],argMap["tone2"]);
    
    //and here are some parameters set through the command line
    int numOfTrials = atoi(argMap["num-of-trials"].c_str());
    int numOfSignals = atoi(argMap["num-of-signals"].c_str());
    //long waitTime = atol(argMap["wait-time"].c_str());
    int mode = atoi(argMap["mode"].c_str());
    vector<long> waitTime(numOfSignals,0l);
    waitTime[0] = 700000l;
    waitTime[1] = 900000l;
    for(int i = 2 ; i < numOfSignals ; i++) waitTime[i] = 1300000l;
    
    manager.setOptionValString ( &OPT_EventLogFileName, argMap["logfile"] );
    //+manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");
    //+nub::soft_ref<EyeTracker> eyet = etc->getET();
    //+d->setEyeTracker(eyet);
    //+eyet->setEventLog(el);


    // let's get all our ModelComponent instances started:
    manager.start();
    for ( map<string,string>::iterator it= argMap.begin(); it!= argMap.end() ; ++it ) d->pushEvent ( "arg:"+ it->first+" value:"+it->second ) ;
    // let's display an ISCAN calibration grid:
    d->clearScreen();
    // d->displayISCANcalib();
    d->waitForMouseClick();
    d->displayText ( "Here the experiment starts! click to start!" );
    d->waitForMouseClick();
    d->clearScreen();
 
    //let's do calibration
    d->displayText ( "CLICK LEFT button to calibrate; RIGHT to skip" );
    int cl = d->waitForMouseClick();
    if ( cl == 1 ) d->displayEyeTrackerCalibration ( 3,5,1 , true );
    d->clearScreen();

    int cr = 0 ;//this is the counter for number of rounds
    d->showCursor ( true );
    d->displayText ( "click one of the  mouse buttons to start!" );
    d->waitForMouseClick() ;
    d->showCursor ( false );
    vector<int> toneSequence(numOfSignals,0);
    vector<bool> responseSequence(numOfSignals,false);
    while ( cr < numOfTrials )
      {
	d->pushEvent("<<<<<<< "+ stringify<int>(cr)+ " >>>>>>>");
	d->clearScreen();
	d->displayFixationBlink(-1,-1,5,1); 
	performNBackTask( d ,mode, numOfSignals, waitTime ,  toneSequence , responseSequence);
	int missed=0;
	int caught = 0 ;
	int falseAlarm=0;
	evaluateThePerformanceOnNBackTask(mode, toneSequence , responseSequence , missed , caught , falseAlarm);
	totalMissed += missed ;
	totalFalseAlarm += falseAlarm ;
	totalAlarmNecessary += missed+caught ;
	totalNotAlarmNecessary += numOfSignals -(missed + caught);
	if((missed+falseAlarm) == 0) {
	  d->displayText("perfect audio task!");
	  d->waitFrames(45);
	  d->clearScreen();
	}else{
	  string feedbackString;
	  if(missed!=0) feedbackString += stringify<int>(missed)+ " Missied   ";
	  
	  if(falseAlarm!=0) {
	    if(feedbackString.size()!=0) feedbackString += " - ";
	    feedbackString += stringify<int>(falseAlarm)+ " False Alarm";
	  }
	  d->displayText(feedbackString);
	  d->waitFrames(90);
	  d->clearScreen();
	}
	d->pushEvent("number of missed events: "+stringify<int>(missed));
	d->pushEvent("number of caught events: "+stringify<int>(caught));
	d->pushEvent("number of false alarm events:" + stringify<int>(falseAlarm));
	
	d->displayFixation();
	d->waitForMouseClick();
	d->pushEvent(">>>>>>> "+ stringify<int>(cr)+ " <<<<<<<<");
	cr++;
      }
      
    d->pushEvent("total number of misses :" + stringify<int>(totalMissed) + " from " + stringify<int>(totalAlarmNecessary));
    d->pushEvent("total number of false alarms :" + stringify<int>(totalFalseAlarm) + " from " + stringify<int>(totalNotAlarmNecessary));
    
    d->clearScreen();
    d->displayText("Experiment complete. Thank you!");
    d->waitForMouseClick();
   
    // stop all our ModelComponents
    manager.stop();


    // all done!
    return 0;
  }
  
  #endif
