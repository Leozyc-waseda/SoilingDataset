/*!@file AppPsycho/psycho-spatial-resolution.C  stimulus presentation program for testing of memory recall with moving objects*/ 

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Psycho/psycho-spatial-resolution.C$

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

#ifndef __PSYCHO_SPC_RESOLUTION__
#define __PSYCHO_SPC_RESOLUTION__
#include <complex>



using namespace std;


std::string getUsageComment()
{
  
  string com = string ( "\nlist of arguments : \n" );

  com += "\nlogfile=[logfilename.psy] {default = psycho-sp-res.psy}\n" ;
  com += "\nmemo=[a_string_without_white_space]\n";
  com += "\nsubject=[subject_name] \n" ;
  com += "\nnum-of-trials=[>=1] (number of trials ) {default=5}\n";
  com += "\nisi-frames=[>0] (number of frames between presenting two items){default=30} \n" ;
  com += "\nsound-dir=[path to wav files directory]{default=..}\n";
  com += "\ntone1=[a valid file name](name of wav file without extension for tone 1){default=sin.wav}\n";
  com += "\ntone2=[a valid file name](name of wav file without extension for tone 2){default=square.wav}\n";
  com += "\nshape=[circle or square](shape of the items){default=circle}\n";
  com += "\nwait-frames={>0}(number of frames for retaining){default=180} \n" ; 
  com += "\nmax-range={>0}(maximum number of pixel from center for dot location){default=180}\n";
  com += "\nmin-range={>0}(minimum number of pixels from center for dot location){default=30}\n";
  com += "\nshow-guideline=[0 or 1](1 for showing the horizontal and the vertical guideline){default=0}\n";
  return com ;
}

void printFormattedResult( vector<int> trials , vector< Point2D<int> > sH ,vector< Point2D<int> > wH, vector< Point2D<int> > rH){
  
  for(uint i = 0 ; i < trials.size() ; i++){
    int o= trials.at(i);
    if(o == 0){
      string oS = stringify<int>(i);
      oS += " "+stringify<int>(o);
      oS += " " + stringify<int>(sH.at(2*i).i)+ " " + stringify<int>(sH.at(2*i).j) ;
      oS += " " + stringify<int>(sH.at(2*i +1).i)+ " " + stringify<int>(sH.at(2*i + 1).j) ;
      oS += " " + stringify<int>(wH.at(i).i)+ " " + stringify<int>(wH.at(i).j) ;
      oS += " " + stringify<int>(rH.at(i).i)+ " " + stringify<int>(rH.at(i).j) ;
      oS += " " + stringify<int>( rH.at(i).i  - wH.at(i).i);
      cout<<oS<<endl;
    }
  }
  
  for(uint i = 0 ; i < trials.size() ; i++){
    int o= trials.at(i);
    if(o == 1){
      string oS = stringify<int>(i);
      oS += " "+stringify<int>(o);
      oS += " " + stringify<int>(sH.at(2*i).i)+ " " + stringify<int>(sH.at(2*i).j) ;
      oS += " " + stringify<int>(sH.at(2*i +1).i)+ " " + stringify<int>(sH.at(2*i + 1).j) ;
      oS += " " + stringify<int>(wH.at(i).i)+ " " + stringify<int>(wH.at(i).j) ;
      oS += " " + stringify<int>(rH.at(i).i)+ " " + stringify<int>(rH.at(i).j) ;
      oS += " " + stringify<int>( rH.at(i).j  - wH.at(i).j);
      cout<<oS<<endl;
    }
  }
}

extern "C" int main ( const int argc, char** argv )
  {
    MYLOGVERB = LOG_INFO;  // suppress debug messages
    ModelManager manager ( "Psycho-Moving-Recall" );
    nub::soft_ref<PsychoDisplay> d ( new PsychoDisplay ( manager ) );
    //let's push the initial value for the parameters
    map<string,string> argMap ;
    argMap["experiment"]="mv-mem";
    argMap["logfile"]="psycho-mv-mem.psy" ;
    argMap["isi-frames"]="30";
    argMap["subject"]="";
    argMap["memo"]="";
    argMap["num-of-trials"] = "1";
    argMap["sound-dir"]="../sounds";
    argMap["tone1"]="sine.wav";
    argMap["tone2"]="square.wav";
    argMap["wait-frames"]="180";
    argMap["max-range"]="180";
    argMap["min-range"]="30";
    argMap["dot-onset-frames"]="60";
    argMap["show-guideline"]="0";
    
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
    int min_range = atoi(argMap["min-range"].c_str());
    int max_range = atoi(argMap["max-range"].c_str());
    int isi_frames = atoi(argMap["isi-frames"].c_str());
    int onsetFrames = atoi(argMap["dot-onset-frames"].c_str());
    int waitFrames = atoi(argMap["wait-frames"].c_str());
    int guidline_flag = atoi(argMap["show-guideline"].c_str());
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
    d->clearScreen();
    d->showCursor ( false );
    vector<int> orientationVector;
    for(int i = 0 ; i < numOfTrials ; i++) {orientationVector.push_back(0);orientationVector.push_back(1);}
    scramble<int>(orientationVector);
    vector<Point2D<int> > stimulusHistory;
    vector<Point2D<int> > responseHistory;
    vector<Point2D<int> > whichOneHistory;
    SDL_EnableKeyRepeat(0, 0);
    d->showCursor(false);
    while ( cr < (int)orientationVector.size() )
      {
	d->showCursor(true);
	d->displayRedDotFixation(d->getWidth()/2,d->getHeight()/2);
	d->waitForMouseClick();
	d->showCursor(false);
	d->waitFrames(30);
	d->pushEvent("<<<<<<< "+ stringify<int>(cr)+ " >>>>>>>");
	//d->waitForMouseClick();
	d->clearScreen();
	int orientation = orientationVector.at(cr);
	if(orientation==0 && guidline_flag==1 ) drawHorizontalGuidline(d);
	if(orientation==1 && guidline_flag==1 ) drawVerticalGuidline(d);
	int p1 = rand()%(max_range - min_range)  + min_range ;
	int p2 = rand()%(max_range - min_range)  + min_range ;
	int whichOneToLeft = rand()%2;
	if(whichOneToLeft==0) {p1 = -p1;}else{ p2=-p2;}
	if(orientation == 0){
	  p1 += d->getWidth()/2;
	  p2 += d->getWidth()/2;
	  stimulusHistory.push_back(Point2D<int>(p1,d->getHeight()/2));
	  stimulusHistory.push_back(Point2D<int>(p2,d->getHeight()/2));
	  d->displayRedDotFixation(p1,d->getHeight()/2);
	  d->waitFrames(onsetFrames);
	  if(guidline_flag==1) {putUpHoizontallyBisectedScreen(d);}else{d->clearScreen();}
	  d->waitFrames(isi_frames);
	  d->displayRedDotFixation(p2,d->getHeight()/2);
	  d->waitFrames(onsetFrames);
	  if(guidline_flag==1) {putUpHoizontallyBisectedScreen(d);}else{d->clearScreen();}
	  d->displayFixation();  
	}else{
	  p1 += d->getHeight()/2;
	  p2 += d-> getHeight()/2;
	  stimulusHistory.push_back(Point2D<int>(d->getWidth()/2,p1));
	  stimulusHistory.push_back(Point2D<int>(d->getWidth()/2,p2));
	  d->displayRedDotFixation(d->getWidth()/2,p1);
	  d->waitFrames(onsetFrames);
	  if(guidline_flag==1) {putUpVerticallyBisectedScreen(d);}else{d->clearScreen();}
	  d->waitFrames(isi_frames);
	  d->displayRedDotFixation(d->getWidth()/2,p2);
	  d->waitFrames(onsetFrames);
	  if(guidline_flag==1) {putUpVerticallyBisectedScreen(d);}else{d->clearScreen();}
	  d->displayFixation();
	}
	
	d->waitFrames(waitFrames);
	int whichOne = rand()%2 ;
	if ( Mix_PlayMusic(tone1,0)==-1) {
        }
        while (Mix_PlayingMusic()==1) {} ;
	d->clearScreen();
	if(orientation==0){
	  if(whichOne==0) {
	    d->displayText(">");
	    whichOneHistory.push_back(Point2D<int>(max(stimulusHistory.at(2*cr).i,stimulusHistory.at(2*cr + 1).i),d->getHeight()/2) );
	  }else{
	    d->displayText("<");
	    whichOneHistory.push_back(Point2D<int>(min(stimulusHistory.at(2*cr).i,stimulusHistory.at(2*cr + 1).i),d->getHeight()/2) );
	  }
	  
	}else{
	  if(whichOne==0) {
	    d->displayText("^");
	    whichOneHistory.push_back(Point2D<int>(d->getWidth()/2, min(stimulusHistory.at(2*cr).j,stimulusHistory.at(2*cr + 1).j)) );
	  }else{
	    d->displayText("v");
	    whichOneHistory.push_back(Point2D<int>(d->getWidth()/2, max(stimulusHistory.at(2*cr).j,stimulusHistory.at(2*cr + 1).j)) );
	  }
	}
	d->waitFrames(onsetFrames/2);
	d->showCursor(true);
	if(orientation == 0){
	  if(guidline_flag==1) {putUpHoizontallyBisectedScreen(d);}else{d->clearScreen();}
	  d->displayFixation();  
	}else{
	  if(guidline_flag==1){putUpVerticallyBisectedScreen(d);}else{d->clearScreen();}
	  d->displayFixation();
	}
	
	
	responseHistory.push_back( waitForMouseClickAndReturnLocation(d));
	d->clearScreen();
	d->pushEvent(">>>>>>> "+ stringify<int>(cr)+ " <<<<<<<<");
	cr++;
	//d->showCursor(false);
      }
      printFormattedResult(orientationVector,stimulusHistory,whichOneHistory,responseHistory);
    d->displayText("Experiment complete. Thank you!");
    d->waitForMouseClick();
   
    // stop all our ModelComponents
    manager.stop();


    // all done!
    return 0;
  }
  
  #endif
