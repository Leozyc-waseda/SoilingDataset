/*!@file AppPsycho/psycho-spatial-attention-bias.C  stimulus presentation program for testing of bias in potential visuospatial attention during an executive memory task*/ 

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Psycho/psycho-spatial-attention-bias.C $

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

#ifndef __PSYCHO_SPC_ATT_BIAS__
#define __PSYCHO_SPC_ATT_BIAS__
#include <complex>



using namespace std;


std::string getUsageComment()
{
  
  string com = string ( "\nlist of arguments : \n" );

  com += "\nlogfile=[logfilename.psy] {default = psycho-spa-bias.psy}\n" ;
  com += "\nmemo=[a_string_without_white_space]\n";
  com += "\nsubject=[subject_name] \n" ;
  com += "\nnum-of-trials=[>=1] (number of trials for each combination of conditions) {default=1}\n";
  com += "\nsound-dir=[path to wav files directory]{default=..}\n";
  com += "\ntone1=[a valid file name](name of wav file without extension for tone 1){default=sin.wav}\n";
  com += "\ntone2=[a valid file name](name of wav file without extension for tone 2){default=square.wav}\n";
  com += "\nmin-cue-wait-frames={>0}(min number of frames for cuing the task after playing the last item in the list){default=60} \n" ; 
  com += "\nmax-cue-wait-frames={>0}(max number of frames for cuing the task after playing the last item in the list){default=120} \n" ;
  com += "\nexecution-frames={>0}(number of frames for executing the task){default=300}\n";
  com += "\nmax-x-range={>0}(maximum number of pixels from center for target or distractor presentation){default=180}\n";
  com += "\nmin-x-range={>0}(minimum number of pixels from center for target or distractor presentation){default=30}\n";
  com += "\nmax-y-range={>0}(maximum number of pixels from center for target or distractor presentation){default=20}\n";
  com += "\nmin-y-range={>0}(minimum number of pixels from center for target or distractor presentation){default=-20}\n";
  com += "\nstimulus-onset-frames={>0}(number of frames for visual stimulus onset){default=3}\n" ;
  com += "\nmin-wait-frames={>0}(minimum number of frames for presenting the target/distractor after the cue for the executive task){default=60}\n";
  com += "\nmax-wait-frames={>0}(maximum number of frames for presenting the target/distractor after the cue for the executive task){default=150}\n";
  com += "\nexecutive-memory-size=[>=3](number of audio stimuli for the executive memory task){default=3}\n" ;
  com += "\naudio-isi=[>=0](number of frames between two consecutive audio stimulus presentation){default=30}\n";
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
    argMap["experiment"]="spa-bias";
    argMap["logfile"]="psycho-spa-bias.psy" ;
    argMap["subject"]="";
    argMap["memo"]="";
    argMap["num-of-trials"] = "1";
    argMap["sound-dir"]="../sounds";
    argMap["tone1"]="sine.wav";
    argMap["tone2"]="square.wav";
    argMap["min-cue-wait-frames"]="60";
    argMap["max-cue-wait-frames"]="120";
    argMap["max-x-range"]="180";
    argMap["min-x-range"]="30";
    argMap["max-y-range"]="20";
    argMap["min-y-range"]="-20";
    argMap["stimulus-onset-frames"]="3";
    argMap["min-wait-frames"]="60";
    argMap["max-wait-frames"]="150";
    argMap["execution-frames"]="300";
    argMap["executive-memory-size"]="3";
    argMap["audio-isi"]="30" ;
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
    
    
    //and here are some parameters set through the command line
    int numOfTrials = atoi(argMap["num-of-trials"].c_str());
    int min_x_range = atoi(argMap["min-x-range"].c_str());
    int max_x_range = atoi(argMap["max-x-range"].c_str());
    int min_y_range = atoi(argMap["min-y-range"].c_str());
    int max_y_range = atoi(argMap["max-y-range"].c_str());
    int max_wait_frames = atoi(argMap["max_wait_frames"].c_str());
    int min_wait_frames = atoi(argMap["min_wait_frames"].c_str());
    int max_cue_wait_frames = atoi(argMap["max_cue_wait_frames"].c_str());
    int min_cue_wait_frames = atoi(argMap["min_cue_wait_frames"].c_str());
    int onsetFrames = atoi(argMap["stimulus-onset-frames"].c_str());
    //int waitFrames = atoi(argMap["wait-frames"].c_str());
    int numOfMemItems= atoi(argMap["executive-memory-size"].c_str());
    int audio_isi = atoi(argMap["audio-isi"].c_str());
    vector<string> soundNames;
    soundNames.push_back("a");soundNames.push_back("b");soundNames.push_back("c");soundNames.push_back("d");soundNames.push_back("e");
    soundNames.push_back("f");soundNames.push_back("g");soundNames.push_back("h");soundNames.push_back("i");soundNames.push_back("j");
    soundNames.push_back("k");soundNames.push_back("l");soundNames.push_back("m");soundNames.push_back("n");soundNames.push_back("o");
    soundNames.push_back("p");soundNames.push_back("q");soundNames.push_back("r");soundNames.push_back("s");soundNames.push_back("t");
    soundNames.push_back("u");soundNames.push_back("v");soundNames.push_back("w");soundNames.push_back("x");soundNames.push_back("y");
    initializeAllSounds(argMap["sound-dir"],argMap["tone1"],argMap["tone2"],soundNames);
    
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
    vector<int> task_target_vector;
    for(int i = 0 ; i < numOfTrials ; i++) {
      task_target_vector.push_back(0);
      task_target_vector.push_back(1);
      task_target_vector.push_back(2);
      task_target_vector.push_back(3);
      task_target_vector.push_back(4);
      task_target_vector.push_back(5);
    }
    scramble<int>(task_target_vector);
    vector<vector<int> > executiveMemoryStimulusHistory;
    vector<int> visualTargetHistory;
    vector<Point2D<int> > visualTargetPlaceHistory;
    vector<Point2D<int> > targetResponseHistory;
    
    d->showCursor(false);
    while ( cr < (int)task_target_vector.size() )
      {
	int task_target_identity = task_target_vector.at(cr);
	vector<int> memStim = getRandomNonRepeatingNumbers((uint)numOfMemItems,(int)soundNames.size());
	executiveMemoryStimulusHistory.push_back(memStim);
	int targetFlag = rand()%2; visualTargetHistory.push_back(targetFlag);
	int tX = rand()%(max_x_range - min_x_range) + min_x_range;
	int tY = rand()%(max_y_range - min_y_range) + min_y_range + d->getHeight()/2;
	if(task_target_identity % 2 == 0) {tX = tX + d->getWidth()/2;}else{tX =  d->getWidth()/2 - tX;}
	visualTargetPlaceHistory.push_back(Point2D<int>(tX,tY));
	int cueWaitFrames = rand()%(max_cue_wait_frames - min_cue_wait_frames) + min_cue_wait_frames ;
	int targetWaitFrames = rand()% (max_wait_frames - min_wait_frames) + min_wait_frames ;
	
	d->displayFixation();
	d->waitFrames(30);
	
	d->pushEvent("<<<<<<< "+ stringify<int>(cr)+ " >>>>>>>");
	
	for(uint i = 0 ; i < (uint)numOfMemItems ; i++){
	  if (Mix_PlayMusic(audioVector.at(memStim.at(i)) ,0)==-1){};
	  while (Mix_PlayingMusic()==1) {} ;
	  d->waitFrames(audio_isi);
	}
	cueWaitFrames = 120 ;
	d->waitFrames(cueWaitFrames);
	
	if(task_target_identity==0 || task_target_identity == 1){
	  if ( Mix_PlayMusic(tone1,0)==-1) {
	  }
	  while (Mix_PlayingMusic()==1) {} ;
	}
	if(task_target_identity==2 || task_target_identity == 3){
	  if ( Mix_PlayMusic(tone2,0)==-1) {
	  }
	  while (Mix_PlayingMusic()==1) {} ;
	}
	targetWaitFrames = 120 ;
	d->waitFrames(targetWaitFrames);
	if(targetFlag==0){
	  d->displayText("6",visualTargetPlaceHistory.at(cr),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128));
	}else{
	  d->displayText("9",visualTargetPlaceHistory.at(cr),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128));
	}
	d->waitFrames(onsetFrames);
	d->clearScreen();
	d->displayFixation();
	
	d->pushEvent(">>>>>>> "+ stringify<int>(cr)+ " <<<<<<<<");
	cr++;
	//d->showCursor(false);
      }

    d->displayText("Experiment complete. Thank you!");
    d->waitForMouseClick();
   
    // stop all our ModelComponents
    manager.stop();


    // all done!
    return 0;
  }
  
  #endif
