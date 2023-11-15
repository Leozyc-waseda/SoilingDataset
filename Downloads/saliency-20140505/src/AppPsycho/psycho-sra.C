/*!@file AppPsycho/psycho-sra.C Psychophysics test to the accuracy in retrieving visuospatial memory */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-sra.C $
// $Id: psycho-spatial-memory.C 13040 2010-03-23 03:59:25Z nnoori $
// paradigm can found at /lab/nnoori/works/experiments/sra/docs/paradigm.txt
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

#ifndef INVT_HAVE_LIBSDL_IMAGE
#include <cstdio>
int main()
{
        fprintf(stderr, "The SDL_image library must be installed to use this program\n");
        return 1;
}

#else



using namespace std;

// ######################################################################

ModelManager manager("Psycho sra");
	nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
 map<string,string> argMap ;

//////////////////////////////////////////////
// a functionf for stringigying things
//////////////////////////////////////////////
template <class T> std::string stringify(T i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}


// ######################################################################


int addArgument(const string st,const string delim="="){
        int i = st.find(delim) ;
        argMap[st.substr(0,i)] = st.substr(i+1);

        return 0 ;
}

std::string getArgumentValue(string arg){
        return argMap[arg] ;
}



std::string getUsageComment(){

        string com = string("\nlist of arguments : \n");

        com += "\nlogfile=[logfilename.psy] {default = psycho-sm-or.psy}\n" ;
        com += "\nmemo=[a_string_without_white_space]\n";
        com += "\nsubject=[subject_name] \n" ;
        com += "\nnum-of-trials=[>1] (number of trials ) {default=10}\n";
        com += "\nmode=[1,2](1 for horizontal and 2 for vertical layout}\n";
        com += "\ndot-onset=[>1](number of frames that the single dot should be presented){default=16}\n";
        com += "\nsound-dir=[path to wav files directory]{default=..}\n";
        com += "\ntone1=[a valid file name](name of wav file without extension for tone 1){default=sin}\n";
        com += "\ntone2=[a valid file name](name of wav file without extension for tone 2){default=square}\n";
        com += "\ndelay=[ delay value in terms of frames befor recall signal]\n";
        com += "\ngaze-park1-x\n";
        com += "\ngaze-park1-y\n";
        com += "\ngaze-park2-x\n";
        com += "\ngaze-park2-y\n";
        return com ;
}

void initialize(){
       argMap["experiment"]="spatial-recall-accuracy(sra)";
        argMap["logfile"]="./psycho-sra.psy" ;
        argMap["num-of-trials"]="10";
        argMap["subject"]="" ;
        argMap["memo"]="" ;
        argMap["delay1"]="60";
        argMap["delay2"]="120";
        argMap["mode"]="1" ;
        argMap["sound-dir"]="..";
        argMap["tone1"]="sine";
        argMap["tone2"]="square";
        argMap["gaze-park-x"]="-1";
        argMap["gaze-park-y"]="-1";
        argMap["dot-onset"]="30";
        argMap["number-of-intervals"]="3";
        argMap["interval-distance"]="120";
        argMap["interval-width"]="120";
        argMap["gaze-park1-x"]="840";
        argMap["gaze-park1-y"]="240";
        argMap["gaze-park2-x"]="1080";
        argMap["gaze-park2-y"]="240";
}

extern "C" int main(const int argc, char** argv)
{
    MYLOGVERB = LOG_INFO;  // suppress debug messages
	initialize();
    
    manager.addSubComponent(d);//
    nub::soft_ref<EventLog> el(new EventLog(manager));//
    manager.addSubComponent(el);//
    d->setEventLog(el);//
    nub::soft_ref<EyeTrackerConfigurator>
                        etc(new EyeTrackerConfigurator(manager));//
    manager.addSubComponent(etc);//
	if (manager.parseCommandLine(argc, argv,"at least one argument needed", 1, -1)==false){
      cout<<getUsageComment()<<endl;
      return(1);
    }//
//and here we see what else is passed through command line
    for(uint i = 0 ; i < manager.numExtraArgs() ; i++){
      addArgument(manager.getExtraArg(i),std::string("=")) ;
    }//
    
    manager.setOptionValString(&OPT_EventLogFileName, argMap["logfile"]);//
    manager.setOptionValString(&OPT_EyeTrackerType, "EL");//
    nub::soft_ref<EyeTracker> eyet = etc->getET();//
    d->setEyeTracker(eyet);//
    eyet->setEventLog(el);//
    manager.start();//
    //here we set default values
    


    

   // let's get all our ModelComponent instances started:
    
    eyet->setBackgroundColor(d);//
    
    //let's write every parameter in the log file
    for(map<string,string>::iterator it= argMap.begin(); it!= argMap.end() ; ++it) d->pushEvent("arg:"+ it->first+" value:"+it->second ) ;//
    
 
    //now let's get the arguments and push them into some variables used for defining the behavior of the program
     //let's see in what mode the user like to run the program
    int mode = atoi(argMap["mode"].c_str());// 1 is horizontal visual presentation 2 for vertical
    int num_of_trials = atoi(argMap["num-of-trials"].c_str());
    int gaze1_x = atoi(argMap["gaze-park1-x"].c_str());
    int gaze1_y = atoi(argMap["gaze-park1-y"].c_str());
    int gaze2_x = atoi(argMap["gaze-park2-x"].c_str());
    int gaze2_y = atoi(argMap["gaze-park2-y"].c_str());
    int delay1 = atoi(argMap["delay1"].c_str());
    int delay2 = atoi(argMap["delay2"].c_str());
    int dot_onset = atoi(argMap["dot-onset"].c_str());
    int interval_distance = atoi(argMap["interval-distance"].c_str());
    int interval_width = atoi(argMap["interval-width"].c_str());
    
   
    //let's load up the audios
    Mix_Music* tone1 = NULL;//tone 1
    Mix_Music* tone2 = NULL;//tone 2
    //now let's open the audio channel
    if( Mix_OpenAudio( 22050, MIX_DEFAULT_FORMAT, 2, 4096 ) == -1 ){
         LINFO( "did not open the mix-audio") ;
                 // return -1 ;
    }
    string tmpstr = argMap["sound-dir"]+"/"+argMap["tone1"]+".wav";
    tone1 = Mix_LoadMUS(tmpstr.c_str());
    tmpstr = argMap["sound-dir"]+"/"+argMap["tone2"]+".wav";
    tone2 = Mix_LoadMUS(tmpstr.c_str());
    
    
    //here we shoot the real experiment
    // let's display an ISCAN calibration grid:
    d->clearScreen();
    d->displayISCANcalib();
    d->waitForMouseClick(); 
    d->clearScreen();
    d->waitForMouseClick();
    d->displayText("Here the experiment starts! click to start!");
    d->waitForMouseClick();
    d->displayText("CLICK for calibration!");
    d->waitForMouseClick();
    eyet->calibrate(d);
    d->clearScreen();
          
    //now let's start trials
    for(int tr = 0 ; tr < num_of_trials ; tr++){
      int intr = rand()%3 -1;
      int gaze_park_x=0;
      int gaze_park_y=0;
      if(intr == -1){
        gaze_park_x = gaze1_x ; gaze_park_y = gaze1_y;
      }
      if(intr == 0){
        int peshk = rand()%2;
        if (peshk == 0){
          gaze_park_x = gaze1_x ; gaze_park_y = gaze1_y;
        }else{
          gaze_park_x = gaze2_x ; gaze_park_y = gaze2_y;
        }
      }
      if(intr == 1){
        gaze_park_x = gaze2_x ; gaze_park_y = gaze2_y;
      }
      
      
      d->displayFixationBlink(gaze_park_x,gaze_park_y,true);
      d->pushEvent("*** trial start : " + stringify(tr) +" ***");
      d->pushEvent("gaze parked at :["+stringify(gaze_park_x)+","+stringify(gaze_park_y)+"]");
      d->waitForMouseClick();
      d->clearScreen();
      
      //these two values keep the location of dot
      int xS=0;
      int yS=0;
      
      //if peresentation_direction is set to 0 let's toss a coin to see for this trial which direction should be tried out
      
      int del = rand()%2;
      int delay=0;
      switch(del){
        case 0 : delay = delay1 ; break ;
        case 1 : delay = delay2 ; break ;
      }
      
      int lft = intr * interval_distance - interval_width/2 + interval_width*intr;
      
      d->pushEvent("delay: "+ stringify(delay) );
      d->pushEvent("interval: "+stringify(intr));
      
     // for(int interval = 0 ; interval < num_of_intervals ; interval++){
       // center[interval] = presentation_direction * ( num_of_intervals/2  - interval)*interval_distance + rand()%interval_width - interval_width/2;
      int thePos = lft+ rand()%interval_width;
        int x = d->getWidth() /2  ; int y = d->getHeight()/2;
        switch(mode){
          case 1 : x += thePos;break;
          case 2 : y += thePos;break;
        }
        xS = x ; yS = y;
        d->pushEvent("presentation "+stringify(intr) + " :  [" +stringify(xS)+","+ stringify(yS)+"]" );
        if( Mix_PlayMusic( tone1, 0 ) == -1 ) { LINFO("tone 1 is not there, what should I play?");return 1;  }
        while( Mix_PlayingMusic() == 1 ){}
        d->displayRedDotFixation(x,y,true);
        d->pushEvent("displayEvent start gazing at : "+stringify(tr)+"." +stringify(intr)+"."+stringify(xS)+"."+stringify(yS));
        eyet->track(true);
        d->waitFrames(dot_onset);
        d->clearScreen();
        eyet->track(false);
         d->pushEvent("displayEvent stop gazing at : "+stringify(tr)+"." +stringify(intr)+"."+stringify(xS)+"."+stringify(yS));
      //}
       if( Mix_PlayMusic( tone2, 0 ) == -1 ) {  LINFO("tone 2 is not there, what should I play?");return 1; }
        while( Mix_PlayingMusic() == 1 ){}
       d->displayFixation(gaze_park_x,gaze_park_y,true);
       d->waitFrames(delay);
       d->clearScreen();
       
     // for(int interval = 0 ; interval < num_of_intervals ; interval++){
       if( Mix_PlayMusic( tone1, 0 ) == -1 ) { LINFO("tone 1 is not there, what should I play?");return 1; }
        while( Mix_PlayingMusic() == 1 ){}
        d->pushEvent("recall "+ stringify(intr) + " :  [" +stringify(xS)+","+stringify(yS)+"]" );
        d->pushEvent("displayEvent start racall gaze at : "+stringify(tr)+"." +stringify(intr)+"."+stringify(xS)+"."+stringify(yS));
        eyet->track(true);
        d->waitFrames(dot_onset);
        d->clearScreen();
        eyet->track(false); 
        d->pushEvent("displayEvent stop recall gaze at : "+stringify(tr)+"." +stringify(intr)+"."+stringify(xS)+"."+stringify(yS));
       //}
       
       if( Mix_PlayMusic( tone2, 0 ) == -1 ) { LINFO("tone 2 is not there, what should I play?");return 1; }
        while( Mix_PlayingMusic() == 1 ){}
       
    }//end of trial
    
    //end of experiment
    d->displayText("That\'s  it!");
    d->waitForMouseClick();
    manager.stop(); 
   return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

