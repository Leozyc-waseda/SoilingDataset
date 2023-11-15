/*!@file AppPsycho/psycho-2way-recall.C  stimulus presentation program for testing of memory recall with moving objects*/ 

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Psycho/psycho-2way-recall.C$

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

#ifndef __PSYCHO_MV_MEME_RECALL__
#define __PSYCHO_MV_MEME_RECALL__
#include <complex>



using namespace std;


std::string getUsageComment()
{
  
  string com = string ( "\nlist of arguments : \n" );

  com += "\nlogfile=[logfilename.psy] {default = psycho-2way-recall.psy}\n" ;
  com += "\nmemo=[a_string_without_white_space]\n";
  com += "\nsubject=[subject_name] \n" ;
  com += "\nnum-of-trials=[>=1] (number of trials ) {default=5}\n";
  com += "\nnum-of-items=[>1](number of items in each trial){default=5}\n" ;
  com += "\ncharacters=[a string of characters]{default=%$#!@?}\n" ;
  com += "\nisi-frames=[>0] (number of frames between presenting two items){default=15} \n" ;
  com += "\nsound-dir=[path to wav files directory]{default=..}\n";
  com += "\ntone1=[a valid file name](name of wav file without extension for tone 1){default=sin.wav}\n";
  com += "\ntone2=[a valid file name](name of wav file without extension for tone 2){default=square.wav}\n";
  com += "\norientation=[1 or 2]( 1  for horizontal and 2 for vertical)(default=1)\n";
  com += "\nspeed=[>0.00001](a float value for speed of movement, this number of pixels by which the item moves){default=1} \n";
  com += "\nshape-size=[>1](size of the shape - diameter for circle){default=20}\n";
  com += "\nclip-size=[>1](size of the black window for presenting the item){default=40}\n";
  com += "\nshape=[circle or square](shape of the items){default=circle}\n";
  com += "\nresponse-time-limit={>0}(number of milliseconds waiting for response){default=4000000} \n" ; 
  com += "\nonset-framse=[>0](number of frames for displaying the whole stimulus){default=180}\n";
  com += "\nwhite-space[>0] (white space between characters){default=40}\n";
  com += "\ndelay-frames=[>0] (number of frames for delay after mask){default=180}\n";
  return com ;
}

void printFormattedResult(int noi, vector<int> trials , vector< vector<string> > sH , vector< vector<string> > aH , vector< vector<string> > rH){
  
  for(uint i = 0 ; i < trials.size() ; i++){
    string oS = "" ;
    if(trials.at(i)==0) {
      oS = stringify<int>(i) ;
      oS += " 0 ";
      vector<string> s = sH.at(i);
      vector<string> a = aH.at(i);
      vector<string> r = rH.at(i);
      
      for(uint ii = 0 ; ii < s.size() ; ii++){
	if(ii >= r.size()){
	  oS += " -1 ";
	}else{
	  if(a.at(ii).compare(r.at(ii))== 0) {
	    oS += " 1 " ;
	  }else{
	    oS += " 0 ";
	  }
	}
      } 
	
      for(uint ii = 0 ; ii < s.size() ; ii++){
	  oS += " " + s.at(ii) ;
      }

      for(uint ii = 0 ; ii < a.size() ; ii++){
	  oS += " " + a.at(ii) ;
	}
      for(uint ii = 0 ; ii < r.size() ; ii++){
	  oS += " " + r.at(ii) ;
	}
      cout<<oS<<endl;
    }
  
  }
  
for(uint i = 0 ; i < trials.size() ; i++){
    string oS = "" ;
    if(trials.at(i)==1) {
      oS = stringify<int>(i) ;
      oS += " 1 ";
      vector<string> s = sH.at(i);
      vector<string> a = aH.at(i);
      vector<string> r = rH.at(i);
      
      for(uint ii = 0 ; ii < s.size() ; ii++){
	if(ii < noi - r.size()){
	  oS += " -1 ";
	}else{
	  if(a.at(noi - ii -1).compare(r.at(noi - ii -1))== 0) {oS += " 1 " ;}else{ oS += " 0 ";}
	}
	} 
      for(uint ii = 0 ; ii < s.size() ; ii++){
	  oS += " " + s.at(ii) ;
	}

      for(uint ii = 0 ; ii < a.size() ; ii++){
	  oS += " " + a.at(ii) ;
	}
      for(uint ii = 0 ; ii < r.size() ; ii++){
	  oS += " " + r.at(ii) ;
	}
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
    argMap["num-of-items"]="5";
    argMap["isi-frames"]="15";
    argMap["subject"]="";
    argMap["memo"]="";
    argMap["num-of-trials"] = "1";
    argMap["speed"] = "1";
    argMap["orientation"] = "1";
    argMap["direction"]="1";
    argMap["clip-size"]="40";
    argMap["shape"]="circle";
    argMap["sound-dir"]="../sounds";
    argMap["tone1"]="sine.wav";
    argMap["tone2"]="square.wav";
    argMap["response-time-limit"]="5000000";
    argMap["characters"]="%$#!@?\\/|*";
    argMap["onset-frames"]="180";
    argMap["white-space"]="40";
    argMap["delay-frames"]="180";
    
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
    int num_of_items = atoi(argMap["num-of-items"].c_str());
    int orientation = atoi(argMap["orientation"].c_str());
    //float speed = atof(argMap["speed"].c_str());
    //int clip_size = atoi(argMap["clip-size"].c_str());
    int isi_frames = atoi(argMap["isi-frames"].c_str());
    long response_time_limit = atol(argMap["response-time-limit"].c_str());
    int onSetFrames = atoi(argMap["onset-frames"].c_str());
    int whiteSpace = atoi(argMap["white-space"].c_str());
    int delay = atoi(argMap["delay-frames"].c_str());
   
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
    vector<int> taskVector;
    for(int i = 0 ; i < numOfTrials ; i++) {taskVector.push_back(0);taskVector.push_back(1);}
    scramble<int>(taskVector);
    vector< vector<string> > sequenceHistory ;
    vector< vector<string> > correctAnswerHistory ;
    vector< vector<string> > responseHistory ;
    vector<string> charsVector;
    for(int i = 0 ; i < (int)argMap["characters"].size() ; i++){
      charsVector.push_back(argMap["characters"].substr(i,1));
    }
    
    
    while ( cr < (int)taskVector.size() )
      {
	vector<string> sequence;
	vector<string> correctAnswer ;
	vector<int> seq ;
	int task = taskVector.at(cr);
	do{
	  int zert = rand() % argMap["characters"].size();
	  if(! itIsInThere(zert,seq)) seq.push_back(zert);
	}while((int)seq.size()< num_of_items);
	
	
	for(int i =0 ; i < num_of_items ; i ++){
	  
	  sequence.push_back(charsVector.at(seq.at(i)));
	  if(task==0){
	    correctAnswer.push_back(charsVector.at(seq.at(i)));
	  }else{
	    correctAnswer.push_back(charsVector.at(seq.at(num_of_items - i - 1)));
	  }
	}
	sequenceHistory.push_back(sequence);
	correctAnswerHistory.push_back(correctAnswer);
	
	
	d->displayText("click to start the next trial!");
	d->waitForMouseClick();
	d->pushEvent("<<<<<<< "+ stringify<int>(cr)+ " >>>>>>>");
	string outString="";
	if (task == 0 )d->pushEvent("the task is forward recall");
	if (task == 1 )d->pushEvent("the task is backward recall") ; 
	for(int i = 0 ; i < (int)sequence.size() ; i++) outString += sequence.at(i)+" " ;
	d->pushEvent("sequence of characters: "+outString);
	outString = "";
	for(int i = 0 ; i < (int)correctAnswer.size() ; i++) outString += correctAnswer.at(i)+" " ;
	d->pushEvent("correct sequence of characters: "+outString);
	d->clearScreen();
	d->displayFixationBlink(-1,-1,5,1); 
	d->clearScreen();
	if(orientation == 1) displayStringHorizontally(d , sequence,onSetFrames,whiteSpace,isi_frames);
	if(orientation == 2) displayStringVertically(d , sequence,onSetFrames,whiteSpace,isi_frames);
	d->clearScreen();
	showMask(d,6,argMap["characters"]); 
	d->displayFixation();
	d->waitFrames(delay);
	d->pushEvent("subject will know what is the task");
	if (task==0 && Mix_PlayMusic(tone1,0)==-1) {
              //  return retVector;
        }
        if (task==1 && Mix_PlayMusic(tone2,0)==-1) {
              //  return retVector;
        }
        while (Mix_PlayingMusic()==1) {} ;
        vector<string> response ;
	
	d->waitForMouseClick();
	d->pushEvent("subject clicked for response");
	if(task==0)response =  getSpeededSquaredKeypadResponse ( d,charsVector , (uint)num_of_items , (uint)num_of_items,"", "",response_time_limit);
	if(task==1)response =  getSpeededSquaredKeypadResponse ( d,charsVector , (uint)num_of_items , (uint)num_of_items, "","",response_time_limit);
	responseHistory.push_back(response);
	outString = "";
	for(int i = 0 ; i < (int)response.size() ; i++) outString += response.at(i)+" " ;
	d->pushEvent("response sequence is : "+outString);
	d->clearScreen();
	d->pushEvent(">>>>>>> "+ stringify<int>(cr)+ " <<<<<<<<");
	cr++;
      }
    printFormattedResult(num_of_items, taskVector , sequenceHistory , correctAnswerHistory , responseHistory);
    d->displayText("Experiment complete. Thank you!");
    d->waitForMouseClick();
   
    // stop all our ModelComponents
    manager.stop();


    // all done!
    return 0;
  }
  
  #endif
