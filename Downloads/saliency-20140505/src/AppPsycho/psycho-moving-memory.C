/*!@file AppPsycho/psycho-moving-memory.C  stimulus presentation program for testing of memory recall with moving objects*/ 

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Psycho/psycho-moving-memory.C$

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

#ifndef __PSYCHO_MV_MEME_TST__
#define __PSYCHO_MV_MEME_TST__
#include <complex>


using namespace std;


std::string getUsageComment()
{

  string com = string ( "\nlist of arguments : \n" );

  com += "\nlogfile=[logfilename.psy] {default = psycho-mv-mem.psy}\n" ;
  com += "\nmemo=[a_string_without_white_space]\n";
  com += "\nsubject=[subject_name] \n" ;
  com += "\nnum-of-trials=[>=1] (number of trials ) {default=5}\n";
  com += "\nmin-num-of-items=[>1](number of items in each trial){default=3}\n" ;
  com += "\nmax-num-of-items=[>min-num-of-items](number of items in each trial){default=6}\n" ;
  com += "\nisi-frames=[>0] (number of frames between presenting two items){default=30} \n" ;
  com += "\nsound-dir=[path to wav files directory]{default=..}\n";
  com += "\ntone1=[a valid file name](name of wav file without extension for tone 1){default=sin.wav}\n";
  com += "\ntone2=[a valid file name](name of wav file without extension for tone 2){default=square.wav}\n";
  com += "\norientation=[0,1 or 2](0 stationary presentation, 1  for horizontal and 2 for vertical)(default=1)\n";
  com += "\ndirection=[1 or -1] {default=1}\n";
  com += "\nspeed=[>0.00001](a float value for speed of movement, this number of pixels by which the item moves){default=1} \n";
  com += "\nshape-size=[>1](size of the shape - diameter for circle){default=20}\n";
  com += "\nclip-size=[>1](size of the black window for presenting the item){default=40}\n";
  com += "\nshape=[circle or square](shape of the items){default=circle}\n";
  com += "\nresponse-time-limit={-1 or >0}(-1 for non-speeded case and a positive number as number of millisecods for speeded case){default=-1} \n" ; 
  return com ;
}

void printFormattedResult(int minNumOfTrials , int maxNumOfTrials , vector<int> trials , vector< vector<int> > cH , vector< vector<int> > tH , vector< vector<int> > rH){
  
  for(int i = minNumOfTrials ; i <= maxNumOfTrials ; i++){
    for(uint j = 0 ; j < trials.size() ; j++){
      string oS = "";
     
      if(trials.at(j)==i) {
	oS = stringify<int>(i) ;
	oS += " " + stringify<int>(j) ;
      	vector<int> c = cH.at(j);
	vector<int> t = tH.at(j);
	vector<int> r = rH.at(j);
	for(uint ii = 0 ; ii < t.size() ; ii++){
	  oS += " " + stringify<int>(t.at(ii)) ;
	}
	for(uint ii = 0 ; ii < r.size() ; ii++){
	  oS += " " + stringify<int>(r.at(ii));
	}
	
	for(uint ii = 0 ; ii < t.size() ; ii++){
	  if(t.at(ii)==r.at(ii)){oS += " "+stringify<int>(1);
	  }else{
	    oS += " "+stringify<int>(0);
	  }
	}
	
	for(int ii = 0 ; ii < maxNumOfTrials ; ii++){
	  if(ii < (int)c.size()) {oS += " "+ stringify<int>(c.at(ii));}else{oS += " NaN";}
	}
	cout<<oS<<endl;
      }
    }
  }

}

extern "C" int main ( const int argc, char** argv )
  {
    MYLOGVERB = LOG_INFO;  // suppress debug messages
    ModelManager manager ( "Psycho-Moving-Memory" );
    nub::soft_ref<PsychoDisplay> d ( new PsychoDisplay ( manager ) );
    //let's push the initial value for the parameters
    map<string,string> argMap ;
    argMap["experiment"]="mv-mem";
    argMap["logfile"]="psycho-mv-mem.psy" ;
    argMap["min-num-of-items"]="3";
    argMap["max-num-of-items"]="6";
    argMap["isi-frames"]="30";
    argMap["subject"]="";
    argMap["memo"]="";
    argMap["num-of-trials"] = "1";
    argMap["speed"] = "1";
    argMap["orientation"] = "1";
    argMap["direction"]="1";
    argMap["shape-size"]="20";
    argMap["clip-size"]="40";
    argMap["shape"]="circle";
    argMap["sound-dir"]="../sounds";
    argMap["tone1"]="sine.wav";
    argMap["tone2"]="square.wav";
    argMap["response-time-limit"]="-1";
    
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
    int min_num_items = atoi(argMap["min-num-of-items"].c_str());
    int max_num_items = atoi(argMap["max-num-of-items"].c_str());
    int orientation = atoi(argMap["orientation"].c_str());
    int direction = atoi(argMap["direction"].c_str());
    float speed = atof(argMap["speed"].c_str());
    int shape_size = atoi(argMap["shape-size"].c_str());
    int clip_size = atoi(argMap["clip-size"].c_str());
    int isi_frames = atoi(argMap["isi-frames"].c_str());
    long response_time_limit = atol(argMap["response-time-limit"].c_str());
   
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
    vector<int> colorPalet ;
    colorPalet.push_back(0);
    colorPalet.push_back(1);
    colorPalet.push_back(3);
    colorPalet.push_back(4);
    colorPalet.push_back(6);
    colorPalet.push_back(8);
    colorPalet.push_back(9);
   // colorPalet.push_back(11);
   // colorPalet.push_back(12);
    colorPalet.push_back(10);
    vector<int> trialSizes;
    for(int i = min_num_items ; i <=max_num_items ; i++)for(int j = 0 ; j < numOfTrials ; j++)trialSizes.push_back(i);
    scramble<int>(trialSizes);
    vector< vector<int> > colorHistory ;
    vector< vector<int> > toBeRememberedHistory ;
    vector< vector<int> > responseHistory ;
    
    while ( cr < (int)trialSizes.size() )
      {
	vector<int> colors;
	vector<int> toBeRememberedColors ;
	int num_of_items = trialSizes.at(cr) ;
	for(int i = 0 ; i < num_of_items ; i++){
	  if(i==0) {
	    colors.push_back(colorPalet.at(rand()%(colorPalet.size()-1) +1));
	  }else{
	    int pc = colors.at(i-1);
	    int tc = pc ;
	    do{
	      tc = colorPalet.at(rand()%(colorPalet.size() -1) +1);
	    }while(tc==pc);
	    colors.push_back(tc);
	  }
	  if(i >= num_of_items - min_num_items) toBeRememberedColors.push_back(colors.at(i));
	}
	colorHistory.push_back(colors);
	toBeRememberedHistory.push_back(toBeRememberedColors);
	
	d->displayText("click to start the next trial!");
	d->waitForMouseClick();
	d->pushEvent("<<<<<<< "+ stringify<int>(cr)+ " >>>>>>>");
	d->clearScreen();
	d->displayFixationBlink(-1,-1,5,1); 
	d->clearScreen();
	displayATrainOfMovingObjects( d ,colors , isi_frames , shape_size, clip_size , speed , argMap["shape"] ,  orientation , direction);
	d->clearScreen();
	d->displayFixationBlink(-1,-1,5,1);
	
	
	string outString="";
	for(int i = 0 ; i < (int)colors.size() ; i++) outString += stringify<int>(colors.at(i))+" " ;
	d->pushEvent("sequence of colors: "+outString);
	outString = "";
	for(int i = 0 ; i < (int)toBeRememberedColors.size() ; i++) outString += stringify<int>(toBeRememberedColors.at(i))+" " ;
	d->pushEvent("sequence of to be remembered colors: "+outString);
	
	
	if (Mix_PlayMusic(tone1,0)==-1) {
              //  return retVector;
            }
        while (Mix_PlayingMusic()==1) {} ;
        vector<int> response ;
	if(response_time_limit == -1){
	  response= drawColorWheelInput( d , colorPalet , 50 ,  100 , min_num_items);
	}else{
	  response= drawColorWheelInputWithClock( d , colorPalet , 50 ,  100 , min_num_items,response_time_limit);
	}
	
	responseHistory.push_back(response);
	d->clearScreen();
	d->pushEvent(">>>>>>> "+ stringify<int>(cr)+ " <<<<<<<<");
	cr++;
      }
    printFormattedResult(min_num_items,max_num_items,trialSizes,colorHistory,toBeRememberedHistory,responseHistory);
    d->displayText("Experiment complete. Thank you!");
    d->waitForMouseClick();
    // stop all our ModelComponents
    manager.stop();


    // all done!
    return 0;
  }
  
  #endif
