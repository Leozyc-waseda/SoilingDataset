/*!@file AppPsycho/psycho-vmc-hv.C  stimulus presentation program which presents items of visual working memory either horizontally or vertically*/ 

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Psycho/psycho-vmc-hv.C$

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

  com += "\nlogfile=[logfilename.psy] {default = psycho-sm.psy}\n" ;
  com += "\nmemo=[a_string_without_white_space]\n";
  com += "\nstring-size=[>0](the size of counter string){default=4} \n";
  com += "\nsubject=[subject_name] \n" ;
  com += "\nnum-of-trials-for-each-case=[>=1] (number of trials ) {default=1}\n";
  com += "\nmax-num-of-items=[>2] (maximum number of patches in stimulus- should be set to a positive value along with min-num-of-items) {default=5} \n" ;
  com += "\nmin-num-of-items=[>1] (minimum number of patches in stimulus- should be set to a positive value along with max-num-of-items) {default=3}  \n" ;
  com += "\nstimulus-presentation-frames=[>0] (number of frames per item that stimulus remains onset) {default=12}\n" ;
  com += "\ndelay-frames=[>0] (delay period in frames between stimulus presenation and blinking for probe onset){default=120} \n" ;
  com += "\nprobe-presentation-frames=[>0] (number of frames per item that probe remains onset) {default=12} \n" ;
  com += "\npatch-pixel-size=[>2] (size of color patches in pixels)  {default=10}\n" ;
  com += "\npatch-center-pixel-distance=[>4] (distance between center of patches) {default=25}\n" ;
  com += "\nFP-feedback=[0/1] ( :( feedback on false positive response, 1 for providing feedback 0 for not providing) {default=1}\n";
  com += "\nFN-feedback=[0/1] ( :( feedback on false negative response, 1 for providing feedback 0 for not providing) {default=0}\n";
  com += "\nTP-feedback=[0/1] ( :) feedback on true positive response, 1 for providing feedback 0 for not providing) {default=0}\n";
  com += "\nFP-feedback=[0/1] ( :) feedback on true negative response, 1 for providing feedback 0 for not providing) {default=0}\n";
  
  return com ;
}

extern "C" int main ( const int argc, char** argv )
  {
    MYLOGVERB = LOG_INFO;  // suppress debug messages
    ModelManager manager ( "Psycho-Spatial-Memory" );
    nub::soft_ref<PsychoDisplay> d ( new PsychoDisplay ( manager ) );
    map<string,string> argMap ;
    //let's push the initial value for the parameters
    argMap["experiment"]="vm-capacity-test";
    argMap["logfile"]="psycho-vmc.psy" ;
    argMap["string-size"]="3" ;
    argMap["num-of-trials-for-each-case"]="1";
    argMap["subject"]="" ;
    argMap["memo"]="" ;
    argMap["max-num-of-items"] = "-1";
    argMap["min-num-of-items"] = "-1";
    argMap["stimulus-presentation-frames"] = "12";
    argMap["delay-frames"] = "120";
    argMap["probe-presentation-frames"] = "12";
    argMap["patch-pixel-size"] = "10";
    argMap["patch-center-pixel-distance"] = "25";
    argMap["FP-feedback"] = "1";
    argMap["FN-feedback"] = "0";
    argMap["TP-feedback"]="0";
    argMap["TN-feedback"]="0";
    argMap["tone1"] = "sine.wav";
    argMap["tone2"] = "square.wav";
    argMap["sound-dir"] = ".." ;
    
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
    int maxNumOfItems = atoi(argMap["max-num-of-items"].c_str());
    int minNumOfItems = atoi(argMap["min-num-of-items"].c_str());
    int numOfIndividualTrials = atoi(argMap["num-of-trials-for-each-case"].c_str());
    int stimulusPresentationDuration = atoi(argMap["stimulus-presentation-frames"].c_str());
    int probePresentationDuration = atoi(argMap["probe-presentation-frames"].c_str());
    int delay = atoi(argMap["delay-frames"].c_str());
    int patchsize = atoi(argMap["patch-pixel-size"].c_str());
    int distanceBetweenPatches = atoi(argMap["patch-center-pixel-distance"].c_str());
    int FPf = atoi(argMap["FP-feedback"].c_str());
    int FNf = atoi(argMap["FN-feedback"].c_str());
    int TPf = atoi(argMap["TP-feedback"].c_str());
    int TNf = atoi(argMap["TN-feedback"].c_str());
    
    //this rather complicated pice of code is just for handling dynamic parameters 
    //at the command line, in this program we have two of them 1.possible values of 
    //our set size (set by s@ prefix) and 2.possible number of changes in colors (set by c@ prefix)
    vector<string> arraySizesString;
    vector<int> arraySizes;
    if(maxNumOfItems!=-1 && minNumOfItems!=-1){
      for(int i = minNumOfItems; i <= maxNumOfItems ; i++) arraySizes.push_back(i);
    }else{
      for ( uint i = 0 ; i < manager.numExtraArgs() ; i++ )
      {
        addDynamicArgument (arraySizesString, manager.getExtraArg ( i ),std::string ( "s@" ) ) ;
      }
      
      for( uint i = 0 ;  i < arraySizesString.size() ; i ++) arraySizes.push_back(atoi(arraySizesString.at(i).c_str()));
    }
    
    vector<int> changeSizes;
    vector<string> changeSizesString;
    for ( uint i = 0 ; i < manager.numExtraArgs() ; i++ )
      {
        addDynamicArgument (changeSizesString, manager.getExtraArg ( i ),std::string ( "c@" ) ) ;
      }
    if(changeSizesString.size()==0) changeSizesString.push_back("1");  
    for( uint i = 0 ;  i < changeSizesString.size() ; i ++) changeSizes.push_back(atoi(changeSizesString.at(i).c_str()));
    
    //now we translate program variables into data structures for guiding the program
    //trialsVector will include three parameters needed to be determined for
    //each trial: 1.oreientation of array, 2. size of the array and 3.change size (if we decide to change)
    //we will have equal numbers of trials for each possible combination 
    vector< vector<int> > trialsVector;
    for(int o = 0 ; o < 2 ; o++){
      for(int s = 0 ; s < (int)(arraySizes.size()) ; s++){
	for(int cs = 0 ; cs < (int)(changeSizes.size());cs++){
	  for(int c = 0 ; c < 2 ; c++){
	    for(int t = 0 ; t < numOfIndividualTrials ; t++){  
	    //trialsVector.push_back(Point3D<int> (o,arraySizes.at(s),changeSizes.at(c)));
	      vector<int> tv(4,0);
	      tv[0] = o ;
	      tv[1] = arraySizes.at(s);
	      tv[2] = changeSizes.at(cs);
	      tv[3] =  c ;
	      trialsVector.push_back(tv);
	    }
	  }
	}
      }
    }
    scramble(trialsVector);
    
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
    //let's see in what mode the user like to run the program
    //int mode = atoi(argMap["mode"].c_str());
 
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
    vector<string> responseTokens;responseTokens.push_back("Yes");responseTokens.push_back("No");
    vector<int> response;
    vector<int> groundTruth ; 
    while ( cr < (int)trialsVector.size() )
      {
	d->pushEvent("<<<<<<< "+ stringify<int>(cr)+ " >>>>>>>");
	int changeFlag = trialsVector.at(cr).at(3);
	groundTruth.push_back(changeFlag);
	int orientation = trialsVector.at(cr).at(0) ;
	int numOfItems = trialsVector.at(cr).at(1) ;
	int numOfTargetChanges = trialsVector.at(cr).at(2) ;
	vector< Point2D<int> > centers ;
	if(orientation == 0 ) {
	  centers = getAHorizontalArrayOfVSCenters(numOfItems,distanceBetweenPatches,Point2D<int> ((d->getWidth()-patchsize)/2,(d->getHeight()-patchsize)/2 ));
	}else{
	 centers = getAVerticalArrayOfVSCenters(numOfItems,distanceBetweenPatches,Point2D<int> ((d->getWidth()-patchsize)/2,(d->getHeight()-patchsize)/2 ));
	 
	 //centers = getARandomArrayOfVSCenters(numOfItems,100,200,distanceBetweenPatches,Point2D<int> ((d->getWidth()-patchsize)/2,(d->getHeight()-patchsize)/2 ));
	}
	vector<int> colors = getRandomNonRepeatingNumbers(numOfItems,myColorMap.size());
	vector<int> newColors = colors;
	vector<int> changedItems ;
	if(changeFlag == 1) newColors = repalceColors(colors,myColorMap.size(),numOfTargetChanges,changedItems);
	string stimulusColorString = "";
	string probeColorString = "";
	for(uint i = 0 ; i < colors.size() ; i++) {
	  stimulusColorString += stringify<int>(colors.at(i))+",";
	  probeColorString += stringify<int>(newColors.at(i))+",";
	}
	if(orientation == 0 ) d->pushEvent("items presented horizontally");
	if(orientation == 1 ) d->pushEvent("items presented vertically");
	d->pushEvent("original stimulus colors : "+stimulusColorString);
	d->pushEvent("probe stimulus colors : " + probeColorString) ;
	d->pushEvent("number of target change: " + numOfTargetChanges);
	
	d->displayFixationBlink(-1,-1,3,1);
	d->clearScreen();
	displayVMStimulus( d,myColorMap , colors ,  centers, patchsize);
	d->waitFrames(stimulusPresentationDuration*numOfItems);
	d->clearScreen(); 
	d->displayFixation();
	d->waitFrames(delay);
	d->displayFixationBlink(-1,-1,3,1); 
	displayVMStimulus( d, myColorMap, newColors ,  centers, patchsize);
	d->waitFrames(probePresentationDuration*numOfItems);
	d->clearScreen();
	vector<string> ans = getKeypadResponse(d,responseTokens,1,1," ","Changed?");
	if(ans.at(0).compare("Yes")==0) {
	  response.push_back(1);
	}else{
	  response.push_back(0);
	}
	
	//let's provide our subject with some feedback (if we decided to do so)
        if(FPf==1 && changeFlag==0 && response.at(response.size()-1)==1){
	  d->displayText(":(");
	  d->waitFrames(30);
	  d->clearScreen();
	}
	if(FNf==1 && changeFlag==1 && response.at(response.size()-1)==0){
	  d->displayText(":(");
	  d->waitFrames(30);
	  d->clearScreen();
	}
	if(TPf==1 && changeFlag==1 && response.at(response.size()-1)==1){
	  d->displayText(":)");
	  d->waitFrames(30);
	  d->clearScreen();
	}
	if(TNf==1 && changeFlag==0 && response.at(response.size()-1)==0){
	  d->displayText(":)");
	  d->waitFrames(30);
	  d->clearScreen();
	}
	
	d->displayFixation();
	d->waitForMouseClick();
	d->pushEvent(">>>>>>> "+ stringify<int>(cr)+ " <<<<<<<<");
	cr++;
      }
    d->clearScreen();
    d->displayText("Experiment complete. Thank you!");
    d->waitForMouseClick();
    vector<string> titles ; 
    titles.push_back("trial-number:"); titles.push_back("orientation");titles.push_back("array-size");titles.push_back("change-size"); titles.push_back("change");titles.push_back("answer");
    printBlockSummary(d,trialsVector,response,titles);
    printBlockSummary(trialsVector,response,titles);
    // stop all our ModelComponents
    manager.stop();


    // all done!
    return 0;
  }
  
  #endif

