/*!@file AppPsycho/psycho-triplecounting.C  stimulus presentation program for parity test mental task*/ 

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Psycho/psycho-triplecounting.C$

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
#include "AppPsycho/psycho-triplecounting-utility.H"

#ifndef __PSYCHO_TR_COUNT__
#define __PSYCHO_TR_COUNT__
#include <map>


using namespace std;


vector<int> fourColors;
vector<Point2D<int> > fourCenters;
int boxSize ;
int discSize ;
int objectPresentationFrames ;
int counterPresenationFrames ;
int boxDistance ;
int numOfIncrements ; 
int repeatBreakPoint = 2 ;
vector< vector<int> > globalBoxHistory;
vector< vector<int> > globalColorHistory;
vector< vector<long> > globalPressHistory ;
vector<int> globalTrialTypeHistory;
vector< vector<int> > globalTrialObjectOrder;
vector< vector<int> > globalTrialInitialCounter;
vector< vector<int> > globalTrialFinalCounter;
vector< vector<char> > globalTrialSubjectResponse;


void printoutHistory( ){
  
  for(int i = 0 ; i < (int)globalTrialTypeHistory.size() ; i++){
    string outstring = stringify<int>(i)+" "+stringify<int>(globalTrialTypeHistory.at(i))+" ";
    for(int j = 0 ; j < (int)(globalBoxHistory.at(i).size()) ; j++) outstring += stringify<int>(globalBoxHistory.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalColorHistory.at(i).size()) ; j++) outstring += stringify<int>(globalColorHistory.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalPressHistory.at(i).size()) ; j++) outstring += stringify<int>(globalPressHistory.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalTrialObjectOrder.at(i).size()) ; j++) outstring += stringify<int>(globalTrialObjectOrder.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalTrialInitialCounter.at(i).size()) ; j++) outstring += stringify<int>(globalTrialInitialCounter.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalTrialFinalCounter.at(i).size()) ; j++) outstring += stringify<int>(globalTrialFinalCounter.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalTrialSubjectResponse.at(i).size()) ; j++) outstring += stringify<char>(globalTrialSubjectResponse.at(i).at(j)) + " " ;outstring+=":";
    cout << outstring<<endl;
  }
  
}


void performSpatialCueingBlock(nub::soft_ref<PsychoDisplay> d, int numOfTrials  ){
  
  d-> displayText("Counting discs in each BOX "+stringify<int>(numOfTrials)+" trials in a row!");
  d-> waitForKey();
  int cr = 0 ; 
  while(cr < numOfTrials){
    vector<int> colorHistory ;
    vector< int > boxHistory ;
    vector<long> pressHistory;
    
    vector<int> myColors = getRandomNonRepeatingNumbers(3,fourColors.size());
    vector<int> myBoxes =  getRandomNonRepeatingNumbers(3,fourCenters.size());
    vector<int> boxCounter ;
    vector< Point2D<int> > boxCenters; 
    vector<int> colors;
    globalTrialObjectOrder.push_back(myBoxes);
    vector<int> initialCounters ; initialCounters.push_back(rand()%3);initialCounters.push_back(rand()%3);initialCounters.push_back(rand()%3);
    for(int i = 0 ; i < 3 ; i++){
      boxCenters.push_back(fourCenters.at(myBoxes.at(i)));
      colors.push_back(fourColors.at(myColors.at(i)));
      boxCounter.push_back(initialCounters.at(i));
    }
    
    globalTrialInitialCounter.push_back(initialCounters);
    d->clearScreen();
    d->displayFixationBlink();
    d->displayFixation();
    drawBoxes(d,  boxCenters, boxSize , 1);
    d->waitFrames(30);
    for(int i = 0 ; i < 3 ; i++){
      drawTextInTheBox( d, boxCenters.at(i), boxSize , 1,stringify<int>(initialCounters.at(i)),-1); 
      d->waitForKey();
      drawTextInTheBox( d, boxCenters.at(i), boxSize , 1," ",-1);
    }
    vector<int> colorIndexHistory ;
    vector<int> boxIndexHistory ;
    for(int i = 0 ; i < numOfIncrements ; i++){
      int boxIndex = rand()%3 ;
      int colorIndex = rand()%3 ;
      //this block is to make sure that colors and boxes are not repeated more than repeatBreakPoint
      if(i >= repeatBreakPoint){
	int repeatFlag = 1 ;
	do{
	  boxIndex = rand()%3 ;
	  for (int j = 1 ; j <= repeatBreakPoint ; j++){
		  if (boxIndexHistory.at(i-j)!= boxIndex) repeatFlag = 0 ; 
	  }
	}while(repeatFlag == 1);
	
	repeatFlag = 1 ;
	do{
	  colorIndex = rand()%3 ;
	  for (int j = 1 ; j <= repeatBreakPoint ; j++){
		  if (colorIndexHistory.at(i-j)!= colorIndex) repeatFlag = 0 ; 
	  }
	}while(repeatFlag == 1);
      }
      colorIndexHistory.push_back(colorIndex);
      boxIndexHistory.push_back(boxIndex);
      colorHistory.push_back(colors.at(colorIndex));
      boxHistory.push_back(myBoxes.at(boxIndex));
      boxCounter.at(boxIndex) = boxCounter.at(boxIndex)+1 ;    
      long onsetTime = drawDiscInTheBox(d,  boxCenters.at(boxIndex), boxSize , 1 , discSize , colors.at(colorIndex), objectPresentationFrames);
      d->waitForKey();
      pressHistory.push_back(d->getTimerValue()-onsetTime);
    }
    globalTrialTypeHistory.push_back(0);
    globalBoxHistory.push_back(boxHistory);
    globalColorHistory.push_back(colorHistory);
    globalPressHistory.push_back(pressHistory);
    globalTrialFinalCounter.push_back(boxCounter);
    vector<char> subjectResponse;
    for(int i = 0 ; i < 3 ; i++){
      drawTextInTheBox( d, boxCenters.at(i), boxSize , 1,"*",-1); 
      char reportedCounter = -1 ;
      char  c ;
      do{
	c = d->waitForKey();
	if(c>='0' && c<'9') {
	  reportedCounter = c;
	}else{if(c != ' ')reportedCounter = -1 ;}
	drawTextInTheBox( d, boxCenters.at(i), boxSize , 1,stringify<char>(c),-1);
	if(c == ' ' && reportedCounter != -1) break;
      }while(true);
      subjectResponse.push_back(reportedCounter);
    }
    globalTrialSubjectResponse.push_back(subjectResponse);
    cr++;
  }
  
}

void performColorCueingBlock(nub::soft_ref<PsychoDisplay> d, int numOfTrials  ){
   d-> displayText("Counting discs for each COLOR "+stringify<int>(numOfTrials)+" trials in a row!");
   d->waitForKey();
  int cr = 0 ; 
  while(cr < numOfTrials){
    vector<int> colorHistory ;
    vector<int> boxHistory ;
    vector<long> pressHistory;
    vector<int> myColors = getRandomNonRepeatingNumbers(3,fourColors.size());
    vector<int> myBoxes =  getRandomNonRepeatingNumbers(3,fourCenters.size());
    vector< Point2D<int> > boxCenters;
    
    vector<int> colors;
    vector<int> initialCounters ; initialCounters.push_back(rand()%3);initialCounters.push_back(rand()%3);initialCounters.push_back(rand()%3);
    vector<int> colorCounter ;
    for(int i =0 ; i < 3 ; i++) {
      boxCenters.push_back(fourCenters.at(myBoxes.at(i)));
      colors.push_back(fourColors.at(myColors.at(i)));
      colorCounter.push_back(initialCounters.at(i));
    }
    globalTrialObjectOrder.push_back(colors);
    
    globalTrialInitialCounter.push_back(initialCounters);
    d->clearScreen();
    d->displayFixationBlink();
    d->clearScreen();
    for(int i = 0 ; i < 3 ; i++){
      drawTextOnTheDisc(d, stringify<int>(initialCounters.at(i)),discSize ,  colors.at(i) , 0);
      d->waitForKey();
    }
    d->clearScreen();
    d->displayFixation();
    drawBoxes(d,  boxCenters, boxSize , 1);
    d->waitFrames(30);
    vector<int> colorIndexHistory ;
    vector<int> boxIndexHistory ;
    for(int i = 0 ; i < numOfIncrements ; i++){
      int boxIndex = rand()%3 ;
      int colorIndex = rand()%3 ;
      //this block is to make sure that colors and boxes are not repeated more than repeatBreakPoint
      if(i >= repeatBreakPoint){
	int repeatFlag = 1 ;
	do{
	  boxIndex = rand()%3 ;
	  for (int j = 1 ; j <= repeatBreakPoint ; j++){
		  if (boxIndexHistory.at(i-j)!= boxIndex) repeatFlag = 0 ; 
	  }
	}while(repeatFlag == 1);
	
	repeatFlag = 1 ;
	do{
	  colorIndex = rand()%3 ;
	  for (int j = 1 ; j <= repeatBreakPoint ; j++){
		  if (colorIndexHistory.at(i-j)!= colorIndex) repeatFlag = 0 ; 
	  }
	}while(repeatFlag == 1);
      }
      colorIndexHistory.push_back(colorIndex);
      boxIndexHistory.push_back(boxIndex);
      colorHistory.push_back(colors.at(colorIndex));
      colorCounter.at(colorIndex) = colorCounter.at(colorIndex)+1;
      boxHistory.push_back(myBoxes.at(boxIndex));
      long onsetTime = drawDiscInTheBox(d,  boxCenters.at(boxIndex), boxSize , 1 , discSize , colors.at(colorIndex), objectPresentationFrames);
      d->waitForKey();
      pressHistory.push_back(d->getTimerValue()-onsetTime);
    }
    globalTrialTypeHistory.push_back(1);
    globalBoxHistory.push_back(boxHistory);
    globalColorHistory.push_back(colorHistory);
    globalPressHistory.push_back(pressHistory);
    globalTrialFinalCounter.push_back(colorCounter);
    d->clearScreen();
    vector<char> subjectResponse;
    for(int i = 0 ; i < 3 ; i++){ 
      drawTextOnTheDisc(d, "*",discSize ,  colors.at(i) , 0);
      char reportedCounter = -1 ;
      char c;
      do{
	c = d->waitForKey();
	if(c>='0' && c<'9') {
	  reportedCounter = c;
	}else{if(c != ' ')reportedCounter = -1 ;}
	drawTextOnTheDisc(d, stringify<char>(c),discSize ,  colors.at(i) , 0);
	if(c == ' ' && reportedCounter != -1) break;
      }while(true);
      subjectResponse.push_back((char)reportedCounter);
    }
    globalTrialSubjectResponse.push_back(subjectResponse);
    
    cr++;
  }
  
}



std::string getUsageComment()
{

  string com = string ( "\nlist of arguments : \n" );

  com += "\nlogfile=[logfilename.psy] {default = psycho-vmc-nbt.psy}\n" ;
  com += "\nmemo=[a_string_without_white_space]\n";
  com += "\nsubject=[subject_name] \n" ;
  com += "\nnum-of-trials=[>=1] (number of trials for each mini blocks of spatial cueing or color cueing) {default=3}\n";
  com += "\nnum-of-blocks=[>=1] (number of blocks, each block includes 2 mini blocks of spatial and color cueing) {default=1}\n";
  com += "\nmode=[0 or 1] (0 for starting with the spatial cueing and 1 for color cueing) {default=0}\n";
  com += "\nbox-size=[>=10] (size of the box) {default=40}\n";
  com += "\ndisc-size=[>=5] (size of color discs){default=30}\n";
  com += "\nobject presentation-frames=[>0] (number of frame to have disks onset){default=60}\n";
  com += "\nobject presentation-frames=[>0] (number of frame to have disks onset){default=60}\n";
  com += "\ncounter-presentation-frames=[-1 or >0](-1 for unlimited time and a number of more than 0 is the number of frames for having the counter onset){default=-1}\n";
  com += "\nbox-distance=[>0](distance between center of boxes in pixel values){defualt=100}\n";
  return com ;
}

extern "C" int main ( const int argc, char** argv )
  {
    MYLOGVERB = LOG_INFO;  // suppress debug messages
    ModelManager manager ( "Psycho-Triple-Counting" );
    nub::soft_ref<PsychoDisplay> d ( new PsychoDisplay ( manager ) );
    //let's push the initial value for the parameters
    map<string,string> argMap ;
    argMap["experiment"]="triplecounting";
    argMap["logfile"]="psycho-tc.psy" ;
    argMap["subject"]="";
    argMap["memo"]="";
    argMap["num-of-blocks"] = "1" ;
    argMap["num-of-increments"] = "9" ;
    argMap["box-size"] = "40";
    argMap["disc-size"]= "30";
    argMap["object-presentation-frames"] = "60" ;
    argMap["counter-presentation-frames"] = "120" ;
    argMap["box-distance"] = "100" ;
    argMap["num-of-increments"] = "9" ;
    argMap["num-of-trials"] = "3" ;
    argMap["mode"]="0";
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
   
    
    manager.setOptionValString ( &OPT_EventLogFileName, argMap["logfile"] );
    //+manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");
    //+nub::soft_ref<EyeTracker> eyet = etc->getET();
    //+d->setEyeTracker(eyet);
    //+eyet->setEventLog(el);


    // let's get all our ModelComponent instances started:
    manager.start();
    
    fourColors.push_back(1);
    fourColors.push_back(3);
    fourColors.push_back(4);
    fourColors.push_back(6);
    fourColors.push_back(8);
    fourColors.push_back(9);
    fourColors.push_back(10);
    boxSize = atoi(argMap["box-size"].c_str());
    discSize = atoi(argMap["disc-size"].c_str());
    objectPresentationFrames = atoi(argMap["object-presentation-frames"].c_str());
    counterPresenationFrames = atoi(argMap["counter-presentation-frames"].c_str());
    boxDistance = atoi(argMap["box-distance"].c_str());
    numOfIncrements = atoi(argMap["num-of-increments"].c_str());
    int numOfBlocks = atoi(argMap["num-of-blocks"].c_str());
    int mode = atoi(argMap["mode"].c_str());
    int numOfTrials = atoi(argMap["num-of-trials"].c_str()) ;
    fourCenters.push_back(Point2D<int>(d->getWidth()/2 - boxDistance , d->getHeight()/2 - boxDistance));
    fourCenters.push_back(Point2D<int>(d->getWidth()/2 - boxDistance , d->getHeight()/2 + boxDistance));
    fourCenters.push_back(Point2D<int>(d->getWidth()/2 + boxDistance , d->getHeight()/2 + boxDistance));
    fourCenters.push_back(Point2D<int>(d->getWidth()/2 + boxDistance , d->getHeight()/2 - boxDistance));
    
//     fourCenters.push_back(Point2D<int>(d->getWidth()/2 - boxDistance , d->getHeight()/2 ));
//     fourCenters.push_back(Point2D<int>(d->getWidth()/2  , d->getHeight()/2 ));
//     fourCenters.push_back(Point2D<int>(d->getWidth()/2 + boxDistance , d->getHeight()/2 ));
  
    for ( map<string,string>::iterator it= argMap.begin(); it!= argMap.end() ; ++it ) d->pushEvent ( "arg:"+ it->first+" value:"+it->second ) ;
    // let's display an ISCAN calibration grid:
    d->clearScreen();
    // d->displayISCANcalib();
    d->waitForKey();
    d->displayText ( "Here the experiment starts! click to start!" );
    d->waitForKey();
    d->clearScreen();
 
    //let's do calibration
    d->displayText ( "CLICK LEFT button to calibrate; RIGHT to skip" );
    int cl = d->waitForKey();
    if ( cl == 1 ) d->displayEyeTrackerCalibration ( 3,5,1 , true );
    d->clearScreen();

    d->showCursor ( true );
    d->displayText ( "click one of the  mouse buttons to start!" );
    d->waitForKey() ;
    d->showCursor ( false );
    
    for(int i = 0 ; i < numOfBlocks ; i++){
      if(mode == 0){
	performSpatialCueingBlock(d,numOfTrials);
	performColorCueingBlock(d,numOfTrials);
      }else{
	performColorCueingBlock(d,numOfTrials);
	performSpatialCueingBlock(d,numOfTrials);
      }
    }
    
    
    d->clearScreen();
    d->displayText("Experiment complete. Thank you!");
    d->waitForKey();
    
    // stop all our ModelComponents
    manager.stop();
    printoutHistory();

    // all done!
    return 0;
  }
  
#endif
