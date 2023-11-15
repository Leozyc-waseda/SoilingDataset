/*!@file AppPsycho/psycho-triplecounting-characters.C  stimulus presentation program for parity test mental task*/ 

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Psycho/psycho-triplecounting-characters.C$

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

#ifndef __PSYCHO_TR_COUNT_CHARS__
#define __PSYCHO_TR_COUNT_CHARS__
#include <map>


using namespace std;


vector<string> fourStrings;
vector<Point2D<int> > fourCenters;
int boxSize ;
int discSize ;
int objectPresentationFrames ;
int counterPresenationFrames ;
int boxDistance ;
int numOfIncrements ; 
int repeatBreakPoint = 2 ;
vector< vector<int> > globalBoxHistory;
vector< vector<int> > globalCharHistory;
vector< vector<long> > globalPressHistory ;
vector<int> globalTrialTypeHistory;
vector< vector<int> > globalTrialObjectOrder;
vector< vector<int> > globalTrialInitialCounter;
vector< vector<int> > globalTrialFinalCounter;
vector< vector<char> > globalTrialSubjectResponse;
string dummyChar;
int etFlag;

void printoutHistory( ){
  
  for(int i = 0 ; i < (int)globalTrialTypeHistory.size() ; i++){
    string outstring = stringify<int>(i)+" "+stringify<int>(globalTrialTypeHistory.at(i))+" ";
    for(int j = 0 ; j < (int)(globalBoxHistory.at(i).size()) ; j++) outstring += stringify<int>(globalBoxHistory.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalCharHistory.at(i).size()) ; j++) outstring += stringify<int>(globalCharHistory.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalPressHistory.at(i).size()) ; j++) outstring += stringify<int>(globalPressHistory.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalTrialObjectOrder.at(i).size()) ; j++) outstring += stringify<int>(globalTrialObjectOrder.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalTrialInitialCounter.at(i).size()) ; j++) outstring += stringify<int>(globalTrialInitialCounter.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalTrialFinalCounter.at(i).size()) ; j++) outstring += stringify<int>(globalTrialFinalCounter.at(i).at(j)) + " " ;outstring+=":";
    for(int j = 0 ; j < (int)(globalTrialSubjectResponse.at(i).size()) ; j++) outstring += stringify<char>(globalTrialSubjectResponse.at(i).at(j)) + " " ;outstring+=":";
    cout << outstring<<endl;
  }
  
}


void performSpatialCueingBlock(nub::soft_ref<PsychoDisplay> d,nub::soft_ref<EyeTracker> et, int numOfTrials ,int numOfTestItems ){
  
  d-> displayText(" BOX counting "+stringify<int>(numOfTrials)+" trials in a row!");
  d-> waitForKey();
  int cr = 0 ; 
  while(cr < numOfTrials){
    vector<int> charsHistory ;
    vector< int > boxHistory ;
    vector<long> pressHistory;
    
    vector<int> myChars = getRandomNonRepeatingNumbers(numOfTestItems,fourStrings.size());
    vector<int> myBoxes =  getRandomNonRepeatingNumbers(numOfTestItems,4);
    vector<int> boxCounter ;
    vector< Point2D<int> > boxCenters; 
    vector<string> chars;
    globalTrialObjectOrder.push_back(myBoxes);
    vector<int> initialCounters ;
    for(int i = 0 ; i < numOfTestItems ; i++ )initialCounters.push_back(rand()%2);
    for(int i = 0 ; i < numOfTestItems ; i++){
      boxCenters.push_back(fourCenters.at(myBoxes.at(i)));
      chars.push_back(fourStrings.at(myChars.at(i)));
      boxCounter.push_back(initialCounters.at(i));
    }
    
    globalTrialInitialCounter.push_back(initialCounters);
    d->clearScreen();
    d->displayFixationBlink();
    d->displayFixation();
    drawBoxes(d,  boxCenters, boxSize , 1);
    d->waitFrames(30);
    for(int i = 0 ; i < numOfTestItems ; i++){
      drawTextInTheBox( d, boxCenters.at(i), boxSize , 1,stringify<int>(initialCounters.at(i)),-1); 
      d->waitForKey();
      drawTextInTheBox( d, boxCenters.at(i), boxSize , 1," ",-1);
    }
    vector<int> charIndexHistory ;
    vector<int> boxIndexHistory ;
    for(int i = 0 ; i < numOfIncrements ; ){
      int boxIndex = 0 ;
      int charIndex = 0 ;
      if(i> 0 ){
        int dummyFlag = dummyChar.size() * (rand()%10);
        if(dummyFlag == 1){
          charIndex = numOfTestItems ;
          int placeShiftFlag = rand()%2;
          if (placeShiftFlag == 0) boxIndex = boxIndexHistory.at(boxIndexHistory.size()-1);
        }else{
          int changeFlag = rand()%4;
          
          if (changeFlag == 0){
            boxIndex = boxIndexHistory.at(boxIndexHistory.size()-1);
            charIndex = charIndexHistory.at(charIndexHistory.size()-1) ;
          }
          if (changeFlag == 1){
             boxIndex = boxIndexHistory.at(boxIndexHistory.size()-1);
             do{
              charIndex = rand()%numOfTestItems ;
            }while(charIndex == charIndexHistory.at(charIndexHistory.size()-1));
          }
          if (changeFlag == 2){
             charIndex = charIndexHistory.at(charIndexHistory.size()-1) ;
             do{
              boxIndex = rand()%numOfTestItems ;
            }while(boxIndex == boxIndexHistory.at(boxIndexHistory.size()-1));
          }
          if(changeFlag == 3){
            do{
              charIndex = rand()%numOfTestItems ;
            }while(charIndex == charIndexHistory.at(charIndexHistory.size()-1));
             do{
              boxIndex = rand()%numOfTestItems ;
            }while(boxIndex == boxIndexHistory.at(boxIndexHistory.size()-1));
          }
        }
        
      }else{
        boxIndex = rand()%numOfTestItems ;
        charIndex = rand()%numOfTestItems ;
      }    
      
      charIndexHistory.push_back(charIndex);
      boxIndexHistory.push_back(boxIndex);
      boxHistory.push_back(myBoxes.at(boxIndex));
      long onsetTime = 0;
      if(charIndex==numOfTestItems){
      charsHistory.push_back(-1);
      drawTextInTheBox( d, boxCenters.at(boxIndex), boxSize , 1,dummyChar,objectPresentationFrames);
      onsetTime = d-> getTimerValue();
       d->pushEvent("Eye-Tracking starts");
      if(etFlag==1) et->track(true);
      d->waitForKey();
      if(etFlag==1) et->track(false);
      d->pushEvent("Eye-Tracking stopped");
      }else{
    charsHistory.push_back(myChars.at(charIndex));
    boxCounter.at(boxIndex) = boxCounter.at(boxIndex)+1 ;    
    drawTextInTheBox( d, boxCenters.at(boxIndex), boxSize , 1,chars.at(charIndex),objectPresentationFrames);
    onsetTime = d-> getTimerValue();
     d->pushEvent("Eye-Tracking starts");
      if(etFlag==1) et->track(true);
      d->waitForKey();
      if(etFlag==1) et->track(false);
      d->pushEvent("Eye-Tracking stopped");
    i++ ;
      }
      pressHistory.push_back(d->getTimerValue()-onsetTime);
    }
    globalTrialTypeHistory.push_back(0);
    globalBoxHistory.push_back(boxHistory);
    globalCharHistory.push_back(charsHistory);
    globalPressHistory.push_back(pressHistory);
    globalTrialFinalCounter.push_back(boxCounter);
    vector<char> subjectResponse;
    for(int i = 0 ; i < numOfTestItems ; i++){
      drawTextInTheBox( d, boxCenters.at(i), boxSize , 1,".",-1); 
      char reportedCounter = -1 ;
      char  c ;
      do{
    c = d->waitForKey();
    if(c>='0' && c<='9') {
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

void performCharcterCueingBlock(nub::soft_ref<PsychoDisplay> d,nub::soft_ref<EyeTracker> et, int numOfTrials , int numOfTestItems){
   d-> displayText("Counting CHARACTERS : "+stringify<int>(numOfTrials)+" trials in a row!");
   d->waitForKey();
  int cr = 0 ; 
  while(cr < numOfTrials){
    vector<int> charsHistory ;
    vector<int> boxHistory ;
    vector<long> pressHistory;
    vector<int> myChars = getRandomNonRepeatingNumbers(numOfTestItems,fourStrings.size());
    vector<int> myBoxes =  getRandomNonRepeatingNumbers(numOfTestItems,4);
    vector< Point2D<int> > boxCenters;
    
    vector<string> chars;
    vector<int> initialCounters ; 
    for(int i = 0 ; i < numOfTestItems ; i++ )initialCounters.push_back(rand()%2);
    vector<int> charCounter ;
    for(int i =0 ; i < numOfTestItems ; i++) {
      boxCenters.push_back(fourCenters.at(myBoxes.at(i)));
      chars.push_back(fourStrings.at(myChars.at(i)));
      charCounter.push_back(initialCounters.at(i));
    }
    globalTrialObjectOrder.push_back(myChars);
    
    globalTrialInitialCounter.push_back(initialCounters);
    d->clearScreen();
    d->displayFixationBlink();
    d->clearScreen();
    for(int i = 0 ; i < numOfTestItems ; i++){
      d-> displayText(chars.at(i)+":"+stringify<int>(initialCounters.at(i)));
      //drawTextOnTheDisc(d, stringify<int>(initialCounters.at(i)),discSize ,  colors.at(i) , 0);
      d->waitForKey();
    }
    d->clearScreen();
    d->displayFixation();
    drawBoxes(d,  boxCenters, boxSize , 1);
    d->waitFrames(30);
    vector<int> charIndexHistory ;
    vector<int> boxIndexHistory ;
    for(int i = 0 ; i < numOfIncrements ; ){
      int boxIndex = 0 ;
      int charIndex = 0 ;
      if(i> 0 ){
        int dummyFlag = dummyChar.size() * (rand()%10);
        if(dummyFlag == 1){
          charIndex = numOfTestItems ;
          int placeShiftFlag = rand()%2;
          if (placeShiftFlag == 0) boxIndex = boxIndexHistory.at(boxIndexHistory.size()-1);
        }else{
          int changeFlag = rand()%4;
          
          if (changeFlag == 0){
            boxIndex = boxIndexHistory.at(boxIndexHistory.size()-1);
            charIndex = charIndexHistory.at(charIndexHistory.size()-1) ;
          }
          if (changeFlag == 1){
             boxIndex = boxIndexHistory.at(boxIndexHistory.size()-1);
             do{
              charIndex = rand()%numOfTestItems ;
            }while(charIndex == charIndexHistory.at(charIndexHistory.size()-1));
          }
          if (changeFlag == 2){
             charIndex = charIndexHistory.at(charIndexHistory.size()-1) ;
             do{
              boxIndex = rand()%numOfTestItems ;
            }while(boxIndex == boxIndexHistory.at(boxIndexHistory.size()-1));
          }
          if(changeFlag == 3){
            do{
              charIndex = rand()%numOfTestItems ;
            }while(charIndex == charIndexHistory.at(charIndexHistory.size()-1));
             do{
              boxIndex = rand()%numOfTestItems ;
            }while(boxIndex == boxIndexHistory.at(boxIndexHistory.size()-1));
          }
        }
        
      }else{
        boxIndex = rand()%numOfTestItems ;
        charIndex = rand()%numOfTestItems ;
      }    
      charIndexHistory.push_back(charIndex);
      boxIndexHistory.push_back(boxIndex);
      boxHistory.push_back(myBoxes.at(boxIndex));
      long onsetTime = 0;
      if(charIndex==numOfTestItems){
      charsHistory.push_back(-1);
      drawTextInTheBox( d, boxCenters.at(boxIndex), boxSize , 1,dummyChar,objectPresentationFrames);
      onsetTime = d-> getTimerValue();
      d->pushEvent("Eye-Tracking starts");
      if(etFlag==1) et->track(true);
      d->waitForKey();
      if(etFlag==1) et->track(false);
      d->pushEvent("Eye-Tracking stopped");
      }else{
    charsHistory.push_back(myChars.at(charIndex));
    charCounter.at(charIndex) = charCounter.at(charIndex)+1;   
    drawTextInTheBox( d, boxCenters.at(boxIndex), boxSize , 1,chars.at(charIndex),objectPresentationFrames);
    onsetTime = d-> getTimerValue();
    d->pushEvent("Eye-Tracking starts");
    if(etFlag==1) et->track(true);
      d->waitForKey();
      if(etFlag==1) et->track(false);
      d->pushEvent("Eye-Tracking stopped");
    i++ ;
      }
      pressHistory.push_back(d->getTimerValue()-onsetTime);
      
    }
    globalTrialTypeHistory.push_back(1);
    globalBoxHistory.push_back(boxHistory);
    globalCharHistory.push_back(charsHistory);
    globalPressHistory.push_back(pressHistory);
    globalTrialFinalCounter.push_back(charCounter);
    d->clearScreen();
    vector<char> subjectResponse;
    for(int i = 0 ; i < numOfTestItems ; i++){ 
//       drawTextOnTheDisc(d, "*",discSize ,  colors.at(i) , 0);
      d-> displayText(chars.at(i)+":.");
      char reportedCounter = -1 ;
      char c;
      do{
    c = d->waitForKey();
    if(c>='0' && c<='9') {
      reportedCounter = c;
    }else{if(c != ' ')reportedCounter = -1 ;}
    d-> displayText(chars.at(i)+":"+stringify<char>(c));
//  drawTextOnTheDisc(d, stringify<char>(c),discSize ,  colors.at(i) , 0);
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
  com += "\ncharacters=[a string of at least four keybord characters](these characters are used for object cueing){default=?#%&}\n" ;
  com += "\ndummy-character=[a character] (this is the character will be shown as the dummy character, the difault is no dummy character) {default=}\n";
  com += "\nnum-of-counting-items=[1,2,3 or 4](number of counting items) {default=2}\n";
  return com ;
}

extern "C" int main ( const int argc, char** argv )
  {
    MYLOGVERB = LOG_INFO;  // suppress debug messages
    ModelManager manager ( "Psycho-Triple-Counting-Chars" );
    
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
    argMap["characters"]="?#%";
    argMap["dummy-character"]="";
    argMap["num-of-counting-items"] = "2";
    manager.addSubComponent ( d );
     nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
    manager.addSubComponent(etc);
    
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
    manager.setOptionValString(&OPT_EyeTrackerType, "EL");
    //+manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");
    nub::soft_ref<EyeTracker> eyet = etc->getET();
    d->setEyeTracker(eyet);
    eyet->setEventLog(el);


    // let's get all our ModelComponent instances started:
    manager.start();
    string charsString = argMap["characters"];
    dummyChar = argMap["dummy-character"];
    for(uint i = 0 ; i < charsString.size() ; i++ ) fourStrings.push_back(charsString.substr(i,1));
    boxSize = atoi(argMap["box-size"].c_str());
    discSize = atoi(argMap["disc-size"].c_str());
    objectPresentationFrames = atoi(argMap["object-presentation-frames"].c_str());
    counterPresenationFrames = atoi(argMap["counter-presentation-frames"].c_str());
    boxDistance = atoi(argMap["box-distance"].c_str());
    numOfIncrements = atoi(argMap["num-of-increments"].c_str());
    int numOfBlocks = atoi(argMap["num-of-blocks"].c_str());
    int mode = atoi(argMap["mode"].c_str());
    int numOfTrials = atoi(argMap["num-of-trials"].c_str()) ;
    int numOfCountingItems = atoi(argMap["num-of-counting-items"].c_str());
    fourCenters.push_back(Point2D<int>(d->getWidth()/2 - boxDistance , d->getHeight()/2 - boxDistance));
    fourCenters.push_back(Point2D<int>(d->getWidth()/2 - boxDistance , d->getHeight()/2 + boxDistance));
    fourCenters.push_back(Point2D<int>(d->getWidth()/2 + boxDistance , d->getHeight()/2 + boxDistance));
    fourCenters.push_back(Point2D<int>(d->getWidth()/2 + boxDistance , d->getHeight()/2 - boxDistance));
  
    
    for ( map<string,string>::iterator it= argMap.begin(); it!= argMap.end() ; ++it ) d->pushEvent ( "arg:"+ it->first+" value:"+it->second ) ;
    // let's display an ISCAN calibration grid:
    d->clearScreen();
    // d->displayISCANcalib();
    d->waitForKey();
    d->displayText ( "Here the experiment starts! click to start!" );
    d->waitForKey();
    d->clearScreen();
 
    //let's do calibration
    d->displayText ( "press the space bar for calibration and eye tracking" );
    int cl = d->waitForKey();
    if (cl == ' ') etFlag = 1;
    d->clearScreen();
    d->showCursor ( false );
    if(etFlag==1) {
        LINFO("Performing calibration");
        eyet->calibrate(d);
    }
        //clear the screen of any junk
        d->clearScreen();
    
    for(int i = 0 ; i < numOfBlocks ; i++){
      if(mode == 0){
    performSpatialCueingBlock(d,eyet,numOfTrials,numOfCountingItems);
    performCharcterCueingBlock(d,eyet,numOfTrials,numOfCountingItems);
      }else{
    performCharcterCueingBlock(d,eyet,numOfTrials,numOfCountingItems);
    performSpatialCueingBlock(d,eyet,numOfTrials,numOfCountingItems);
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
