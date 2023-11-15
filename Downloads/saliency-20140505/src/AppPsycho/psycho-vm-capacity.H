/*!@file AppPsycho/psycho-vm-capacity.H  header file for functions commonly used for visual memory capacity experiments*/ 

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Psycho/psycho-vm-capacity.H$

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
#include <SDL/SDL_mixer.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "Psycho/PsychoKeypad.H"
#include "Image/Point3D.H"

#ifndef __PSYCHO_VM_CAPACITY_H_DEFINED_
#define __PSYCHO_VM_CAPACITY_H_DEFINED_
using namespace std;


//////////////////////////////////////////////
// a functionf for stringigying things
//////////////////////////////////////////////
template <class T> std::string stringify ( T i )
{
  ostringstream o ;
  o << i ;
  return o.str();
}

////////////////////////////////////////////////////////
///////this will change the order of elements in a vector to a random order
////////////////////////////////////////////////////////

template <class T> void scramble(std::vector<T>& v){
        vector<T> tv = vector<T>() ;
        while(v.size()>0){
                tv.push_back(v[0]);
                v.erase(v.begin());
        }
        int i = 0 ;
        while(tv.size()>0){
                i = rand()%tv.size() ;
                v.push_back(tv[i]);
                tv.erase(tv.begin()+i);
        }
}

bool itIsInThere ( int x , vector<int> bag );

vector<int> getRandomNonRepeatingNumbers ( uint l,int maxVal );


vector< Point2D<int> > getAHorizontalArrayOfVSCenters(int numOfItems,int dist,Point2D<int> center);

vector< Point2D<int> > getAVerticalArrayOfVSCenters(int numOfItems,int dist,Point2D<int> center);

vector< Point2D<int> > getARandomArrayOfVSCenters(int numOfItems,int minRadius,int maxRadius,int dist,Point2D<int> center);

vector< Point2D<int> > getACircularArrayOfVSCenters(int numOfItems,int vs_radius, int alpha0,Point2D<int> center);

void addArgument (std::map<string,string>& argMap, const string st,const string delim="=" );

void addDynamicArgument (vector<string>& bag , string st,const string delim="s#" );

void displayVMStimulus( nub::soft_ref<PsychoDisplay> d, map<int , PixRGB <byte> > myColorMap, vector<int> colors , vector<Point2D<int> > centers,int size , string shape="square");

void diplayTargetDistractors(nub::soft_ref<PsychoDisplay> d,string target,string distractor ,vector<Point2D<int> > centers,int targetLocation );

vector<int> repalceColors(vector<int> oldColors,int maxPalet, int numOfChange, vector<int>& changedColors);

bool listenForClicks(nub::soft_ref<PsychoDisplay> d , long miliseconds , int butt);

//this will initilize all global variabls we might need 
void initialize_vmc();

//this is a map of colors that are used for color patchs
extern  map<int , PixRGB <byte> > myColorMap;
extern  map<int,vector<vector<float> > > modeProbabilityMap ;
extern Mix_Music* tone1;
extern Mix_Music* tone2;
extern vector<Mix_Music*> audioVector;

void displayMovingObject(nub::soft_ref<PsychoDisplay> d , PixRGB<byte> color , int shapeSize, int clipSize , float speed , string shape , int orientation , int direction);

void displayATrainOfMovingObjects(nub::soft_ref<PsychoDisplay> d ,vector< int > colors , int isiFrames , int shapeSize, int clipSize , float speed , string shapeName , int orientation , int direction);

void displayATrainOfMovingCharacters(nub::soft_ref<PsychoDisplay> d ,vector< string > characters , int isiFrames , int clipSize , float speed  , int orientation , int direction);

void initializeSounds(string soundDir,string tone1 , string tone2);

void initializeAllSounds(string soundDir,string tone1Str , string tone2Str,vector<string> names);

void performParityTestingTask( nub::soft_ref<PsychoDisplay> d , int mode ,int numberOfSignals, vector<long> waitingTime , vector<int>& toneSequence , vector<bool> responseSequence);

void performNBackTask( nub::soft_ref<PsychoDisplay> d , int mode ,int numberOfSignals, vector<long> waitingTime , vector<int>& toneSequence , vector<bool>& responseSequence);

void evaluateThePerformanceOnParityTestTask(int typeOfTheTask,vector<int> toneSequence , vector<bool> responseSequence , int& missed , int& caught , int& falseAlarm);

void evaluateThePerformanceOnNBackTask(int typeOfTheTask,const vector<int>& toneSequence , const vector<bool>& responseSequence , int& missed , int& caught , int& falseAlarm);

void printBlockSummary(nub::soft_ref<PsychoDisplay> d ,vector< vector <int> > trialsVector , vector<int> responseVecotr , vector<string> titles);

void printBlockSummary(vector< vector <int> > trialsVector , vector<int> responseVector , vector<string> titles);

vector<int> drawColorWheelInput(nub::soft_ref<PsychoDisplay> d , vector<int> colorInecis , int minRadius , int maxRadius, int numOfResponds);

vector<int> drawColorWheelInputWithClock(nub::soft_ref<PsychoDisplay> d , vector<int> colorIndecis , int minRadius , int maxRadius, int numOfResponds,long timeInterval);

void drawHorizontalGuidline(nub::soft_ref<PsychoDisplay> d );

void drawVerticalGuidline(nub::soft_ref<PsychoDisplay> d );

void putUpHoizontallyBisectedScreen(nub::soft_ref<PsychoDisplay> d );

void putUpVerticallyBisectedScreen(nub::soft_ref<PsychoDisplay> d ) ;

Point2D<int> waitForMouseClickAndReturnLocation(nub::soft_ref<PsychoDisplay> d ) ;

void displayStringHorizontally(nub::soft_ref<PsychoDisplay> d, vector<string> s , int onsetFrames , int wsd, int isi);

void displayStringVertically(nub::soft_ref<PsychoDisplay> d,vector<string> s , int onsetFrames , int wsd,int isi);

void showMask(nub::soft_ref<PsychoDisplay> d,int frames, string alphabet);

#endif


