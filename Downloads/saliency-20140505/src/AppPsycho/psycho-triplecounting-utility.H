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

#ifndef __PSYCHO_TRIPLE_COUNTING_UTILITY_H_DEFINED_
#define __PSYCHO_TRIPLE_COUNTING_UTILITY_H_DEFINED_
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


void addArgument (std::map<string,string>& argMap, const string st,const string delim="=" );

void addDynamicArgument (vector<string>& bag , string st,const string delim="s#" );


//this will initilize all global variabls we might need 
void initialize_vmc();
void initialize_patches(string stumuliDir , vector<int> patchCatch);

//this is a map of colors that are used for color patchs
extern  map<int , PixRGB <byte> > myColorMap;

void drawPatchWithoutTheBox(nub::soft_ref<PsychoDisplay> d, int classId, int patchId , Point2D<int> boxCenter);
void drawPatchWithoutTheBoxInFrames(nub::soft_ref<PsychoDisplay> d, int classId, int patchId , Point2D<int> boxCenter,int frames);
void drawBoxes(nub::soft_ref<PsychoDisplay> d, vector< Point2D<int> > boxCenters, int size , int boarderSize);
void drawBoxesWithColoredBorders(nub::soft_ref<PsychoDisplay> d,vector< vector<Point2D<int> > > boxCenters, vector<PixRGB<byte>  > boarderColors , int size , int boarderSize);
long drawDiscInTheBox(nub::soft_ref<PsychoDisplay> d,  Point2D<int> boxCenter, int boxsize , int boarderSize,int discSize, int color,int frames);
long drawTextInTheBox(nub::soft_ref<PsychoDisplay> d,  Point2D<int> boxCenter, int boxsize , int boarderSize,string theText, int frames);
void drawTextOnTheDisc(nub::soft_ref<PsychoDisplay> d, string theText,int discSize ,  int color , int frames);
long drawPatchInTheBox(nub::soft_ref<PsychoDisplay> d, int classId, int patchId , Point2D<int> boxCenter, int boxsize , int boarderSize , int frames);
#endif

