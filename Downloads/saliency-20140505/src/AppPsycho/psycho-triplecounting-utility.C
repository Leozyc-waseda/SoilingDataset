/*!@file AppPsycho/psycho-triplecounting-utility.C implementation of functions used for triple counting experiments*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-triplecounting-utility.C $


/*
This file contains implementations of functions commonly used for visual working memory experiments
just to keep our code cleaner and modular

*/

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
#include "GameBoard/ColorDef.H"
#include "Image/DrawOps.H"

#ifndef __PSYCHO_TRIPLE_COUNTING_UTILITY_C_DEFINED_
#define __PSYCHO_TRIPLE_COUNTING_UTILITY_C_DEFINED_

#ifndef M_PI
#define M_PI    3.14159265359
#endif

using namespace std;

map<int , PixRGB <byte> > myColorMap;


void initialize_vmc(){
myColorMap[0] = black;
myColorMap[1] = white;
myColorMap[2] = lime ;
myColorMap[3] = green;
myColorMap[4] = blue ;
myColorMap[5] = navy ;
myColorMap[6] = red ;
myColorMap[7] = maroon ;
myColorMap[8] = yellow ;
myColorMap[9] = violet;
myColorMap[10] = orange ;
myColorMap[11] = pink ;
myColorMap[12] = lightblue ;
}

map<string , SDL_Surface* >  patchMaps;

void initialize_patches(string stumuliDir , vector<int> patchCatch){
  
  for(int i = 0 ; i < (int)patchCatch.size() ; i++){
    for(int j = 0 ; j < 3 ; j++){
        patchMaps[ stringify<int>(patchCatch.at(i)) + "-"+stringify<int>(j) ] = load_image(stumuliDir+"/"+ stringify<int>(patchCatch.at(i)) + "-"+stringify<int>(j)+".png");
    }  
  } 
  
}

long drawPatchInTheBox(nub::soft_ref<PsychoDisplay> d, int classId, int patchId , Point2D<int> boxCenter, int boxsize , int boarderSize,int frames){
  
  long onsetTime;
  SDL_Surface* box = getABlankSurface ( boxsize,boxsize);
  SDL_Surface* graypatch = getABlankSurface(boxsize , boxsize);
  fillRectangle(graypatch , d->getUint32color(d->getGrey()) , 0  , 0  , boxsize , boxsize) ;
 // SDL_Surface* shapepatch = getABlankSurface ( boxsize,boxsize);
 // fillRectangle(shapepatch,d->getUint32color(d->getGrey()),0,0,boxsize,boxsize);
 // fillOval(shapepatch , d->getUint32color(myColorMap[color]) , (boxsize - discSize)/2  , (boxsize - discSize)/2  , discSize  , discSize ,d->getUint32color(d->getGrey())) ;
  SDL_Rect offs ; offs.x = boxCenter.i - boxsize/2; offs.y = boxCenter.j - boxsize/2 ;
  SDL_Rect clip;
      clip.y = 0 ;
      clip.x = 0 ;
      clip.w = boxsize - 2*boarderSize;
      clip.h = boxsize - 2*boarderSize ;
  drawRectangle(box , d->getUint32color(white) , 0  , 0  , boxsize-1 , boxsize-1 , 4);
  apply_surface(boarderSize,boarderSize, *(patchMaps[stringify<int>(classId)+"-"+ stringify<int>(patchId)]),*box,clip);
  d->displaySDLSurfacePatch( box, &offs , NULL , -2 , true ,true ) ;
  onsetTime = d->getTimerValue();
  d->waitFrames(frames);
  apply_surface(boarderSize,boarderSize,*graypatch,*box,clip);
  d->displaySDLSurfacePatch( box , &offs , NULL , -2 , true ,true ) ;
  dumpSurface(box);
  dumpSurface(graypatch);
  //dumpSurface(shapepatch);
  return onsetTime ;
}

void drawPatchWithoutTheBox(nub::soft_ref<PsychoDisplay> d, int classId, int patchId , Point2D<int> boxCenter ){
  SDL_Rect offs ; offs.x = boxCenter.i ; offs.y = boxCenter.j  ;
 
  d->displaySDLSurfacePatch( patchMaps[stringify<int>(classId)+"-"+ stringify<int>(patchId)], &offs , NULL , -2 , true ,true ) ;
}


void drawPatchWithoutTheBoxInFrames(nub::soft_ref<PsychoDisplay> d, int classId, int patchId , Point2D<int> boxCenter,int frames){
  SDL_Rect offs ; offs.x = boxCenter.i ; offs.y = boxCenter.j  ;
 
  d->displaySDLSurfacePatch( patchMaps[stringify<int>(classId)+"-"+ stringify<int>(patchId)], &offs , NULL , -2 , true ,true ) ;
  d->waitFrames(frames);
  d->clearScreen();
}

bool itIsInThere ( int x , vector<int> bag )
{
  for ( uint i=0 ; i < bag.size(); i++ )
    {
      if ( x == bag[i] ) return true ;
    }
  return false ;
}

vector<int> getRandomNonRepeatingNumbers ( uint l,int maxVal )
{
  vector<int> pickedones = vector<int>() ;
  for ( uint i = 0 ; i < l ; i++ )
    {
      int nd;
      do { nd= rand() % maxVal ; }
      while ( itIsInThere ( nd,pickedones ) ) ;
      pickedones.push_back ( nd );
    }
  return pickedones ;
}




void addArgument (std::map<string,string>& argMap, const string st,const string delim )
{
  int i = st.find ( delim ) ;
  if(i>0) argMap[st.substr ( 0,i ) ] = st.substr ( i+1 );
}

void addDynamicArgument (vector<string>& bag , string st,const string delim )
{
  int i = st.find ( delim ) ; 
  if(i>=0) bag.push_back(st.substr(delim.size(),st.size()-delim.size()));
}

void drawBoxes(nub::soft_ref<PsychoDisplay> d,vector<Point2D<int> > boxCenters, int size , int boarderSize){
  
  SDL_Surface* box = getABlankSurface ( size,size);
  SDL_Surface* graypatch = getABlankSurface(size , size);
  fillRectangle(graypatch , d->getUint32color(d->getGrey()) , 0  , 0  , size , size) ;
  SDL_Rect clip;
      clip.y = 0 ;
      clip.x = 0 ;
      clip.w = size - 2*boarderSize;
      clip.h = size - 2*boarderSize ;
  drawRectangle(box , d->getUint32color(white) , 0  , 0  , size-1 , size-1 , 4);
  apply_surface(boarderSize,boarderSize,*graypatch,*box,clip);
  for(uint i = 0 ; i < boxCenters.size() ; i++) {
    SDL_Rect offs ; offs.x = boxCenters.at(i).i - size/2; offs.y = boxCenters.at(i).j - size/2 ;
    d->displaySDLSurfacePatch( box , &offs , NULL , -2 , true ,true ) ;
  }
  dumpSurface(box);
  dumpSurface(graypatch);
}

void drawBoxesWithColoredBorders(nub::soft_ref<PsychoDisplay> d,vector< vector<Point2D<int> > > boxCenters, vector<PixRGB<byte>  > boarderColors , int size , int boarderSize){
  
  for(uint i = 0 ; i < boxCenters.size() ; i++) {
    for(uint j = 0 ; j < boxCenters.at(i).size() ; j++) {
    SDL_Surface* box = getABlankSurface ( size,size);
    SDL_Surface* graypatch = getABlankSurface(size , size);
    fillRectangle(graypatch , d->getUint32color(d->getGrey()) , 0  , 0  , size , size) ;
    SDL_Rect clip;
      clip.y = 0 ;
      clip.x = 0 ;
      clip.w = size - 2*boarderSize;
      clip.h = size - 2*boarderSize ;
    drawRectangle(box , d->getUint32color(boarderColors.at(i)) , 0  , 0  , size-1 , size-1 , 4);
    apply_surface(boarderSize,boarderSize,*graypatch,*box,clip);
    SDL_Rect offs ; offs.x = boxCenters.at(i).at(j).i - size/2; offs.y = boxCenters.at(i).at(j).j - size/2 ;
    d->displaySDLSurfacePatch( box , &offs , NULL , -2 , true ,true ) ;
    dumpSurface(box);
    dumpSurface(graypatch);
   }
  }
  
}



long drawDiscInTheBox(nub::soft_ref<PsychoDisplay> d,  Point2D<int> boxCenter, int boxsize , int boarderSize,int discSize, int color,int frames){
  long onsetTime;
  SDL_Surface* box = getABlankSurface ( boxsize,boxsize);
  SDL_Surface* graypatch = getABlankSurface(boxsize , boxsize);
  fillRectangle(graypatch , d->getUint32color(d->getGrey()) , 0  , 0  , boxsize , boxsize) ;
  SDL_Surface* shapepatch = getABlankSurface ( boxsize,boxsize);
  fillRectangle(shapepatch,d->getUint32color(d->getGrey()),0,0,boxsize,boxsize);
  fillOval(shapepatch , d->getUint32color(myColorMap[color]) , (boxsize - discSize)/2  , (boxsize - discSize)/2  , discSize  , discSize ,d->getUint32color(d->getGrey())) ;
  SDL_Rect offs ; offs.x = boxCenter.i - boxsize/2; offs.y = boxCenter.j - boxsize/2 ;
  SDL_Rect clip;
      clip.y = 0 ;
      clip.x = 0 ;
      clip.w = boxsize - 2*boarderSize;
      clip.h = boxsize - 2*boarderSize ;
  drawRectangle(box , d->getUint32color(white) , 0  , 0  , boxsize-1 , boxsize-1 , 4);
  apply_surface(boarderSize,boarderSize,*shapepatch,*box,clip);
  d->displaySDLSurfacePatch( box , &offs , NULL , -2 , true ,true ) ;
  onsetTime = d->getTimerValue();
  d->waitFrames(frames);
  apply_surface(boarderSize,boarderSize,*graypatch,*box,clip);
  d->displaySDLSurfacePatch( box , &offs , NULL , -2 , true ,true ) ;
  dumpSurface(box);
  dumpSurface(graypatch);
  dumpSurface(shapepatch);
  return onsetTime ;
}




long drawTextInTheBox(nub::soft_ref<PsychoDisplay> d,  Point2D<int> boxCenter, int boxsize , int boarderSize,string theText,int frames){
  long onsetTime;
  Image<PixRGB<byte> > textIm  ( d->getWidth(),d->getHeight(),ZEROS );
    textIm.clear ( PixRGB<byte> ( 128,128,128 ) );
    writeText ( textIm, Point2D<int> ( 0, 0),theText.c_str(),PixRGB<byte> ( 0,0,0 ),PixRGB<byte> ( 128,128,128 ));
    SDL_Surface *surf = d->makeBlittableSurface ( textIm , true );
    
    SDL_Surface* graypatch1 = getABlankSurface(boxsize - 2*boarderSize , boxsize - 2*boarderSize);
    fillRectangle(graypatch1 , d->getUint32color(d->getGrey()) , 0  , 0  , boxsize - 2*boarderSize , boxsize - 2*boarderSize) ;
    SDL_Surface* graypatch2 = getABlankSurface(boxsize - 2*boarderSize , boxsize - 2*boarderSize);
    fillRectangle(graypatch2 , d->getUint32color(d->getGrey()) , 0  , 0  , boxsize - 2*boarderSize , boxsize - 2*boarderSize) ;
    SDL_Rect clip;
    clip.x = 0 ;
    clip.y = 0 ;
    clip.w = 10 ;
    clip.h = 20 ;
    apply_surface ( boxsize/2 - 5 ,boxsize/2 - 10 ,*surf,*graypatch1,clip );
    
    //SDL_Rect offs1 ; offs1.x = boxCenter.i - 5; offs1.y = boxCenter.j - 10 ;
    SDL_Rect offs ; offs.x = boxCenter.i - (boxsize-boarderSize)/2; offs.y = boxCenter.j - (boxsize-boarderSize)/2 ;
    d->displaySDLSurfacePatch(graypatch1, &offs , NULL , -2 , true ,true );
    onsetTime = d->getTimerValue();
    if(frames>=0){
      d->waitFrames(frames);
      d->displaySDLSurfacePatch(graypatch2, &offs , NULL , -2 , true ,true );
    }
    
    dumpSurface(surf);
    dumpSurface(graypatch1);
    dumpSurface(graypatch2);
    return onsetTime ;
}

void drawTextOnTheDisc(nub::soft_ref<PsychoDisplay> d, string theText,int discSize ,  int color , int frames){
  Image<PixRGB<byte> > textIm  ( d->getWidth(),d->getHeight(),ZEROS );
    textIm.clear ( myColorMap[color] );
    writeText ( textIm, Point2D<int> ( 0, 0),theText.c_str(),PixRGB<byte> ( 0,0,0 ),myColorMap[color]);
    SDL_Surface *surf = d->makeBlittableSurface ( textIm , true );
    SDL_Surface* shapepatch = getABlankSurface ( discSize+2,discSize+2);
    fillRectangle(shapepatch,d->getUint32color(d->getGrey()),0,0,discSize+2,discSize+2);
    fillOval(shapepatch , d->getUint32color(myColorMap[color]) , 1  , 1  , discSize  , discSize ,d->getUint32color(d->getGrey())) ;
    SDL_Rect clip;
    clip.x = 0 ;
    clip.y = 0 ;
    clip.w = 10 ;
    clip.h = 20 ;
    apply_surface ( discSize/2 - 5 ,discSize/2 - 10 ,*surf,*shapepatch,clip );
    SDL_Rect offs ; offs.x = d->getWidth()/2 - discSize/2; offs.y = d->getHeight()/2 - discSize/2 ;
    d->displaySDLSurfacePatch(shapepatch, &offs , NULL , -2 , true ,true );
    d->waitFrames(frames);
    
    dumpSurface(surf);
    dumpSurface(shapepatch);
    
}

#endif
//#endif // INVT_HAVE_LIBSDL_IMAGE

