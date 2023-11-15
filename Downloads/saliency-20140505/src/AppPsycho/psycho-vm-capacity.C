/*!@file AppPsycho/psycho-vm-capacity.C implementation of functions used for visual memory capacity experiments*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-vm-capacity.C $


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
#include "AppPsycho/psycho-vm-capacity.H"
#include "GameBoard/ColorDef.H"
#include "Image/DrawOps.H"

#ifndef __PSYCHO_VM_CAPACITY_C_DEFINED_
#define __PSYCHO_VM_CAPACITY_C_DEFINED_

#ifndef M_PI
#define M_PI    3.14159265359
#endif

using namespace std;

map<int , PixRGB <byte> > myColorMap;
map<int, vector<vector<float> > > modeProbabilityMap ;
Mix_Music* tone1;
Mix_Music* tone2;
vector<Mix_Music*> audioVector;
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

vector<float> _4_0(4,0.0f);
_4_0[0]=0.0f;
_4_0[1]=0.5f;
_4_0[2]=0.5f;
_4_0[3]=0.5f;

vector<float> _4_1(4,0.0f);
_4_1[0]=0.674f ;
_4_1[1]=0.674f ;
_4_1[2]=0.674f ;
_4_1[3]=0.674f ;

vector<float> _4_2(4,0.0f);
_4_2[0]=0.5f;
_4_2[1]=0.5f;
_4_2[2]=0.5f;
_4_2[3]=0.5f;

vector< vector<float> > _4signals;
_4signals.push_back(_4_0);
_4signals.push_back(_4_1);
_4signals.push_back(_4_2);

modeProbabilityMap[4] = _4signals;
}

Point2D<int> rand2DVector(int minR,int maxR){
  int x ; 
  int y ;
  int l ;
  int ourMinR = (int)(minR*sqrt(2)/2);
  do{
    x = rand()%(maxR - ourMinR) + ourMinR ;
    y = rand()%(maxR - ourMinR) + ourMinR ;
    x = (2*(rand()%2) -1)*x;
    y = (2*(rand()%2) -1)*y;
    l = (int) sqrt(squareOf<int>(x)+squareOf<int>(y));
  }while(l >maxR || l < minR);
  return Point2D<int>(x,y);
}


bool checkDistances(vector<Point2D<int> >& bag , Point2D<int> v , int d){
  
  bool flag = true;
  for(uint i = 0 ; i < bag.size() ; i++){
      Point2D<int> tv = bag.at(i) ;
      int dis = (int)sqrt(squareOf<int>(tv.i - v.i)+squareOf<int>(tv.j - v.j)) ;
      if(dis < d ) flag = false ;
  }
  
  return flag ;
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



vector< Point2D<int> > getAHorizontalArrayOfVSCenters(int numOfItems,int dist,Point2D<int> center){
  vector< Point2D<int> > array;
  int my = center.j ;
  int mx = center.i - ((numOfItems -1) * dist)/2 ;
  for(int i = 0 ; i < numOfItems ; i++){
    array.push_back(Point2D<int>(mx+ dist*i , my));
  }
  return array;
}


vector< Point2D<int> > getAVerticalArrayOfVSCenters(int numOfItems,int dist,Point2D<int> center){
  vector< Point2D<int> > array;
  int my = center.j - ((numOfItems -1) * dist)/2 ;
  int mx = center.i ;
  for(int i = 0 ; i < numOfItems ; i++){
    array.push_back(Point2D<int>(mx , my+ dist*i));
  }
  return array;
}

vector< Point2D<int> > getARandomArrayOfVSCenters(int numOfItems,int minRadius,int maxRadius,int dist,Point2D<int> center){
  vector< Point2D<int> > tmparray;
  vector< Point2D<int> > array;
  
  while(tmparray.size() < (uint)numOfItems){
      Point2D<int> newVector = rand2DVector(minRadius,maxRadius);
      if(checkDistances(tmparray,newVector,dist)) tmparray.push_back(newVector);
    }
    
  for(uint i = 0 ; i < tmparray.size() ; i++){
    Point2D<int> tP = tmparray.at(i);
    array.push_back(Point2D<int>(tP.i + center.i , tP.j + center.j));
  }
    
    return array ;
}


vector< Point2D<int> > getACircularArrayOfVSCenters(int numOfItems,int vs_radius, int alpha0,Point2D<int> center){
  vector< Point2D<int> > array;
  
  for(int i = 0 ; i < numOfItems ; i++){
    int alpha = alpha0 + i*360/numOfItems;
    int x = (int)(vs_radius*cos(alpha*M_PI/180)) ;
    int y = (int)(vs_radius*sin(alpha*M_PI/180)) ;
    array.push_back(Point2D<int> (center.i+ x, center.j + y));
  }
  
  return array ;
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


void diplayTargetDistractors(nub::soft_ref<PsychoDisplay> d,string target,string distractor,vector<Point2D<int> > centers,int targetLocation ){
  Image<PixRGB<byte> > textIm ( d->getWidth(),d->getHeight(),ZEROS );
  textIm.clear ( d->getGrey() );
  for (uint i = 0 ; i < centers.size() ; i++){
    if((int)i == targetLocation){
      writeText ( textIm, Point2D<int> ( centers.at(i).i ,centers.at(i).j ),target.c_str(),PixRGB<byte> ( 0,0,0 ),d->getGrey() );
    }else{
      writeText ( textIm, Point2D<int> ( centers.at(i).i ,centers.at(i).j ),distractor.c_str(),PixRGB<byte> ( 0,0,0 ),d->getGrey() );
    }
    
  }
  SDL_Surface *surf = d->makeBlittableSurface ( textIm , true );
  SDL_Rect offs ; offs.x = 0 ; offs.y = 0 ;
   d->displaySDLSurfacePatch ( surf , &offs , NULL , -2 , false ,true ) ;
  dumpSurface ( surf ) ;
}

void displayVMStimulus( nub::soft_ref<PsychoDisplay> d,map<int , PixRGB <byte> > myColorMap , vector<int> colors , vector<Point2D<int> > centers,int size,  string shape){
  SDL_Surface* pad= getABlankSurface ( d->getWidth(),d->getHeight());
  fillRectangle(pad , d->getUint32color(d->getGrey()) , 0  , 0  , d->getWidth() , d->getHeight() );
    for(int i = 0 ; i < (int)colors.size(); i++){
      PixRGB<byte> c = myColorMap[colors.at(i)];
      int x = centers.at(i).i ;
      int y = centers.at(i).j ;
      SDL_Surface *patch = getABlankSurface ( size,size);
      if(shape.compare("square") == 0) fillRectangle(patch , d->getUint32color(c) , 0  , 0  , size  , size ) ;
      if(shape.compare("circle") == 0) fillOval(patch , d->getUint32color(c) , 0  , 0  , size  , size ,d->getUint32color(d->getGrey())) ;
      SDL_Rect clip;
      clip.x = 0 ;
      clip.y = 0 ;
      clip.w = size ;
      clip.h = size ;
      apply_surface(x,y,*patch,*pad,clip);
      dumpSurface(patch);
    }
    
    SDL_Rect offs ; offs.x = 0 ; offs.y = 0 ;
    d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , false ,true ) ;
    dumpSurface(pad);
}

void moveRight(nub::soft_ref<PsychoDisplay> d , PixRGB<byte> color , int shapeSize, int clipSize , float speed , string shapeName ){
  SDL_Surface* shape = getABlankSurface ( shapeSize,shapeSize);
      if(shapeName.compare("square") == 0) fillRectangle(shape , d->getUint32color(color) , 0  , 0  , shapeSize  , shapeSize ) ;
      if(shapeName.compare("circle") == 0) fillOval(shape , d->getUint32color(color) , 0  , 0  , shapeSize  , shapeSize ,d->getUint32color(myColorMap[0])) ;
      SDL_Rect clip;
      clip.y = 0 ;
      clip.x = 0 ;
      clip.w = clipSize;
      clip.h = clipSize ;
      
      SDL_Rect offs ; offs.x = (d->getWidth() - clipSize)/2 ; offs.y = (d->getHeight() - clipSize)/2 ;
      float posIndex = -shapeSize;
      
      while(posIndex <= clipSize){
	SDL_Surface* pad= getABlankSurface ( clipSize,clipSize);
	apply_surface((int)posIndex,(clipSize - shapeSize)/2,*shape,*pad,clip);
	d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , true ,true ) ;
	posIndex += speed ;
	dumpSurface(pad);
      }
      SDL_Surface* pad= getABlankSurface ( clipSize,clipSize);
      d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , true ,true ) ;
      dumpSurface(pad);
  
  dumpSurface(shape);
}

void moveLeft(nub::soft_ref<PsychoDisplay> d , PixRGB<byte> color , int shapeSize, int clipSize , float speed , string shapeName ){
  SDL_Surface* shape = getABlankSurface ( shapeSize,shapeSize);
      if(shapeName.compare("square") == 0) fillRectangle(shape , d->getUint32color(color) , 0  , 0  , shapeSize  , shapeSize ) ;
      if(shapeName.compare("circle") == 0) fillOval(shape , d->getUint32color(color) , 0  , 0  , shapeSize  , shapeSize ,d->getUint32color(myColorMap[0])) ;
      SDL_Rect clip;
      clip.y = 0 ;
      clip.x = 0 ;
      clip.w = clipSize;
      clip.h = clipSize ;
      
      SDL_Rect offs ; offs.x = (d->getWidth() - clipSize)/2 ; offs.y = (d->getHeight() - clipSize)/2 ;
      float posIndex = clipSize;
      
      while(posIndex >= -shapeSize){
	SDL_Surface* pad= getABlankSurface ( clipSize,clipSize);
	apply_surface((int)posIndex,(clipSize - shapeSize)/2,*shape,*pad,clip);
	d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , true ,true ) ;
	posIndex -= speed ;
	dumpSurface(pad);
      }
      SDL_Surface* pad= getABlankSurface ( clipSize,clipSize);
      d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , true ,true ) ;
      dumpSurface(pad);
  
  dumpSurface(shape);
}

void moveUp(nub::soft_ref<PsychoDisplay> d , PixRGB<byte> color , int shapeSize, int clipSize , float speed , string shapeName ){
  SDL_Surface* shape = getABlankSurface ( shapeSize,shapeSize);
      if(shapeName.compare("square") == 0) fillRectangle(shape , d->getUint32color(color) , 0  , 0  , shapeSize  , shapeSize ) ;
      if(shapeName.compare("circle") == 0) fillOval(shape , d->getUint32color(color) , 0  , 0  , shapeSize  , shapeSize ,d->getUint32color(myColorMap[0])) ;
      SDL_Rect clip;
      clip.y = 0 ;
      clip.x = 0 ;
      clip.w = clipSize;
      clip.h = clipSize ;
      
      SDL_Rect offs ; offs.x = (d->getWidth() - clipSize)/2 ; offs.y = (d->getHeight() - clipSize)/2 ;
      float posIndex = clipSize;
      
      while(posIndex >= -shapeSize){
	SDL_Surface* pad= getABlankSurface ( clipSize,clipSize);
	apply_surface((clipSize - shapeSize)/2,(int)posIndex,*shape,*pad,clip);
	d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , true ,true ) ;
	posIndex -= speed ;
	dumpSurface(pad);
      }
      SDL_Surface* pad= getABlankSurface ( clipSize,clipSize);
      d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , true ,true ) ;
      dumpSurface(pad);
  dumpSurface(shape);
}

void moveDown(nub::soft_ref<PsychoDisplay> d , PixRGB<byte> color , int shapeSize, int clipSize , float speed , string shapeName ){
  SDL_Surface* shape = getABlankSurface ( shapeSize,shapeSize);
      if(shapeName.compare("square") == 0) fillRectangle(shape , d->getUint32color(color) , 0  , 0  , shapeSize  , shapeSize ) ;
      if(shapeName.compare("circle") == 0) fillOval(shape , d->getUint32color(color) , 0  , 0  , shapeSize  , shapeSize ,d->getUint32color(myColorMap[0])) ;
      SDL_Rect clip;
      clip.y = 0 ;
      clip.x = 0 ;
      clip.w = clipSize;
      clip.h = clipSize ;
      
      SDL_Rect offs ; offs.x = (d->getWidth() - clipSize)/2 ; offs.y = (d->getHeight() - clipSize)/2 ;
      float posIndex = -shapeSize;
      
      while(posIndex <= clipSize){
	SDL_Surface* pad= getABlankSurface ( clipSize,clipSize);
	apply_surface((clipSize - shapeSize)/2,(int)posIndex,*shape,*pad,clip);
	d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , true ,true ) ;
	posIndex += speed ;
	dumpSurface(pad);
      }
  SDL_Surface* pad= getABlankSurface ( clipSize,clipSize);
  d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , true ,true ) ;
  dumpSurface(pad);
  dumpSurface(shape);
}

void showStill(nub::soft_ref<PsychoDisplay> d , PixRGB<byte> color , int shapeSize, int clipSize , float speed , string shapeName ){
  
  SDL_Surface* shape = getABlankSurface ( shapeSize,shapeSize);
      if(shapeName.compare("square") == 0) fillRectangle(shape , d->getUint32color(color) , 0  , 0  , shapeSize  , shapeSize ) ;
      if(shapeName.compare("circle") == 0) fillOval(shape , d->getUint32color(color) , 0  , 0  , shapeSize  , shapeSize ,d->getUint32color(myColorMap[0])) ;
      SDL_Rect clip;
      clip.y = 0 ;
      clip.x = 0 ;
      clip.w = clipSize;
      clip.h = clipSize ;
      
      SDL_Rect offs ; offs.x = (d->getWidth() - clipSize)/2 ; offs.y = (d->getHeight() - clipSize)/2 ;
      float posIndex = -shapeSize;
      SDL_Surface* pad= getABlankSurface ( clipSize,clipSize);
      apply_surface((clipSize - shapeSize)/2,(clipSize - shapeSize)/2,*shape,*pad,clip);
      while(posIndex <= clipSize){
	d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , true ,true ) ;
	posIndex += speed ;
      }
      SDL_Surface* emptypad= getABlankSurface ( clipSize,clipSize);
      d->displaySDLSurfacePatch ( emptypad , &offs , NULL , -2 , true ,true ) ;
  dumpSurface(pad);
  dumpSurface(emptypad);
  dumpSurface(shape);
  
}
//////////////////////////
void moveRightCharacter(nub::soft_ref<PsychoDisplay> d , string character , int clipSize , float speed  ){
    Image<PixRGB<byte> > textIm  ( d->getWidth(),d->getHeight(),ZEROS );
    textIm.clear ( myColorMap[0] );
    writeText ( textIm, Point2D<int> ( 0, 0),character.c_str(),myColorMap[1],myColorMap[0] );
    SDL_Surface *surf = d->makeBlittableSurface ( textIm , true ); 
    SDL_Rect clip;
    clip.x = 0 ;
    clip.y = 0 ;
    clip.w = 10 ;
    clip.h = 20 ;
    
      SDL_Rect offs ; offs.x = (d->getWidth() - clipSize)/2 ; offs.y = (d->getHeight() - clipSize)/2 ;
      float posIndex = -20 ;
      while(posIndex <= clipSize){
	SDL_Surface* blank = getABlankSurface ( clipSize,clipSize);
	 apply_surface ( posIndex,clipSize/2 - 10 ,*surf,*blank,clip );
	d->displaySDLSurfacePatch ( blank , &offs , NULL , -2 , true ,true ) ;
	posIndex += speed ;
	dumpSurface(blank);
      }
  dumpSurface(surf);
}

void moveLeftCharacter(nub::soft_ref<PsychoDisplay> d , string character, int clipSize , float speed  ){
    Image<PixRGB<byte> > textIm  ( d->getWidth(),d->getHeight(),ZEROS );
    textIm.clear ( myColorMap[0] );
    writeText ( textIm, Point2D<int> ( 0, 0),character.c_str(),myColorMap[1],myColorMap[0] );
    SDL_Surface *surf = d->makeBlittableSurface ( textIm , true ); 
    SDL_Rect clip;
    clip.x = 0 ;
    clip.y = 0 ;
    clip.w = 10 ;
    clip.h = 20 ;
    
      SDL_Rect offs ; offs.x = (d->getWidth() - clipSize)/2 ; offs.y = (d->getHeight() - clipSize)/2 ;
      float posIndex = clipSize ;
      while(posIndex >= -20){
	SDL_Surface* blank = getABlankSurface ( clipSize,clipSize);
	apply_surface ( posIndex,clipSize/2 - 10 ,*surf,*blank,clip );
	d->displaySDLSurfacePatch ( blank , &offs , NULL , -2 , true ,true ) ;
	posIndex -= speed ;
	dumpSurface(blank);
      }
  dumpSurface(surf);
}

void moveUpCharacter(nub::soft_ref<PsychoDisplay> d , string character, int clipSize , float speed  ){
   Image<PixRGB<byte> > textIm  ( d->getWidth(),d->getHeight(),ZEROS );
    textIm.clear ( myColorMap[0] );
    writeText ( textIm, Point2D<int> ( 0, 0),character.c_str(),myColorMap[1],myColorMap[0] );
    SDL_Surface *surf = d->makeBlittableSurface ( textIm , true ); 
    SDL_Rect clip;
    clip.x = 0 ;
    clip.y = 0 ;
    clip.w = 10 ;
    clip.h = 20 ;
    
      SDL_Rect offs ; offs.x = (d->getWidth() - clipSize)/2 ; offs.y = (d->getHeight() - clipSize)/2 ;
      float posIndex = clipSize ;
      while(posIndex >= -20){
	SDL_Surface* blank = getABlankSurface ( clipSize,clipSize);
	apply_surface ( clipSize/2 - 5,posIndex ,*surf,*blank,clip );
	d->displaySDLSurfacePatch ( blank , &offs , NULL , -2 , true ,true ) ;
	posIndex -= speed ;
	dumpSurface(blank);
      }
  dumpSurface(surf);
}

void moveDownCharacter(nub::soft_ref<PsychoDisplay> d ,string character, int clipSize , float speed  ){
   Image<PixRGB<byte> > textIm  ( d->getWidth(),d->getHeight(),ZEROS );
    textIm.clear ( myColorMap[0] );
    writeText ( textIm, Point2D<int> ( 0, 0),character.c_str(),myColorMap[1],myColorMap[0] );
    SDL_Surface *surf = d->makeBlittableSurface ( textIm , true ); 
    SDL_Rect clip;
    clip.x = 0 ;
    clip.y = 0 ;
    clip.w = 10 ;
    clip.h = 20 ;
    
      SDL_Rect offs ; offs.x = (d->getWidth() - clipSize)/2 ; offs.y = (d->getHeight() - clipSize)/2 ;
      float posIndex = -20 ;
      while(posIndex <= clipSize){
	SDL_Surface* blank = getABlankSurface ( clipSize,clipSize);
	 apply_surface ( clipSize/2 - 5,posIndex ,*surf,*blank,clip );
	d->displaySDLSurfacePatch ( blank , &offs , NULL , -2 , true ,true ) ;
	posIndex += speed ;
	dumpSurface(blank);
      }
  dumpSurface(surf);
}

void showStillCharacter(nub::soft_ref<PsychoDisplay> d , string character, int clipSize , float speed  ){
  Image<PixRGB<byte> > textIm  ( d->getWidth(),d->getHeight(),ZEROS );
    textIm.clear ( myColorMap[0] );
    writeText ( textIm, Point2D<int> ( 0, 0),character.c_str(),myColorMap[1],myColorMap[0] );
    SDL_Surface *surf = d->makeBlittableSurface ( textIm , true ); 
    SDL_Rect clip;
    clip.x = 0 ;
    clip.y = 0 ;
    clip.w = 10 ;
    clip.h = 20 ;
    
      SDL_Rect offs ; offs.x = (d->getWidth() - clipSize)/2 ; offs.y = (d->getHeight() - clipSize)/2 ;
      float posIndex = -10 ;
      while(posIndex <= clipSize){
	SDL_Surface* blank = getABlankSurface ( clipSize,clipSize);
	 apply_surface ( clipSize/2 - 5,clipSize/2 - 10 ,*surf,*blank,clip );
	d->displaySDLSurfacePatch ( blank , &offs , NULL , -2 , true ,true ) ;
	posIndex += speed ;
	dumpSurface(blank);
      }
  dumpSurface(surf);
  
}


//////////////////////////

void displayMovingObject(nub::soft_ref<PsychoDisplay> d , PixRGB<byte> color , int shapeSize, int clipSize , float speed , string shapeName , int orientation , int direction){
  if(orientation == 0 ) showStill( d ,  color ,  shapeSize,  clipSize ,  speed ,  shapeName);
  if(orientation == 1 && direction == 1) moveRight( d ,  color ,  shapeSize,  clipSize ,  speed ,  shapeName);
  if(orientation == 1 && direction == -1) moveLeft( d ,  color ,  shapeSize,  clipSize ,  speed ,  shapeName);
  if(orientation == 2 && direction == 1) moveUp( d ,  color ,  shapeSize,  clipSize ,  speed ,  shapeName);
  if(orientation == 2 && direction == -1) moveDown( d ,  color ,  shapeSize,  clipSize ,  speed ,  shapeName);
}

void displayMovingCharacter(nub::soft_ref<PsychoDisplay> d , string character,  int clipSize , float speed  , int orientation , int direction){
  if(orientation == 0 ) showStillCharacter( d ,  character ,  clipSize ,  speed );
  if(orientation == 1 && direction == 1) moveRightCharacter( d ,  character,  clipSize ,  speed );
  if(orientation == 1 && direction == -1) moveLeftCharacter( d ,  character ,  clipSize ,  speed );
  if(orientation == 2 && direction == 1) moveUpCharacter( d ,  character ,  clipSize ,  speed );
  if(orientation == 2 && direction == -1) moveDownCharacter( d ,  character ,  clipSize ,  speed );
}



void displayATrainOfMovingObjects(nub::soft_ref<PsychoDisplay> d ,vector< int > colors , int isiFrames , int shapeSize, int clipSize , float speed , string shapeName , int orientation , int direction){
  
  for(int i = 0 ; i < (int)colors.size() ; i++){
    displayMovingObject(d,myColorMap[colors.at(i)], shapeSize,clipSize,speed,shapeName,orientation,direction);
    d->waitFrames(isiFrames);
  }
  
}


void displayATrainOfMovingCharacters(nub::soft_ref<PsychoDisplay> d ,vector< string > characters , int isiFrames , int clipSize , float speed  , int orientation , int direction){
  
  for(int i = 0 ; i < (int)characters.size() ; i++){
    displayMovingCharacter(d,characters.at(i), clipSize,speed,orientation,direction);
    d->waitFrames(isiFrames);
  }
  
}

vector<int> repalceColors(vector<int> oldColors,int maxPalet, int numOfChange, vector<int>& changedColors){
  vector<int> leftovers;
  for(int i = 0 ; i < maxPalet ; i++) {
    int flag = 0;
    for(uint j = 0 ; j < oldColors.size() ; j++){
      if(oldColors.at(j)==i) flag=1 ;
    }
    if(flag==0) leftovers.push_back(i);
  }
  vector<int> newColors;
  
  vector<int> pickedOnes ;
  while((int)pickedOnes.size()<min(numOfChange,(int)oldColors.size())){
    int proposedNumber = rand()%(int)(oldColors.size());
    if(!itIsInThere(proposedNumber,pickedOnes)) pickedOnes.push_back(proposedNumber);
  }
  
  for(uint i = 0 ; i < oldColors.size() ; i++){   
    if(itIsInThere((int)i,pickedOnes)){
      scramble(leftovers);
      int replace = leftovers.at(0);
      leftovers.erase(leftovers.begin());
      newColors.push_back(replace);
      changedColors.push_back(i);
    }else{
      newColors.push_back(oldColors.at(i));
    }
  }
  
  return newColors;
}

void initializeSounds(string soundDir,string tone1Str , string tone2Str)
{
  //now let's open the audio channel
    if ( Mix_OpenAudio( 22050, MIX_DEFAULT_FORMAT, 2, 4096 ) == -1 ) {
        LINFO( "did not open the mix-audio") ;
    }
    
    string tmpstr = soundDir+"/"+tone1Str;
    tone1 = Mix_LoadMUS(tmpstr.c_str());
    tmpstr = soundDir+"/"+tone2Str;
    tone2 = Mix_LoadMUS(tmpstr.c_str());
}

void initializeAllSounds(string soundDir,string tone1Str , string tone2Str,vector<string> names)
{
  //now let's open the audio channel
    if ( Mix_OpenAudio( 22050, MIX_DEFAULT_FORMAT, 2, 4096 ) == -1 ) {
        LINFO( "did not open the mix-audio") ;
    }
    
    string tmpstr = soundDir+"/"+tone1Str;
    tone1 = Mix_LoadMUS(tmpstr.c_str());
    tmpstr = soundDir+"/"+tone2Str;
    tone2 = Mix_LoadMUS(tmpstr.c_str());
    for (uint i = 0; i < names.size() ; i++) {
        string str = soundDir+"/"+names.at(i)+".wav";
        audioVector.push_back(Mix_LoadMUS(str.c_str()));
    }
}

bool listenForClicks(nub::soft_ref<PsychoDisplay> d , long miliseconds , int butt){
  long start = d->getTimerValue();
  long end = start ;
  int mcFlag=-1;
  bool didItFlag = false ;
  while(end-start < miliseconds){
    mcFlag = d->checkForMouseClick();
    end = d->getTimerValue();
    if(mcFlag!= -1 ){
      if(mcFlag == butt){
	d-> pushEvent("It took "+stringify<long>(end-start));
	didItFlag = true ;
      }else{
	d-> pushEvent("Wrong button in "+ stringify<long>(end - start));
      } 
      mcFlag = -1 ;
    }
  }
  return didItFlag ;
}

int flip(float p){
  int p1 = (int)(p*10000.0f) ;
  int tn = rand()%10000;
  if(tn < p1){
    return 0 ;
  }else{
    return 1 ;
  }
}


//to save our cycles here we will have some redundant information here, we pass a vector for tone sequence and response sequence
// at this stage we will play a sequence of signals and wait listen to mouse clicks afterwards and write everything in two vectors,
// the content of these vectors will be inspected later on
void performNBackTask( nub::soft_ref<PsychoDisplay> d ,int mode ,int numberOfSignals, vector<long> waitingTime , vector<int>& toneSequence , vector<bool>& responseSequence){

  
  if (mode == 0) {
    toneSequence[0] = 1 ;
    toneSequence[1] = 1 ;
    for(int i = 2 ; i < numberOfSignals ; i++){
      toneSequence[i] = rand()%2;
    }
  }
  
  if (mode == 1) {
    toneSequence[0] = rand()%2 ;
    if(toneSequence[0] == 0 ) {toneSequence[1] = 1 ;}else{toneSequence[1] = 0 ;}
    for(int i = 2 ; i < numberOfSignals ; i++){
      toneSequence[i] = rand()%2;
    }
  }
  
   if (mode == 2) {
    for(int i = 0 ; i < numberOfSignals ; i++){
      toneSequence[i] = rand()%2;
    }
  }
  
  bool flag;

  for (int i = 0 ; i < numberOfSignals ; i++) {
        
        if (toneSequence[i]==0) {
            d->pushEvent("tone1 playing");
            //t1c++;
            if (Mix_PlayMusic(tone1,0)==-1) {
              //  return retVector;
            }
            flag = listenForClicks(d,waitingTime[i],1);
            //while (Mix_PlayingMusic()==1) {} ;
        } else {
            d->pushEvent("tone2 playing");
            //t2c++ ;
            if (Mix_PlayMusic(tone2,0)==-1) {
             //   return retVector;
            }
            flag = listenForClicks(d,waitingTime[i],1);
            //while (Mix_PlayingMusic()==1) {} ;
        }
        responseSequence[i] = flag ;
    }  
}


//to save our cycles here we will have some redundant information here, we pass a vector for tone sequence and response sequence
// at this stage we will play a sequence of signals and wait listen to mouse clicks afterwards and write everything in two vectors,
// the content of these vectors will be inspected later on
void performParityTestingTask( nub::soft_ref<PsychoDisplay> d ,int mode ,int numberOfSignals, vector<long> waitingTime , vector<int>& toneSequence , vector<bool> responseSequence){

  
  for(int i = 0 ; i < numberOfSignals ; i++){
    toneSequence[i]= flip(modeProbabilityMap[numberOfSignals].at(mode).at(i));
  }
  bool flag;

  for (int i = 0 ; i < numberOfSignals ; i++) {
        
        if (toneSequence[i]==0) {
            d->pushEvent("tone1 playing");
            //t1c++;
            if (Mix_PlayMusic(tone1,0)==-1) {
              //  return retVector;
            }
            flag = listenForClicks(d,waitingTime[i],1);
            //while (Mix_PlayingMusic()==1) {} ;
        } else {
            d->pushEvent("tone2 playing");
            //t2c++ ;
            if (Mix_PlayMusic(tone2,0)==-1) {
             //   return retVector;
            }
            flag = listenForClicks(d,waitingTime[i],1);
            //while (Mix_PlayingMusic()==1) {} ;
        }
        responseSequence[i] = flag ;
    }  
}


void evaluateThePerformanceOnNBackTask(int typeOfTheTask,const vector<int>& toneSequence , const vector<bool>& responseSequence , int& missed , int& caught , int& falseAlarm){
  
  if(typeOfTheTask == 0 ){
    for(uint i = 0 ; i < toneSequence.size(); i++){
      if(toneSequence[i]== 0 ){
	if(responseSequence[i]){
	  caught++;
	}else{
	  missed++;
	}
      }else{
	if(responseSequence[i]) falseAlarm++;
      } 
    }
  }
  
  
  if(typeOfTheTask == 1){
    for(uint i = 2 ; i < toneSequence.size() ; i++){
      if(toneSequence[i]== toneSequence[i-1]){
	if(responseSequence[i]){
	  caught++;
	}else{
	  missed++;
	}
      }else{
	if(responseSequence[i]) falseAlarm++;
      }
    }
  }
  
  if(typeOfTheTask == 2){
    for(uint i = 2 ; i < toneSequence.size() ; i++){
      if(toneSequence[i]== toneSequence[i-2]){
	if(responseSequence[i]){
	  caught++;
	}else{
	  missed++;
	}
      }else{
	if(responseSequence[i]) falseAlarm++;
      }
    }
  }
}

vector<int> getParityEventVector(vector<int> toneSequence, int whichOne){
  vector<int> tmp(toneSequence.size(),0);
  vector<int> parityEvent(toneSequence.size(),0);
  for(uint i = 0 ; i < toneSequence.size(); i++){
    if(toneSequence[i]== whichOne){
      if(i==0){tmp[0] = 1;}else{tmp[i]=tmp[i-1] + 1 ;}
    }else{
      if(i==0){tmp[0] = 0 ;} else{tmp[i] = tmp[i-1];} 
    }
  }
  
  vector<int> x(toneSequence.size(),0) ;
  vector<int> d(toneSequence.size(),0) ;
  for(uint i = 0 ; i < toneSequence.size() ; i++){
    x[i] = 1 - tmp[i]%2 ;
    if(i==0) {d[i] = tmp[i];}else{d[i]=tmp[i] - tmp[i-1];}
    parityEvent[i] = d[i]*x[i];
  }
  return parityEvent ;
}


void evaluateThePerformanceOnParityTestTask(int typeOfTheTask,vector<int> toneSequence , vector<bool> responseSequence , int& missed , int& caught , int& falseAlarm){
  
  if(typeOfTheTask == 0 ){
    for(uint i = 0 ; i < toneSequence.size(); i++){
      if(toneSequence[i]== 0 ){
	if(responseSequence[i]){
	  caught++;
	}else{
	  missed++;
	}
      }else{
	if(responseSequence[i]) falseAlarm++;
      } 
    }
  }
  
  
  if(typeOfTheTask == 1){
    vector<int> pE0 = getParityEventVector(toneSequence,0);
    for(uint i = 0 ; i < toneSequence.size(); i++) {
      if(pE0[i]==1){
	if(responseSequence[i]) {
	  caught++;
	}else{
	  missed++;
	}
      }else{
	if(responseSequence[i]) falseAlarm++;
      }
    }
  }
  
  
  if(typeOfTheTask == 2){
    vector<int> pE0 = getParityEventVector(toneSequence,0);
    vector<int> pE1 = getParityEventVector(toneSequence,1);
    
    for(uint i = 0 ; i < toneSequence.size(); i++) {
      if(pE0[i]==1 || pE1[i]==1){
	if(responseSequence[i]) {
	  caught++;
	}else{
	  missed++;
	}
      }else{
	if(responseSequence[i]) falseAlarm++;
      }
    }
    
  }

}


void printBlockSummary(nub::soft_ref<PsychoDisplay> d ,vector< vector <int> > trialsVector , vector<int> responseVector , vector<string> titles){
  
  string titleString = "" ;
  for(uint i = 0 ; i < titles.size() ; i++) titleString += " " + titles.at(i) ;
  d->pushEvent(titleString);
  for(uint i = 0 ; i < trialsVector.size() ; i++){
    string trialString = stringify<int>((int)i) + " : ";
    for(uint j = 0 ; j < trialsVector.at(i).size() ; j++){
      trialString += stringify<int>(trialsVector.at(i).at(j))+" " ;
    }
    trialString += stringify<int>(responseVector.at(i));
    d->pushEvent(trialString);
  }
}

void printBlockSummary(vector< vector <int> > trialsVector , vector<int> responseVector , vector<string> titles){
  
  string titleString = "" ;
  for(uint i = 0 ; i < titles.size() ; i++) titleString += " " + titles.at(i) ;
  cout<<titleString<<endl;
  for(uint i = 0 ; i < trialsVector.size() ; i++){
    string trialString = stringify<int>((int)i) + " ";
    for(uint j = 0 ; j < trialsVector.at(i).size() ; j++){
      trialString += stringify<int>(trialsVector.at(i).at(j))+" " ;
    }
    trialString += stringify<int>(responseVector.at(i));
    cout<<trialString<<endl;
  }
}


int getOneKeyPressForColorWheel(nub::soft_ref<PsychoDisplay> d , vector<int> colorInecis , int minRadius , int maxRadius){
  int k = -1 ;
   SDL_Event event ;
  while (SDL_PollEvent(&event)) {}
  while(k<0){
  
     while ( SDL_PollEvent ( &event ) )
        {
          if ( event.type == SDL_MOUSEBUTTONUP  && event.button.button == SDL_BUTTON_LEFT )
            {
	      d->pushEvent("color wheel keypad button-press");
	      float x = (float)(event.button.x - (d->getWidth()/2 -maxRadius));
	      float y = (float)(event.button.y - (d->getHeight()/2 -maxRadius));
	      float alpha0= 2*3.14159265/colorInecis.size();
	      float yp = 2*maxRadius - y ;
	      float r = sqrt((x-maxRadius)*(x-maxRadius) + (yp-maxRadius)*(yp-maxRadius)) ;
	      if(r<maxRadius && r >minRadius){
		float alpha;
		  alpha = acos((yp-(float)maxRadius)/r); 
		  if(x-maxRadius <0 ) alpha =  2*3.14159265 - alpha;
		  int sector = (int)(alpha/alpha0);
		  k = colorInecis.at(sector);
	      }
            }
        }
  }
  return k ;
}

int getOneKeyPressForClockedColorWheel(nub::soft_ref<PsychoDisplay> d , vector<int> colorInecis , int minRadius , int maxRadius, long waitTime){
  int k = -1 ;
  SDL_Event event ;
  long start = d->getTimerValue();
  long end = start ;
     while ( end - start < waitTime )
        {
	   while (SDL_PollEvent(&event)) {if (event.type == SDL_MOUSEBUTTONDOWN) break;}
	   end = d-> getTimerValue();
          if ( event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT )
            {
	      d->pushEvent("color wheel keypad button-press");
	      float x = (float)(event.button.x - (d->getWidth()/2 -maxRadius));
	      float y = (float)(event.button.y - (d->getHeight()/2 -maxRadius));
	      float alpha0= 2*3.14159265/colorInecis.size();
	      float yp = 2*maxRadius - y ;
	      float r = sqrt((x-maxRadius)*(x-maxRadius) + (yp-maxRadius)*(yp-maxRadius)) ;
	      if(r<maxRadius && r >minRadius){
		float alpha;
		  alpha = acos((yp-(float)maxRadius)/r); 
		  if(x-maxRadius <0 ) alpha =  2*3.14159265 - alpha;
		  int sector = (int)(alpha/alpha0);
		  k = colorInecis.at(sector);
	      }
            }
            return k ;
        }
  return k ;
}

vector<int>  getKeyPressesForColorWheel(nub::soft_ref<PsychoDisplay> d , vector<int> colorInecis , int minRadius , int maxRadius,int numOfKeyPresses){
  vector<int> keyPressed;
  int k = 0 ;
  while(k<numOfKeyPresses){
    int kp = getOneKeyPressForColorWheel(d,colorInecis,minRadius,maxRadius);
    keyPressed.push_back(kp);
    SDL_Surface* shape = getABlankSurface ( (int)((float)2*minRadius/sqrt(2)),(int)((float)2*minRadius/sqrt(2)));
    fillOval(shape , d->getUint32color(myColorMap[kp]) , 0  , 0  , (int)((float)2*minRadius/sqrt(2))  , (int)((float)2*minRadius/sqrt(2)) ,d->getUint32color(d->getGrey()));
    SDL_Rect offs ; offs.x = (d->getWidth()-(int)((float)2*minRadius/sqrt(2)))/2  ; offs.y = (d->getHeight()-(int)((float)2*minRadius/sqrt(2)))/2 ;
    d->displaySDLSurfacePatch ( shape , &offs , NULL , -2 , true ,true ) ;
    dumpSurface(shape);
    k++;
  }
  
return keyPressed;
}


vector<int>  getKeyPressesForClockedColorWheel(nub::soft_ref<PsychoDisplay> d , vector<int> colorInecis , int minRadius , int maxRadius,int numOfKeyPresses , long waitTime){
  vector<int> keyPressed;
  int k = 0 ;
  long start = d->getTimerValue();
  long end = start ;
  SDL_Event event ;
  while (SDL_PollEvent(&event)){}
  while(end - start < waitTime){
    int kp = -2 ; 
    if((int)keyPressed.size() < numOfKeyPresses){
      kp = getOneKeyPressForClockedColorWheel(d,colorInecis,minRadius,maxRadius,500000l);
      if (kp >=0 ) keyPressed.push_back(kp);
    }
    end = d->getTimerValue();
    SDL_Surface* shape = getABlankSurface ( (int)((float)2*minRadius/sqrt(2)),(int)((float)2*minRadius/sqrt(2)));
    
    
    if((int)keyPressed.size()>0){
    fillOval(shape , d->getUint32color(myColorMap[keyPressed.at(keyPressed.size()-1)]) , 10  , 10  , (int)((float)2*minRadius/sqrt(2))-20  , (int)((float)2*minRadius/sqrt(2))-20 ,d->getUint32color(myColorMap[0]));
    }
    
    drawArc(shape , d->getUint32color(myColorMap[1]) , (int)((float)2*minRadius/sqrt(2))/2  , (int)((float)2*minRadius/sqrt(2))/2  , (int)((float)2*minRadius/sqrt(2))/2 , 8 , 0, (float)((double)(end-start)/(double)waitTime)*6.283185);
    
    SDL_Rect offs ; offs.x = (d->getWidth()-(int)((float)2*minRadius/sqrt(2)))/2  ; offs.y = (d->getHeight()-(int)((float)2*minRadius/sqrt(2)))/2 ;
    d->displaySDLSurfacePatch ( shape , &offs , NULL , -2 , true ,true ) ;
    dumpSurface(shape);
    k++;
  }
  
return keyPressed;
}



vector<int> drawColorWheelInput(nub::soft_ref<PsychoDisplay> d , vector<int> colorIndecis , int minRadius , int maxRadius, int numOfResponds){
  d->showCursor(true);
  SDL_Surface* shape = getABlankSurface ( 2*maxRadius,2*maxRadius);
  vector<Uint32> colors;
  for (int i = 0 ; i < (int)colorIndecis.size() ; i++) colors.push_back(d->getUint32color(myColorMap[colorIndecis.at(i)]));
  drawColorWheel(shape , colors, maxRadius  , maxRadius  , minRadius ,  maxRadius ,  d->getUint32color(d->getGrey()));
  SDL_Rect offs ; offs.x = d->getWidth()/2 -maxRadius ; offs.y = d->getHeight()/2 -maxRadius ;
  d->displaySDLSurfacePatch ( shape , &offs , NULL , -2 , true ,true ) ;
  vector<int> keypress = getKeyPressesForColorWheel(d,colorIndecis,minRadius,maxRadius,numOfResponds);
  dumpSurface(shape);
  d->showCursor(false);
  return keypress;
}

vector<int> drawColorWheelInputWithClock(nub::soft_ref<PsychoDisplay> d , vector<int> colorIndecis , int minRadius , int maxRadius, int numOfResponds,long timeInterval){
  
  
  d->showCursor(true);
  SDL_Surface* shape = getABlankSurface ( 2*maxRadius,2*maxRadius);
  vector<Uint32> colors;
  for (int i = 0 ; i < (int)colorIndecis.size() ; i++) colors.push_back(d->getUint32color(myColorMap[colorIndecis.at(i)]));
  drawColorWheel(shape , colors, maxRadius  , maxRadius  , minRadius ,  maxRadius ,  d->getUint32color(d->getGrey()));
  SDL_Rect offs ; offs.x = d->getWidth()/2 -maxRadius ; offs.y = d->getHeight()/2 -maxRadius ;
  d->displaySDLSurfacePatch ( shape , &offs , NULL , -2 , true ,true ) ;
  vector<int> keypress = getKeyPressesForClockedColorWheel(d,colorIndecis,minRadius,maxRadius,numOfResponds,timeInterval);
  if((int)keypress.size() < numOfResponds){
    for( int i = (int)keypress.size() ; i < numOfResponds ; i++) keypress.push_back(-1);
  }
  dumpSurface(shape);
  d->showCursor(false);
  return keypress;

}


void drawHorizontalGuidline(nub::soft_ref<PsychoDisplay> d ){
  SDL_Surface* pad= getABlankSurface ( d->getWidth(),1);
  SDL_Rect offs ; offs.x = 0 ; offs.y = d->getHeight()/2;
  d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , false ,true ) ;
  dumpSurface(pad);
}

void drawVerticalGuidline(nub::soft_ref<PsychoDisplay> d ){
  SDL_Surface* pad= getABlankSurface (1, d->getHeight());
  SDL_Rect offs ; offs.y = 0 ; offs.x = d->getWidth()/2;
  d->displaySDLSurfacePatch ( pad , &offs , NULL , -2 , false ,true ) ;
  dumpSurface(pad);
}

void putUpHoizontallyBisectedScreen(nub::soft_ref<PsychoDisplay> d ){
  SDL_Surface* backPad = getABlankSurface ( d->getWidth(),d->getHeight());
  fillRectangle(backPad ,  d->getUint32color(d->getGrey()) , 0  , 0  , d->getWidth() , d->getHeight() ) ;
  SDL_Surface* pad= getABlankSurface ( d->getWidth(),1);
  SDL_Rect clip;
      clip.x = 0 ;
      clip.y = 0 ;
      clip.w = d->getWidth() ;
      clip.h = 1 ;
      apply_surface(0,d->getHeight()/2,*pad,*backPad,clip);
  
  SDL_Rect offs ; offs.x = 0 ; offs.y = 0;
  d->displaySDLSurfacePatch ( backPad , &offs , NULL , -2 , false ,true ) ;
  dumpSurface(pad);
  dumpSurface(backPad);
}
void putUpVerticallyBisectedScreen(nub::soft_ref<PsychoDisplay> d ){
  SDL_Surface* backPad = getABlankSurface ( d->getWidth(),d->getHeight());
  fillRectangle(backPad ,  d->getUint32color(d->getGrey()) , 0  , 0  , d->getWidth() , d->getHeight() ) ;
  SDL_Surface* pad= getABlankSurface (1, d->getHeight());
  SDL_Rect clip;
      clip.x = 0 ;
      clip.y = 0 ;
      clip.w = 1 ;
      clip.h = d->getHeight() ;
      apply_surface( d->getWidth()/2,0,*pad,*backPad,clip);
  
  SDL_Rect offs ; offs.y = 0 ; offs.x = 0;
  d->displaySDLSurfacePatch ( backPad , &offs , NULL , -2 , false ,true ) ;
  dumpSurface(pad);
  dumpSurface(backPad);
}

Point2D<int> waitForMouseClickAndReturnLocation(nub::soft_ref<PsychoDisplay> d ) {
 
  SDL_Event event;
  while(SDL_PollEvent(&event)){}
  
  do { SDL_PollEvent(&event); } while (event.type != SDL_MOUSEBUTTONDOWN ||  event.button.button != SDL_BUTTON_LEFT);
  
return Point2D<int>(event.button.x,event.button.y);
}

void displayStringHorizontally(nub::soft_ref<PsychoDisplay> d, vector<string> s , int onsetFrames , int wsd,int isi){
        int x = (d->getWidth()-s.size()*wsd)/2 ;
        int y = (d->getHeight())/2 -10;
	Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
	textIm.clear(PixRGB<byte>(128,128,128));
	SDL_Rect offs ; offs.x = 0 ; offs.y = 0 ;
        for( uint k = 0 ; k < s.size() ; k++ ){
               // d->displayText(s.substr(k,1),Point2D<int>(x,y+k*10),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
		
                writeText(textIm, Point2D<int>(x+k*wsd,y),s.at(k).c_str(),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128));
		SDL_Surface *surf = d->makeBlittableSurface(textIm , true); 
		d->displaySDLSurfacePatch(surf , &offs , NULL , -2 , false ,true ) ;
		dumpSurface(surf);
		d->waitFrames(isi);
        }
        
        d->waitFrames(onsetFrames);
        d->clearScreen() ;
}

void displayStringVertically(nub::soft_ref<PsychoDisplay> d,vector<string> s , int onsetFrames , int wsd, int isi){
        int x = (d->getWidth())/2 ;
        int y = (d->getHeight()-s.size()*wsd)/2 ;
        SDL_Rect offs ; offs.x = 0 ; offs.y = 0 ;
	Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
	textIm.clear(PixRGB<byte>(128,128,128));
        for( uint k = 0 ; k < s.size() ; k++ ){
              writeText(textIm, Point2D<int>(x,y+k*wsd),s.at(k).c_str(),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128));
	      SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
	      d->displaySDLSurfacePatch(surf , &offs , NULL , -2 , false ,true ) ;
	      dumpSurface(surf);
	      d->waitFrames(isi);
        }
        
        d->waitFrames(onsetFrames);
        d->clearScreen() ;
}
//and this is the function which creates and displays a mask of randomly positioned numbers
void showMask(nub::soft_ref<PsychoDisplay> d,int frames, string alphabet){
        Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
        PixRGB<byte> bgcolor = PixRGB<byte>(128,128,128);
        PixRGB<byte> txtcolor = PixRGB<byte>(0,0,0);
        textIm.clear(bgcolor);
        for(int i = 0 ;  i < 800 ; i++)
                writeText(textIm, Point2D<int>((int)random()%(d->getWidth()),(int)random()%(d->getHeight())),alphabet.substr(random()%(int)alphabet.size(),1).c_str(),txtcolor,bgcolor);
        SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
        SDL_Rect offs ; offs.x = 0 ; offs.y = 0 ;
        d->displaySDLSurfacePatch(surf , &offs , NULL , -2 , false ,true ) ;
        d->waitFrames(frames) ;
        d->clearScreen();
        dumpSurface(surf) ;
}


#endif
//#endif // INVT_HAVE_LIBSDL_IMAGE

