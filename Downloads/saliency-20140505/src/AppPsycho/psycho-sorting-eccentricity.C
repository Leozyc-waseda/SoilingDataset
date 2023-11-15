/*!@file AppPsycho/psycho-sorting-eccentricity.C Psychophysics test to measure the influence of eyemovement on memory task performance */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-sorting-eccentricity.C $
// $Id: psycho-sorting-cost.C 12962 2010-03-06 02:13:53Z irock $
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

ModelManager manager("Psycho-sorting-eccentricity");
nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
map<uint,uint> testMap ;
map<string,string> argMap ;
map<string,vector<SDL_Rect*>*> clipsmap;

//////////////////////////////////////////////
// a functionf for stringigying things
//////////////////////////////////////////////
template <class T> std::string stringify(T i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}


double getAvarage(vector<long> v){
        double f = 0.0 ;
        for( uint i = 0 ; i <  v.size() ; i++ ){
                f += v[i] ;
        }
        if (v.size()!=0) return f/v.size() ;
        return -1 ;
}

double getVariance(vector<long> v){
        double m = getAvarage(v);
        double var = 0.0 ;
        for( uint i = 0 ; i < v.size(); i++ ){
                var += (v[i]-m)*(v[i]-m) ;
        }
        if (v.size()!=0) return var/v.size() ;
        return -1 ;
}

bool itIsInThere(int x , vector<int> bag){
        for( uint i=0 ; i < bag.size(); i++ ){
                if(x == bag[i]) return true ;
        }
        return false ;
}


string get34012(string s){
        string t = "";
        t = s.substr(3,1)+s.substr(4,1)+s.substr(0,1)+s.substr(1,1)+s.substr(2,1);
        return t ;
}

string get21043(string s){
        string t = "";
        t = s.substr(2,1)+s.substr(1,1)+s.substr(0,1)+s.substr(4,1)+s.substr(3,1);
        return t ;
}

string get41230(string s){
        string t = "";
        t = s.substr(4,1)+s.substr(1,1)+s.substr(2,1)+s.substr(3,1)+s.substr(0,1);
        return t ;
}

string get03214(string s){
        string t = "";
        t = s.substr(0,1)+s.substr(3,1)+s.substr(2,1)+s.substr(1,1)+s.substr(4,1);
        return t ;
}

string get42130(string s){
        string t="";
        t = s.substr(4,1)+s.substr(2,1)+s.substr(1,1)+s.substr(3,1)+s.substr(0,1);
        return t ;
}


string get03124(string s){
        string t = "" ;
        t = s.substr(0,1)+s.substr(3,1)+s.substr(1,1)+s.substr(2,1)+s.substr(4,1);
        return t ;
}


string get2013(string s){
        string t = "" ;
        t = s.substr(2,1)+s.substr(0,1)+s.substr(1,1)+s.substr(3,1);
        return t ;
}

string get3201(string s){
        string t = "" ;
        t = s.substr(3,1)+s.substr(2,1)+s.substr(0,1)+s.substr(1,1);
        return t ;
}

string get3120(string s){
        string t = "" ;
        t = s.substr(3,1)+s.substr(1,1)+s.substr(2,1)+s.substr(0,1);
        return t ;
}

////////////////////////////////////////////////////////////////////
//// gets a string as the argument and returns a string composed of
//// characters of the first string sorted in the ascending order
///////////////////////////////////////////////////////////////////
string ascSort(string st)
{
        string res = "" ;
        vector<string> v = vector<string>();
        for(uint i = 0 ; i < st.size() ; i++) v.push_back(st.substr(i,1)) ;

        std::sort(v.begin(), v.end());

        for ( uint i = 0 ; i < v.size() ; i++ ){
                res += v[i] ;
        }
        return res;
}



////////////////////////////////////////////////////////////////////
//// gets a string as the argument and returns a string composed of
//// characters of the first string sorted in the descending order
///////////////////////////////////////////////////////////////////
string desSort(string st)
{
        string res = "" ;
        vector<string> v = vector<string>();
        for(uint i = 0 ; i < st.size() ; i++) v.push_back(st.substr(i,1)) ;
        std::sort(v.begin(), v.end());
        std::reverse(v.begin(), v.end());
        for ( uint i = 0 ; i < v.size() ; i++ ){
                res += v[i] ;
        }
        return res;
}





////////////////////////////////////////////////////////
///// simply generates a sequence of digits with given alphabet with length of l with given threshold, if the given threshold is not achieved the best answer with
//highest metric value in 1000000 times try will be returned
////////////////////////////////////////////////////////

string getARandomString(uint l, string alphabet="0123456789"){

        string test = string("") ;
        test = "" ;
        string tp = string("") ;
        vector<int> pickedones = vector<int>() ;
        for(uint i = 0 ; i < l ; i++){
                int nd;
                do{ nd= rand()% alphabet.size() ; }while(itIsInThere(nd,pickedones) && pickedones.size() <= alphabet.size()) ;
                pickedones.push_back(nd);
                tp = alphabet.substr(nd,1) ;
                test += tp ;
        }

        return test ;
}

///////////////////////////////////////////////////////
//this function is not called in this program, but it generates a random string and it will show it in
//a random place on the screen.
//////////////////////////////////////////////////////
string digitMemorizationTask(uint l, string alphabet="0123456789" , int displayFrame = 10  ){
        d->clearScreen() ;
        vector<int> pickedones = vector<int>() ;
        string test = string("") ;
        string tp = string("") ;
        for(uint i = 0 ; i < l ; i++){
                int nd;
                do{ nd= rand()% alphabet.size() ; }while(itIsInThere(nd,pickedones) && pickedones.size() <= alphabet.size()) ;
                pickedones.push_back(nd);
                tp = alphabet.substr(nd,1) ;
                test += tp ;
        }
        d->displayText(test,true,0) ;
        d->waitFrames(displayFrame) ;
        d->clearScreen() ;
        return test ;
}


////////////////////////////////////////////////////////
///////this will change the order of elements in a vector to a random order
////////////////////////////////////////////////////////
void scramble(vector<string>& v){
        vector<string> tv = vector<string>() ;
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

void scramble(vector<int>& v){
        vector<int> tv = vector<int>() ;
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

////////////////////////////////////////////////////////////////
////This is our button factory
////////////////////////////////////////////////////////////////
SDL_Surface* getButtonImage(string label , PixRGB<byte> txtcolor=PixRGB<byte>(0,0,0) , PixRGB<byte> bgcolor=PixRGB<byte>(255,255,255) ,Point2D<int> size = Point2D<int>(100,100) ,PixRGB<byte> bordercolor=PixRGB<byte>(0,0,0) , int border=3){
        Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
        textIm.clear(bgcolor);
        writeText(textIm, Point2D<int>((size.i - label.length()*10)/2,(size.j-20) /2),label.c_str(),txtcolor,bgcolor);
        SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
        Uint32 bc = d->getUint32color(bordercolor);
        drawRectangle(surf,bc,0,0,size.i -1,size.j -1 ,border);
        SDL_Surface* blank =getABlankSurface(size.i , size.j);
        SDL_Rect clip;
        clip.x = 0 ;
        clip.y = 0 ;
        clip.w = size.i ;
        clip.h = size.j ;
        apply_surface(0,0,*surf,*blank,clip);
        dumpSurface(surf) ;
        return blank ;
}

////////////////////////////////////////////////////////////////////////
////This is the function for creating the keypad, in fact it generates
////12 buttons and associates the actions to the region for each button
////////////////////////////////////////////////////////////////////////

SDL_Surface* getKeyPad(string alphabet,map<string , SDL_Rect>& buttmap){
        SDL_Surface* pad= getABlankSurface(d->getWidth()/4,d->getHeight()/3);
        SDL_Rect clip;
        clip.x=0;
        clip.y=0;
        int numofrows = alphabet.size()/3 +1;
        if(alphabet.size()%3 != 0 ) numofrows++ ;
        int numofcolumns = 3 ;
        clip.w= pad->w / numofcolumns ;
        clip.h = pad->h / numofrows ;

  //keys for 1 to 9
        for( int i = 0 ; i < numofrows*3 ; i++){
                SDL_Surface* but ;
                if((uint)i < alphabet.size()){
                        but = getButtonImage(alphabet.substr(i,1),PixRGB<byte>(0,0,0),PixRGB<byte>(255,255,255),Point2D<int>(pad->w / numofcolumns , pad->h / numofrows),PixRGB<byte>(255, 98 , 25),3);
                }else{
                        but = getButtonImage(" ",PixRGB<byte>(0,0,0),PixRGB<byte>(255,255,255),Point2D<int>(pad->w / numofcolumns , pad->h / numofrows),PixRGB<byte>(255, 98 , 25),3);
                }

                SDL_Rect cl ;
                cl.x = ((i)%numofcolumns)*(pad->w)/numofcolumns ; cl.y= ((i)/numofcolumns)*((pad->h)/numofrows) ;
                cl.w = clip.w ;
                cl.h = clip.h ;
                apply_surface( cl.x , cl.y ,*but,*pad,clip);
                if((uint)i < alphabet.size()) buttmap[alphabet.substr(i,1)] = cl ;
                dumpSurface(but);
        }
        SDL_Rect cl1 ;
        cl1.x = 0 ; cl1.y= (numofrows-1)*((pad->h)/numofrows) ;
        cl1.w = clip.w ;
        cl1.h = clip.h ;
        buttmap["!"] = cl1 ;
        SDL_Surface* but = getButtonImage(string("<-"),PixRGB<byte>(0,0,0),PixRGB<byte>(255,255,255),Point2D<int>(pad->w / numofcolumns , pad->h / numofrows),PixRGB<byte>(255, 98 , 25),3);
        apply_surface(0, (numofrows-1)*((pad->h)/numofrows),*but,*pad,clip);
        dumpSurface(but);
        SDL_Rect cl2 ;
        cl2.x = (pad->w)/numofcolumns ; cl2.y= (numofrows-1)*((pad->h)/numofrows) ;
        cl2.w = clip.w ;
        cl2.h = clip.h ;
        buttmap[" "] = cl2 ;
        but = getButtonImage(string("spc"),PixRGB<byte>(0,0,0),PixRGB<byte>(255,255,255),Point2D<int>(pad->w / numofcolumns , pad->h / numofrows),PixRGB<byte>(255, 98 , 25),3);
        apply_surface((pad->w)/numofcolumns, (numofrows-1)*((pad->h)/numofrows),*but,*pad,clip);
        dumpSurface(but);
        SDL_Rect cl3 ;
        cl3.x = 2*(pad->w)/numofcolumns ; cl3.y= (numofrows-1)*((pad->h)/numofrows) ;
        cl3.w = clip.w ;
        cl3.h = clip.h ;
        buttmap["*"] = cl3 ;
        but = getButtonImage(string("Ok"),PixRGB<byte>(0,0,0),PixRGB<byte>(255,255,255),Point2D<int>(pad->w / numofcolumns , pad->h / numofrows),PixRGB<byte>(255, 98 , 25),3);
        apply_surface(2*(pad->w)/numofcolumns, (numofrows-1)*((pad->h)/numofrows),*but,*pad,clip);
        dumpSurface(but);
        return pad ;
}




///////////////////////////////////////////////////////////////////////////
/////this function listens to mouse clicks and then finds the region of the screen
/////associated with the action, buttmap is the map of the region, offset is the offset of
/////buttons
///////////////////////////////////////////////////////////////////////////
string getPressedButtonCommand(map<string , SDL_Rect>& buttmap,Point2D<int> offset=Point2D<int>(0,0)){
        int quit = 0 ;
        string s ;
        SDL_Event event ;
        while( quit!=2 ){
                while( SDL_PollEvent( &event ) ) {
                        if(event.type == SDL_MOUSEBUTTONDOWN  && event.button.button == SDL_BUTTON_LEFT ){
                                for( map<string , SDL_Rect>::iterator it = buttmap.begin() ; it!=buttmap.end() ; ++it){
                                        if(event.button.x >= (it->second).x + offset.i && event.button.x <= (it->second).x + (it->second).w + offset.i  && event.button.y >= (it->second).y+ offset.j && event.button.y <= (it->second).y + (it->second).h + offset.j) {
                                                quit = 2 ;
                                                s = it->first ;
                                                break;
                                        }

                                }
                        }

                }
        }
        return s ;

}


////////////////////////////////////////////////////
////This function creates a virtual keypad, creates a map of buttons
////and their representation area and listens to the button press and at
////the end returns the keyed digits
////////////////////////////////////////////////////
string getDigitSequenceFromSubject(string alphabet="0123456789" , uint maxl = 7 ){
        d->showCursor(true) ;
        //let's creat a map to map actions to regions of the screen, each region is represented as an SDL_Rect
        map<string , SDL_Rect>* buttmap = new map<string , SDL_Rect>();
        //now let's get the keypad surface while we get the actions map to regions
        SDL_Surface * keypad = getKeyPad(alphabet,*buttmap);
        //this will be the offset of displaying the keypad on the screen
        SDL_Rect offset ;
        offset.x = (d->getWidth() - keypad->w) /2;
        offset.y = (d-> getHeight() - keypad->h) /2;
        //now let's display the keypad
        d->displaySDLSurfacePatch(keypad , &offset,NULL , -2,false, true);
        //this will hold the final string keyed be the subject
        string p = string("") ;
        //this is a temporary string holding the last action related to the pressed key
        string tp = string("");
        //now let's record subject's key press
        while( tp.compare("*")!=0 ){
                //this button is actually the display for the current string
                SDL_Surface* dp = getButtonImage(p ,PixRGB<byte>(195,60,12) ,PixRGB<byte>(255,255,255) ,Point2D<int>(d->getWidth()/6,d->getHeight() /15) ,PixRGB<byte>(0,25,180) , 4) ;
                SDL_Rect offs ; offs.x = (d->getWidth() - dp->w) /2 ; offs.y = d->getHeight()/6 ;
                d->displaySDLSurfacePatch(dp , &offs , NULL , -2 , false ,true ) ;
                //now let's listen to button events
                tp = getPressedButtonCommand(*buttmap,Point2D<int>(offset.x,offset.y)) ;
                dumpSurface(dp) ;
                if(tp.compare("!")==0 && p.size()>=0 ) {
                        if (p.size()>0) p = p.substr(0,p.size()-1) ;
                }else{
                        if(p.size() < maxl && tp.compare("*")!=0) {
                                p +=tp ;
                        }

                }

        }
        buttmap = 0 ;
        dumpSurface(keypad) ;
        d->clearScreen() ;
        return p ;

}



///////////////////////////////////////////////////////////////
//////gets the test string, answer and the mode and identifies if
//////the answer matches the thing it should be, mode=0 checks if
//////if the answer and test string simply the same, mode=1 matches
//////the answer against the ascending sorted string of the test string
//////mode=2 compares the answer against the descending sorted of
//////the test string
///////////////////////////////////////////////////////////////
bool isAnswerCorrect(string test , string answer , int mode){

        if(mode == 0 && answer.compare(test)==0) return true ;

        if(mode == 1 && answer.compare(ascSort(test))==0) return true ;

        if(mode == 2 && answer.compare(desSort(test))==0) return true ;

        return false;
}



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

        com += "\nlogfile=[logfilename.psy] {default = psycho-stroop-concurrent.psy}\n" ;
        com += "\nmemo=[a_string_without_white_space]\n";
        com += "\nstring-size=[>0](the size of string){default=5} \n";
        com += "\nsubject=[subject_name] \n" ;
        com += "\nnum-of-sorting-trials=[>1](number of trials){default=10}\n";
        com += "\nnum-of-mem-trials=[>1](number of trials){default=10}\n";
        com += "\ndigit-onset=[>1] (number of frames that the string will remain onset ){default=10}\n";
        com += "\nalphabet=[a string of characters](a string of characters){default=0123456789}\n";
        com += "\nmode=[1,2,3,4](1 for displaying the whole number, 2 for random display , 3 for linear flashing disply , 4 for linear reverse flashing){default=1}\n";
        com += "\ncue-wait-frames=[<0](number of frames to show the cue){default=0}\n";
        com += "\nmask=[y/n](whether present a mask after presentation, n for no, y  for yes ){default=n}\n" ;
        com += "\nmask-onset-frames=[<0](number of frames that mask will be onset){default=0}\n";
        com += "\nwhite-space=[>0](the distance between digits in display){default=20}\n" ;
        com += "\ncue-type=[a/v](a for audio v for visual){default=v}\n";
        com += "\nsound-dir=[path to wav files directory]{default=..}\n";

        return com ;
}


void displayWholeNumber(string s , int onsetTime , int wsd){
        int x = (d->getWidth()-s.size()*wsd)/2 ;
        int y = (d->getHeight())/2 -10;
        Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
        textIm.clear(PixRGB<byte>(128,128,128));
        for( uint k = 0 ; k < s.size() ; k++ ){
               // d->displayText(s.substr(k,1),Point2D<int>(x,y+k*10),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
                writeText(textIm, Point2D<int>(x+k*wsd,y),s.substr(k,1).c_str(),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128));
        }
        SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
        SDL_Rect offs ; offs.x = 0 ; offs.y = 0 ;
        d->displaySDLSurfacePatch(surf , &offs , NULL , -2 , false ,true ) ;
        dumpSurface(surf);
        d->waitFrames(onsetTime);
        d->clearScreen() ;
}

void displayWholeNumberVertically(string s , int onsetTime , int wsd){
        int x = (d->getWidth())/2 ;
        int y = (d->getHeight()-s.size()*wsd)/2 ;
        Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
        textIm.clear(PixRGB<byte>(128,128,128));
        for( uint k = 0 ; k < s.size() ; k++ ){
               // d->displayText(s.substr(k,1),Point2D<int>(x,y+k*10),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
                writeText(textIm, Point2D<int>(x,y+k*wsd),s.substr(k,1).c_str(),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128));
        }
        SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
        SDL_Rect offs ; offs.x = 0 ; offs.y = 0 ;
        d->displaySDLSurfacePatch(surf , &offs , NULL , -2 , false ,true ) ;
        dumpSurface(surf);
        d->waitFrames(onsetTime);
        d->clearScreen() ;
}

void displayRandom(string s , int onsetTime){
        for( uint k = 0 ; k < s.size() ; k++ ){
                int x = 9*d->getWidth()/20 + rand()%(d->getWidth()/10);
                int y = 9*d->getHeight()/20 + rand()%(d->getHeight()/10) ;
                d->displayText(s.substr(k,1),Point2D<int>(x,y),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
                d->waitFrames(onsetTime);
                d->clearScreen() ;
        }
}

void displayLinear(string s , int onsetTime){
        int x = (d->getWidth()-s.size()*10)/2 ;
        int y = d->getHeight()/2 - 10;
        for( uint k = 0 ; k < s.size() ; k++ ){
                d->displayText(s.substr(k,1),Point2D<int>(x+k*10,y),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
                d->waitFrames(onsetTime);
                d->clearScreen() ;
        }
}

void displayLinearReverse(string s , int onsetTime){
        int x = (d->getWidth()-s.size()*10)/2 ;
        int y = d->getHeight()/2 - 10;
        for( uint k = 0 ; k < s.size() ; k++ ){
                d->displayText(s.substr(k,1),Point2D<int>(x+(s.size()-k)*10,y),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
                d->waitFrames(onsetTime);
                d->clearScreen() ;
        }
}
void displayLinearRandom(string s , int onsetTime){
        int x = (d->getWidth()-s.size()*10)/2 ;
        int y = d->getHeight()/2 - 10;
        for( uint k = 0 ; k < s.size() ; k++ ){
                d->displayText(s.substr(k,1),Point2D<int>(x+ (random()%s.size())*10,y),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
                d->waitFrames(onsetTime);
                d->clearScreen() ;
        }
}

void displayLinearRandomVertically(string s , int onsetTime){
        int x = (d->getWidth())/2 ;
        int y = (d->getHeight()-s.size()*10)/2 - 10;
        for( uint k = 0 ; k < s.size() ; k++ ){
                d->displayText(s.substr(k,1),Point2D<int>(x, (random()%s.size())*10+y),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
                d->waitFrames(onsetTime);
                d->clearScreen() ;
        }
}

void displayLinearRandomNoRepeating(string s , int onsetTime){
        int x = (d->getWidth()-s.size()*10)/2 ;
        int y = d->getHeight()/2 - 10;
        for( uint k = 0 ; k < s.size() ; k++ ){
                d->displayText(s.substr(k,1),Point2D<int>(x+ (random()%s.size())*10,y),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
                d->waitFrames(onsetTime);
                d->clearScreen() ;
        }
}

//and this is the function which creates and displays a mask of randomly positioned numbers
void showMask(int frames, string alphabet="0123456789"){
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

extern "C" int main(const int argc, char** argv)
{

        MYLOGVERB = LOG_INFO;  // suppress debug messages
        //let's push the initial value for the parameters
        argMap["experiment"]="working memory single task - sorting";
        argMap["logfile"]="psycho-sorting-const.psy" ;
        argMap["string-size"]="4" ;
        argMap["subject"]="" ;
        argMap["memo"]="" ;
        argMap["digit-onset"]="10" ;
        argMap["alphabet"]="bcdfghjkmprvx";
        argMap["mode"]="1" ;
        argMap["cue-wait-frames"]="0" ;
        argMap["mask"]="n" ;
        argMap["cue-onset-frames"] = "3" ;
        argMap["white-space"] = "20" ;
        argMap["cue-type"] = "v" ;
        argMap["sound-dir"]="..";
        argMap["num-of-trials"] = "27" ;
        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        d->setEventLog(el);
        nub::soft_ref<EyeTrackerConfigurator>
                        etc(new EyeTrackerConfigurator(manager));
          manager.addSubComponent(etc);

        if (manager.parseCommandLine(argc, argv,
            "at least one argument needed", 1, -1)==false){
                    cout<<getUsageComment()<<endl;
                    return(1);
            }

            for(uint i = 0 ; i < manager.numExtraArgs() ; i++){
                    addArgument(manager.getExtraArg(i),std::string("=")) ;
            }

            manager.setOptionValString(&OPT_EventLogFileName, argMap["logfile"]);
        manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");
        nub::soft_ref<EyeTracker> eyet = etc->getET();
        d->setEyeTracker(eyet);
        eyet->setEventLog(el);


  // let's get all our ModelComponent instances started:
            manager.start();
            for(map<string,string>::iterator it= argMap.begin(); it!= argMap.end() ; ++it) d->pushEvent("arg:"+ it->first+" value:"+it->second ) ;
  // let's display an ISCAN calibration grid:
            d->clearScreen();
            d->displayISCANcalib();
            d->waitForMouseClick();
            d->displayText("Here the experiment starts! click to start!");
            d->waitForMouseClick();
            d->clearScreen();
        //let's see in what mode the user like to run the program
            int mode = atoi(argMap["mode"].c_str());
            vector<long> correctAnswersTiming;
            vector<long> incorrectAnswersTiming;
            vector<long> allTiming ;
            int correctMemory = 0 ;
            int incorrectMemory = 0 ;
            int stringSize = atoi(argMap["string-size"].c_str());
            int onsetDel = atoi(argMap["digit-onset"].c_str()) ;
            int cue_wait_frames = atoi(argMap["cue-wait-frames"].c_str()) ;
            int mask_onset_frames = atoi(argMap["mask-onset-frames"].c_str()) ;
            int white_sapece_distance = atoi(argMap["white-space"].c_str());
            int num_of_trials = atoi(argMap["num-of-trials"].c_str()) ;
            vector<string>* testVector = new vector<string>();
            string se1="cfhk"; testVector->push_back(get3201(se1));testVector->push_back(get2013(se1));testVector->push_back(get3120(se1));
            string se2="bdhj"; testVector->push_back(get3201(se2));testVector->push_back(get2013(se2));testVector->push_back(get3120(se2));
            string se3="dfgk"; testVector->push_back(get3201(se3));testVector->push_back(get2013(se3));testVector->push_back(get3120(se3));
            string le1="dkrv"; testVector->push_back(get3201(le1));testVector->push_back(get2013(le1));testVector->push_back(get3120(le1));
            string le2="bjmx"; testVector->push_back(get3201(le2));testVector->push_back(get2013(le2));testVector->push_back(get3120(le2));
            string le3="ckpv"; testVector->push_back(get3201(le3));testVector->push_back(get2013(le3));testVector->push_back(get3120(le3));
            scramble(*testVector);
            //let's do calibration
            d->displayText("CLICK LEFT button to calibrate; RIGHT to skip");
            int cl = d->waitForMouseClick();
            if (cl == 1) d->displayEyeTrackerCalibration(3,5,1 , true);
            d->clearScreen() ;

            Mix_Music* recallCueMusic = NULL;
            Mix_Music* sortCueMusic = NULL;
            if(argMap["cue-type"].compare("a")==0){
                     //now let's open the audio channel
                    if( Mix_OpenAudio( 22050, MIX_DEFAULT_FORMAT, 2, 4096 ) == -1 ){
                            LINFO( "did not open the mix-audio") ;
                            return -1 ;
                    }

                        string str = argMap["sound-dir"]+"/recall.wav" ;
                        recallCueMusic = Mix_LoadMUS(str.c_str());
                        str = argMap["sound-dir"]+"/sort.wav" ;
                        sortCueMusic = Mix_LoadMUS(str.c_str());
            }
            vector<int>* taskVector = new vector<int>();
            for(int i = 0 ; i < 18 ; i++) taskVector->push_back(1);
            for(int i = 0 ; i < 9 ; i++)  taskVector->push_back(0);
            scramble(*taskVector);
            int testCounter = 0 ;
            for( int r = 0 ; (int)r < min((int)taskVector->size(),num_of_trials) ; r++ ){
                            
                            int task = taskVector->at(r) ;
                            d->pushEvent("**************************************") ;
                            d->showCursor(true);
                            d->displayText("click one of the  mouse buttons to start!");
                            d->waitForMouseClick() ;
                            d->showCursor(false) ;
                            string testString ;
                            
                            switch( task ){
                                case 0 : testString = getARandomString(stringSize, argMap["alphabet"]);break ;
                                case 1 : testString = testVector->at(testCounter);testCounter++;break ;
                            }
                            d->clearScreen() ;
                            d->displayFixationBlink();
                            switch( mode ){
                                    case 1 : displayWholeNumber(testString,onsetDel,white_sapece_distance);break ;
                                    case 2 : displayRandom(testString,onsetDel) ; break ;
                                    case 3 : displayLinear(testString,onsetDel) ; break ;
                                    case 4 : displayLinearRandom(testString,onsetDel) ; break ;
                                    case 5 : displayLinearReverse(testString,onsetDel) ; break ;
                                    case 6 : displayLinearRandomVertically(testString,onsetDel) ; break ;
                                    case 7 : displayWholeNumberVertically(testString,onsetDel,white_sapece_distance) ; break ;
                            }
                            if(argMap["mask"].compare("y")==0) showMask(mask_onset_frames,argMap["alphabet"]);
                            if(cue_wait_frames != 0) d->waitFrames(cue_wait_frames) ;
                            d->displayFixation() ;
                            if(argMap["cue-type"].compare("v")==0){
                                    d->displayRedDotFixation();
                            }else{
                                if(task == 0){if( Mix_PlayMusic( recallCueMusic, 0 ) == -1 ) { return 1; }
                                    while( Mix_PlayingMusic() == 1 ){}
                                }else{
                                    if( Mix_PlayMusic( sortCueMusic, 0 ) == -1 ) { return 1; }
                                    while( Mix_PlayingMusic() == 1 ){}
                                }

                            }

                            d->clearScreen() ;
                            string imst ;
                            if(task!=0){
                                imst= "===== Showing image: def_"+stringify(task)+"_"+stringify(testCounter)+"_"+testString+".png =====";
                                d->pushEvent("the sequence for operation is : "+testString) ;
                                d->pushEvent(imst);
                                eyet->track(true);
                                d->waitForMouseClick();
                                eyet->track(false);
                                d->pushEvent("task ends") ;
                                string  answer = getDigitSequenceFromSubject(argMap["alphabet"] , testString.size());
                                d->pushEvent("subject keyed : "+answer);
                                if(answer.compare(ascSort(testString))==0){
                                    d->pushEvent("answer was correct");
                                    d->displayText(":)");
                                }else{
                                    d->pushEvent("answer was incorrect");
                                    d->displayText(":(");
                                }
                            }else{
                                 string  answer = getDigitSequenceFromSubject(argMap["alphabet"] , testString.size());
                                d->pushEvent("subject keyed : "+answer);
                                if(answer.compare(testString)==0){
                                    d->pushEvent("recall was correct");
                                    d->displayText(":)");
                                    correctMemory++ ;
                                }else{
                                    d->pushEvent("recall was incorrect");
                                    d->displayText(")");
                                    incorrectMemory++ ;
                                }
                            }

            }
            d->pushEvent("number of correct memory recall : "+ stringify(correctMemory)) ;
            d->pushEvent("number of incorrect memory recall : "+ stringify(incorrectMemory)) ;
            d->clearScreen();
            d->displayText("Experiment complete. Thank you!");
            d->waitForMouseClick();
            taskVector = 0 ;
          // stop all our ModelComponents
            manager.stop();


          // all done!
            return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

