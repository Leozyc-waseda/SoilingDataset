/*!@file AppPsycho/psycho-spatial-orientation.C Psychophysics test to measure the effect of executive tasks on spatial memory */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-spatial-memory.C $
// $Id: psycho-spatial-memory.C 13040 2010-03-23 03:59:25Z nnoori $
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

ModelManager manager("Psycho-Spatial-Orientation");
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
///// simply generates a sequence of characters out of master string
////////////////////////////////////////////////////////

string getARandomString(uint l, string alphabet="0123456789" ){

        string test = string("") ;
        string retString ;
        test = "" ;
        string tp = string("") ;
        vector<int> pickedones = vector<int>() ;
        for(uint i = 0 ; i < l ; i++){
                        int nd;
			do{ nd= rand()% alphabet.size() ; }while(itIsInThere(nd,pickedones)) ;
                        pickedones.push_back(nd);
                        tp = alphabet.substr(nd,1) ;
                        test += tp ;
        }
        retString = test ;
        return retString ;
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
string getDigitSequenceFromSubject(string alphabet="0123456789" , uint maxl = 7 , string message=""){
        d->showCursor(true) ;
        //let's creat a map to map actions to regions of the screen, each region is represented as an SDL_Rect
        map<string , SDL_Rect>* buttmap = new map<string , SDL_Rect>();
        //now let's get the keypad surface while we get the actions map to regions
        SDL_Surface * keypad = getKeyPad(alphabet,*buttmap);
        //this will be the offset of displaying the keypad on the screen
        SDL_Rect offset ;
        offset.x = (d->getWidth() - keypad->w) /2;
        offset.y = (d-> getHeight() - keypad->h) /2;
        //d->displayText(message, Point2D<int>(d->getWidth()/3,d->getHeight()*2 /16) , PixRGB<byte>(0,0,0) ,PixRGB<byte>(127,127,127)) ;
        SDL_Surface* msgp = getButtonImage(message ,PixRGB<byte>(195,60,12) ,PixRGB<byte>(127,127,127) ,Point2D<int>(d->getWidth()/6,d->getHeight() /15) ,PixRGB<byte>(127,127,127) , 4) ;
        SDL_Rect msgoffs ; msgoffs.x = (d->getWidth() - msgp->w) /2 ; msgoffs.y = 2*d->getHeight()/9 ;
        d->displaySDLSurfacePatch(msgp , &msgoffs , NULL , -2 , false ,true ) ;
        dumpSurface(msgp) ;
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
        for(int i = 0 ;  i < 200 ; i++) 
                writeText(textIm, Point2D<int>((int)random()%(d->getWidth()),(int)random()%(d->getHeight())),alphabet.substr(random()%(int)alphabet.size(),1).c_str(),txtcolor,bgcolor);
        SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
        SDL_Rect offs ; offs.x = 0 ; offs.y = 0 ;
        d->displaySDLSurfacePatch(surf , &offs , NULL , -2 , false ,true ) ;
        d->waitFrames(frames) ;
        d->clearScreen();
        dumpSurface(surf) ;
}

vector<int> spatial_memory_task(int onsetTime, int isd , int dir ,  int ma=30 ,int Ma=120){

    int w = d->getWidth();
    int h = d->getHeight();
    int neg = rand()%(Ma-ma) +ma ;
    int pos = rand()%(Ma-ma) +ma ;
    //d->displayFixation();
    if(dir==0) {
	    d->displayRedDotFixation(w/2+pos,h/2);
	    d->waitFrames(onsetTime);
	    d->clearScreen();
	    d->waitFrames(isd);
	    d->displayRedDotFixation(w/2-neg,h/2);
	    d->waitFrames(onsetTime);
	    d->clearScreen();
    }
    if(dir==1) {
	    d->displayRedDotFixation(w/2,h/2+pos);
	    d->waitFrames(onsetTime);
	    d->clearScreen();
	    d->waitFrames(isd);
	    d->displayRedDotFixation(w/2,h/2-neg);
	    d->waitFrames(onsetTime);
	    d->clearScreen();
    }
    vector<int> locations;
    locations.push_back(neg);
    locations.push_back(pos);
    return locations ;
}

int spatial_memory_retrival(vector<int> dotsVector,int dir ,float cr = 0.5 ){
    
    int w = d->getWidth();
    int h = d->getHeight();
    int change = rand()%2;
    int neg = dotsVector.at(0); int pos = dotsVector.at(1);
    switch( change ){
        case 0 : d->pushEvent("same target");break;
        case 1 : d->pushEvent("target changed");break;
    }
    if(change == 1){
        int chDir = rand()%2;
        int c=0;
        switch(chDir){
            case 0:  c =((rand()%2 -0.5)/0.5 ); neg = neg + c* cr*neg ; break ;
	        case 1:  c =((rand()%2 -0.5)/0.5 ); pos = pos + c* cr*pos ; break ;
        }
    }
    
    d->pushEvent("for retrieval dots showed at negative side :"+ stringify(neg)+" positive side:" + stringify(pos));
    if(dir == 0){
	    d->displayRedDotFixation(w/2+pos,h/2);
	    d->displayRedDotFixation(w/2-neg,h/2);
    }
    if(dir == 1){
	    d->displayRedDotFixation(w/2,h/2+pos);
	    d->displayRedDotFixation(w/2,h/2-neg);
    }
	d->waitFrames(30);
	d->clearScreen();
	string ans = getDigitSequenceFromSubject("yn-",1);
	int res = 0 ;
	if(change==0 && ans.compare("y")==0) res=1;
	if(change==1 && ans.compare("n")==0) res=1;
	return res ;
}

vector<int> executive_memory_task(Mix_Music* tone1 , Mix_Music* tone2 ,string initial_string,int counter_length, int min_tone_wait, int max_tone_wait ){
        int t1c = 0 ;
        int t2c = 0 ;
        vector<int> wFrames;
        vector<int> toneSequence ;
	vector<int> retVector ;
        map<int,string> strMap;

        int flag=0;
        float milisum=0;
        do{
            int rtmp = rand()%(max_tone_wait - min_tone_wait) + min_tone_wait;
            milisum += rtmp*33.33 + 240;
            if(milisum > counter_length*33.33) {
                milisum = milisum -rtmp*33.33 - 240; 
                flag=1;
            }else{
                wFrames.push_back(rtmp);
                toneSequence.push_back(rand()%2);
            }
        }while(flag==0);
        int lagFrames = (int)((counter_length - milisum)/33.33);

        for(uint i = 0 ; i < wFrames.size() ; i++){
            d->pushEvent("will wait for "+stringify(wFrames[i]));
            d->waitFrames(wFrames[i]);
            if(toneSequence[i]==0){
                d->pushEvent("tone1 playing");
                t1c++;
                if(Mix_PlayMusic(tone1,0)==-1){return retVector;}
                while(Mix_PlayingMusic()==1){} ;
            }else{
                d->pushEvent("tone2 playing");
                t2c++ ;
                if(Mix_PlayMusic(tone2,0)==-1){return retVector;}
                while(Mix_PlayingMusic()==1){} ;
            }
        }
        d->waitFrames(lagFrames);
	retVector.push_back(t1c);
        retVector.push_back(t2c);
        d->clearScreen();
        d->displayFixation();
        return retVector ;
        
}



std::string getUsageComment(){

        string com = string("\nlist of arguments : \n");

        com += "\nlogfile=[logfilename.psy] {default = psycho-sm-or.psy}\n" ;
        com += "\nmemo=[a_string_without_white_space]\n";
        com += "\nstring-size=[>0](the size of counter string){default=4} \n";
        com += "\nsubject=[subject_name] \n" ;
        com += "\nnum-of-trials=[>1] (number of trials ) {default=10}\n";
        com += "\nalphabet=[a string of characters](a string of characters){default=abcdefghijklmnopqrstuvwxyz}\n";
        com += "\nmode=[1,2,3](1 for spatial memory task 2 for single counter task, 2 for concurrent task){default=1}\n";
	com += "\nsingle-dot-onset=[>1](number of frames that the single dot should be presented){default=16}\n";
	com += "\ndots_ISI=[>1](number of frames between dots presentations){default=16}\n";
	com += "\ndots_radius=[>1](the radius for circle of dots in pixel){default=100}\n";
	com += "\ndots_min_angle=[>0](minimum angle between dots on the circle){default=45}\n";
	com += "\ndots_max_angle=[>0] (maximum angle between dots on the circle){default=90}\n";
	com += "\nspatial-delay=[>0](number of frames for spatial memory task ){default=180}\n";
	com += "\n spatial-counter-ISI=[>1](numer of frames between last dot presentation and start of counter task){default=16}\n";
	com += "\n counter-length=[>1](total lenght of counter experiment in frames){default=300}\n";
	com += "\n counter-spatial-query-ISI=[>1](the delay between end of counter trail and start of spatial memory query in number of frames){default=16}\n";
	com += "\n spatial-query-length=[>1](the total length for showing the spatial memory test in number of frames){default=60}\n";
	com += "\n spatial-query-counter-query-ISI=[>1](number of frames between the unspeeded spatial memory query and showing the counter query){default=30}\n";
	com += "\n counter-query-length=[>1](number of frames for presenting the counter query){default=60}\n";
	com += "\n cue-onset-frames=[>1](){default=3}\n";;
        com += "\nsound-dir=[path to wav files directory]{default=..}\n";
	com += "\ntone1=[a valid file name](name of wav file without extension for tone 1){default=sin}\n";
	com += "\ntone2=[a valid file name](name of wav file without extension for tone 2){default=square}\n";
	com += "\ncue-file=[a valid file name](name of a wav file without extension for the audio cue){default=cue1}\n";
	com += "\nmin-tone-wait=[>1](minimum number of frames waiting on each tone){default=60}\n";
	com += "\nmax-tone-wait=[>1](maximum number of frames waiting on each tone){default=90}\n";
    com += "\ndot-shift=[>1](amount of shift in dots position in pixel)[default=16]\n";
        return com ;
}

extern "C" int main(const int argc, char** argv)
{

        MYLOGVERB = LOG_INFO;  // suppress debug messages
        //let's push the initial value for the parameters
        argMap["experiment"]="spatial-memory-test";
        argMap["logfile"]="psycho-sm.psy" ;
        argMap["string-size"]="3" ;
        argMap["num-of-trials"]="10";
        argMap["subject"]="" ;
        argMap["memo"]="" ;
        argMap["single-dot-onset"]="16" ;
	    argMap["dots-ISI"]="16";
	    argMap["spatial-delay"]="180";
	    argMap["spatial-counter-ISI"]="16";
	    argMap["counter-length"]="300";
	    argMap["counter-spatial-query-ISI"]="16";
	    argMap["spatial-query-length"]="60";
	    argMap["spatial-query-counter-query-ISI"]="30";
	    argMap["counter-query-length"]="60";
       	argMap["alphabet"]="0123456789";
        argMap["mode"]="1" ;
        argMap["cue-onset-frames"] = "3" ;
        argMap["sound-dir"]="..";
	    argMap["tone1"]="sine";
	    argMap["tone2"]="square";
	    argMap["cue-file"]="cue1";
	    argMap["min-tone-wait"]="50";
	    argMap["max-tone-wait"]="80";
	    argMap["dots_min_radius"]="12";
	    argMap["dots_max_radius"]="96";
        argMap["dot-shift"] = "0.75";
        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        d->setEventLog(el);
       //? nub::soft_ref<EyeTrackerConfigurator>
          //?              etc(new EyeTrackerConfigurator(manager));
         //? manager.addSubComponent(etc);

        if (manager.parseCommandLine(argc, argv,
            "at least one argument needed", 1, -1)==false){
                    cout<<getUsageComment()<<endl;
                    return(1);
            }

            for(uint i = 0 ; i < manager.numExtraArgs() ; i++){
                    addArgument(manager.getExtraArg(i),std::string("=")) ;
            }

            manager.setOptionValString(&OPT_EventLogFileName, argMap["logfile"]);
        	//+manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");
        	//+nub::soft_ref<EyeTracker> eyet = etc->getET();
        	//+d->setEyeTracker(eyet);
        	//+eyet->setEventLog(el);


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
            //int mode = atoi(argMap["mode"].c_str());
            string masterString=argMap["alphabet"];
            
            
            int numOfTests = atoi(argMap["num-of-trials"].c_str()) ;
            int stringSize = atoi(argMap["string-size"].c_str());
            int dots_ISI = atoi(argMap["dots-ISI"].c_str()) ;
	    int dot_onset = atoi(argMap["single-dot-onset"].c_str());
	    int dots_min_radius = atoi(argMap["dots_min_radius"].c_str());
	    int dots_max_radius = atoi(argMap["dots_max_radius"].c_str());
	    int mode = atoi(argMap["mode"].c_str());
            int counter_length = atoi(argMap["counter-length"].c_str());
 	    int min_tone_wait = atoi(argMap["min-tone-wait"].c_str());
 	    int max_tone_wait = atoi(argMap["max-tone-wait"].c_str());
            float dot_max_change = atof(argMap["dot-shift"].c_str());
	    vector<int> orVector;
	    for( int i = 0 ; i< numOfTests/2 ; i++ ){
		    orVector.push_back(0);
	    }
	    for( int i =numOfTests/2; i<numOfTests ; i++){
		    orVector.push_back(1);
	    }
	    scramble(orVector);
            //let's do calibration        
           
            
            d->clearScreen();
            
            cout<< stringify(d->getWidth())<<" x "<<stringify(d->getHeight())<<endl;
            Mix_Music* cueMusic = NULL;//this is the cue for the start of trial 
			Mix_Music* tone1 = NULL;//tone 1
			Mix_Music* tone2 = NULL;//tone 2
            map<int,Mix_Music*>  audio_map ;//we will have access to the audios for stimulus presentation using this map
			map<string,int>  charmap ;
			map<int,string>  charinvmap ;
 			//now let's open the audio channel
	        if( Mix_OpenAudio( 22050, MIX_DEFAULT_FORMAT, 2, 4096 ) == -1 ){
                  LINFO( "did not open the mix-audio") ;
                  return -1 ;
			}
			//now that everyting is ok, let's load the main string audio files and push them in a map
	    	for(uint i = 0; i < masterString.size() ; i++){
				string str = argMap["sound-dir"]+"/"+masterString.substr(i,1)+".wav";
				audio_map[i]=Mix_LoadMUS(str.c_str());
				charmap[masterString.substr(i,1)]=i;
				charinvmap[i]=masterString.substr(i,1) ;
			}

            for( uint i=10;i<100 ;i++ ){
                charmap[stringify(i)]=i;
                charinvmap[i]=stringify(i);
            }
			
            if(argMap["cue-type"].compare("a")==0){
                        string str = argMap["sound-dir"]+"/"+argMap["cue-file"]+".wav" ;
                        cueMusic = Mix_LoadMUS(str.c_str());
            }
			
			string tmpstr = argMap["sound-dir"]+"/"+argMap["tone1"]+".wav";
			tone1 = Mix_LoadMUS(tmpstr.c_str());
			tmpstr = argMap["sound-dir"]+"/"+argMap["tone2"]+".wav";
			tone2 = Mix_LoadMUS(tmpstr.c_str());
			
            int cr = numOfTests ;//this is the counter for number of rounds
			d->showCursor(true);
			d->displayText("click one of the  mouse buttons to start!");
            d->waitForMouseClick() ;
			d->showCursor(false);
           
	int correctHSpatialMemory=0;
    int correctVSpatialMemory=0;
    int incorrectHSpatialMemory=0;
    int incorrectVSpatialMemory=0;
	int correctCounting=0;
	int incorrectCounting=0;
    int missCounting=0;
	    cr=0;
            while( cr < numOfTests ){
                string inStr = getARandomString(stringSize,"34567");
		d->showCursor(false);
		d->clearScreen() ;
		d->pushEvent("**************************************") ;
		d->pushEvent("initial string :" + inStr);
                for(uint i = 0 ;  i < inStr.size(); i++){
		    d->pushEvent("playing character "+inStr.substr(i,1));
                    if(Mix_PlayMusic(audio_map[charmap[inStr.substr(i,1)]],0)==-1){return 1;}
                    while(Mix_PlayingMusic()==1){} ;
		    d->pushEvent("ended playing character "+inStr.substr(i,1));
                }
                d->waitFrames(60);
		d->displayFixation();
		d->waitFrames(10);
		d->clearScreen() ;
		d->pushEvent("presentation orientation: " + stringify(orVector.at(cr)));
                vector<int> dotsVector = spatial_memory_task(dot_onset,dots_ISI,orVector.at(cr),dots_min_radius,dots_max_radius);
		d->pushEvent("dots showed at negative:"+ stringify(dotsVector.at(0))+" positive:" + stringify(dotsVector.at(1)));
		d->waitFrames(10);
		d->displayFixation();
		vector<int> tonesVector = executive_memory_task( tone1 , tone2 , inStr, counter_length, min_tone_wait, max_tone_wait );
		string  newStr ;
		if (stringSize == 2)
		newStr = charinvmap[charmap[inStr.substr(0,1)]+tonesVector.at(0)]+charinvmap[charmap[inStr.substr(inStr.size()-1,1)]+tonesVector.at(1)];
		if (stringSize>2)
			newStr = charinvmap[charmap[inStr.substr(0,1)]+tonesVector.at(0)]+inStr.substr(1,inStr.size()-2)+charinvmap[charmap[inStr.substr(inStr.size()-1,1)]+tonesVector.at(1)];
		if(mode==1 || mode==3){
			int spMemAns = spatial_memory_retrival(dotsVector,orVector.at(cr),dot_max_change);
			if(orVector.at(cr)==0){
				if (spMemAns ==1){
                    d->pushEvent("correct horizontal spatial memory recall");
					correctHSpatialMemory++;
				}else{
                    d->pushEvent("incorrect horizontal spatial memory recall");
					incorrectHSpatialMemory++;
				}
			}else{
				if (spMemAns ==1){
                    d->pushEvent("correct horizontal spatial memory recall");
					correctVSpatialMemory++;
				}else{
                    d->pushEvent("incorrect horizontal spatial memory recall");
					incorrectVSpatialMemory++;
				}
				
			}
		}
		
		
		if(mode ==1){
            	//string ans = getDigitSequenceFromSubject(masterString,3," 3 digits");
			string ans1 = getDigitSequenceFromSubject(masterString,1,"first digit");
			d->pushEvent("subject keyed:" + ans1 + " for first digit"); 
			string ans2 = getDigitSequenceFromSubject(masterString,1,"second digit");
			d->pushEvent("subject keyed:" + ans2 + " for second digit");
            	//d->pushEvent("subject keyed:" + ans);
			if(ans1.compare(inStr.substr(0,1))==0 && ans2.compare(inStr.substr(inStr.size()-1,1))==0 ) {
				correctCounting++;
			}else{
				incorrectCounting++ ;
			}
			
		}
		
		if(mode ==2 || mode==3){
			int sineCounter = tonesVector.at(0)+atoi(inStr.substr(0,1).c_str());
			int squareCounter = tonesVector.at(1)+atoi(inStr.substr(inStr.size()-1,1).c_str());
			d->pushEvent("sine signal counter is :"+stringify(sineCounter));
			d->pushEvent("square signal counter is :"+stringify(squareCounter));
			string ans1 = getDigitSequenceFromSubject(masterString,2,"soft tone counter");
			d->pushEvent("subject keyed:" + ans1 + " for sine counter");
			string ans2 = getDigitSequenceFromSubject(masterString,2,"rough tone counter");
			d->pushEvent("subject keyed:" + ans2 + " for square counter");
			d->pushEvent("trial misscounting: " + stringify(abs(atoi(ans1.c_str()) - sineCounter) + abs(atoi(ans2.c_str()) - squareCounter)));
			missCounting += abs(atoi(ans1.c_str()) - sineCounter) + abs(atoi(ans2.c_str()) - squareCounter);
			
		}	
		cr++;
            }
	    d->pushEvent("correct horizontal spatial memory: " +stringify(correctHSpatialMemory));
	    d->pushEvent("incorrect horizontal spatial memory: " +stringify(incorrectHSpatialMemory));
	    
        d->pushEvent("correct vertical spatial memory: " +stringify(correctVSpatialMemory));
	d->pushEvent("incorrect vertical spatial memory: " +stringify(incorrectVSpatialMemory));
	    d->pushEvent("correct counting: " +stringify(correctCounting));
	    d->pushEvent("incorrect counting: " +stringify(incorrectCounting));
        d->pushEvent("total misscounting: "+stringify(missCounting));
            d->displayText("Experiment complete. Thank you!");
            d->waitForMouseClick();

          // stop all our ModelComponents
            manager.stop();


          // all done!
            return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

