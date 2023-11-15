/*!@file AppPsycho/psychoWM.C Psychophysics test to measure the influence of eyemovement on memory task performance */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psychoWM-M.C $
// $Id: psychoWM-M.C 12962 2010-03-06 02:13:53Z irock $
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
#include "Devices/SimpleMotor.H"
#include <SDL/SDL_mixer.h>

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

ModelManager manager("Psycho-Concurrent-Digit");
nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
nub::soft_ref<SimpleMotor> motor(new SimpleMotor(manager));
map<uint,uint> testMap ;
map<string,string> argMap ;
Mix_Chunk *audio_chunk = NULL;

//////////////////////////////////////////////
// a functionf for stringigying things
//////////////////////////////////////////////
template <class T> std::string stringify(T i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}




bool itIsInThere(int x , vector<int> bag){
        for( uint i=0 ; i < bag.size(); i++ ){
                if(x == bag[i]) return true ;
        }
        return false ;
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


/////////////////////////////////////////////////////////////
//this function checks how many subblocks of the sorted string can be found in
//original string, the bigger chunks found the bigger number will be return
////////////////////////////////////////////////////////////

int mydist(string str , string sorted){
        size_t found;
        int m = 0 ;
        for(uint i = 2 ; i <= sorted.size() ; i++ ){
                for(uint j = 0 ; j <= sorted.size()-i  ; j++ ) {
                        found = str.find(sorted.substr(j,i));
                        if (found!=string::npos) m += i ;
                }
        }
        return m ;
}

/////////////////////////////////////////////////////////////
//this function checks out the difficulty of sorting the string assigns a number which
//reflects how many moves are needed for sorting,
////////////////////////////////////////////////////////////

int getDisMetric(string st){
        int m = 0 ;
        size_t found;
        string asString = ascSort(st) ;
        for(uint i = 0 ; i < st.size() ; i++ ){
                found = asString.find(st.substr(i,1));
                m += abs((int)i-int(found)) ;
        }

        return m- 2*mydist(st,asString) ;
}



////////////////////////////////////////////////////////
///// simply generates a sequence of digits with given alphabet with length of l with given threshold, if the given threshold is not achieved the best answer with
//highest metric value in 1000000 times try will be returned
////////////////////////////////////////////////////////

string getARandomString(uint l, string alphabet="0123456789" , int thresh=0){

        string test = string("") ;
        string retString ;
        int maxDist = -1000000;
        int it = 0 ;
        int d = 0 ;
        do{
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
                it++ ;
                d = getDisMetric(test);
                maxDist=max(maxDist,d) ;
                if (d==maxDist) retString = test ;

        }while( maxDist<thresh && it < 1000000);

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
        com += "\nlogfile=[logfilename.psy] {default =psycho-wm-m.psy}\n" ;
        com += "\nmemo=[a_string_without_white_space]\n";
        com += "\nstring-size=[>0](the size of string){default=5} \n";
        com += "\nsubject=[subject_name] \n" ;
        com += "\ntest-rounds=[>1] (number of tests ) {default=10}\n";
        com += "\ndigit-onset=[>1] (number of frames that the string will remain onset ){default=10}\n";
        com += "\nalphabet=[a string of characters](a string of characters){default=0123456789}\n";
        com += "\nmetric-thresholdt=[>-30 for string of lenght of 5](a string of characters){default=10}\n";
        com += "\nmaintain-prb=[a number between 0 to 1], probability of challenging the subject with a memorization task, default=0.2\n"  ;
        com += "\nmin-reaction-time=[>0](minimum value for avarage reaction time in microsecond in order to consider a trial valid){default=1000000}\n" ;
        com += "\nmax-miss=[>0](maximum misses in a trial in order to be  considered as a valid one){default=0}";
        com += "\nmax-false-click=[>0](maximum false clicks in a trial in order to be  considered as a valid one){default=0}";
        com += "\ninterrupt-time-range=[x-y](this defines a range of uniform radom distribution by which the perceptual interruption happens){default=500000-5000000}\n" ;
        com += "\ncue-wait-frames=[<0](number of frames to show the cue){default=0}\n";
        com += "\ncue-onset-frames=[<0](number of frames to show the cue onset){default=3}\n";
        com += "\nmask=[y/n](whether present a mask after presentation, n for no, y  for yes ){default=n}\n" ;
        com += "\nmask-onset-frames=[<0](number of frames that mask will be onset){default=0}\n";
        com += "\nsound-dir=[path to wav files directory]{default=..}\n";
        com += "\audio-file=[white noise file]{default=audio.wav}\n";
        com += "\npuase-frames=[number of frames for stopping the the audio playback]{default=5}\n";
        return com ;
}

int myCheckForMouseClick()
{
        SDL_Event event;

        while(SDL_PollEvent(&event))
        {
                if (event.type == SDL_MOUSEBUTTONDOWN)
                {
                        if(event.button.button == SDL_BUTTON_LEFT) {
                                return 1 ;
                        }
                        if(event.button.button == SDL_BUTTON_RIGHT) {
                                return 2 ;
                        }

                }
      // ignore other events
        }

  // there was no event in the event queue:
        return -1;
}


int getClick(){
    // while (myCheckForMouseClick() != -1) ;
        SDL_Event event;
        bool report = false ;
        int i = 0;  // will be returned if any other button than left or right
        do {
                do { SDL_WaitEvent(&event); } while (event.type != SDL_MOUSEBUTTONDOWN);
                if (event.button.button == SDL_BUTTON_LEFT) {
                        i = 0 ;
                        report = true ;
                }
                if (event.button.button == SDL_BUTTON_RIGHT) {
                        i = 1 ;
                        report = true ;
                }

        }while(!report) ;

        return i ;
}

int getAllKindOfClick(){
    // while (myCheckForMouseClick() != -1) ;
        SDL_Event event;
        bool report = false ;
        int i = 0;  // will be returned if any other button than left or right
        do {
                do { SDL_WaitEvent(&event); } while (event.type != SDL_MOUSEBUTTONDOWN);
                long st = d->getTimerValue();
                long et = st ;

                if (event.button.button == SDL_BUTTON_LEFT) {
                        i = 0 ;
                        report = true ;
                        while( et-st < 300000){
                                et = d->getTimerValue() ;
                                if(myCheckForMouseClick()==1) return 2 ;
                        }
                }
                if (event.button.button == SDL_BUTTON_RIGHT) {
                        i = 1 ;
                        report = true ;
                }

        }while(!report) ;

        return i ;

}

int getNext(int i){

        int r = 1 ;
        if(i == 0 || i == 2) return 1 ;
        if(i==1) return 0;
        return r ;
}

extern "C" int main(const int argc, char** argv)
{

        MYLOGVERB = LOG_INFO;  // suppress debug messages
        //let's push the initial value for the parameters
        argMap["experiment"]="sorting_strings_in_working_memory-with-somatosensory-task";
        argMap["logfile"]="psycho-wm-m.psy" ;
        argMap["string-size"]="5" ;
        argMap["test-rounds"]="10";
        argMap["subject"]="" ;
        argMap["memo"]="" ;
        argMap["digit-onset"]="10" ;
        argMap["alphabet"]="0123456789";
        argMap["metric-threshold"]="10" ;
        argMap["maintain-prb"]="0.2" ;
        argMap["min-reaction-time"]="1000000" ;
        argMap["max-miss"]="0" ;
        argMap["interrupt-time-range"]= "500000-5000000";
        argMap["cue-wait-frames"]="0" ;
        argMap["mask"]="n" ;
        argMap["mask-onset-frames"]="0";
        argMap["cue-onset-frames"] = "3" ;
        argMap["sound-dir"]="..";
        argMap["audio-file"]="audio.wav";
        argMap["pause-frames"]="5" ;
        argMap["max-false-click"]="0";
        manager.addSubComponent(d);
        manager.addSubComponent(motor) ;
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        d->setEventLog(el);
        //mmanager.parseCommandLine(argc, argv, "", 0,0);
        //mmanager.start();

        if (manager.parseCommandLine(argc, argv,
            "at least one argument needed", 1, -1)==false){
                    cout<<getUsageComment()<<endl;
                    return(1);
            }

            for(uint i = 0 ; i < manager.numExtraArgs() ; i++){
                    addArgument(manager.getExtraArg(i),std::string("=")) ;
            }

            manager.setOptionValString(&OPT_EventLogFileName, argMap["logfile"]);

            if( Mix_OpenAudio( 22050, MIX_DEFAULT_FORMAT, 2, 4096 ) == -1 ){
                    LINFO( "did not open the mix-audio") ;
                    return -1 ;
            }
            string noisestr = argMap["sound-dir"] + "/" + argMap["audio-file"];
            audio_chunk = Mix_LoadWAV(noisestr.c_str());
            if( audio_chunk == NULL  )
            {
                    LINFO("did not find the indicated wav files!");
                    return -1;
            }


  // let's get all our ModelComponent instances started:
            manager.start();
            for(map<string,string>::iterator it= argMap.begin(); it!= argMap.end() ; ++it) d->pushEvent("arg:"+ it->first+" value:"+it->second ) ;
            SDL_Rect offs ; offs.x = 0 ; offs.y = 0 ;

  // let's display an ISCAN calibration grid:
            d->clearScreen();
            d->displayISCANcalib();
            d->waitForMouseClick();
            d->displayText("Here the experiment starts! click to start!");
            d->waitForMouseClick();
            d->clearScreen();
        //let's see in what mode the user like to run the program

            int numOfTests = atoi(argMap["test-rounds"].c_str()) ;
            int stringSize = atoi(argMap["string-size"].c_str());
            int meticThreshold = atoi(argMap["metric-threshold"].c_str()) ;
            int onsetDel = atoi(argMap["digit-onset"].c_str()) ;
            float memPrb = atof(argMap["maintain-prb"].c_str()) ;
            int max_miss = atoi(argMap["max-miss"].c_str());
            int max_wrong = atoi(argMap["max-false-click"].c_str()) ;
            int in  = argMap["interrupt-time-range"].find("-") ;
            long iub = atol(argMap["interrupt-time-range"].substr(in+1).c_str());
            long ilb = atol(argMap["interrupt-time-range"].substr(0,in).c_str()) ;
            long min_reaction_time = atol(argMap["min-reaction-time"].c_str()) ;
            int cue_onset_frames = atoi(argMap["cue-onset-frames"].c_str()) ;
            int cue_wait_frames = atoi(argMap["cue-wait-frames"].c_str()) ;
            int mask_onset_frames = atoi(argMap["mask-onset-frames"].c_str()) ;
            int pause_frames = atoi(argMap["pause-frames"].c_str());
            vector<int> speedVector;
            speedVector.push_back(110) ;
            speedVector.push_back(-110) ;
            speedVector.push_back(150) ;
            Mix_PlayChannel( -1, audio_chunk , 500 ) ;////Mix_Pause(-1);
            int cr = 0 ;
            int fm =0 ; //keeps number of failed memory tests
            int sm = 0 ; //keeps number of successful memory tests
            int ssfr=0 ; //keeps number of successful sorts but failed reactions
            int fssr=0 ; //keeps number of failed sorting and suuccessful reaction
            int sssr=0 ; //keeps number of successful sorting and successful reaction
            int fsfr=0 ; //keeps number of failed sorting and failed reactions
            vector<long> correctAnswersTiming;
            vector<long> incorrectAnswersTiming;
            vector<long> allTiming ;
            while( cr <numOfTests ){
                    float mP = (float) rand()/RAND_MAX  ;
                    if(mP > memPrb){

                            d->pushEvent("**************************************") ;
                            d->showCursor(true);
                            d->displayText("click one of the  mouse buttons to start!");
                            d->waitForMouseClick() ;
                            d->showCursor(false) ;
                            string testString ;
                            vector<long> reactionTimes ;
                            testString = getARandomString(stringSize, argMap["alphabet"] , meticThreshold);// digitMemorizationTask(stringSize, argMap["alphabet"] , wr , hr , onsetDel) ;
                            d->clearScreen() ;
                            d->displayFixationBlink();
                            d->displayText(testString,true,0) ;
                            d->waitFrames(onsetDel) ;
                            d->clearScreen() ;
                            if(argMap["mask"].compare("y")==0) showMask(mask_onset_frames,argMap["alphabet"]);
                            if(cue_wait_frames != 0) d->waitFrames(cue_wait_frames) ;
                            d->displayRedDotFixation();
                            d->waitFrames(cue_onset_frames);
                            int cs = 0 ;//this is the index of the speed presented initially starts from 0
                            //Mix_Resume(-1) ;
                            motor->setMotor(speedVector[cs]) ;
                            long st = d->getTimerValue() ;
                            d->clearScreen() ;
                            d->pushEvent("manipulation starts");
                            d->pushEvent("the sequence for manipulation is : "+testString) ;
                            long tst =0 ;
                            long tet =0 ;
                            long dst = 0 ;
                            long det = 0 ;
                            long dl = 0 ;
                            dl = ilb+ random() % (iub-ilb);//we get a value for next stop
                            dst = d->getTimerValue() ;
                            det = dst ;
                            int missed = 0 ;
                            bool exitFlag = false ;
                            bool clickFlag = false ;
                            int ms = -1 ;//holds  mouseclick status
                            int wrongclick = 0 ;
                            while( !exitFlag ){

                                    if (det - dst > dl ){
                                            cs = getNext(cs) ;//choose the index of next speed
                                            motor->setMotor(0) ;
                                            d->waitFrames(pause_frames);
                                            motor->setMotor(speedVector[cs]) ;
                                            tst = d->getTimerValue() ;
                                            det=tst ;
                                            if(clickFlag){
                                                    missed++ ;
                                                    d->pushEvent("missed one change");
                                            }
                                            clickFlag=true ;
                                            dst = det ;
                                            dl = ilb+ random() % (iub-ilb);//we get a value for next stop
                                    }
                                    ms = myCheckForMouseClick() ;
                                    if(ms==2) exitFlag = true ;
                                    det = d->getTimerValue() ;

                                    if(clickFlag && ms==1){
                                            clickFlag = false ;
                                            tet = d->getTimerValue() ;
                                            reactionTimes.push_back(tet-tst);
                                            d->pushEvent("reaction time :" + stringify(tet-tst));
                                    }else{if(ms==1) wrongclick++ ;}
                            }

                            long et = d->getTimerValue();
                            motor->setMotor(0) ;
                            //Mix_Pause(-1);
                            d->pushEvent("manipulation ends") ;
                            d->pushEvent("maniupulation time : "+stringify(et-st)) ;allTiming.push_back(et-st);
                            d->clearScreen();
                            string  answer = getDigitSequenceFromSubject(argMap["alphabet"] , testString.size());

                            bool af = false ;
                            af = isAnswerCorrect(testString,answer,1);
                            d->pushEvent("subject keyed : "+answer);
                            d->pushEvent("avarage reaction time : "+ stringify(getAvarage(reactionTimes))) ;
                            d->pushEvent("number of missed events : "+stringify(missed));
                            d->pushEvent("number of caught events : "+stringify(reactionTimes.size())) ;
                            if(missed <= max_miss && getAvarage(reactionTimes)<= min_reaction_time && wrongclick <= max_wrong){
                                    cr++;
                                    d->pushEvent("valid trial");
                                    if(af){
                                            d->pushEvent("answer was correct");sssr++;correctAnswersTiming.push_back(et-st) ;
                                    }else{
                                            d->pushEvent("answer was incorrect");fssr++; incorrectAnswersTiming.push_back(et-st);
                                    }
                            }else{
                                    if(wrongclick > max_wrong) {
                                                d->displayText("Trial failed, too many FALSE REPORT! Click to start over!");
                                                d->waitForMouseClick();
                                        }
                                    if(missed > max_miss) {
                                            d->displayText("Trial failed, too many events missed! Click to start over!");
                                            d->waitForMouseClick();
                                    }
                                    if(getAvarage(reactionTimes) > min_reaction_time){
                                            d->displayText("Trial failed, reaction slower than limit! Click to start over!");
                                            d->waitForMouseClick();
                                    }
                                    if(af){
                                            d->pushEvent("answer was correct");ssfr++;
                                    }else{
                                            d->pushEvent("answer was incorrect");fsfr++;
                                    }
                                    d->pushEvent("invalid trial");
                            }

                    }else{
                            d->pushEvent("+++++++++++++++++++++++++++++++++++++++") ;
                            d->showCursor(true);
                            d->displayText("click one of the  mouse buttons to start!");
                            d->waitForMouseClick() ;
                            d->showCursor(false) ;
                            string testString ;
                            testString = getARandomString(stringSize, argMap["alphabet"] , meticThreshold);// digitMemorizationTask(stringSize, argMap["alphabet"] , wr , hr , onsetDel) ;
                            d->clearScreen() ;
                            d->displayFixationBlink();
                            d->displayText(testString,true,0) ;
                            d->waitFrames(onsetDel) ;
                            d->clearScreen() ;
                            if(argMap["mask"].compare("y")==0) showMask(mask_onset_frames,argMap["alphabet"]);
                            if(cue_wait_frames != 0) d->waitFrames(cue_wait_frames) ;
                            d->displayFixationBlink(d->getWidth()/2,d->getHeight()/2,1,3);
                            d->pushEvent("the memorization sequence is : "+testString) ;

                            d->pushEvent("subject is being challenged for simple momorization");
                            d->clearScreen();

                            string  answer = getDigitSequenceFromSubject(argMap["alphabet"] , testString.size());

                            bool af = false ;
                            af = isAnswerCorrect(testString,answer,0);
                            d->pushEvent("subject keyed : "+answer);
                            if(af){
                                    d->pushEvent("correct answer");
                                    sm++;
                            }else{
                                    d->pushEvent("incorrect answer");
                                    fm++ ;
                            }

                    }
            }

            d->pushEvent("number of successful memory tests :" + stringify(sm));
            d->pushEvent("number of failed memory tests :" + stringify(fm)) ;
            d->pushEvent("number of successful sorting and successful reaction trials :"+stringify(sssr));
            d->pushEvent("number of successful sorting and failed reaction trials :"+stringify(ssfr));
            d->pushEvent("number of failed sorting and successful reaction trials :"+stringify(fssr));
            d->pushEvent("number of failed sorting and failed reaction trials :"+stringify(fsfr));
            d->pushEvent("avarage time for respond :"+ stringify(getAvarage(allTiming)));
            d->pushEvent("variance of respond :"+ stringify(getVariance(allTiming)));
            d->pushEvent("avarage time for correct answers "+stringify(getAvarage(correctAnswersTiming))) ;
            d->pushEvent("variance of correct responds :"+ stringify(getVariance(correctAnswersTiming)));
            d->clearScreen();
            d->displayText("Experiment complete. Thank you!");
            d->waitForMouseClick();
            Mix_FreeChunk( audio_chunk );
            Mix_CloseAudio();
        // stop all our ModelComponents
            manager.stop();


        // all done!
            return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

