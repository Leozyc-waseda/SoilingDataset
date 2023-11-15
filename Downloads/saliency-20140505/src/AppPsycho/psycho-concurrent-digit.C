/*!@file AppPsycho/psycho-concurrent-digit.C Psychophysics display of still images for concurrent task */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-concurrent-digit.C $
// $Id: psycho-concurrent-digit.C 10794 2009-02-08 06:21:09Z itti $
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
map<uint,uint> testMap ;
map<string,string> argMap ;
map<string,vector<SDL_Rect*>*> clipsmap;

//////////////////////////////////////////////

// a functionf for stringigying things
template <class T> std::string stringify(T i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}


//////////////////////////////////////////////
void getMouseEvent(){
        bool quit = false ;
        SDL_Event event ;
         while( quit == false ) {
                while( SDL_PollEvent( &event ) ) {
                        if( event.type == SDL_MOUSEBUTTONDOWN  ) { quit = true; }
                }
        }
}
//pushes back the name of files in the directory into the given vector
int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    string fn = "" ;
    size_t found;
    string extension = "" ;
    while ((dirp = readdir(dp)) != NULL) {
        fn = string(dirp->d_name) ;
        found = fn.find_last_of(".");
        if(found > 0 && found <1000){
                extension = fn.substr(found) ;
                if(extension.compare(".png")== 0 || extension.compare(".jpg")==0 )
                files.push_back(dir+"/"+fn);
        }
    }
    closedir(dp);
    return 0;
}

bool itIsInThere(int x , vector<int> bag){
        for( uint i=0 ; i < bag.size(); i++ ){
          if(x == bag[i]) return true ;
        }
        return false ;
}

////////////////////////////////////////////////////////
///// simply generates a sequence of digits displayed on random places on the screen
////////////////////////////////////////////////////////
string digitMemorizationTask(uint l, int maxForDigit=10 ,float wp = 1.0f, float hp = 1.0f, int displayFrame = 10 , int delayFrame = 30 ){
  d->clearScreen() ;
  vector<int> pickedones = vector<int>() ;
  string test = string("") ;
  string tp = string("") ;
  int widthRange = (int)((float)(d->getWidth()) * wp);
  int heightRange = (int)((float)(d->getHeight()) * hp);
  int wMargin = (d->getWidth() - widthRange)/2 ;
  int hMargin = (d->getHeight() - heightRange) / 2 ;
  for(uint i = 0 ; i < l ; i++){
    int nd;
    do{ nd= rand()% maxForDigit ; }while(itIsInThere(nd,pickedones) && pickedones.size() < 11) ;
    pickedones.push_back(nd);
    tp = stringify(nd) ;
//     int x = (rand()%widthRange)  + wMargin;
//     int y = (rand()%heightRange ) + hMargin ;
    //d->displayText(tp,Point2D<int>(x,y),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
    test += tp ;
    //d->waitFrames(displayFrame) ;
    d->clearScreen() ;
    //if(i<l-1)
    //d->waitFrames(delayFrame) ;
  }

        int x = (rand()%widthRange)  + wMargin;
        int y = (rand()%heightRange ) + hMargin ;
        d->displayText(test,Point2D<int>(x,y),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
        d->waitFrames(displayFrame*l) ;
        return test ;
}


////////////////////////////////////////////////////////
///// simply generates a sequence of digits displayed on random places on the screen
////////////////////////////////////////////////////////
string unlimitedDigitMemorizationTask(int maxForDigit=10 ,float wp = 1.0f, float hp = 1.0f, int displayFrame = 30 , int delayFrame = 30){
        vector<int> pickedones = vector<int>() ;

        string test = string("") ;
        string tp = string("") ;
        int widthRange = (int)((float)(d->getWidth()) * wp);
        int heightRange = (int)((float)(d->getHeight()) * hp);
        int wMargin = (d->getWidth() - widthRange)/2 ;
        int hMargin = (d->getHeight() - heightRange) / 2 ;
        //bool flag = true ;
        SDL_Event event;


        while( true ){
                d->clearScreen() ;
                while( SDL_PollEvent(&event) != 0 ){
                        if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) return test ;
                }
                d->waitFrames(delayFrame) ;
                int nd;
                do{ nd= rand()% maxForDigit ; }while(itIsInThere(nd,pickedones) && pickedones.size() < 11) ;
                pickedones.push_back(nd);
                tp = stringify(nd) ;
//                 int x = (rand()%widthRange)  + wMargin;
//                 int y = (rand()%heightRange ) + hMargin ;
//                 d->displayText(tp,Point2D<int>(x,y),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
                test += tp ;
//                 d->waitFrames(displayFrame) ;
//                 d->clearScreen() ;
        }
        int x = (rand()%widthRange)  + wMargin;
        int y = (rand()%heightRange ) + hMargin ;
        d->displayText(test,Point2D<int>(x,y),PixRGB<byte>(0,0,0),PixRGB<byte>(128,128,128),true) ;
        d->waitFrames(displayFrame) ;
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

SDL_Surface* getKeyPad(map<string , SDL_Rect>& buttmap){
  SDL_Surface* pad= getABlankSurface(d->getWidth()/4,d->getHeight()/3);
  SDL_Rect clip;
  clip.x=0;
  clip.y=0;
  clip.w= pad->w / 3 ;
  clip.h = pad->h / 4 ;
  //keys for 1 to 9
  for( int i = 1 ; i < 10 ; i++){
    SDL_Surface* but = getButtonImage(stringify(i),PixRGB<byte>(0,0,0),PixRGB<byte>(255,255,255),Point2D<int>(pad->w / 3 , pad->h / 4),PixRGB<byte>(255, 98 , 25),3);
    SDL_Rect cl ;
    cl.x = ((i-1)%3)*(pad->w)/3 ; cl.y= ((i-1)/3)*((pad->h)/4) ;
    cl.w = clip.w ;
    cl.h = clip.h ;
    apply_surface( cl.x , cl.y ,*but,*pad,clip);
    buttmap[stringify(i)] = cl ;
    dumpSurface(but);
  }
  SDL_Rect cl1 ;
  cl1.x = 0 ; cl1.y= 3*((pad->h)/4) ;
  cl1.w = clip.w ;
  cl1.h = clip.h ;
  buttmap["z"] = cl1 ;
  SDL_Surface* but = getButtonImage(string("<-"),PixRGB<byte>(0,0,0),PixRGB<byte>(255,255,255),Point2D<int>(pad->w / 3 , pad->h / 4),PixRGB<byte>(255, 98 , 25),3);
  apply_surface(0, 3*((pad->h)/4),*but,*pad,clip);
  dumpSurface(but);
  SDL_Rect cl2 ;
  cl2.x = (pad->w)/3 ; cl2.y= 3*((pad->h)/4) ;
  cl2.w = clip.w ;
  cl2.h = clip.h ;
  buttmap["0"] = cl2 ;
  but = getButtonImage(string("0"),PixRGB<byte>(0,0,0),PixRGB<byte>(255,255,255),Point2D<int>(pad->w / 3 , pad->h / 4),PixRGB<byte>(255, 98 , 25),3);
  apply_surface((pad->w)/3, 3*((pad->h)/4),*but,*pad,clip);
  dumpSurface(but);
  SDL_Rect cl3 ;
  cl3.x = 2*(pad->w)/3 ; cl3.y= 3*((pad->h)/4) ;
  cl3.w = clip.w ;
  cl3.h = clip.h ;
  buttmap["o"] = cl3 ;
  but = getButtonImage(string("Ok"),PixRGB<byte>(0,0,0),PixRGB<byte>(255,255,255),Point2D<int>(pad->w / 3 , pad->h / 4),PixRGB<byte>(255, 98 , 25),3);
  apply_surface(2*(pad->w)/3, 3*((pad->h)/4),*but,*pad,clip);
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
string getDigitSequenceFromSubject(uint maxl = 7 ){
        d->showCursor(true) ;
        //let's creat a map to map actions to regions of the screen, each region is represented as an SDL_Rect
        map<string , SDL_Rect>* buttmap = new map<string , SDL_Rect>();
        //now let's get the keypad surface while we get the actions map to regions
        SDL_Surface * keypad = getKeyPad(*buttmap);
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
        while( tp.compare("o")!=0 ){
                //this button is actually the display for the current string
                SDL_Surface* dp = getButtonImage(p ,PixRGB<byte>(195,60,12) ,PixRGB<byte>(255,255,255) ,Point2D<int>(d->getWidth()/6,d->getHeight() /15) ,PixRGB<byte>(0,25,180) , 4) ;
                SDL_Rect offs ; offs.x = (d->getWidth() - dp->w) /2 ; offs.y = d->getHeight()/6 ;
                d->displaySDLSurfacePatch(dp , &offs , NULL , -2 , false ,true ) ;
                //now let's listen to button events
                tp = getPressedButtonCommand(*buttmap,Point2D<int>(offset.x,offset.y)) ;
                dumpSurface(dp) ;
                if(tp.compare("z")==0 && p.size()>=0 ) {
                        if (p.size()>0) p = p.substr(0,p.size()-1) ;
                }else{
                  if(p.size() < maxl && tp.compare("o")!=0) {
                                p +=tp ;
                        }

                }

        }
        buttmap = 0 ;
        dumpSurface(keypad) ;
        d->clearScreen() ;
        return p ;

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

        com += "\ndelay-interval=[>1]:[>1] (the interval for random delay after stroop task) {default=10}\n";
        com += "\nfixation-blink=[y/n] (show the blink for fixation after stroop task or no) {defaule y}\n";
        com += "\nimage-dir=[path to image directory] (needed for mode 2 and 3) {default=60:160} \n"  ;
        com += "\nlogfile=[logfilename.psy] {default = psycho-stroop-concurrent.psy}\n" ;
        com += "\nmemo=[a_string_without_white_space]\n";
        com += "\nmode=[1..5] (1 memorization only , 2 dynamic memorization only (ending with evens ascending and ending with odds discending)"
                        ", 3 memorization + image display , 4 for dynamic memorization + image display "
                        "5 for image display only ) {default=1}\n";
        com += "\ndigit-test-size=[<1](number of words in the stroop task){default=6} \n" ;
        com += "\nsubject=[subject_name] \n" ;
        com += "\ntest-rounds=[>1] (needed for mode1) {default=5}\n";
        com += "\ndigit-onset=[>1] (number of frames that the digit will remain onset per digit){default=10}\n";
        com += "\ndigit-offset=[>1](number of frames that between two consequent digit onset){default=30}\n";
        com += "\nwidth-range=[1..100](the percentage of the width of screen for showing the digits){default=100}\n";
        com += "\nheight-range=[1..100](the percentage of the height of screen for showing the digits){default=100}\n";
        com += "\nblink-num=[>1](number of blinking of fixation cross){default=3} \n";
        com += "\nblink-delay=[>1](delay for blinking cross apprearnce on the screen){default=2}\n";
        com += "\ncalib-refresh-period=[>1](number of tests between two calibration){default=12} \n";
        return com ;
}


extern "C" int main(const int argc, char** argv)
{

          MYLOGVERB = LOG_INFO;  // suppress debug messages
        //let's push the initial value for the parameters
        argMap["mode"] = "1" ;
        argMap["logfile"]="psycho-digit-concurrent.psy" ;
        argMap["digit-test-size"]="6" ;
        argMap["image-dir"]="..";
        argMap["test-rounds"]="5";
        argMap["delay-interval"]="60:160" ;
        argMap["subject"]="" ;
        argMap["memo"]="" ;
        argMap["fixation-blink"]="y" ;
        argMap["digit-onset"]="10" ;
        argMap["digit-offset"]="30" ;
        argMap["width-range"]="100" ;
        argMap["height-range"]="100" ;
        argMap["blink-num"]="3" ;
        argMap["blink-delay"]="2" ;
        argMap["calib-refresh-period"] = "12" ;
        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
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
          // hook our various babies up and do post-command-line configs:
        nub::soft_ref<EyeTracker> et = etc->getET();
        d->setEyeTracker(et);
        d->setEventLog(el);
         et->setEventLog(el);


  // let's get all our ModelComponent instances started:
          manager.start();
        for(map<string,string>::iterator it= argMap.begin(); it!= argMap.end() ; ++it) d->pushEvent("arg:"+ it->first+" value:"+it->second ) ;
  // let's display an ISCAN calibration grid:
          d->clearScreen();
          d->displayISCANcalib();
          d->waitForMouseClick();

  // let's do an eye tracker calibration:
          d->displayText("CLICK LEFT button to calibrate; RIGHT to skip");
          int c = d->waitForMouseClick();
          if (c == 1) d->displayEyeTrackerCalibration(3,5,1 , true);

          d->clearScreen();
          int calibPeriod = atoi(argMap["calib-refresh-period"].c_str());

        int blinkNum = atoi(argMap["blink-num"].c_str());
        int blinkDelay = atoi(argMap["blink-delay"].c_str());
        //let's see in what mode the user like to run the program
        int mode = atoi(argMap["mode"].c_str()) ;
        //mode 1 and 2 inlude single memorization task, 1 for just memorization and 2 for dynamic memorization
        //in dynamic memorization the subject has to sort the string in ascending order when the last digit is even
        //and descending order if the last digit is odd
        if(mode==1 || mode==2 ){
          int numOfTests = atoi(argMap["test-rounds"].c_str()) ;
          int stringSize = atoi(argMap["digit-test-size"].c_str()) ;
          string::size_type position = argMap["delay-interval"].find(":");
          int minDel = atoi(argMap["delay-interval"].substr(0,position).c_str()) ;
          int maxDel = atoi(argMap["delay-interval"].substr(position+1).c_str()) ;
          float wr = ((float)(atoi(argMap["width-range"].c_str())))/100.0f;
          float hr = ((float)(atoi(argMap["height-range"].c_str())))/100.0f;
          int onsetDel = atoi(argMap["digit-onset"].c_str()) ;
          int offsetDel = atoi(argMap["digit-offset"].c_str()) ;

          for(int i = 0 ; i < numOfTests ; i++){
            d->pushEvent("**************************************") ;
            d->showCursor(true);
            d->displayText("click one of the  mouse buttons to start!");
            d->waitForMouseClick() ;
            d->showCursor(false);
           // d->clearScreen();
            string  testString ;
            if(stringSize == 0 ){
                    testString = unlimitedDigitMemorizationTask( 10 , wr , hr ,  onsetDel ,  offsetDel);
            }else{
                    testString = digitMemorizationTask(stringSize, 10 , wr , hr ,  onsetDel ,  offsetDel) ;
            }
            //
            d->pushEvent("the memorization sequence is : "+testString) ;
            d->waitFrames((rand()%(maxDel - minDel)) +minDel);
            string  answer = getDigitSequenceFromSubject(testString.size());
            bool af = false ;
            if(mode==1){
              af = isAnswerCorrect(testString,answer,0);
            }else{
              int m = atoi(testString.substr(testString.size()-1).c_str()) % 2 ;
              switch( m ){
                case 0 : af = isAnswerCorrect(testString , answer , 1 ) ; break ;
                case 1 : af = isAnswerCorrect(testString , answer , 2 ) ; break ;
              }

            }
            d->pushEvent("subject keyed : "+answer);
            if(af){
              d->pushEvent("answer was correct");
            }else{
              d->pushEvent("answer was incorrect");
            }

          }
        }

        //mode 3 and 4 is reserved for concurrent task; 3 for simple memorization + free viewing
        //; 4 for dynamic memorization + free viewing again in dynamic one . Again for strings ending with odd numbers descending
        //sorting is required while for strings ending with even digits ascending order is required
        if(mode == 3 || mode == 4){
                string::size_type position = argMap["delay-interval"].find(":");
                int minDel = atoi(argMap["delay-interval"].substr(0,position).c_str()) ;
                int maxDel = atoi(argMap["delay-interval"].substr(position+1).c_str()) ;
                int stringSize = atoi(argMap["digit-test-size"].c_str()) ;
                string dir = argMap["image-dir"];
                vector<string> files = vector<string>();
                getdir(dir,files);
                float wr = ((float)(atoi(argMap["width-range"].c_str())))/100.0f;
                float hr = ((float)(atoi(argMap["height-range"].c_str())))/100.0f;
                int onsetDel = atoi(argMap["digit-onset"].c_str()) ;
                int offsetDel = atoi(argMap["digit-offset"].c_str()) ;
                int c = 0 ;
                while(files.size()>0){
                        c++ ;
                        if(c%calibPeriod == 0){
// let's do an eye tracker calibration:
                                d->displayText("CLICK LEFT button to calibrate; RIGHT to skip");
                                int cl = d->waitForMouseClick();
                                if (cl == 1) d->displayEyeTrackerCalibration(3,5,1 , true);
                                d->clearScreen();
                        }
                        int imageIndex = rand()%files.size() ;
                        SDL_Surface *surf = load_image(files[imageIndex]);
                        float r1 = (float)d->getWidth()/(float)surf->w ;
                        float r2 = (float)d->getHeight() / (float)surf->h ;
                        surf = SDL_Resize(surf ,min(r1,r2) , 5 );
                        SDL_Rect offset;
                        offset.x = (d->getWidth() - surf->w) /2;
                        offset.y = (d-> getHeight() - surf->h) /2;
                        d->pushEvent("**************************************") ;
                        d->showCursor(true);
                        d->displayText("click a mouse button to start!");
                        d->waitForMouseClick() ;
                        d->showCursor(false);
                        string  testString ;
                        if(stringSize == 0 ){
                                testString = unlimitedDigitMemorizationTask( 10 , wr , hr ,  onsetDel ,  offsetDel);
                        }else{
                                testString = digitMemorizationTask(stringSize, 10 , wr , hr ,  onsetDel ,  offsetDel) ;
                        }
                        d->pushEvent("the memorization sequence is : "+testString) ;

                        d->clearScreen() ;
                        d->waitNextRequestedVsync(false, true);
                        d->pushEvent(std::string("===== Showing image: ") +
                                           files[imageIndex] + " =====");
                        // start the eye tracker:
                        et->track(true);
                              //blink the fixation:
                        if(argMap["fixation-blink"].compare("y")==0) d->displayFixationBlink(-1,-1,blinkNum,blinkDelay);
                        d->pushEvent("image will go up");
                        d->displaySDLSurfacePatch(surf , &offset,NULL , -2,false, true);
                        d->waitFrames((rand()%(maxDel-minDel)) +minDel );
                        d->clearScreen() ;
                        d->pushEvent("image is down");
                        dumpSurface(surf);
                        // stop the eye tracker:
                        usleep(50000);
                             et->track(false);
                        //see if the subject was looking at the screen!
                        string  answer = getDigitSequenceFromSubject(testString.size());
                        bool af = false ;
                        if(mode==3){
                          af = isAnswerCorrect(testString,answer,0);
                        }else{
                          int m = atoi(testString.substr(testString.size()-1).c_str()) % 2 ;
                          switch( m ){
                            case 0 : af = isAnswerCorrect(testString , answer , 1 ) ; break ;
                            case 1 : af = isAnswerCorrect(testString , answer , 2 ) ; break ;
                          }

                        }
                        d->pushEvent("subject keyed : "+answer);
                        if(af){
                          d->pushEvent("answer was correct");
                        }else{
                          d->pushEvent("answer was incorrect");
                        }
                        d->clearScreen() ;
                        files.erase(files.begin()+imageIndex);
                }
        }

        //mode 5 is reserved for simple free viewing task
        if(mode == 5){
                string::size_type position = argMap["delay-interval"].find(":");
                int minDel = atoi(argMap["delay-interval"].substr(0,position).c_str()) ;
                int maxDel = atoi(argMap["delay-interval"].substr(position+1).c_str()) ;
                string dir = argMap["image-dir"];
                vector<string> files = vector<string>();
                getdir(dir,files);
                int c = 0 ;

                while(files.size()>0){
                        c++ ;
                        if(c%calibPeriod == 0){
// let's do an eye tracker calibration:
                                  d->displayText("CLICK LEFT button to calibrate; RIGHT to skip");
                                  int cl = d->waitForMouseClick();
                                  if (cl == 1) d->displayEyeTrackerCalibration(3,5,1 , true);
                                  d->clearScreen();
                        }
                        int imageIndex = rand()%files.size() ;
                        //this block turned out to be buggy! so I decided to switch to working with SDL_Surface instead
                        //Image< PixRGB<byte> > image =
                                        //Raster::ReadRGB(files[imageIndex]);

                        //SDL_Surface *surf = d->makeBlittableSurface(image, true);
                        SDL_Surface *surf = load_image(files[imageIndex]);
                        float r1 = (float)d->getWidth()/(float)surf->w ;
                        float r2 = (float)d->getHeight() / (float)surf->h ;
                        surf = SDL_Resize(surf ,min(r1,r2) , 5 );
                        SDL_Rect offset;
                        offset.x = (d->getWidth() - surf->w) /2;
                        offset.y = (d-> getHeight() - surf->h) /2;
                        d->waitFrames(45);
                        d->waitNextRequestedVsync(false, true);
                        d->pushEvent(std::string("===== Showing image: ") +
                                           files[imageIndex] + " =====");

                        // start the eye tracker:
                        et->track(true);
                              //blink the fixation:
                        if(argMap["fixation-blink"].compare("y")==0) d->displayFixationBlink(-1,-1,blinkNum,blinkDelay);

                        d->displaySDLSurfacePatch(surf , &offset,NULL , -2,false, true);
                        d->waitFrames((rand()%(maxDel-minDel)) +minDel );
                        dumpSurface(surf);
                        // stop the eye tracker:
                        usleep(50000);
                             et->track(false);
                        d->clearScreen() ;
                        files.erase(files.begin()+imageIndex);
                }

        }

          d->clearScreen();
          d->displayText("Experiment complete. Thank you!");
          d->waitForMouseClick();
        //getEvent();
          // stop all our ModelComponents
          manager.stop();


          // all done!
          return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

