/*!@file AppPsycho/psychoWMS.C Psychophysics test to measure the influence of eyemovement on memory task performance */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psychoWMS.C $
// $Id: psychoWMS.C 12962 2010-03-06 02:13:53Z irock $
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

ModelManager manager("Psycho-WM");
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

        com += "\nlogfile=[logfilename.psy] {default = psycho-wm-subtraction.psy}\n" ;
        com += "\nmemo=[a_string_without_white_space]\n";
        com += "\nrange=[x-y](the size of string){default=200-500} \n";
        com += "\nsubject=[subject_name] \n" ;
        com += "\ndelay=[>0] (number of micro seconds){default=30000000}\n";
        com += "\ntest-rounds=[>1] (number of tests ) {default=5}\n";
        com += "\ndigit-onset=[>1] (number of frames that the string will remain onset ){default=10}\n";
        com += "\nsubtraction-step=[>0] (the subtraction number){default=3} ";
        return com ;
}


extern "C" int main(const int argc, char** argv)
{

        MYLOGVERB = LOG_INFO;  // suppress debug messages
        //let's push the initial value for the parameters
        argMap["experiment"]="subtraction-working-memory";
        argMap["logfile"]="psycho-wm-subtraction.psy" ;
        argMap["string-size"]="5" ;
        argMap["test-rounds"]="5";
        argMap["subject"]="" ;
        argMap["memo"]="" ;
        argMap["digit-onset"]="10" ;
        argMap["range"]="200-500";
        argMap["delay"]="30000000" ;
        argMap["subtraction-step"]="3" ;
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

            int numOfTests = atoi(argMap["test-rounds"].c_str()) ;
            int onsetDel = atoi(argMap["digit-onset"].c_str()) ;
            long opDelay = atol(argMap["delay"].c_str());
            int in = argMap["range"].find("-") ;
            int rangeU = atoi( argMap["range"].substr(in+1).c_str()) ;
            int rangeL = atoi(argMap["range"].substr(0,in).c_str()) ;
            int step = atoi(argMap["subtraction-step"].c_str());
        //let's count the rounds
            d->displayText("CLICK LEFT button to calibrate; RIGHT to skip");
            int cl = d->waitForMouseClick();
            if (cl == 1) d->displayEyeTrackerCalibration(3,5,1 , true);
            d->clearScreen();
            int cr = 0 ;
            while( cr <numOfTests ){

                            cr++;
                            d->pushEvent("**************************************") ;
                            d->showCursor(true);
                            d->displayText("click one of the  mouse buttons to start!");
                            d->waitForMouseClick() ;
                            d->showCursor(false) ;
                            int initialNum = rangeL+ random()%(rangeU-rangeL) ;
                            d->clearScreen() ;
                            d->pushEvent("the initial number is: "+stringify(initialNum)) ;
                            d->displayFixationBlink();
                            d->displayText(stringify(initialNum),true,0) ;
                            d->waitFrames(onsetDel) ;
                            d->clearScreen() ;
                            d->displayFixationBlink(d->getWidth()/2,d->getHeight()/2,2,3);
                            d->pushEvent(std::string("===== Showing image: def.mpg ====="));
                            eyet->track(true);
                            d->pushEvent("manupulation starts") ;
                            long st = d->getTimerValue();
                            long et = st ;
                            while( et - st < opDelay ){
                                et = d->getTimerValue();
                            }
                            usleep(50000);
                            eyet->track(false);
                            d->pushEvent("manipulation ends") ;
                            string  answer = getDigitSequenceFromSubject("0123456789" , 3);
                            d->pushEvent("the reported number: "+answer);
                            int finalNum = atoi(answer.c_str()) ;
                            d->pushEvent("number of operations: "+stringify((initialNum-finalNum)/step));
            }

            d->clearScreen();
            d->displayText("Experiment complete. Thank you!");
            d->waitForMouseClick();

        // stop all our ModelComponents
            manager.stop();


        // all done!
            return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

