/*!@file AppPsycho/psycho-concurrent-still.C Psychophysics display of still for concurrent task images */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-auditory-concurrent.C $
// $Id: psycho-auditory-concurrent.C 10794 2009-02-08 06:21:09Z itti $
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
ModelManager manager("Psycho Auditory");
nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
map<uint,uint> testMap ;
map<string,string> argMap ;
Mix_Chunk *highchunk = NULL;
Mix_Chunk *mediumchunk = NULL;
Mix_Chunk *lowchunk = NULL;

//////////////////////////////////////////////

// a functionf for stringigying things
template <class T> std::string stringify(T i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}

//////////////////////////////////////////////

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

void getMouseEvent(){
        bool quit = false ;
        SDL_Event event ;
         while( quit == false ) {
                while( SDL_PollEvent( &event ) ) {
                        if( event.type == SDL_MOUSEBUTTONDOWN  ) { quit = true; }
                }
        }
}


vector<int> getAuditoryTask(string ts){
        //0 stands for the first highchunk (high pitch)
        //1 stands for the second chunk (medium pitch)
        //2 stands for the third chunk (low pitch)
        string::size_type position = ts.find(":");
        int minTaskSize = atoi(ts.substr(0,position).c_str()) ;
        int maxTaskSize = atoi(ts.substr(position+1).c_str()) ;
        int taskSize = rand()%((maxTaskSize-minTaskSize)/2);
        taskSize += (minTaskSize-1)/2 ;
        taskSize = 2*taskSize +1 ;
        vector<int> task = vector<int>() ;
        for(int i = 0 ; i < taskSize ; i++){
                int current_task = rand()%3;
                task.push_back(current_task) ;
        }

        return task ;
}


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
//this function is used for the memory task, it takes in two lists of file names, one as the target list and
//the second as test images, then it scramble the aggragated list and displays a random array of images, upon
//click on one image a fram apears around the image and by clicking on the framed image the image is selected
//as subject's choice, the choice and correctness will be logged
bool memoryTest(vector<string> dil , vector<string> cil){
        d->showCursor(true);
        bool flag = false ;
        //here we decide what is size of grid of images, number of rows and columns are equal
        int sf = (int)sqrt(dil.size()+cil.size()) ;
        if((uint)(sf*sf) < dil.size()+cil.size() ) sf++ ;
        float sizefactor = 1.0f/(float)sf ;
        //fls is the aggregated list of file names
        vector<string>* fls = new vector<string>() ;
        //let's put everything in fls
        for(uint i = 0 ; i < dil.size() ; i++){
                fls->push_back(dil[i]);
        }
        for(uint i = 0 ; i < cil.size() ; i++){
                fls->push_back(cil[i]);
        }
        // and then scramble the list in a random fashion
        scramble(*fls) ;
        //this surface is used for final display
        SDL_Surface* blank = getABlankSurface(d->getWidth(),d->getHeight()) ;
        //this surface keeps the original display images in the case that we need to refresh the
        //the display with original image this one will be copied to blank
        SDL_Surface* oblank =  getABlankSurface(d->getWidth(),d->getHeight()) ;
        d->displayText("Time for a memory test!",true, 1);
        //cout<<cil.size()<< "  "<<dil.size()<<"  "<<fls->size()<<endl;

        //this surface is used as the progress bar
        SDL_Surface* pbar = getABlankSurface((d->getWidth())/2,(d->getHeight())/25);
        SDL_Rect pbarOffset = SDL_Rect();
        pbarOffset.x = (d->getWidth() - pbar->w )/2 ;
        pbarOffset.y = (d->getHeight() - pbar->h)/2 ;
        d->displaySDLSurfacePatch(pbar , &pbarOffset,NULL , -2,false, true);

        //Now let's load up the images, resize them and put them in the right place of the display
        int c = 0 ;
        for(int i = 0 ; i < sf ; i++){
                for(int j = 0 ; j < sf ; j++){
                        if((uint)(j*sf + i) < fls->size()){
                                c++ ;
                                //let's load the image
                                SDL_Surface* ts = load_image((*fls)[j*sf + i]);
                                sizefactor = min(((float)(d->getWidth()))/sf/((float)(ts->w)),((float)(d->getHeight()))/sf/((float)(ts->h)));
                                //here we resize the image
                                ts = SDL_Resize(ts,sizefactor,5) ;
                                //cout<<(j*sf + i)<<" " <<(*fls)[j*sf + i]<<endl
                                SDL_Rect* clip = new SDL_Rect() ;
                                clip->x = 0 ;
                                clip->y = 0 ;
                                clip->w = ts->w ;
                                clip->h = ts->h ;
                                SDL_Rect offset;
                                offset.x = i*(d->getWidth())/sf + ((d->getWidth())/sf - ts->w)/2;
                                offset.y = j*(d->getHeight())/sf + ((d->getHeight())/sf - ts->h)/2;

                                //here we apply the surface patch on both copies of the displaying image
                                apply_surface(offset.x, offset.y,*ts,*blank ,*clip);
                                apply_surface(offset.x, offset.y,*ts,*oblank ,*clip);
                                //let's free up the pointers
                                dumpSurface(ts);
                                delete clip ;
                                //let's draw the progress bar
                                Uint32 color = d->getUint32color(PixRGB<byte>(255, 98 , 25));
                                fillRectangle(pbar,color,0,0,(int)((float)((j*sf + i + 1)*(pbar->w))/(float)(fls->size())) , pbar->h);
                                fillRectangle(pbar,color,0,0,(int)((float)c/float(fls->size())) , pbar->h);
                                 d->displaySDLSurfacePatch(pbar  , &pbarOffset,NULL , -2,false, true);
                        }


                }
        }
        //here we are ready for displaying the image so let's ask the subject to continue
        d->displayText("Click to start!",true);
        SDL_Rect offset;
        offset.x = 0;
        offset.y = 0;
        getMouseEvent() ;
        d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);
        uint quit = 0 ;
        SDL_Event event ;
        int fmx = 0 ;
        int fmy = 0 ;
        int smx = 0 ;
        int smy = 0 ;
        int choice1 = 0 ;
        int choice2 = 0 ;
        //here we interact with the subject in order to get the subject's choice
        while( quit != 2 ) {
                while( SDL_PollEvent( &event ) ) {

                  if( event.type == SDL_MOUSEBUTTONDOWN  && event.button.button == SDL_BUTTON_LEFT
                      && event.button.x > offset.x && event.button.x <= offset.x + blank->w
                      && event.button.y > offset.y && event.button.y <= offset.y + blank->h
                      && quit == 1
                    ) {
                        smx = event.button.x ;
                        smy = event.button.y ;
                        choice2 = (int)(sf*(smx-offset.x)/(d->getWidth())) + sf*(int)(sf*(smy-offset.y)/(d->getHeight())) ;
                        if(choice1 == choice2){
                                if((uint)choice1 < fls->size()){
                                        quit = 2 ;
                                        d->pushEvent("memory test choice : "+(*fls)[choice1]);

                                        for(uint f = 0 ; f < dil.size() ; f++){
                                                if((*fls)[choice1].compare(dil[f])==0) {
                                                        flag = true ;
                                                        break ;
                                                }
                                        }

                                        if (flag==true){
                                                d->pushEvent("memory test result : correct" );
                                        }else{
                                                d->pushEvent("memory test result : incorrect" );
                                        }

                                        cout<< "second choice "<< choice2 <<endl;
                                }else{
                                        quit = 3 ;
                                        SDL_Rect* clip = new SDL_Rect() ;
                                        clip->x = 0 ;
                                        clip->y = 0 ;
                                        clip->w = d->getWidth() ;
                                        clip->h = d->getHeight() ;
                                        apply_surface(0,0,*oblank,*blank,*clip) ;
                                        delete clip ;
                                        d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);

                                }

                        }else{
                                quit = 3 ;
                                SDL_Rect* clip = new SDL_Rect() ;
                                clip->x = 0 ;
                                clip->y = 0 ;
                                clip->w = d->getWidth() ;
                                clip->h = d->getHeight() ;
                                apply_surface(0,0,*oblank,*blank,*clip) ;
                                delete clip ;
                                d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);
                                }
                        }

                        if( event.type == SDL_MOUSEBUTTONDOWN  && event.button.button == SDL_BUTTON_LEFT
                                                 && event.button.x > offset.x && event.button.x <= offset.x + blank->w
                                                 && event.button.y > offset.y && event.button.y <= offset.y + blank->h
                                                 && quit == 0
                          ) {
                                SDL_Rect* clip = new SDL_Rect() ;
                                clip->x = 0 ;
                                clip->y = 0 ;
                                clip->w = d->getWidth() ;
                                clip->h = d->getHeight() ;
                                apply_surface(0,0,*oblank,*blank,*clip) ;
                                delete clip ;
                                quit = 1;
                                fmx = event.button.x ;
                                fmy = event.button.y ;
                                int i = sf*(fmx-offset.x)/(d->getWidth()) ;
                                int j = sf*(fmy-offset.y)/(d->getHeight());
                                cout<<"("<<sf*(fmx-offset.x)/(d->getWidth())<<","<<sf*(fmy-offset.y)/(d->getHeight())<<")"<<endl ;
                                choice1 = (int)(sf*(fmx-offset.x)/(d->getWidth())) + sf*(int)(sf*(fmy-offset.y)/(d->getHeight())) ;
                                cout<<"first choice:" << choice1 <<endl;
                                Uint32 color = d->getUint32color(PixRGB<byte>(255, 98 , 25));
                                drawRectangle(blank , color , i*(d->getWidth())/sf , j*(d->getHeight())/sf ,(d->getWidth())/sf  , (d->getHeight())/sf , 5) ;
                                d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);
                          }

                        if(quit==3) quit=0 ;

                }
        }


        //d->waitFrames(200 );
        dumpSurface(blank);
        dumpSurface(oblank);
        dumpSurface(pbar) ;
        delete fls ;
        return flag ;
}


void firstTask(const uint l){
        Uint32 green = d->getUint32color(PixRGB<byte>(0, 255, 0));
        Uint32 blue = d->getUint32color(PixRGB<byte>(0, 0, 255));
        Uint32 red = d->getUint32color(PixRGB<byte>(255,0,0));
        Uint32 color = 0 ;
        int c = 0 ;
        SDL_Rect offset;
        d->clearScreen() ;
        //Give the offsets to the rectangle
        SDL_Surface* blankSurface = getABlankSurface(d->getWidth(),d->getHeight()) ;
        offset.x = (d->getWidth() - blankSurface->w) /2;
        offset.y = (d-> getHeight() - blankSurface->h) /2;
        for (uint i = 0 ; i < l ; i++){
                c = rand()%3 ;
                switch(c){
                        case 0 : color = red ;break;
                        case 1 : color = green ; break ;
                        case 2 : color = blue ; break ;
                }

                fillQuadricRadiant(blankSurface,color,rand()%(blankSurface->w -100 ) + 50 ,rand()%(blankSurface->h -100)+50,20) ;
                d->displaySDLSurfacePatch(blankSurface , &offset,NULL , -2,false, true);
                d->waitFrames(10);
        }
        dumpSurface(blankSurface) ;
}

///////////////////////////////////////////////////////////////////
void volumeAdjustment(const int randBase , const string extraMessage = "continue" ){
        int rndBase = randBase+1 ;
        map<int,string> imap = map<int,string>();
        imap[0] = "high";
        imap[1] = "medium" ;
        imap[2] = "low" ;
        imap[3] = extraMessage ;
        d->clearScreen();
        d->pushEvent(std::string("voluem adjustment called"));
        d->displayText(std::string("click on the botton to play or cancel?"),true,1);
        //we need the subject be able to interact with the mouse
        d->showCursor(true);
        SDL_Surface* blank =getABlankSurface((d->getWidth())/2 , d->getHeight()/10);
        //we make a scrap image to write down the numbers on
        Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
        textIm.clear(PixRGB<byte>(182, 145, 225));
        //here we write the numbers
        for(int i = 0 ; i < rndBase ; i ++){
                writeText(textIm, Point2D<int>((blank->w)/(2*rndBase) -2,10+100*i), imap[i].c_str());
        }
        //and here we make a blittabl surface out of that
        SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
        //and take out differnt clips out of that
        SDL_Rect clip;
        clip.x = 0 ;
        clip.w = (blank->w)/rndBase - 2 ;
        clip.h =blank->h ;
        for(int i = 0 ; i < rndBase ; i++){
                clip.y = i*100 ;
                apply_surface(i*(clip.w + 2),0,*surf,*blank,clip);
        }

        SDL_Rect offset;
        offset.x = (d->getWidth() - blank->w) /2;
        offset.y = (d-> getHeight() - blank->h) /2;
        d->pushEvent("the button panel is going up");
        d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);

        bool quit = false ;
        SDL_Event event ;
        int mx = 0 ;
        int my = 0 ;
        while( quit == false ) {
                while( SDL_PollEvent( &event ) ) {
                        if( event.type == SDL_MOUSEBUTTONDOWN  && event.button.button == SDL_BUTTON_LEFT
                                                 && event.button.x > offset.x && event.button.x <= offset.x + blank->w
                                                 && event.button.y > offset.y && event.button.y <= offset.y + blank->h
                          ) {
                                //quit = true;
                                mx = event.button.x ;
                                my = event.button.y ;
                                switch( (mx - offset.x) / ((blank->w)/rndBase) ){
                                        case 0 :  Mix_PlayChannel( -1, highchunk, 0 ) ;break ;
                                        case 1 :  Mix_PlayChannel( -1, mediumchunk, 0 ) ;break ;
                                        case 2 :  Mix_PlayChannel( -1, lowchunk, 0 ) ;break ;
                                        case 3 : quit=true ; break ;
                                }
                          }
                }

        }

        dumpSurface(blank);
        dumpSurface(surf) ;
        d->showCursor(false);

}



///////////////////////////////////////////////////////////////////
void quiz(const int rndBase  , const int answer ){
        map<int,string> imap = map<int,string>();
        imap[0] = "high";
        imap[2] = "low" ;
        imap[1] = "medium" ;
        //let's display  the number in a random place
        //>displayText(stringify(confNum),true,-2);
        //>waitFrames(delay);
        d->clearScreen();
        d->pushEvent(std::string("asking for confirmation number"));
        d->displayText(std::string("which one was played more?"),true,1);
        //we need the subject be able to interact with the mouse
        d->showCursor(true);
        SDL_Surface* blank =getABlankSurface((d->getWidth())/2 , d->getHeight()/10);
        //we make a scrap image to write down the numbers on
        Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
        textIm.clear(PixRGB<byte>(255, 156, 120));
        //here we write the numbers
        for(int i = 0 ; i < rndBase ; i ++){
                writeText(textIm, Point2D<int>((blank->w)/(2*rndBase) -2,10+100*i), imap[i].c_str());
        }
        //and here we make a blittabl surface out of that
        SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
        //and take out differnt clips out of that
        SDL_Rect clip;
        clip.x = 0 ;
        clip.w = (blank->w)/rndBase - 2 ;
        clip.h =blank->h ;
        for(int i = 0 ; i < rndBase ; i++){
                clip.y = i*100 ;
                apply_surface(i*(clip.w + 2),0,*surf,*blank,clip);
        }

        SDL_Rect offset;
        offset.x = (d->getWidth() - blank->w) /2;
        offset.y = (d-> getHeight() - blank->h) /2;
        d->pushEvent("the button panel is going up");
        d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);

        bool quit = false ;
        SDL_Event event ;
        int mx = 0 ;
        int my = 0 ;
        while( quit == false ) {
                while( SDL_PollEvent( &event ) ) {
                        if( event.type == SDL_MOUSEBUTTONDOWN  && event.button.button == SDL_BUTTON_LEFT
                                                 && event.button.x > offset.x && event.button.x <= offset.x + blank->w
                                                 && event.button.y > offset.y && event.button.y <= offset.y + blank->h
                          ) {
                                quit = true;
                                mx = event.button.x ;
                                my = event.button.y ;
                          }
                }
        }

        switch( (mx - offset.x) / ((blank->w)/rndBase) ){
                case 0 : d->pushEvent("answered high") ; break ;
                case 1 : d->pushEvent("answered medium") ; break ;
                case 2 : d->pushEvent("answered low") ; break ;
        }
        //d->pushEvent("confirmed: " + stringify((mx - offset.x) / ((blank->w)/rndBase))) ;
        if((mx - offset.x) / ((blank->w)/rndBase) == answer) {
                d->pushEvent("correct answer") ;
        }else{
                d->pushEvent("incorrect answer") ;
        }

        dumpSurface(blank);
        dumpSurface(surf) ;
        d->showCursor(false);

}

///////////////////////////////////////////////////////////////////

//displays 0 or 1 in a random place on the screen
void finalTask(const int rndBase = 2 , const uint delay = 10 ){
        int confNum = rand() % rndBase ;
        d->pushEvent(std::string("confirmation number : ")+stringify(confNum));
        //let's display  the number in a random place
        d->displayText(stringify(confNum),true,-2);
        d->waitFrames(delay);
        d->clearScreen();
        d->pushEvent(std::string("asking for confirmation number"));
        d->displayText(std::string("what was the number ? "),true,1);
        //we need the subject be able to interact with the mouse
        d->showCursor(true);
        SDL_Surface* blank =getABlankSurface((d->getWidth())/2 , d->getHeight()/10);
        //we make a scrap image to write down the numbers on
        Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
        textIm.clear(PixRGB<byte>(255, 255, 255));
        //here we write the numbers
        for(int i = 0 ; i < rndBase ; i ++){
                writeText(textIm, Point2D<int>((blank->w)/(2*rndBase) -2,10+100*i), stringify(i).c_str());
        }
        //and here we make a blittabl surface out of that
        SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
        //and take out differnt clips out of that
        SDL_Rect clip;
        clip.x = 0 ;
        clip.w = (blank->w)/rndBase - 2 ;
        clip.h =blank->h ;
        for(int i = 0 ; i < rndBase ; i++){
                clip.y = i*100 ;
                apply_surface(i*(clip.w + 2),0,*surf,*blank,clip);
        }

        SDL_Rect offset;
        offset.x = (d->getWidth() - blank->w) /2;
        offset.y = (d-> getHeight() - blank->h) /2;
        d->pushEvent("the button panel is going up");
        d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);

        bool quit = false ;
        SDL_Event event ;
        int mx = 0 ;
        int my = 0 ;
        while( quit == false ) {
                while( SDL_PollEvent( &event ) ) {
                        if( event.type == SDL_MOUSEBUTTONDOWN  && event.button.button == SDL_BUTTON_LEFT
                                                 && event.button.x > offset.x && event.button.x <= offset.x + blank->w
                                                 && event.button.y > offset.y && event.button.y <= offset.y + blank->h
                          ) {
                                quit = true;
                                mx = event.button.x ;
                                my = event.button.y ;
                          }
                }
        }

        d->pushEvent("confirmed: " + stringify((mx - offset.x) / ((blank->w)/rndBase))) ;
        if((mx - offset.x) / ((blank->w)/rndBase) == (int)confNum) {
                d->pushEvent("correct confirmation") ;
        }else{
                d->pushEvent("incorrect confirmation") ;
        }

        dumpSurface(blank);
        dumpSurface(surf) ;
        d->showCursor(false);

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

        com += "\nconf-num=[2..10] (random numbers in confirmation task) {default=2} \n" ;
        com += "\nconf-delay=[>1] (the delay in showing the confirmation number) {default=10}\n";
        com += "\ndelay-interval=[>1]:[>1] (the interval for random delay after stroop task) {default=20:30}\n";
        com += "\nfixation-blink=[y/n] (show the blink for fixation after stroop task or no) {defaule y}\n";
        com += "\nimage-dir=[path to image directory] (needed for mode 2 and 3) {default=60:160} \n"  ;
        com += "\nlogfile=[logfilename.psy] {default = psycho-stroop-concurrent.psy}\n" ;
        com += "\nmemo=[a_string_without_white_space]\n";
        com += "\nmode=[1..6] (1 for auditory test only, 2 for auditory test + image display , "
                        "3 for image display only ,4 for volume adjustment, 5 for volume adjustment and then auditory test"
                        " , 6 for volume adjustment and then auditory test + image display ) {default=1}\n";
        com += "\ntask-size=[>1]:[>1](number of beats in a round of task){default=7:17} \n" ;
        com += "\nsubject=[subject_name] \n" ;
        com += "\ntest-rounds=[>1] (needed for mode1) {default=5}\n";
        com += "\nsound-dir=[path to wav files directory]{default=y}\n";
        com += "\nhigh-chunk=[first chunk sound]{default=high.wav}\n";
        com += "\nmedium-chunk=[second chunk sound]{default=medium.wav}\n";
        com += "\nlow-chunk=[third chunk sound]{default=low.wav}\n";
        com += "\nimage-onset-num=[>=0 and <=task-size] {default=2}" ;
        return com ;
}


extern "C" int main(const int argc, char** argv)
{

          MYLOGVERB = LOG_INFO;  // suppress debug messages
        argMap["mode"] = "1" ;
        argMap["logfile"]="psycho-stroop-concurrent.psy" ;
        argMap["conf-number"] = "2" ;
        argMap["task-size"]="7:17" ;
        argMap["image-dir"]="..";
        argMap["conf-delay"]="10" ;
        argMap["test-rounds"]="5";
        argMap["delay-interval"]="20:30" ;
        argMap["subject"]="" ;
        argMap["memo"]="" ;
        argMap["fixation-blink"]="y" ;
        argMap["sound-dir"]="..";
        argMap["high-chunk"]="high.wav";
        argMap["medium-chunk"]="medium.wav";
        argMap["low-chunk"]="low.wav";
        argMap["image-onset-num"]="2";

        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        nub::soft_ref<EyeTrackerConfigurator>        etc(new EyeTrackerConfigurator(manager));
          manager.addSubComponent(etc);

        if (manager.parseCommandLine(argc, argv,
            "at least one argument needed", 1, -1)==false){
                   //since the length of string that can be passed to LINFO is limited and our message is longer than this limit
                    //we have to use std::cout for displaying the message
                    cout<<getUsageComment()<<endl;
                        return(1);
            }

        for(uint i = 0 ; i < manager.numExtraArgs() ; i++){
                addArgument(manager.getExtraArg(i),std::string("=")) ;
        }
        if( Mix_OpenAudio( 22050, MIX_DEFAULT_FORMAT, 2, 4096 ) == -1 ){
                LINFO( "did not open the mix-audio") ;
                return -1 ;
            }
        string highstr = argMap["sound-dir"] + "/" + argMap["high-chunk"];
        string mediumstr = argMap["sound-dir"] + "/" + argMap["medium-chunk"];
        string lowstr = argMap["sound-dir"] + "/" + argMap["low-chunk"];
        highchunk = Mix_LoadWAV(highstr.c_str());
        mediumchunk = Mix_LoadWAV(mediumstr.c_str());
        lowchunk = Mix_LoadWAV(lowstr.c_str());
        if( ( highchunk == NULL ) || ( mediumchunk == NULL ) || ( lowchunk == NULL ) )
        {
                LINFO("did not find the indicated wav files!");
                return -1;
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
          d->waitForKey();

  // let's do an eye tracker calibration:
          d->displayText("<SPACE> to calibrate; other key to skip");
          int c = d->waitForKey();
          if (c == ' ') d->displayEyeTrackerCalibration(3,3);

          d->clearScreen();
          d->displayText("<SPACE> for random play; other key for ordered");
          c = d->waitForKey();

        int mode = atoi(argMap["mode"].c_str()) ;

        if(mode == 1 || mode == 5){
                string::size_type position = argMap["delay-interval"].find(":");
                int minDel = atoi(argMap["delay-interval"].substr(0,position).c_str()) ;
                int maxDel = atoi(argMap["delay-interval"].substr(position+1).c_str()) ;
                int numOfTests = atoi(argMap["test-rounds"].c_str()) ;
                int confNum = atoi(argMap["conf-number"].c_str()) ;
                int confDelay = atoi(argMap["conf-delay"].c_str()) ;
                //this is for the volume adjustment
                if(mode == 5){
                        volumeAdjustment(3,"continue") ;
                }
                for(int i = 0 ; i < numOfTests ; i++){
                        d->clearScreen() ;
                        d->pushEvent("**************************************") ;
                        d->showCursor(true);
                        d->displayText("click one of the  mouse buttons to start!");
                        getMouseEvent() ;
                        d->showCursor(false);
                        d->clearScreen() ;
                        vector<int> task = getAuditoryTask(argMap["task-size"]) ;
                        string ts = string("");
                        int high = 0 ;
                        int medium = 0 ;
                        int low = 0 ;
                        int answ = 0 ;
                        for( uint j = 0 ; j < task.size() ; j++){
                                ts = ts + stringify(task[j]) ;
                                switch( task[j] ){
                                        case 0 : high++;break ;
                                        case 1 : medium++;break;
                                        case 2 : low++;break ;
                                }
                        }
                        int m = max(max(low,medium),high);
                        if( m == high) answ = 0 ;
                        if (m == medium ) answ = 1 ;
                        if (m == low ) answ = 2 ;
                        d->pushEvent("task sequence : "+ts);
                        for( uint j = 0 ; j < task.size() ; j++ ){
                                switch( task[j] ){
                                        case 0 :  Mix_PlayChannel( -1, highchunk , 0 ) ;break ;
                                        case 1 :  Mix_PlayChannel( -1, mediumchunk , 0 ); break ;
                                        case 2 :  Mix_PlayChannel( -1, lowchunk , 0 ) ;break ;
                                }
                                d->waitFrames((rand()%(maxDel-minDel)) +minDel);
                        }
                        finalTask(confNum,confDelay) ;
                        quiz(3,answ) ;

                }
        }

        if(mode == 2 || mode == 6){
                string::size_type position = argMap["delay-interval"].find(":");
                int minDel = atoi(argMap["delay-interval"].substr(0,position).c_str()) ;
                int maxDel = atoi(argMap["delay-interval"].substr(position+1).c_str()) ;
                string dir = argMap["image-dir"];
                vector<string> files = vector<string>();
                getdir(dir,files);
                int confNum = atoi(argMap["conf-number"].c_str()) ;
                int confDelay = atoi(argMap["conf-delay"].c_str()) ;
                //this is for the volume adjustment
                if(mode == 6){
                        volumeAdjustment(3,"continue") ;
                }
                while(files.size()>0){
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
                        getMouseEvent() ;
                        d->showCursor(false);
                        d->clearScreen() ;
                        vector<int> task = getAuditoryTask(argMap["task-size"]) ;
                        string ts = string("");
                        int high = 0 ;
                        int medium = 0 ;
                        int low = 0 ;
                        int answ = 0 ;
                        for( uint j = 0 ; j < task.size() ; j++){
                                ts = ts + stringify(task[j]) ;
                                switch( task[j] ){
                                        case 0 : high++;break ;
                                        case 1 : medium++;break;
                                        case 2 : low++;break ;
                                }
                        }
                        int m = max(max(low,medium),high);
                        if( m == high) answ = 0 ;
                        if (m == medium ) answ = 1 ;
                        if (m == low ) answ = 2 ;
                        d->pushEvent("task sequence : "+ts);
                        for( uint j = 0 ; j < task.size() ; j++ ){
                                switch( task[j] ){
                                        case 0 :  Mix_PlayChannel( -1, highchunk , 0 ) ;break ;
                                        case 1 :  Mix_PlayChannel( -1, mediumchunk , 0 ); break ;
                                        case 2 :  Mix_PlayChannel( -1, lowchunk , 0 ) ;break ;
                                }

                                if (j==2){
                                        d->waitNextRequestedVsync(false, true);
                                        d->pushEvent(std::string("===== Showing image: ") +
                                           files[imageIndex] + " =====");

                                        // start the eye tracker:
                                        et->track(true);
                                              //blink the fixation:
                                        if(argMap["fixation-blink"].compare("y")==0) d->displayFixationBlink();

                                        d->displaySDLSurfacePatch(surf , &offset,NULL , -2,false, true);
                                        files.erase(files.begin()+imageIndex);
                                }else{
                                        d->waitFrames((rand()%(maxDel-minDel)) +minDel);
                                }
                        }


                        dumpSurface(surf);
                        // stop the eye tracker:
                        usleep(50000);
                        et->track(false);
                        //see if the subject was looking at the screen!
                        finalTask(confNum,confDelay) ;
                        quiz(3,answ) ;
                        d->clearScreen() ;
                        //now quiz time!
                        d->clearScreen() ;
                }
        }

        if(mode == 3){
                string::size_type position = argMap["delay-interval"].find(":");
                int minDel = atoi(argMap["delay-interval"].substr(0,position).c_str()) ;
                int maxDel = atoi(argMap["delay-interval"].substr(position+1).c_str()) ;
                string dir = argMap["image-dir"];
                vector<string> files = vector<string>();
                getdir(dir,files);
                int c = 0 ;

                while(files.size()>0){
                        c++ ;
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
                        d->waitNextRequestedVsync(false, true);
                        d->pushEvent(std::string("===== Showing image: ") +
                                           files[imageIndex] + " =====");

                        // start the eye tracker:
                        et->track(true);
                              //blink the fixation:
                        if(argMap["fixation-blink"].compare("y")==0) d->displayFixationBlink();
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
        //for the case that the subject wants to adjust the volume
        if(mode == 4){
                volumeAdjustment(3,"exit") ;
        }


          d->clearScreen();
          d->displayText("Experiment complete. Thank you!");
          d->waitForKey();
        //getEvent();
          // stop all our ModelComponents
          manager.stop();
        //Quit SDL_mixer
        Mix_FreeChunk( highchunk );
        Mix_FreeChunk( lowchunk );
        Mix_FreeChunk( mediumchunk );
        Mix_CloseAudio();
          // all done!
          return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

