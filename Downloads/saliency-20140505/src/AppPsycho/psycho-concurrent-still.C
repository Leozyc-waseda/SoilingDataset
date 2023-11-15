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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-concurrent-still.C $
// $Id: psycho-concurrent-still.C 10794 2009-02-08 06:21:09Z itti $
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
ModelManager manager("Psycho-Concurrent");
nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
map<uint,uint> testMap ;
map<string,string> argMap ;


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
//////////////////////////////////////////////
//inputs: l ; number of words to be shown (1..5)
//        :delay; delay between displaying words in number of frames

void colorStroopTask(const uint l,const int delay = 10){
        //0 stands for white
        //1 stands for yellow
        //2 stands for red
        //3 stands for green
        //4 stands for blue
        vector<uint> word = vector<uint>() ;
        vector<uint> color = vector<uint>() ;
        for(uint i = 0 ; i < 5 ; i++){
                word.push_back(i) ;
                color.push_back(i) ;
        }
        for(uint i = 0 ; i < l ; i++){
                int wi = rand() % word.size() ;
                int ci = rand() % color.size() ;
                //let's make sure that the color word and color are not the same
                while( ci==wi ){
                        ci = rand() % color.size() ;
                }

                testMap[word[wi]] = color[ci] ;
                word.erase(word.begin()+wi);
                color.erase(color.begin()+ci) ;
        }

        PixRGB<byte> white(255, 255, 255);
        PixRGB<byte> yellow(255, 255, 0);
        PixRGB<byte> red(255,0,0);
        PixRGB<byte> green(0,255,0) ;
        PixRGB<byte> blue(0,0,255) ;
        SDL_Surface* blank = getABlankSurface(d->getWidth(),d->getHeight()) ;
        for(map<uint,uint>:: iterator  it = testMap.begin() ; it != testMap.end() ; ++it  ){
                Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
                textIm.clear(PixRGB<byte>(0, 0, 0));
                int x = rand()% (d->getWidth()-60) ;
                int y = rand() % (d-> getHeight() - 60);
                string myword = "" ;
                PixRGB<byte> mycolor(0,0,0) ;
                switch (it->first){
                        case 0 : myword = "white"; break ;
                        case 1 : myword = "yellow" ; break ;
                        case 2 : myword = "red" ; break ;
                        case 3 : myword = "green" ; break ;
                        case 4 : myword = "blue" ; break ;
                }

                switch ( it->second){
                        case 0 : mycolor = white ; break ;
                        case 1 : mycolor = yellow ; break ;
                        case 2 : mycolor = red ; break ;
                        case 3 : mycolor = green; break ;
                        case 4 : mycolor = blue ; break ;
                }
                writeText(textIm, Point2D<int>(x,y), myword.c_str(),mycolor, PixRGB<byte>(0,0,0), SimpleFont::FIXED(10), true);
                SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
                d->pushEvent(string("stroop test: ")+stringify(it->first)+" " +stringify(it->second)) ;
                d->pushEvent(std::string("image is going up"));
                d->displaySurface(surf, -2);
                d->waitFrames(delay);
                d->pushEvent(std::string("image is going down"));
                d->displaySurface(blank, -2);
                dumpSurface(surf) ;
        }
        dumpSurface(blank);
}
/////////////////////////////////////////////
//this function poses the question based on the testMap previously set,at the end the testMap will be cleared
void colorStroopQuiz(){
        //since the subject is supposed to interact with the mouse we make it appear!
        d->showCursor(true);
        map<uint,string> abrMap ;
        abrMap[2] = "red" ;
        abrMap[4] = "blue" ;
        abrMap[3] = "green" ;
        abrMap[0] = "white" ;
        abrMap[1] = "yellow" ;
        int rd = rand()%testMap.size() ;
        int c = -1 ;
        uint word = 0 ;
        uint color= 0 ;

        for(map<uint,uint>:: iterator  it = testMap.begin() ; it != testMap.end() ; ++it  ){
                c++ ;
                if(c==rd){
                        word = it->first;
                        color = it->second;
                }
        }
        string disWord = abrMap[word] ;
        string disColor = abrMap[color]  ;

        //we toss a coin to see what question to ask; asking the color of a word (rd=0) or the word in a give color (rd=1)
        rd = rand()%2 ;
        Uint32 white = d->getUint32color(PixRGB<byte>(255, 255, 255));
        Uint32 yellow = d->getUint32color(PixRGB<byte>(255, 255, 0));
        Uint32 red = d->getUint32color(PixRGB<byte>(255,0,0));
        Uint32 green = d->getUint32color(PixRGB<byte>(0,255,0));
        Uint32 blue = d->getUint32color(PixRGB<byte>(0,0,255));
        if(rd == 0){
                //let's ask the question
                d->pushEvent("asked "+stringify(word)+":-") ;
                d->displayText("'"+disWord+"'",true,1);
                //this will serve as the panle for showing the buttons
                SDL_Surface* blank =getABlankSurface(d->getWidth()/2 , d->getHeight()/10);
                fillRectangle(blank , white , 0  , 0  , blank->w/5 , blank->h );
                fillRectangle(blank , yellow , (blank->w)/5 ,0   , (blank->w)/5 , blank->h );
                fillRectangle(blank , red , 2*(blank->w)/5 ,0   , (blank->w)/5 , blank->h );
                fillRectangle(blank , green , 3*(blank->w)/5 ,0   , (blank->w)/5 , blank->h );
                fillRectangle(blank , blue , 4*(blank->w)/5 ,0   , (blank->w)/5 , blank->h );
                //this is the offset of buttons panel
                SDL_Rect offset;
                offset.x = (d->getWidth() - blank->w) /2;
                offset.y = (d-> getHeight() - blank->h) /2;
                d->pushEvent("the answer panel is going up");
                d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);
                bool quit = false ;
                SDL_Event event ;
                int mx = 0 ; //the  x location of mouse pointer at click time
                int my = 0 ; //the  y location of mouse pointer at click time

                //in this block we will wait until left burron is down
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
                //let's save the answer in the logfile!
                d->pushEvent("answered: "+ stringify((mx - offset.x) / ((blank->w)/5)) ) ;
                if((mx - offset.x) / ((blank->w)/5) == (int)color) {
                        d->pushEvent("correct answer") ;
                }else{
                        d->pushEvent("incorrect answer") ;
                }
                dumpSurface(blank) ;

        }else{
                //let's ask the question
                d->pushEvent("asked -:"+stringify(color)) ;
                SDL_Surface* cblank =getABlankSurface((d->getWidth())/10 , d->getHeight()/10);
                Uint32 cl = 0 ;
                switch(color){
                        case 0 : cl = white ;break ;
                        case 1 : cl = yellow ; break ;
                        case 2 : cl = red ; break ;
                        case 3 : cl = green ; break ;
                        case 4 : cl = blue ; break ;
                }
                fillRectangle(cblank , cl , 0  , 0  , cblank->w , cblank->h );
                SDL_Rect coffset;
                coffset.x = (d->getWidth() - cblank->w) /2;
                coffset.y = (d-> getHeight() - cblank->h) /6;
                d->displaySDLSurfacePatch(cblank , &coffset,NULL , -2,false, true);
                //d->displayText("what word was written in '"+disColor+"'? ",true,1);
                //this is a panel for the buttons
                SDL_Surface* blank =getABlankSurface(3*(d->getWidth())/4 , d->getHeight()/10);
                //and this is a scrap image for writing lables of the buttons
                Image<PixRGB<byte> > textIm(d->getWidth(),d->getHeight(),ZEROS);
                textIm.clear(PixRGB<byte>(255, 255, 255));
                writeText(textIm, Point2D<int>(10,10), "white");
                writeText(textIm, Point2D<int>(10,110), "yellow");
                writeText(textIm, Point2D<int>(10,210), "red");
                writeText(textIm, Point2D<int>(10,310), "green");
                writeText(textIm, Point2D<int>(10,410), "blue");
                //now let's make a SDL_Surface out of our scrap image
                SDL_Surface *surf = d->makeBlittableSurface(textIm , true);
                //we will cut out different clips from the scrap image
                SDL_Rect clip;
                clip.x = 0 ;
                clip.y = 0 ;
                clip.w = (blank->w)/5 - 2 ;
                clip.h =blank->h ;
                //and apply it to the button panel
                apply_surface(0,0,*surf,*blank,clip);
                clip.y = 100 ;
                apply_surface(clip.w + 2 ,0,*surf,*blank,clip);
                clip.y = 200 ;
                apply_surface(2* (clip.w + 2) ,0,*surf,*blank,clip);
                clip.y = 300 ;
                apply_surface(3* (clip.w + 2) ,0,*surf,*blank,clip);
                clip.y = 400 ;
                apply_surface(4* (clip.w + 2) ,0,*surf,*blank,clip);
                //and this the offset for the button panel
                SDL_Rect offset;
                offset.x = (d->getWidth() - blank->w) /2;
                offset.y = (d-> getHeight() - blank->h) /2;
                d->pushEvent("the answer panel is going up");
                d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);
                bool quit = false ;
                SDL_Event event ;
                int mx = 0 ; //the  x location of mouse pointer at click time
                int my = 0 ; //the  y location of mouse pointer at click time

                //in this block we will wait until left burron is down
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
                d->pushEvent("answered: "+ stringify((mx - offset.x) / ((blank->w)/5)) ) ;
                if((mx - offset.x) / ((blank->w)/5) == (int)word) {
                        d->pushEvent("correct answer") ;
                }else{
                        d->pushEvent("incorrect answer") ;
                }
                dumpSurface(blank) ;
                dumpSurface(cblank);
                dumpSurface(surf) ;
        }
        //cleaning up the questions and answers
        while(testMap.size()>0){
                map<uint,uint>::iterator it=testMap.begin() ;
                testMap.erase(it);
        }
        //we make the mouse pointer disappear again!
        d->showCursor(false);
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
bool memoryTest(vector<string>& dil , vector<string>& cil){
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

        //cframe is a frame on a transparent background put on top of the las choice of the subject
        SDL_Surface* cframe = getABlankSurface((d->getWidth())/sf,(d->getHeight())/sf);
        Uint32 color = d->getUint32color(PixRGB<byte>(255, 98 , 25));
        drawRectangle(cframe , color , 0 , 0 ,cframe->w -2   , cframe->h -2 , 5) ;
        if( cframe != NULL )
        {
            //Map the color key
          Uint32 colorkey = SDL_MapRGB( cframe->format, 0x00, 0x00, 0x00 );

            //Set all pixels of color R 0, G 0, B 0 to be transparent
          SDL_SetColorKey( cframe, SDL_SRCCOLORKEY, colorkey );
        }

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
        SDL_FreeSurface( pbar );
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
                SDL_Rect tempClip = SDL_Rect();
                tempClip.x = 0 ; tempClip.y=0 ;
                tempClip.w = 10 ; tempClip.h = 10;

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

                                }else{
                                        quit = 3 ;
                                        SDL_Rect* clip = new SDL_Rect() ;
                                        SDL_Rect* tof = new SDL_Rect ;
                                        tof->x = 0 ; tof->y=0 ;
                                        SDL_BlitSurface( oblank, NULL, blank, tof );
                                        clip = 0 ;
                                        tof=0;
                                        d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);

                                }

                        }else{
                                quit = 3 ;
                                SDL_Rect* clip = new SDL_Rect() ;
                                clip->x = 0 ;
                                clip->y = 0 ;
                                clip->w = d->getWidth() ;
                                clip->h = d->getHeight() ;
                                SDL_Rect* tof = new SDL_Rect ;
                                tof->x = 0 ; tof->y=0 ;
                                SDL_BlitSurface( oblank, NULL, blank, tof );
                                clip = 0 ;
                                tof=0;
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
                                clip->w = oblank->w ;
                                clip->h = oblank->h ;
                                SDL_Rect* tof = new SDL_Rect ;
                                tof->x = 0 ; tof->y=0 ;
                                SDL_BlitSurface( oblank, NULL, blank, tof );
                                clip = 0 ;
                                tof=0;
                                quit = 1;
                                fmx = event.button.x ;
                                fmy = event.button.y ;
                                int i = sf*(fmx-offset.x)/(d->getWidth()) ;
                                int j = sf*(fmy-offset.y)/(d->getHeight());
                                choice1 = (int)(sf*(fmx-offset.x)/(d->getWidth())) + sf*(int)(sf*(fmy-offset.y)/(d->getHeight())) ;
                                tempClip.x =  i*(d->getWidth())/sf; tempClip.y= j*(d->getHeight())/sf;tempClip.w=(d->getWidth())/sf;tempClip.h=(d->getHeight())/sf;
                                SDL_BlitSurface(cframe,NULL,blank,&tempClip) ;
                                d->displaySDLSurfacePatch(blank , &offset,NULL , -2,false, true);
                          }

                        if(quit==3) quit=0 ;

                }
        }

        if(oblank != NULL)
        SDL_FreeSurface( oblank );
        if(blank != NULL)
        SDL_FreeSurface( blank );
        SDL_FreeSurface( cframe );
        fls = 0 ;
        d->clearScreen();
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


void getARandomScoop(vector<string> mainlist , vector<string>& targetlist , int s){
  //cleaning up the target list
  if(targetlist.size()>0) {
    while( targetlist.size()!=0 ){
      targetlist.erase(targetlist.begin()) ;
    }
  }
 //let's duplicate the mainlist and make a temporary vector
  LINFO("scoop1");
  vector<string> tv = vector<string>() ;
  for(uint i = 0 ; i < mainlist.size() ; i++){
    tv.push_back(mainlist[i]);
  }

  for(int i = 0 ; i < s ; i++){
        int index = rand()%tv.size() ;
        targetlist.push_back(tv[index]);
        tv.erase(tv.begin());
  }
  string st = stringify(targetlist.size());
  LINFO(st.c_str());
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
        com += "\ndelay-interval=[>1]:[>1] (the interval for random delay after stroop task) {default=10}\n";
        com += "\nfixation-blink=[y/n] (show the blink for fixation after stroop task or no) {defaule y}\n";
        com += "\nimage-dir=[path to image directory] (needed for mode 2 and 3) {default=60:160} \n"  ;
        com += "\nextra-image-dir=[path to extra images directory](needed for lineup memory test)";
        com += "\nlogfile=[logfilename.psy] {default = psycho-stroop-concurrent.psy}\n" ;
        com += "\nstroop-display-delay=[>1] (number of frames to display the stroop word){default=60}\n";
        com += "\nmemo=[a_string_without_white_space]\n";
        com += "\nmode=[1..5] (1 for stroop test only, 2 for stroop test + image display + digit memorizetion"
                        ", 3 for image display only, 4 for stroop test + image display + image memorizetion, "
                        "5 for image display + image memorization ) {default=1}\n";
        com += "\nstroop-size=[1..5](number of words in the stroop task){default=3} \n" ;
        com += "\nsubject=[subject_name] \n" ;
        com += "\ntest-rounds=[>1] (needed for mode1) {default=5}\n";


        return com ;
}


extern "C" int main(const int argc, char** argv)
{

          MYLOGVERB = LOG_INFO;  // suppress debug messages
        //let's push the initial value for the parameters
        argMap["mode"] = "1" ;
        argMap["logfile"]="psycho-stroop-concurrent.psy" ;
        argMap["conf-number"] = "2" ;
        argMap["stroop-size"]="3" ;
        argMap["image-dir"]="..";
        argMap["extra-image-dir"] = ".." ;
        argMap["conf-delay"]="10" ;
        argMap["test-rounds"]="5";
        argMap["stroop-display-delay"] = "60" ;
        argMap["delay-interval"]="60:160" ;
        argMap["subject"]="" ;
        argMap["memo"]="" ;
        argMap["fixation-blink"]="y" ;
        argMap["lineup-size"]="4";
        argMap["image-memory-stack"]="4";

        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        //*nub::soft_ref<EyeTrackerConfigurator>
        //*                etc(new EyeTrackerConfigurator(manager));
          //*manager.addSubComponent(etc);

        if (manager.parseCommandLine(argc, argv,
            "at least one argument needed", 1, -1)==false){
                    LINFO(getUsageComment().c_str());
                        return(1);
            }

        for(uint i = 0 ; i < manager.numExtraArgs() ; i++){
                addArgument(manager.getExtraArg(i),std::string("=")) ;
        }

        manager.setOptionValString(&OPT_EventLogFileName, argMap["logfile"]);
        manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");
          // hook our various babies up and do post-command-line configs:
        //*nub::soft_ref<EyeTracker> et = etc->getET();
        //*d->setEyeTracker(et);
        d->setEventLog(el);
         //*et->setEventLog(el);


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
          if (c == ' ') d->displayEyeTrackerCalibration(3,5);

          d->clearScreen();
          d->displayText("<SPACE> for random play; other key for ordered");
          c = d->waitForKey();

        int mode = atoi(argMap["mode"].c_str()) ;

        if(mode == 1){
                string::size_type position = argMap["delay-interval"].find(":");
                int minDel = atoi(argMap["delay-interval"].substr(0,position).c_str()) ;
                int maxDel = atoi(argMap["delay-interval"].substr(position+1).c_str()) ;
                int numOfTests = atoi(argMap["test-rounds"].c_str()) ;
                int stroopSize = atoi(argMap["stroop-size"].c_str()) ;
                //int confNum = atoi(argMap["conf-number"].c_str()) ;
                //int confDelay = atoi(argMap["conf-delay"].c_str()) ;
                int stroopDelay = atoi(argMap["stroop-display-delay"].c_str());
                for(int i = 0 ; i < numOfTests ; i++){
                        d->pushEvent("**************************************") ;
                        d->showCursor(true);
                        d->displayText("click one of the  mouse buttons to start!");
                        getMouseEvent() ;
                        d->showCursor(false);
                        colorStroopTask(stroopSize,stroopDelay);
                        d->waitFrames((rand()%(maxDel - minDel)) +minDel );
                        //finalTask(confNum,confDelay) ;
                        d->clearScreen() ;
                        colorStroopQuiz();
                }
        }

        if(mode == 2 ){
                string::size_type position = argMap["delay-interval"].find(":");
                int minDel = atoi(argMap["delay-interval"].substr(0,position).c_str()) ;
                int maxDel = atoi(argMap["delay-interval"].substr(position+1).c_str()) ;
                string dir = argMap["image-dir"];
                vector<string> files = vector<string>();
                getdir(dir,files);
                int stroopSize = atoi(argMap["stroop-size"].c_str()) ;
                int confNum = atoi(argMap["conf-number"].c_str()) ;
                int confDelay = atoi(argMap["conf-delay"].c_str()) ;
                int stroopDelay = atoi(argMap["stroop-display-delay"].c_str());

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
                        colorStroopTask(stroopSize,stroopDelay);
                        d->clearScreen() ;
                        d->waitNextRequestedVsync(false, true);
                        d->pushEvent(std::string("===== Showing image: ") +
                                           files[imageIndex] + " =====");

                        // start the eye tracker:
                        //*et->track(true);
                              //blink the fixation:
                        if(argMap["fixation-blink"].compare("y")==0) d->displayFixationBlink();

                        d->displaySDLSurfacePatch(surf , &offset,NULL , -2,false, true);
                        d->waitFrames((rand()%(maxDel-minDel)) +minDel );
                        dumpSurface(surf);
                        // stop the eye tracker:
                        usleep(50000);
                             //*et->track(false);
                        //see if the subject was looking at the screen!
                        finalTask(confNum,confDelay) ;
                        d->clearScreen() ;
                        //now quiz time!
                        colorStroopQuiz();
                        d->clearScreen() ;
                        files.erase(files.begin()+imageIndex);
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
                        d->waitFrames(45);
                        d->waitNextRequestedVsync(false, true);
                        d->pushEvent(std::string("===== Showing image: ") +
                                           files[imageIndex] + " =====");

                        // start the eye tracker:
                        //*et->track(true);
                              //blink the fixation:
                        if(argMap["fixation-blink"].compare("y")==0) d->displayFixationBlink();

                        d->displaySDLSurfacePatch(surf , &offset,NULL , -2,false, true);
                        d->waitFrames((rand()%(maxDel-minDel)) +minDel );
                        dumpSurface(surf);
                        // stop the eye tracker:
                        usleep(50000);
                             //*et->track(false);
                        d->clearScreen() ;
                        files.erase(files.begin()+imageIndex);
                }

        }
        if(mode==5){
          //let's read the parameters related to this part
          string::size_type position = argMap["delay-interval"].find(":");
          int minDel = atoi(argMap["delay-interval"].substr(0,position).c_str()) ;
          int maxDel = atoi(argMap["delay-interval"].substr(position+1).c_str()) ;
          vector<string> files = vector<string>();
          vector<string> extras = vector<string>();
          vector<string> memoryStack = vector<string>();
          string dir = argMap["image-dir"];
          getdir(dir,files);
          dir = argMap["extra-image-dir"];
          getdir(dir,extras);
          int stroopSize = atoi(argMap["stroop-size"].c_str()) ;
          int stroopDelay = atoi(argMap["stroop-display-delay"].c_str());
          int lineupSize = atoi(argMap["lineup-size"].c_str()) ;
          int stackSize = atoi(argMap["image-memory-stack"].c_str());
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
            colorStroopTask(stroopSize,stroopDelay);
            d->clearScreen() ;
            d->waitNextRequestedVsync(false, true);
            d->pushEvent(std::string("===== Showing image: ") +
                files[imageIndex] + " =====");

                        // start the eye tracker:
                        //*et->track(true);
                        //blink the fixation:
            if(argMap["fixation-blink"].compare("y")==0) d->displayFixationBlink();

            d->displaySDLSurfacePatch(surf , &offset,NULL , -2,false, true);
            d->waitFrames((rand()%(maxDel-minDel)) +minDel );
            dumpSurface(surf);
                        // stop the eye tracker:
            usleep(50000);
                        //*et->track(false);
            d->clearScreen() ;
                        //now quiz time!
            colorStroopQuiz();
            d->clearScreen() ;
            memoryStack.push_back(files[imageIndex]);

            if(memoryStack.size()%stackSize == 0 && memoryStack.size()!=0){

              while( memoryStack.size() != 0 ){
                vector<string> imVector = vector<string>();
                imVector.push_back(memoryStack[0]);
                vector<string> distractorsVector =  vector<string>() ;
                memoryStack.erase(memoryStack.begin());
                vector<string> tv = vector<string>() ;
                for(uint i = 0 ; i < extras.size() ; i++){
                        tv.push_back(extras[i]);
                }

                for(int i = 0 ; i < lineupSize -1 ; i++){
                        int index = rand()%tv.size() ;
                        distractorsVector.push_back(tv[index]);
                        tv.erase(tv.begin());
                }

                //getARandomScoop(extras,distractorsVector,lineupSize -1 ) ;
                string st = stringify(distractorsVector.size());
                LINFO(st.c_str()) ;
                memoryTest(imVector , distractorsVector);
                //delete distractorsVector;
              }
            }

            files.erase(files.begin()+imageIndex);
          }

        }

        if(mode==6){

        }
          d->clearScreen();
          d->displayText("Experiment complete. Thank you!");
          d->waitForKey();
        //getEvent();
          // stop all our ModelComponents
          manager.stop();

          // all done!
          return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

