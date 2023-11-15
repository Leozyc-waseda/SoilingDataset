/*!@file GameBoard/indexing-playback.C the application for playing back the indexing experiment*/

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
//
// 2008-02-20 16:14:27Z nnoori $
//

#ifndef INVT_HAVE_LIBSDL_IMAGE

#include <cstdio>
int main()
{
        fprintf(stderr, "The SDL_image library must be installed to use this program\n");
        return 1;
}

#else



/*
Attention: SDL_image is needed for linking

*/

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
#include <iostream>
#include <fstream>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include "AppPsycho/psycho-skin-resize.h"
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <time.h>
#include "AppPsycho/psycho-skin-mapgenerator.h"
#include <sstream>


using namespace std;

// ModelManager manager("Psycho Skin");
// nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
bool programQuit = false ;
map<string,string> skinFileMap ;
const int NUMBER_OF_CLASSES = 6;
const int IMAGE_WIDTH = 128 ;
const string classes[NUMBER_OF_CLASSES]={"class1","class2","class3","class4","class5","class6"};


////////////////////////////////////

template <class T>
                string stringify(T &i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}

///////////////////////////////////


SDL_Surface *getAFreshSurface(int w ,  int h){

        SDL_Surface *sur = SDL_CreateRGBSurface(SDL_SWSURFACE, w, h, 32,
                        0x00000000, 0x00000000, 0x00000000, 0x00000000);

        return sur ;

}

/////////////////////////////////////////////
SDL_Surface *load_image( string filename )
{
//The image that's loaded
        SDL_Surface* loadedImage = NULL;

//The optimized image that will be used
        SDL_Surface* optimizedImage = NULL;

//Load the image
        loadedImage = IMG_Load( filename.c_str() );

//If the image loaded
        if( loadedImage != NULL )
        {
//Create an optimized image
                optimizedImage = SDL_DisplayFormat( loadedImage );

//Free the old image
                SDL_FreeSurface( loadedImage );

//If the image was optimized just fine
                if( optimizedImage != NULL )
                {
        //Map the color key
                        Uint32 colorkey = SDL_MapRGB( optimizedImage->format, 0, 0xFF, 0xFF );

        //Set all pixels of color R 0, G 0xFF, B 0xFF to be transparent
                        SDL_SetColorKey( optimizedImage, SDL_SRCCOLORKEY, colorkey );
                }
        }/*else{
                emergencyExit() ;
        }*/

//Return the optimized image
        return optimizedImage;
}

///////////////////////////////////
void dumpSurface(SDL_Surface& surface){
        SDL_FreeSurface( &surface );
}

/////////////////////////////////////////////
void apply_surface( int x, int y, SDL_Surface& source, SDL_Surface& destination , SDL_Rect& clip )
{
//Make a temporary rectangle to hold the offsets
        SDL_Rect offset;

//Give the offsets to the rectangle
        offset.x = x;
        offset.y = y;

//Blit the surface
        SDL_BlitSurface( &source, &clip, &destination, &offset );
}


/////////////////////////////////////////
extern "C" int main( int argc, char* argv[] ){

        ModelManager manager("Psycho Skin");
        nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));

        MYLOGVERB = LOG_INFO;  // suppress debug messages

        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        d->setEventLog(el);
        // Parse command-line:
        if (manager.parseCommandLine(argc, argv,"<profile.psymap> <path-to-skin> <logfilename>", 3, 3)==false)
                return(1);
        manager.setOptionValString(&OPT_EventLogFileName,manager.getExtraArg(2).c_str() );
        manager.start();
        d->clearScreen() ;
        int trialSize = 0 ;

        ////
        vector<string>* evalVector = new vector<string>() ;
        vector<int>* elapsedTimeVector = new vector<int>() ;
        vector<int>* classVector = new vector<int>() ;
        vector<Matrix*>* patternVector = new vector<Matrix*>() ;
        vector<Matrix*>* mapVector = new vector<Matrix*>() ;
        ////
        //we read the psymap file and keep the trial information in seperate maps
        const char* filename = manager.getExtraArg(0).c_str();
        ifstream inFile(filename, ios::in);
        string pathToSkin = manager.getExtraArg(1);
        map<int,SDL_Surface*>* classesMap = new map<int,SDL_Surface*>();
        int maxMapRows = 0 ;
        int maxMapCols = 0 ;
        int maxPatternRows = 0 ;
        int maxPatternCols = 0 ;
        bool exitFlag = false ;
        if (!inFile)
        {
                LINFO("profile  not found!");
                exitFlag = true ;
        } else{
                char ch[1000];
                while (inFile.getline(ch , 1000)){
                        string line = ch;
                        LINFO("reads : %s", line.c_str());
                        if(line[0]=='1' && line[2]=='1') evalVector->push_back("TP");
                        if(line[0]=='1' && line[2]=='0') evalVector->push_back("FN");
                        if(line[0]=='0' && line[2]=='1') evalVector->push_back("FP");
                        if(line[0]=='0' && line[2]=='0') evalVector->push_back("TN");
                        line = line.substr(4);
                        uint position = line.find(",");
                        char rf[10] ;
                        strcpy(rf,line.substr(0,position).c_str());
                        elapsedTimeVector->push_back(atoi(rf)) ;
                        line = line.substr(position+1);
                        position = line.find(",");
                        strcpy(rf,line.substr(position-1,1).c_str());
                        classVector->push_back(atoi(rf)) ;
                        line = line.substr(position+1) ;
                        position = line.find(",");
                        patternVector->push_back(getPattenByString(line.substr(0,position+1)));
                        maxPatternRows = max(maxPatternRows,(patternVector->back())->getNumOfRows());
                        maxPatternCols = max(maxPatternCols,(patternVector->back())->getNumOfColumns());
                        line = line.substr(position+1) ;
                        mapVector->push_back(getMapByString(line)) ;
                        maxMapRows = max(maxMapRows , (mapVector->back())->getNumOfRows()) ;
                        maxMapCols = max(maxMapCols,(mapVector->back())->getNumOfColumns()) ;
                        d -> pushEvent("profile  "+line.substr(position+1)+" started reading" ) ;
                        trialSize++;
                }

        }

        float resizeFactor = (float)(d->getHeight())*0.7f/(float)((maxMapRows+maxPatternRows+2)*IMAGE_WIDTH);

        for( int i = 0 ; i < 6 ; i++ ){
                string filepath ;
                filepath += pathToSkin+"/"+classes[i]+"/ball.png" ;
                SDL_Surface* tmpSurface = load_image(filepath) ;
                if(tmpSurface == NULL){
                        exitFlag = true ;
                        LINFO("path to skin images is invalid! ");
                        break ;
                }
                tmpSurface = SDL_Resize(tmpSurface , resizeFactor , 6 );
                (*classesMap)[i] = tmpSurface;
        }

        if(exitFlag){
                manager.stop();
                evalVector->~vector<string>() ;
                elapsedTimeVector-> ~vector<int>();
                patternVector -> ~vector<Matrix*>() ;
                mapVector -> ~vector<Matrix*>() ;
                classesMap ->~map<int,SDL_Surface*>() ;
                classVector -> ~vector<int>() ;
                return(0) ;
        }
        int bgWidth = (int)((max(maxMapCols,maxPatternCols))*IMAGE_WIDTH*resizeFactor);
        int bgHeight = (int)((maxMapRows+maxPatternRows+2)*IMAGE_WIDTH*resizeFactor);
        SDL_Rect offset;
        offset.x = (d->getWidth() -bgWidth) /2;
        offset.y = (d-> getHeight() - bgHeight) /2;

        SDL_Rect theClip ;
        theClip.x = 0 ;
        theClip.y = 0 ;
        theClip.w = (int)(IMAGE_WIDTH*resizeFactor) ;
        theClip.h = (int)(IMAGE_WIDTH*resizeFactor) ;

        SDL_Rect patternOffset;
        patternOffset.x = (bgWidth - (int)(maxPatternCols*IMAGE_WIDTH*resizeFactor))/2;
        patternOffset.y = (int)((maxMapRows+1)*IMAGE_WIDTH*resizeFactor);

        //here we listen for keystrokes and react based on the keystroke
        d->showCursor(true);
        int trialIndex =0 ;
        int key = 1 ;
        while(true){

                SDL_Surface* nbg = getAFreshSurface(bgWidth,bgHeight);

                d->displayText(stringify(trialIndex)+":"+(*evalVector)[trialIndex]+", et:"+"  " + stringify((*elapsedTimeVector)[trialIndex])+" micro sec",true ,1 );

                for( int r = 0 ; r < (*mapVector)[trialIndex]->getNumOfRows() ;r++ ){
                        for( int c = 0 ; c < (*mapVector)[trialIndex]->getNumOfColumns() ;c++ ){
                                apply_surface((int)(c*IMAGE_WIDTH*resizeFactor),(int)(r*IMAGE_WIDTH*resizeFactor),*((*classesMap)[(*mapVector)[trialIndex]->get(r,c) -1]),*nbg,theClip) ;

                        }
                }

                for( int r = 0 ; r < (*patternVector)[trialIndex]->getNumOfRows() ;r++ ){
                        for( int c = 0 ; c < (*patternVector)[trialIndex]->getNumOfColumns() ;c++ ){
                                apply_surface((int)(c*IMAGE_WIDTH*resizeFactor)+patternOffset.x,(int)(r*IMAGE_WIDTH*resizeFactor)+patternOffset.y,*((*classesMap)[(*classVector)[trialIndex] * (*patternVector)[trialIndex]->get(r,c) -1]),*nbg,theClip) ;

                        }
                }

                d->displaySDLSurfacePatch(nbg , &offset,NULL , -2,false, true);
                do{
                        key = d->waitForKey() ;
                }while(key != 49 && key != 50 && key!= 51);

                if( key == 49 && trialIndex !=0 ){
                        --trialIndex ;
                }else{
                        if( key == 49 && trialIndex ==0  ){
                                trialIndex = trialSize -1 ;
                        }
                }

                if( key == 50 ){
                        trialIndex = (trialIndex+1)%trialSize ;
                }
                dumpSurface(*nbg);
                if( key == 51 ) break ;
        }

        evalVector->~vector<string>() ;
        elapsedTimeVector-> ~vector<int>();
        patternVector -> ~vector<Matrix*>() ;
        mapVector -> ~vector<Matrix*>() ;
        classesMap ->~map<int,SDL_Surface*>() ;
        classVector -> ~vector<int>() ;
        manager.stop();
        return(0) ;
}
#endif // INVT_HAVE_LIBSDL_IMAGE

