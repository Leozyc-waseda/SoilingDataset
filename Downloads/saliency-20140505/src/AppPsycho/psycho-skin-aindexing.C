/*!@file AppPsycho/psycho-skin-asindexing.c Psychophysics main application for indexing experiment */

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
//written by Nader Noori

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
#include "psycho-skin-resize.h"
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <time.h>
#include "psycho-skin-mapgenerator.h"
#include <sstream>


using namespace std;

ModelManager manager("Psycho Parallel Search");
nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
bool programQuit = false ;
map<string,string> skinFileMap ;
const int NUMBER_OF_CLASSES = 6;
const int IMAGE_WIDTH = 128 ;
const string classes[NUMBER_OF_CLASSES]={"class1","class2","class3","class4","class5","class6"};
int targetExposure ;
int emptyTime ;
//////////////////////////////////////////
int emergencyExit(){
        time_t rawtime;
        time ( &rawtime );
        LINFO("...emergencyExit called @%s .", ctime( &rawtime ));
        d->pushEvent(std::string("experiment ended at ")+ ctime( &rawtime ) );
        manager.stop();
        return -1;
}


//////////////////////////////////////////
// Here we collect subjects response
uint getStaticResponse(){

        int c = -1;
        int conf = -1 ;
        int quit = -1 ;
        do{
                //
                c = -1 ;
                while( c!=49 && c!=50 ){
                        c = d->waitForKey();
                        d->pushEvent("subject decided to report");
                        d->clearScreen() ;
                        if(c!=49 && c!=50  ){
                                d-> displayText("Press 1 for 'yes', 2 for 'no'");
                        }
                }
                if( c == 49){
                        d-> displayText("You chose 'yes' press space-bar to confirm");
                } else {
                        d-> displayText("You chose 'no' press space-bar to confirm");
                }
                conf = d-> waitForKey() ;
                if( conf == 127){
                        quit =  d-> waitForKey() ;
                        if( quit == 127 ){
                                emergencyExit() ;
                        }
                }
        }while(conf!=32) ;

        if( c == 49 ){
                d->pushEvent("positive identification confirmed");
                return 1 ;
        } else {
                d->pushEvent("negative identification confirmed");
                return 0 ;
        }
}


         //////////////////////////////////////////
// Here we collect subjects response
uint getDynamicResponse(int c = -1){

        //d->clearScreen();
        int conf = -1 ;
        int quit = -1 ;
        do{

                while( c!=49 && c!=50 ){
                        d-> displayText("Press 1 for 'yes', 2 for 'no'");
                        c = d->waitForKey();
                        d->pushEvent("subject decided to report");
                }
                if( c == 49){
                        d-> displayText("You chose 'yes' press space-bar to confirm");
                } else {
                        d-> displayText("You chose 'no' press space-bar to confirm");
                }
                conf = d-> waitForKey() ;
                if( conf == 127){
                        quit =  d-> waitForKey() ;
                        if( quit == 127 ){
                                emergencyExit() ;
                        }
                }
                c = -1 ;
        }while(conf!=32) ;

        if( c == 49 ){
                d->pushEvent("positive identification confirmed");
                return 1 ;
        } else {
                d->pushEvent("negative identification confirmed");
                return 0 ;
        }
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
        }else{
                emergencyExit() ;
        }

//Return the optimized image
        return optimizedImage;
}
///////////////////////////////////

SDL_Surface *getAFreshSurface(int w ,  int h){

        SDL_Surface *sur = SDL_CreateRGBSurface(SDL_SWSURFACE, w, h, 32,
                        0x00000000, 0x00000000, 0x00000000, 0x00000000);

        return sur ;

}
////////////////////////////////////

template <class T>
                string stringify(T &i)
{
        ostringstream o ;
        o << i ;
        return o.str();
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
/////////////////////////////////////////////
int readTop(ifstream& inFile){
        char ch[1000] ;
        while( inFile.getline(ch , 1000) ){
                string line  = ch ;
                LINFO("reads : %s" , line.c_str()) ;
                if( line.compare("end_top") == 0 ) break;
                if( line.substr(0,4).compare("skin")==0 ){
                        string::size_type position = line.find("=") ;
                        string skinName = line.substr(0,position) ;
                        string path = line.substr(position+1) ;
                        skinFileMap[skinName] = path ;
                }
                if( line.substr(0,15).compare("target_exposure")==0 ){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        targetExposure = atoi(rf);
                        d->pushEvent(line.c_str());
                }
                if( line.substr(0,10).compare("empty_time")==0 ){
                        string::size_type position = line.find("=") ;
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        emptyTime = atoi(rf);
                }

        }
        return 0 ;
}
/////////////////////////////////////////////
void showStaticTarget(Matrix& pattern , SDL_Surface* classSkin){
        int pw = classSkin->w;
        int ph = classSkin->h ;
        SDL_Rect theClip ;
        theClip.x = 0 ;
        theClip.y = 0 ;
        theClip.w = pw ;
        theClip.h = pw ;
        SDL_Surface* blankSurface = getAFreshSurface(pw*pattern.getNumOfColumns(),pw*pattern.getNumOfRows());
        int nv = ph / pw ;
        for( int r = 0 ; r < pattern.getNumOfRows() ; r++){
                for( int c = 0 ; c < pattern.getNumOfColumns() ; c++){
                        if( pattern.get(r,c)== 1 ){
                                theClip.y = (rand()%nv)*pw ;
                                apply_surface(c*pw,r*pw, *classSkin , *blankSurface ,theClip)        ;
                        }
                }
        }
        d->clearScreen();
        SDL_Rect offset;

        //Give the offsets to the rectangle
        offset.x = (d->getWidth() - blankSurface->w) /2;
        offset.y = (d-> getHeight() - blankSurface->h) /2;
        d->displaySDLSurfacePatch(blankSurface , &offset,NULL , -2,false, true);
        d->waitFrames(targetExposure);
        SDL_FreeSurface(blankSurface);
        theClip.~SDL_Rect();
}
/////////////////////////////////////////
void showDynamicTarget(Matrix& pattern , SDL_Surface* classSkin){
        int pw = classSkin->w;
        int ph = classSkin->h ;
        SDL_Rect theClip ;
        theClip.x = 0 ;
        theClip.y = 0 ;
        theClip.w = pw ;
        theClip.h = pw ;
        SDL_Surface* blankSurface = getAFreshSurface(pw*pattern.getNumOfColumns(),pw*pattern.getNumOfRows());
        int nv = ph / pw ;
        Matrix* phaseMatrix = getARandomMap(pattern.getNumOfRows(),pattern.getNumOfColumns(),nv) ;
        int counter = 0 ;
        while(counter < targetExposure  ){
                for( int r = 0 ; r < pattern.getNumOfRows() ; r++){
                        for( int c = 0 ; c < pattern.getNumOfColumns() ; c++){
                                if( pattern.get(r,c)== 1 ){
                                        theClip.y = ((counter+phaseMatrix->get(r,c)) %nv)*pw ;
                                        apply_surface(c*pw,r*pw, *classSkin , *blankSurface ,theClip)        ;
                                }
                        }
                }


                SDL_Rect offset;

        //Give the offsets to the rectangle
                offset.x = (d->getWidth() - blankSurface->w) /2;
                offset.y = (d-> getHeight() - blankSurface->h) /2;
                d->displaySDLSurfacePatch(blankSurface , &offset,NULL , -2,true, true);
                counter++;
        }

        SDL_FreeSurface(blankSurface);
        phaseMatrix->~Matrix();
        theClip.~SDL_Rect();
}
/////////////////////////////////////////
uint showDynamicSlide(Matrix& theMap , map<string,SDL_Surface*>& cmap ){
        int cs = theMap.getNumOfColumns();
        int rs = theMap.getNumOfRows() ;
        map<string , int> clipNumMap;
        int imageWidth = -1 ;
        int imageHeigth = -1 ;
        int pw = 0 ;
        int maxForPhase = 0;
        //int sT = 100 ;
        for( map<string,SDL_Surface*>:: iterator it = cmap.begin() ; it != cmap.end() ; ++it ){
                string clName = it->first ;
                SDL_Surface* surf = it->second;
                int numOfClips = (surf->h) / (surf->w);
                clipNumMap[clName] = numOfClips;
                maxForPhase = max(maxForPhase,numOfClips);
                if(imageWidth == -1){
                        pw = surf->w ;
                        imageWidth = theMap.getNumOfColumns()*pw ;
                        imageHeigth = theMap.getNumOfRows()*pw ;
                }

        }
        Matrix* phaseMatrix = getARandomMap(rs,cs,maxForPhase) ;

        SDL_Rect* theClip = new SDL_Rect();
        theClip->x = 0 ;
        theClip->y = 0 ;
        theClip->w = pw ;
        theClip->h = pw ;
        //
        SDL_Surface* nbg = getAFreshSurface(imageWidth,imageHeigth);
        string cname("") ;
        int key = -1 ;
        int counter = 0 ;
        while( key < 0 ){
                for( int r = 0 ; r < rs ; r++){
                        for( int c = 0 ; c < cs ; c++){
                                switch( theMap.get(r,c) ){
                                        case 1 : cname = "class1" ;break ;
                                        case 2 : cname = "class2" ; break ;
                                        case 3 : cname = "class3" ; break ;
                                        case 4 : cname = "class4" ; break ;
                                        case 5 : cname = "class5" ; break ;
                                        case 6 : cname = "class6" ; break ;
                                }
                                SDL_Surface* csurf = cmap[cname] ;
                                int y = ( (counter+phaseMatrix->get(r,c)) % clipNumMap[cname])*pw;
                                theClip->y = y ;
                                apply_surface(c*pw,r*pw, *csurf , *nbg ,*theClip)        ;
                        }
                }
        //d->clearScreen();
                SDL_Rect offset;
                offset.x = (d->getWidth() - nbg->w) /2;
                offset.y = (d-> getHeight() - nbg->h) /2;
                d->pushEvent("displaying frame "+ stringify(counter));
                d->displaySDLSurfacePatch(nbg , &offset,NULL , -2,true, true);
                counter++;
                key = d->checkForKey();
                if( key!=-1 ) {
                        d->pushEvent("subject decided to report");
                        d->clearScreen();
                        break ;
                }
        }

        delete phaseMatrix;
        delete theClip;
        SDL_FreeSurface(nbg);
        return key ;

}

/////////////////////////////////////////
void showStaticSlide(Matrix& theMap , map<string,SDL_Surface*>& cmap ){
        int cs = theMap.getNumOfColumns();
        int rs = theMap.getNumOfRows() ;
        map<string , int> clipNumMap;
        int imageWidth = -1 ;
        int imageHeigth = -1 ;
        int pw = 0 ;

        for( map<string,SDL_Surface*>:: iterator it = cmap.begin() ; it != cmap.end() ; ++it ){
                string clName = it->first ;
                SDL_Surface* surf = it->second;
                int numOfClips = (surf->h) / (surf->w);
                clipNumMap[clName] = numOfClips;
                if(imageWidth == -1){
                        pw = surf->w ;
                        imageWidth = theMap.getNumOfColumns()*pw ;
                        imageHeigth = theMap.getNumOfRows()*pw ;

                }

        }
        SDL_Rect* theClip = new SDL_Rect();
        theClip->x = 0 ;
        theClip->y = 0 ;
        theClip->w = pw ;
        theClip->h = pw ;
        //
        SDL_Surface* nbg = getAFreshSurface(imageWidth,imageHeigth);
        for( int r = 0 ; r < rs ; r++){
                for( int c = 0 ; c < cs ; c++){
                        const int ti = theMap.get(r,c);
                        string cname = "class"+stringify(ti);
                        SDL_Surface* csurf = cmap[cname] ;
                        int y = (rand() % clipNumMap[cname])*pw;
                        theClip->y = y ;
                        apply_surface(c*pw,r*pw, *csurf , *nbg ,*theClip)        ;
                }
        }
        d->clearScreen();
        SDL_Rect offset;
        offset.x = (d->getWidth() - nbg->w) /2;
        offset.y = (d-> getHeight() - nbg->h) /2;
        d->clearScreen();
        d->pushEvent("displaying the test image");
        d->displaySDLSurfacePatch(nbg , &offset,NULL , -2,false, true);
        SDL_FreeSurface(nbg);
        delete theClip;
}
/////////////////////////////////////////////
void doStaticTraining( map<string,SDL_Surface*>& cmap,vector<string>& messages){
        d->pushEvent("training starts") ;
        d->displayText("Training Time! Press any key to start");
        d->showCursor(true);
        d->waitForKey() ;
        //////
        for( uint i = 0 ; i< messages.size() ; i++){
                d->displayText(messages[i]);
                d->pushEvent("displayed training message : " + messages[i]);
                d->waitFrames(300);
        }

        /////
        d->pushEvent("training ends");
//d->pushEvent("Congradulations! You are ready for the real test");
        d->waitFrames(45);
        d->displayText("Press any key to start!") ;
        d->waitForKey();
//d->showCursor(false);
}

void doDynamicTraining( map<string,SDL_Surface*>& cmap,vector<string>& messages){
        d->pushEvent("training starts") ;
        d->displayText("Training Time! Press any key to start");
        d->showCursor(true);
        d->waitForKey() ;
        //////
        for( uint i = 0 ; i< messages.size() ; i++){
                d->pushEvent(messages[i]);
        }

        /////
        d->pushEvent("training ends");
//d->pushEvent("Congradulations! You are ready for the real test");
        d->waitFrames(45);
        d->displayText("Press any key to start!") ;
        d->waitForKey();
        d->showCursor(false);
}
/////////////////////////////////////////////
int readBlock(ifstream& inFile){
        string block_name("") ;
        string pattern("") ;
        string skin_name("") ;
        int px = 0 ;
        int py = 0 ;
        int rows = 0 ;
        int columns = 0 ;
        float rf = 1.0f ;
        float prb = 0.5f ;
        bool staticFlag = true ;
        bool pureFlag = true ;
        char ch[1000];
        bool trainingFlag = false ;
        bool pureFlag = true ;
        string trainingMessage("") ;
        vector<string>* messages = new vector<string>() ;
        vector<string>* exclusionList = new vector<string>();
        float minAccuracy = 0 ;
        float minSensitivity = 0 ;
        int minPositiveCase = 75;
        int maxNumOfTrials = 180 ;

        /*let's read attributes of the block*/
        while( inFile.getline(ch , 1000) ){
                string line = ch ;
                LINFO("reads : %s" , line.c_str()) ;

                if(line.compare("dynamic")==0) {
                        staticFlag = false ;
                }

                if(line.compare("mixed")==0) {
                        pureFlag = false ;
                        d ->pushEvent("mixed stimuli");
                }

                if(line.compare("end_block")==0) {
                        d -> pushEvent("ended reading block "+ block_name);
                        break ;
                }
                if( line.substr(0,10).compare("block_name") == 0 ){
                        string::size_type position = line.find("=") ;
                        block_name = line.substr(position+1) ;
                        d -> pushEvent("started reading block "+block_name) ;
                }

                if( line.substr(0,14).compare("target_pattern") == 0 ){
                        string::size_type position = line.find("=") ;
                        pattern = line.substr(position+1) ;
                        d -> pushEvent("target pattern "+pattern) ;
                }

                if( line.substr(0,16).compare("excluded_pattern") == 0 ){
                        string::size_type position = line.find("=") ;
                        string exPattern = line.substr(position+1);
                        exclusionList->push_back(exPattern ) ;
                        d -> pushEvent("excluded pattern "+exPattern) ;
                }

                if( line.substr(0,4).compare("skin") == 0 ){
                        string::size_type position = line.find("=") ;
                        skin_name = line.substr(position+1) ;
                }

                if(line.substr(0,2).compare("rf")==0){
                        string::size_type position = line.find("=");
                        char x[10] ;
                        strcpy(x,line.substr(position+1).c_str());
                        rf = atof(x);
                }
                if(line.substr(0,11).compare("probability")==0){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        prb = atof(rf);
                }
                if(line.substr(0,2).compare("px")==0){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        px = atoi(rf);
                }
                if(line.substr(0,2).compare("py")==0){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        py = atoi(rf);
                }
                if(line.substr(0,2).compare("rs")==0){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        rows = atoi(rf);
                }
                if(line.substr(0,2).compare("cs")==0){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        columns = atoi(rf);
                }
                if(line.substr(0,16).compare("minimum_positive")==0){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        minPositiveCase = atoi(rf);
                }

                if(line.substr(0,13).compare("maximum_trial")==0){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        maxNumOfTrials = atoi(rf);
                }


                if( line.substr(0,12).compare("min_accuracy")==0 ){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position + 1 ).c_str());
                        minAccuracy = atof(rf);
                }
                if( line.substr(0,15).compare("min_sensitivity")==0 ){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position + 1 ).c_str());
                        minSensitivity = atof(rf);
                }
                if( line.substr(0,16).compare("training_message")==0 ){
                        string::size_type position = line.find("=");
                        trainingFlag = true ;
                        trainingMessage = line.substr(position +1) ;
                        messages->push_back(trainingMessage);
                }

        }

        /*let's load up the sprites associated with the skin and resize them and store them in a map*/
        map<string,SDL_Surface*>* classesMap = new map<string,SDL_Surface*>();
        for( int i = 0 ; i < 6 ; i++ ){
                string filepath ;
                filepath += skinFileMap[skin_name]+"/"+classes[i]+"/ball.png" ;
                SDL_Surface* tmpSurface = load_image(filepath) ;
                tmpSurface = SDL_Resize(tmpSurface , rf , 6 );
                (*classesMap)[classes[i]] = tmpSurface;
        }

/*now we find all Euclidean transormations of the patten and store them in a vector
        in every trial one of these pattens will be used (either included or excluded)
*/
        Matrix *rawPattenMatrix = getPattenByString(pattern) ;
        vector<Matrix*>* vars = getAllVariations(*rawPattenMatrix);
        int vs = vars->size() ;

        /*Let's start training the subject*/
        d->displayText("Press any key to start!") ;
        d->waitForKey() ;
        if( trainingFlag ){
                if( staticFlag ){
                        doStaticTraining(*classesMap,*messages) ;
                } else {
                        doDynamicTraining(*classesMap,*messages) ;
                }

        }

/*the real test starts here and unless the required sensitivity is reached or prbability of having
        positive cases it will continue
*/
        int positiveCases = 0 ;
        vector<int>* experiment = new vector<int>();
        vector<int>* report = new vector<int>();
        float sensitivity = 0.0f;
        float accuracy = 0.0f ;
        float totalNumOfTrials = 0 ;
        do{


                /*here we start different trials and we will have as many as block_size trials*/
                d->clearScreen();
                d->displayText("Please wait!") ;
                srand ( time(NULL) );
                int patternChannel = rand()%NUMBER_OF_CLASSES ;
                string targetClass = classes[patternChannel];
                srand ( time(NULL) );
                /*here we pick one of the transformed pattern to be used*/
                Matrix *patternMatrix = (*vars)[rand()%vs] ;
                vector<Matrix*>* mapsVector ;
                bool positiveFalg = true ;

                /*here we filp a coin to see if we include the pattern or exclude it*/
                if( rand()%10000 < prb*10000 ){

                        if(pureFlag){
                                mapsVector =  getPureMapsWithExactPatternAndExactChannel(rows,columns,NUMBER_OF_CLASSES , patternChannel+1,*patternMatrix,1,1);
                        }else{
                                mapsVector =  getMapsWithExactPattenAndExactChannel(rows,columns,NUMBER_OF_CLASSES , patternChannel+1,*patternMatrix,1,1);
                        }


                        /*we keep in mind that we chose to include the target in the board*/
                        experiment-> push_back(1) ;
                }else{
                        mapsVector =  getMapsWithExactPattenAndExactChannel(rows,columns,NUMBER_OF_CLASSES , patternChannel+1,*patternMatrix,0,1);
                        /*we keep in mind that we chose to exclude the target from the board*/
                        experiment -> push_back(0) ;
                        positiveFalg = false ;
                }
                if( positiveFalg ) positiveCases++ ;
                Matrix *map = (*mapsVector)[0] ;
                d->pushEvent("class picked : " + targetClass) ;
                d->pushEvent("pattern picked : "+patternMatrix->toFormattedString()) ;
                d->pushEvent("map picked :" +map->toFormattedString());

                /*now that we have picked the target patten and the board map let's start showing the
                target
                */
                d->clearScreen();
                d->pushEvent("started showing target image");
                if( staticFlag ){
                        showStaticTarget(*patternMatrix,(*classesMap)[targetClass]) ;
                } else {
                        showDynamicTarget(*patternMatrix,(*classesMap)[targetClass]) ;
                }
                /*we let's delay showing the board after the target*/
                d->pushEvent("started showing blink");
                d->displayFixationBlink(-1, -1, emptyTime/2, 2);
                /*now that the target is shown we can show the board*/

                if( staticFlag ){
                        showStaticSlide(*map,*classesMap );
                        report->push_back(getStaticResponse());
                } else {
                        int resp = showDynamicSlide(*map,*classesMap);
                        report->push_back(getDynamicResponse(resp));
                }


        /*now that we are done with trials let's do some book keeping and see what was subject's sensitivity and
                accuracy
        */

                float truePositives = 0.0f ;
                float trueNegatives = 0.0f ;
                float falsePositives = 0.0f ;
                float falseNegatives = 0.0f ;
                float positives = 0.0f ;
                float negatives = 0.0f ;
                vector<int>::size_type trialSize = experiment->size() ;
                for( uint i = 0 ; i < trialSize ; i++ ){
                        if( (*experiment)[i]==1 && (*report)[i]==1 ) {truePositives++ ; positives++ ; }
                        if( (*experiment)[i]==1 && (*report)[i]==0 ) {falseNegatives++ ; positives++ ; }
                        if( (*experiment)[i]==0 && (*report)[i]==1 ) {falsePositives++ ;negatives++ ; }
                        if( (*experiment)[i]==0 && (*report)[i]==0 ) {trueNegatives++ ; negatives++ ;}
                }
                if( positives != 0 ){
                        sensitivity = truePositives / positives ;
                }
                totalNumOfTrials = (positives+negatives) ;
                accuracy = (truePositives + trueNegatives) / totalNumOfTrials ;


                d->pushEvent("positiveCases: "+stringify(positiveCases)+"  minPositiveCase:"+stringify(minPositiveCase));
                d->pushEvent("accuracy: "+stringify(accuracy)+"   minAccuracy:"+stringify(minAccuracy)) ;
                d->pushEvent("sensitivity: "+stringify(sensitivity)+"    minSensitivity"+stringify(minSensitivity));

        }while( (sensitivity < minSensitivity || accuracy < minAccuracy || positiveCases < minPositiveCase ) && (totalNumOfTrials <= (float)maxNumOfTrials));

        string reportString("") ;
        string experimentString("");
        for(uint i = 0 ; i < experiment->size() ; i++){
                experimentString += stringify((*experiment)[i]) ;
                reportString += stringify((*report)[i]) ;
        }
        d->pushEvent("experiment string :" + experimentString);
        d->pushEvent("report string:" + reportString) ;

        /*and here we do necessary cleanups before finishing up the block*/
        for( map<string,SDL_Surface*>::iterator it=classesMap->begin() ; it!= classesMap->end() ; ++it){
                SDL_FreeSurface(it->second);
        }
        delete classesMap;
        delete rawPattenMatrix;
        delete vars;
        delete messages;
        delete exclusionList;
        d-> displayText("You may take a short break!") ;
        d->waitForKey() ;
        return 0 ;
}
/////////////////////////////////////////////
int readprofile(const char* filename){
        ifstream inFile(filename, ios::in);
        if (! inFile)
        {
                LINFO("profile '%s' not found!" , filename);
                return -1;
        }

        char ch[1000];
        while (inFile.getline(ch , 1000)){
                string line = ch;
                LINFO("reads : %s", line.c_str());
                if( line.substr(0,11).compare("profilename") == 0 ){
                        string::size_type position = line.find("=");
                        d -> pushEvent("profile  "+line.substr(position+1)+" started reading" ) ;
                }
                if(line.compare("start_top")==0 ) readTop(inFile);
                if(line.compare("start_block")==0) readBlock(inFile);
                if(programQuit == true) break ;
        }

        return 0 ;
}
        /////////////////////////////////////////////


extern "C" int main( int argc, char* argv[] )

{

        MYLOGVERB = LOG_INFO;
        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        d->setEventLog(el);
        // Parse command-line:
        if (manager.parseCommandLine(argc, argv,"<profile.prf> <logfileName.psy>", 2, 2)==false)
                return(1);
        manager.setOptionValString(&OPT_EventLogFileName,manager.getExtraArg(1).c_str() );
        manager.start();
        d->clearScreen() ;
        readprofile(manager.getExtraArg(0).c_str());
        d->displayText("Experiment is done! Thanks for participating! ") ;
        d->waitForKey() ;
        manager.stop();
        return(0) ;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

