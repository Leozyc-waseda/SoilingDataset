/*!@file AppPsycho/psycho-skin-bsindexing.c Psychophysics main application for indexing experiment */

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
#include "GameBoard/basic-graphics.H"


using namespace std;


ModelManager manager("Psycho Constellation");
nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
bool programQuit = false ;
map<string,Uint32> colorMap ;
map<string,string> starColorMap ;
map<string,int> starRadiusMap ;
map<string,string> starTypeMap ;
map<string,int> starDensityMap ;
map<string,int> starFlickingMap ;
int vx , vy ;
int sw=800 ;
int sh=600 ;


//////////////////////////////////////////

void startExperiment(){
        cout<<" experiment started "<< starColorMap.size()<<endl;
        SDL_Surface* blankSurface = getABlankSurface(sw,sh) ;
        int screenSize = sw * sh ;

        for( map<string,string>:: iterator it = starColorMap.begin() ; it != starColorMap.end() ; ++it ){
                string starName = it->first ;
                int starCount = screenSize*starDensityMap[starName]/300000 ;
                for(int i = 0 ; i < starCount ; i++){
                        int x = rand()% (sw-150);
                        int y = rand()% (sh-150);
                        //cout<<"starcolor: "<< colorMap[starColorMap[starName]]<<"    "<<x<<"  "<<y<<endl;
                        //fillOval(blankSurface,colorMap[starColorMap[starName]],x,y,2,2) ;
                        fillQuadricRadiant(blankSurface,colorMap[starColorMap[starName]],x+50,y+50,starRadiusMap[starName]) ;
                }


        }

        SDL_Rect offset;
        d->clearScreen() ;
        //Give the offsets to the rectangle
        for (int i = 0 ; i < 100 ; i++){
                offset.x = i + (d->getWidth() - blankSurface->w) /2;
                offset.y = i+ (d-> getHeight() - blankSurface->h) /2;
                d->displaySDLSurfacePatch(blankSurface , &offset,NULL , -2,false, true);
                d->waitFrames(1);
        }



        d->waitForKey() ;
}

//////////////////////////////////////////
int emergencyExit(){
        time_t rawtime;
        time ( &rawtime );
        LINFO("...emergencyExit called @%s .", ctime( &rawtime ));
        d->pushEvent(std::string("experiment ended at ")+ ctime( &rawtime ) );
        manager.stop();
        return -1;
}


////////////////////////////////////

template <class T>
                string stringify(T &i)
{
        ostringstream o ;
        o << i ;
        return o.str();
}


/////////////////////////////////////////////
int readTop(ifstream& inFile){
        char ch[1000] ;
        while( inFile.getline(ch , 1000) ){
                string line  = ch ;
                LINFO("reads : %s" , line.c_str()) ;
                if( line.compare("end_top") == 0 ) break;

                if( line.substr(0,6).compare("color:")==0 ){
                        char rf[3] ;
                        string::size_type colPos = line.find(":") ;
                        string::size_type eqPos = line.find("=") ;
                        string colorName = line.substr(colPos+1,eqPos-colPos-1) ;
                        string colorString = line.substr(eqPos+1);
                        string::size_type  _Pos = colorString.find("_") ;
                        strcpy(rf,colorString.substr(0,_Pos).c_str()) ;
                        int rL = atoi(rf) ;
                        colorString = colorString.substr(_Pos+1) ;
                        _Pos = colorString.find("_") ;
                        strcpy(rf,colorString.substr(0,_Pos).c_str()) ;
                        int gL = atoi(rf) ;
                        colorString = colorString.substr(_Pos+1) ;
                        strcpy(rf,colorString.c_str()) ;
                        int bL = atoi(rf) ;
                        Uint32 color = d->getUint32color(PixRGB<byte>(rL, gL, bL));
                        colorMap[colorName] = color ;
                }
                if( line.substr(0,2).compare("vx")==0 ){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        vx = atoi(rf);
                        d->pushEvent(line.c_str());
                }
                if( line.substr(0,2).compare("vy")==0 ){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        vy = atoi(rf);
                        d->pushEvent(line.c_str());
                }

        }
        return 0 ;
}

/////////////////////////////////////////////

int readStarBlock(ifstream& inFile){
        char ch[1000] ;
        string starName = "" ;
        while( inFile.getline(ch , 1000) ){
                string line  = ch ;
                LINFO("reads : %s" , line.c_str()) ;
                if(line.compare("end_star")==0) {
                        d -> pushEvent("ended reading star block ");
                        break ;
                }

                if(line.substr(0,8).compare("starname")== 0 ) {
                        string::size_type eqPos = line.find("=");
                        starName =line.substr(eqPos+1) ;
                }
                if(line.substr(0,6).compare("color=")==0){
                        string::size_type eqPos = line.find("=");
                        string colorStr = line.substr(eqPos+1) ;
                        starColorMap[starName] = colorStr ;
                        cout<<"local star Name :" <<starName<<" and color:"<< colorStr<<endl ;
                }
                if(line.substr(0,4).compare("type")==0){
                        string::size_type eqPos = line.find("=");
                        string starType = line.substr(eqPos+1) ;
                        starTypeMap[starName] = starType ;
                }

                if( line.substr(0,6).compare("radius")==0 ){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        int s = atoi(rf);
                        starRadiusMap[starName] = s ;
                }
                if( line.substr(0,7).compare("density")==0 ){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        int s = atoi(rf);
                        starDensityMap[starName] = s ;
                }
                if( line.substr(0,8).compare("flicking")==0 ){
                        string::size_type position = line.find("=");
                        char rf[10] ;
                        strcpy(rf,line.substr(position+1).c_str());
                        int s = atoi(rf);
                        starFlickingMap[starName] = s ;
                }


        }
        return 0 ;
}



/////////////////////////////////////////////
int readBlock(ifstream& inFile){
        char ch[1000] ;
        while( inFile.getline(ch,1000) ){
                string line = ch ;
                LINFO("reads :  %s" , line.c_str());
                if(line.compare("end_block")== 0) {
                        d -> pushEvent("ended reading block") ;
                        break ;
                }
                if( line.compare("start_star")==0 ){
                        readStarBlock(inFile) ;
                }
        }
        return 0;
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

        MYLOGVERB = LOG_INFO;  // suppress debug messages

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
        if(readprofile(manager.getExtraArg(0).c_str())== 0 ) startExperiment();
        d->displayText("Experiment is done! Thanks for participating! ") ;
        d->waitForKey() ;
        manager.stop();
        return(0) ;
//         time_t rawtime;
//         time ( &rawtime );
//         LINFO("@%s psycho-skin-bsindexing started:...", ctime(&rawtime));
//         manager.start();
//         d->pushEvent(std::string("started the experiment at ")+ ctime( &rawtime ) );
//         SDL_Surface* free = getAFreshSurface(40,80) ;
//         SDL_Rect offset;
//             //Give the offsets to the rectangle
//         d->clearScreen();
//             offset.x = 100;
//             offset.y = 60;
//         d->displaySDLSurfacePatch(free,&offset,NULL , -2,false, true);
//         d->waitFrames(10);
                //         ////////////////////////////
//         d->clearScreen();
//         offset.x = 100;
//             offset.y = 160;
//         d->displaySDLSurfacePatch(free , &offset,NULL , -2,false, true);
//         d->waitFrames(10);
//           d->displayText("Experiment complete. Thank you!");
//         //d->waitFrames(1);
//           //d->waitForKey();
//         time ( &rawtime );
//         LINFO("...psycho-skin-bsindexing ended @%s .", ctime( &rawtime ));
//         d->pushEvent(std::string("experiment ended at ")+ ctime( &rawtime ) );
//         manager.stop();
                //
//         return 0 ;
}

#endif // INVT_HAVE_LIBSDL_IMAGE

