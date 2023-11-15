/*!@file AppPsycho/parallel-color-search.c Psychophysics main application for indexing experiment */

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
// 2008-4-24 16:14:27Z nnoori $
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
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <time.h>
#include <sstream>
#include "basic-graphics.H"
#include "AppPsycho/psycho-skin-mapgenerator.h"


 using namespace std;

SDL_Surface* drawTheBoard(Matrix& bmap,map<int,Uint32>& colorMap , uint w , uint h ){

        SDL_Surface* surface = getABlankSurface(w,h) ;
        uint nr = (uint)bmap.getNumOfRows() ;
        uint nc = (uint) bmap.getNumOfColumns() ;
        uint pw = w / ( 2* nc +1);
        uint ph = h / (2 * nr +1) ;
        uint offX = (w - pw*(2*nc-1))/2 ;
        uint offY = (h - ph*(2*nr-1))/2 ;
        for( uint r = 0 ; r < nr ; r++){
                for( uint c = 0 ; c < nc ; c++){
                        fillRectangle(surface,colorMap[bmap.get(r,c)],offX+ (2*c )*pw,offY+ (2*r)*ph  ,pw,ph) ;

                }
        }
        return surface ;
}

extern "C" int main(int argc, char* argv[] ){

        ModelManager manager("Psycho Skin");
        nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));

        MYLOGVERB = LOG_INFO;  // suppress debug messages

        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        d->setEventLog(el);
        // Parse command-line:
        if (manager.parseCommandLine(argc, argv,"<logfilename>", 1 , 3)==false)
                return(1);
        manager.setOptionValString(&OPT_EventLogFileName,manager.getExtraArg(2).c_str() );
        manager.start();

        SDL_Rect offset;
        SDL_Rect targetOffset ;
        targetOffset.x = (d->getWidth()-(d->getWidth())/10)/2 ;;
        targetOffset.y = (d->getHeight() - (d->getHeight())/10)/2 ;
        offset.x = 0 ;
        offset.y = 0 ;
        int k = -1 ;
        map<int,Uint32>* cm = new map<int , Uint32>() ;
        for( int i = 1 ; i <=10; i++){
                        (*cm)[i] = d->getUint32color(PixRGB<byte>(0, 255-12.5*i, 0));
                }

        do{
                d->clearScreen() ;
                srand ( time(NULL) );
                uint displacement = rand()%220 + 35 ;
                Matrix* bm = getARandomMap(30,30,5) ;
                Matrix* tm = getARandomMap(1,2,2) ;
                tm->set(0,0,12);
                tm->set(0,1,13);
                (*cm)[11] = d->getUint32color(PixRGB<byte>(displacement,20,displacement)) ;
                (*cm)[12] = d->getUint32color(PixRGB<byte>(displacement,20,0)) ;
                (*cm)[13] = d->getUint32color(PixRGB<byte>(0,20,displacement)) ;

                SDL_Surface* target = drawTheBoard(*tm,*cm,(d->getWidth())/10,(d->getHeight())/10) ;

                uint distractorNum = rand()%15 ;

                for(uint i = 0 ; i < distractorNum ; i++){
                        (*bm).set(rand()%(bm->getNumOfRows())+1,rand() % bm->getNumOfColumns()+1,11) ;
                }
                int targetFlag = rand()%3 ;
                switch( targetFlag ){
                        case 0 : (*bm).set(rand()%(bm->getNumOfRows())+1,rand() % bm->getNumOfColumns()+1,12) ; break ;
                        case 1 : (*bm).set(rand()%(bm->getNumOfRows())+1,rand() % bm->getNumOfColumns()+1,13) ; break ;
                        default : break ;
                }

                SDL_Surface* board = drawTheBoard(*bm,*cm,d->getWidth(),d->getHeight());

                d->pushEvent("started showing target image");
                d->displaySDLSurfacePatch( target , &targetOffset ,NULL , -2,false, true);
                d->waitFrames(5);
                d->clearScreen();
                /*we let's delay showing the board after the target*/
                d->pushEvent("started showing blink");
                d->displayFixationBlink(-1, -1, 3, 2);
                d->pushEvent("displaying the test image");
                d->displaySDLSurfacePatch(board , &offset,NULL , -2,false, true);

                delete bm ;
                delete tm ;
                k = d->waitForKey() ;
                dumpSurface(board) ;
                dumpSurface(target);

        }while(k!=32);
        delete cm ;
        manager.stop();
        return 0 ;
}


#endif // INVT_HAVE_LIBSDL_IMAGE


