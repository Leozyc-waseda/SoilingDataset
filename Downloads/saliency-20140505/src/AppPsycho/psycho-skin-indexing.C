/*!@file AppPsycho/psycho-skin-indexing.C Psychophysics display of different skins for a gameboard */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-skin-indexing.C $
// $Id: psycho-skin-indexing.C 14376 2011-01-11 02:44:34Z pez $
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


using namespace std;


//The attributes of the screen
const int SCREEN_WIDTH = 840;
const int SCREEN_HEIGHT = 680;
const int SCREEN_BPP = 32;
const int NUMBER_OF_CLASSES = 6;
const int MAX_CELLS = 400 ;
const int IMAGE_WIDTH = 128 ;
const string classes[NUMBER_OF_CLASSES]={"class1","class2","class3","class4","class5","class6"};
SDL_Event event ;
int delay = 15 ;
bool programQuit = false ;
map<string,map<string,SDL_Surface*> > skinsmap ;
ModelManager manager("Psycho Skin Indexing");
nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));

//////////////////////////////////////////

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
    }else{cout<<filename;}

    //Return the optimized image
    return optimizedImage;
}
///////////////////////////////////

SDL_Surface *getAFreshSurface(int w ,  int h){

        SDL_Surface *sur = SDL_CreateRGBSurface(SDL_SWSURFACE, w, h, 32,
                                   0x00000000, 0x00000000, 0x00000000, 0x00000000);

        return sur ;

}

///////////////////////////////////


void dumpSurface(SDL_Surface * surface){

 SDL_FreeSurface( surface );
}

/////////////////////////////////////////////

void apply_surface( int x, int y, SDL_Surface* source, SDL_Surface* destination , SDL_Rect* clip = NULL)
{
    //Make a temporary rectangle to hold the offsets
    SDL_Rect offset;

    //Give the offsets to the rectangle
    offset.x = x;
    offset.y = y;

    //Blit the surface
    SDL_BlitSurface( source, clip, destination, &offset );
}


void clean_up()
{

    for( map<string,map<string,SDL_Surface*> >::iterator ii=skinsmap.begin(); ii!=skinsmap.end(); ++ii)
   {
           for(map<string,SDL_Surface*>::iterator jj=((*ii).second).begin() ; jj!= ((*ii).second).end() ; ++jj){
                    SDL_FreeSurface(jj->second);
           }

   }

    SDL_Quit();
}
/////////////////////////////////////
void readTop(ifstream* inFile){
//        tempSur = load_image("skin1/class1/ball.png");
        char ch[1000];
   while ((*inFile).getline(ch , 1000)){
                   string line = ch;
                LINFO("top reader reads : '%s'", line.c_str());
                   if(line.compare("end top")==0) break;
                string::size_type position = line.find("=");
                string skinname = line.substr(0, position);
                string path = line.substr(position+1);
                map<string,SDL_Surface*> themap;

                for(int i = 0 ; i < NUMBER_OF_CLASSES ; i++){
                        string filepath ;
                        filepath += path+"/"+classes[i]+"/ball.png" ;

                          themap[classes[i]]= load_image(filepath) ;
                }
                skinsmap[skinname] = themap;

        }

}

void showStaticSlide(string *slideMap,string skinName , int col , int row , int px , int py , float rf){

d->clearScreen();
        map<string,SDL_Surface*> resizedSurfacesMap;
                  map<string,int> clipNumMap;
                  SDL_Rect theClip ;
                  theClip.x = 0 ;
                theClip.y = 0 ;
                theClip.w = (int)(rf* IMAGE_WIDTH) ;
                theClip.h = (int)(rf* IMAGE_WIDTH) ;
                  //bool slideQuit = false ;
                SDL_Surface* nbg = getAFreshSurface((int)(IMAGE_WIDTH*col*rf), (int)(IMAGE_WIDTH*row*rf));


                  //this block discovers how many clips are there in a sprite and put the number in a  map
                  //moreover it will construct required clips and put them in a map too
                  for(map<string,SDL_Surface*>::iterator jj=(skinsmap[skinName]).begin() ; jj!= (skinsmap[skinName]).end() ; ++jj )
                  {
                          int numOfImages = ((jj->second)->h)/IMAGE_WIDTH ;
                    clipNumMap[jj->first] = numOfImages ;
                    //cout<<"Number of clips : "<<(jj->second)->h<<endl;

                 }

                //this block makes a map from name of classes to the resized surface of each class
                for(map<string,SDL_Surface*>::iterator jj=(skinsmap[skinName]).begin() ; jj!= (skinsmap[skinName]).end() ; ++jj )
                  {
                          SDL_Surface *tmpSurface = getAFreshSurface(IMAGE_WIDTH,IMAGE_WIDTH);
                          *tmpSurface = *(jj->second);
                          tmpSurface = SDL_Resize(tmpSurface , rf , 3 );
                          resizedSurfacesMap[jj->first] = tmpSurface;

                 }




                 for(map<string,SDL_Surface*>::iterator jj=(skinsmap[skinName]).begin() ; jj!= (skinsmap[skinName]).end() ; ++jj){
                                string className = jj->first;



                                   for(int i = 0 ; i < col*row ; i++){

                                           if(slideMap[i].compare( className )==0){
                                                   int clipY = (int)((rand()% clipNumMap[className])*IMAGE_WIDTH*rf);
                                                   theClip.y = clipY ;
                                                apply_surface((int)((i%col)*IMAGE_WIDTH*rf),(int)((i/col)*IMAGE_WIDTH*rf),resizedSurfacesMap[className],nbg,&theClip) ;
                                           }
                                   }



                           }
                SDL_Rect offset;

                    //Give the offsets to the rectangle
                    offset.x = px;
                    offset.y = py;
                d->displaySDLSurfacePatch(nbg , &offset,NULL , -2,false, true);
                      d->waitNextRequestedVsync(false, true);
                d->waitForKey();


                for(map<string,SDL_Surface*>::iterator jj=(skinsmap[skinName]).begin() ; jj!= (skinsmap[skinName]).end() ; ++jj )
                                                                  {
                                                                    dumpSurface(resizedSurfacesMap[jj->first]);
                                                                 }
                dumpSurface(nbg) ;

}


void showDynamicSlide(string *slideMap,string skinName , int col , int row , int px , int py , float rf){

                  map<string,SDL_Surface*> resizedSurfacesMap;
                  map<string,int> clipNumMap;
                int phase[col*row] ;
                  SDL_Rect theClip ;
                  theClip.x = 0 ;
                theClip.y = 0 ;
                theClip.w = (int)(rf* IMAGE_WIDTH) ;
                theClip.h = (int)(rf* IMAGE_WIDTH) ;
                  bool slideQuit = false ;
                  SDL_Surface *nbg = getAFreshSurface((int)(IMAGE_WIDTH*col*rf), (int)(IMAGE_WIDTH*row*rf));

                  //this block discovers how many clips are there in a sprite and put the number in a  map
                  //moreover it will construct required clips and put then in a map too
                  for(map<string,SDL_Surface*>::iterator jj=(skinsmap[skinName]).begin() ; jj!= (skinsmap[skinName]).end() ; ++jj )
                  {
                          int numOfImages = ((jj->second)->h)/IMAGE_WIDTH ;
                            clipNumMap[jj->first] = numOfImages ;

                 }

                //this block makes a map from name of classes to the resized surface of each class
                for(map<string,SDL_Surface*>::iterator jj=(skinsmap[skinName]).begin() ; jj!= (skinsmap[skinName]).end() ; ++jj )
                  {
                          SDL_Surface* tmpSurface = getAFreshSurface(IMAGE_WIDTH,IMAGE_WIDTH );
                          *tmpSurface = *(jj->second);
                          tmpSurface = SDL_Resize(tmpSurface , rf , 3 );
                          resizedSurfacesMap[jj->first] = tmpSurface;

                 }

                for(int i = 0 ; i < col*row ; i++){

                        phase[i] = rand()% clipNumMap[*(slideMap+i)];
                }

                int l = 0 ;

                 while( slideQuit == false ) {

                        while(SDL_PollEvent( &event )){

                                if(event.type == SDL_KEYDOWN){
                                        if( event.key.keysym.sym == SDLK_SPACE )
                                    {
                                        slideQuit = true ;
                            }
                            if( event.key.keysym.sym == SDLK_ESCAPE )
                                    {
                                        slideQuit = true ;
                                        programQuit = true ;
                            }
                            if(event.key.keysym.sym == SDLK_UP){
                                                if (delay > 2 )delay -= 2 ;
                                        }

                                        if(event.key.keysym.sym == SDLK_DOWN){
                                                if (delay < 500 )delay += 2 ;
                                        }

                                }


                            if( event.type == SDL_QUIT || event.key.keysym.sym == SDLK_ESCAPE )
                            {
                                programQuit = true;
                                slideQuit = true ;
                            }


                        }

                        for(map<string,SDL_Surface*>::iterator jj=(skinsmap[skinName]).begin() ; jj!= (skinsmap[skinName]).end() ; ++jj){
                                string className = jj->first;

                                   for(int i = 0 ; i < col*row ; i++){

                                        int clipY = (int)(((l+phase[i]) % clipNumMap[className])*IMAGE_WIDTH*rf);
                                        theClip.y = clipY ;

                                           if(slideMap[i].compare( className )==0){
                                                   apply_surface((int)((i%col)*IMAGE_WIDTH*rf),(int)((i/col)*IMAGE_WIDTH*rf),resizedSurfacesMap[className],nbg,&theClip);
                                           }
                                   }


                           }

                SDL_Rect offset;

                    //Give the offsets to the rectangle
                    offset.x = px;
                    offset.y = py;
                d->displaySDLSurfacePatch(nbg , &offset ,NULL,  -2,true, true);

                        l++ ;
                        SDL_Delay( delay );
                }


                for(map<string,SDL_Surface*>::iterator jj=(skinsmap[skinName]).begin() ; jj!= (skinsmap[skinName]).end() ; ++jj )
                                                                  {
                                                                    dumpSurface(resizedSurfacesMap[jj->first]);
                                                                 }
                dumpSurface(nbg) ;

}


/*
 * Reads part of profile starting with slide till end of the part indicated by end slide
 * */
void readSlide(ifstream* inFile){
        char ch[1000];
        string skinName;
        float resizefactor = 1.0f;
        int px = 0, py = 0, rows = 0, columns = 0;
        string slideMap[MAX_CELLS];
        bool isDynamic = false ;
        d->clearScreen() ;
        d->displayRedDotFixationBlink();
    d->pushEvent(std::string("===== Reading Slide ====="));
   while ((*inFile).getline(ch , 1000)){
           string line = ch;
        LINFO("slide reader reads : '%s'...", line.c_str());
           d->pushEvent(line.c_str());
           if(line.substr(0,4).compare("skin")==0){
                   string::size_type position = line.find("=");
                skinName = line.substr(position+1);

           }
        if(line.substr(0,2).compare("rf")==0){
                   string::size_type position = line.find("=");
                   char rf[10] ;
                   strcpy(rf,line.substr(position+1).c_str());
                resizefactor = atof(rf);
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

           if(line.substr(0,6).compare("static")==0){
                   isDynamic = false ;
           }

           if(line.substr(0,7).compare("dynamic")==0){
                   isDynamic = true ;
           }

           if(line.substr(0,3).compare("map")==0){
                   int p = 3 ;
                   string cs = "class";
                   for (int i =0 ; i < rows*columns ; i++){
                           int pos = p+1 +i*2 ;
                           string cn = line.substr(pos ,1);
                           slideMap[i] = cs+cn;
                   }
                   if(isDynamic == true){

                           showDynamicSlide(&slideMap[0],skinName,columns,rows,px,py,resizefactor) ;

                   }else{
                           showStaticSlide(&slideMap[0],skinName,columns,rows,px,py,resizefactor) ;
                   }


           }



           if(line.compare("end slide")==0) break;


        }

}

int readprofile(const char* filename){
        ifstream inFile(filename, ios::in);
        if (! inFile)
   {
      return -1;
   }

   char ch[1000];
   while (inFile.getline(ch , 500)){
           string line = ch;
        LINFO("profile reader reads : '%s'...", line.c_str());
           if(line.compare("top")==0 ) readTop(&inFile);

           if(line.compare("slide")==0) readSlide(&inFile);
           if(programQuit == true) break ;
        }

        return 0 ;
}


extern "C" int main( int argc, char* args[] ){
        MYLOGVERB = LOG_INFO;

          manager.addSubComponent(d);
          nub::soft_ref<EventLog> el(new EventLog(manager));
          manager.addSubComponent(el);

          manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");

          d->setEventLog(el);

          manager.start();

          // let's display an ISCAN calibration grid:
          d->clearScreen();
          d->displayISCANcalib();
          d->waitForKey();
        // let's do an eye tracker calibration:
          d->displayText("<SPACE> to calibrate; other key to skip");
          //int c = 
	     d->waitForKey();
         // if (c == ' ') d->displayEyeTrackerCalibration();

          d->clearScreen();
          d->displayText("<SPACE> for random play; other key for ordered");
          //c = 
	    d->waitForKey();

        d->displayFixation();

              // ready to go whenever the user is ready:
              d->waitForKey();
              d->waitNextRequestedVsync(false, true);


        string filename;
        if(argc == 2) {
                if(readprofile(args[1])< 0) return -1;
        }else{
                cerr << "usage : " << args[0] << " file_name.exp \n";
                return -1 ;
        }

        clean_up();
        manager.stop() ;
        return 0;
}

#endif // INVT_HAVE_LIBSDL_IMAGE
