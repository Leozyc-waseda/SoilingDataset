/*!@file GameBoard/basic-graphics.C some utilities for displaying stimuli*/
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
//written by Nader Noori, April 2008
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include "basic-graphics.H"
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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sstream>
#include "Psycho/PsychoDisplay.H"
#include "Component/ModelOptionDef.H"
#include "Component/OptionManager.H"
#include "GUI/SDLdisplay.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Util/sformat.H"

using namespace std;
/*
 * Set the pixel at (x, y) to the given value
 * NOTE: The surface must be locked before calling this!
 */
void putpixel(SDL_Surface* surface, int x, int y, Uint32 pixel)
{
        int bpp = surface->format->BytesPerPixel;
        /* Here p is the address to the pixel we want to set */
        Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

        switch(bpp) {
                case 1:
                        *p = pixel;
                        break;

                case 2:
                        *(Uint16 *)p = pixel;
                        break;

                case 3:
                        if(SDL_BYTEORDER == SDL_BIG_ENDIAN) {
                                p[0] = (pixel >> 16) & 0xff;
                                p[1] = (pixel >> 8) & 0xff;
                                p[2] = pixel & 0xff;
                        } else {
                                p[0] = pixel & 0xff;
                                p[1] = (pixel >> 8) & 0xff;
                                p[2] = (pixel >> 16) & 0xff;
                        }
                        break;

                case 4:
                        *(Uint32 *)p = pixel;
                        break;
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

        }

//Return the optimized image
        return optimizedImage;
}
///////////////////////////////////

SDL_Surface *getABlankSurface(int w ,  int h){


        SDL_Surface *sur = SDL_CreateRGBSurface(SDL_SWSURFACE, w, h, 32,
                        0x00000000, 0x00000000, 0x00000000, 0x00000000);
        return sur ;

}


///////////////////////////////////
void dumpSurface(SDL_Surface* surface){
        SDL_FreeSurface( surface );
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
void fillRectangle(SDL_Surface* surface , const Uint32 pc , uint offX  , uint offY  , const uint w , const uint h ) {

        /* Lock the screen for direct access to the pixels */
        if ( SDL_MUSTLOCK(surface) ) {
                if ( SDL_LockSurface(surface) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }

        for( uint x = offX ; x <= offX + w ; x++){
                for( uint y = offY ; y < offY + h ; y++){
                        if( (x >= offX) && (x <= offX + w) && (y >= offY ) && (y <= offY + h)){
                                putpixel(surface, x, y, pc);

                        }
                }

        }

        if ( SDL_MUSTLOCK(surface) ) {
                SDL_UnlockSurface(surface);
        }
}
/////////////////////////////////////////////

void drawCircle(SDL_Surface* surface , const Uint32 pc , uint offX  , uint offY  , const uint r , const uint d ){
        /* Lock the screen for direct access to the pixels */
        if ( SDL_MUSTLOCK(surface) ) {
                if ( SDL_LockSurface(surface) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }

        for( float teta = 0.0f ; teta < 6.3f ; teta += 0.05f ){
                for(uint rho = r-d ; rho < r ; rho++ )
                putpixel(surface, offX + (int)(rho * cos(teta))  , offY + (int)(rho * sin(teta)), pc);
        }
        /*
        for( uint x = offX - r ; x <= offX + r ; x++){
                int yt = (int)sqrt(r*r - (x- offX)*(x-offX)) ;
                putpixel(surface, x, yt+offY, pc);
                putpixel(surface, x, offY-yt, pc);
        }*/

        if ( SDL_MUSTLOCK(surface) ) {
                SDL_UnlockSurface(surface);
        }


}

/////////////////////////////////////////////

void drawArc(SDL_Surface* surface , const Uint32 pc , uint offX  , uint offY  , const uint r , const uint d , float teta1, float teta2 , float deltaAlpha){
        /* Lock the screen for direct access to the pixels */
        if ( SDL_MUSTLOCK(surface) ) {
                if ( SDL_LockSurface(surface) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }

        for( float teta =  teta1; teta <= teta2 ; teta += deltaAlpha ){
                for(uint rho = r-d ; rho < r ; rho++ )
                putpixel(surface, offX + (int)(rho * cos(teta))  , offY + (int)(rho * sin(teta)), pc);
        }
        /*
        for( uint x = offX - r ; x <= offX + r ; x++){
                int yt = (int)sqrt(r*r - (x- offX)*(x-offX)) ;
                putpixel(surface, x, yt+offY, pc);
                putpixel(surface, x, offY-yt, pc);
        }*/

        if ( SDL_MUSTLOCK(surface) ) {
                SDL_UnlockSurface(surface);
        }


}
/////////////////////////////////////////////
void drawRectangleFromImage(SDL_Surface* surface , SDL_Surface* patch , uint offX  , uint offY  , const uint w , const uint h , const uint f){
        /* Lock the screen for direct access to the pixels */
        if ( SDL_MUSTLOCK(surface) ) {
                if ( SDL_LockSurface(surface) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }

        if ( SDL_MUSTLOCK(patch) ) {
                if ( SDL_LockSurface(patch) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }

        for(uint x = offX ; x <= offX + w ; x++){

                for( uint d = 1 ; d <= f ; d++){
                        Uint32 pixel = get_pixel32( patch, x, offY+d-1 );
                        put_pixel32(surface, x, offY+d-1, pixel);
                        pixel = get_pixel32( patch, x,  offY+h-d+1 );
                        put_pixel32(surface, x , offY+h-d+1 , pixel) ;
                }
        }
        for( uint y = offY ; y <= offY + h ; y++){
                for( uint d = 1 ; d <= f ; d++){

                        Uint32 pixel = get_pixel32( patch, offX+d-1 , y );
                        putpixel(surface,offX+d-1,y,pixel) ;
                        pixel = get_pixel32( patch, offX+w-d+1 , y );
                        put_pixel32(surface,offX+w-d+1 ,y ,pixel) ;
                }

        }

        if ( SDL_MUSTLOCK(surface) ) {
                SDL_UnlockSurface(surface);
        }
        if ( SDL_MUSTLOCK(patch) ) {
                SDL_UnlockSurface(patch);
        }

}
/////////////////////////////////////////////////////////
Uint32 get_pixel32( SDL_Surface *surface, int x, int y )
{
    //Convert the pixels to 32 bit
        Uint32 *pixels = (Uint32 *)surface->pixels;

    //Get the requested pixel
        return pixels[ ( y * surface->w ) + x ];
}
///////////////////////////////////////////////////////////
void put_pixel32( SDL_Surface *surface, int x, int y, Uint32 pixel )
{
    //Convert the pixels to 32 bit
        Uint32 *pixels = (Uint32 *)surface->pixels;

    //Set the pixel
        pixels[ ( y * surface->w ) + x ] = pixel;
}


/////////////////////////////////////////////

void drawRectangle(SDL_Surface* surface , const Uint32 pc , uint offX  , uint offY  , const uint w , const uint h , const uint f){
        /* Lock the screen for direct access to the pixels */
        if ( SDL_MUSTLOCK(surface) ) {
                if ( SDL_LockSurface(surface) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }

        for(uint x = offX ; x <= offX + w ; x++){
                for( uint d = 1 ; d <= f ; d++){
                        putpixel(surface, x, offY+d-1, pc);
                        putpixel(surface, x , offY+h-d+1 , pc) ;
                }
        }
        for( uint y = offY ; y <= offY + h ; y++){
                for( uint d = 1 ; d <= f ; d++){
                        putpixel(surface,offX+d-1,y,pc) ;
                        putpixel(surface,offX+w-d+1 ,y ,pc) ;
                }

        }

        if ( SDL_MUSTLOCK(surface) ) {
                SDL_UnlockSurface(surface);
        }

}

////////////////////////////////////////////
void fillOval(SDL_Surface* surface , const Uint32 pc , int offX  , uint offY  , const int w , const int h , const Uint32 bgpc) {
        /* Lock the screen for direct access to the pixels */
        if ( SDL_MUSTLOCK(surface) ) {
                if ( SDL_LockSurface(surface) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }

        float w_2 =((float)w*w)/4.0f ;
        float h_2 = ((float) h*h)/4.0f ;
        for( float x = offX ; x < offX+w  ; x++){
                for( float y = offY ; y < offY +h  ; y++){
                        if( (float)((x-offX-(float)(w)/2)*(x-offX-(float)(w)/2))/w_2 + (float)((y-offY-(float)(h)/2)*(y-offY-(float)(h)/2))/h_2 < 1 ){
                                putpixel(surface, (int)x, (int)y, pc);

                        }else{
			  putpixel(surface,(int)x,(int)y,bgpc);
			}
                }

        }

        if ( SDL_MUSTLOCK(surface) ) {
                SDL_UnlockSurface(surface);
        }
}


////////////////////////////////////////////
void drawColorWheel(SDL_Surface* surface ,const vector<Uint32> colors,  int cX  , uint cY  , const int minRadius , const int maxRadius , const Uint32 bgpc) {
        /* Lock the screen for direct access to the pixels */
        if ( SDL_MUSTLOCK(surface) ) {
                if ( SDL_LockSurface(surface) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }
	
	float alpha0= 2*3.14159265/colors.size();
	//int n = (int)colors.size() ;
        for( float x = 0 ; x < surface->w  ; x++){
                for( float y = 0 ; y < surface->h ; y++){
		  float yp = 2*maxRadius - y ;
		  float r = sqrt((x-cX)*(x-cX) + (yp-cY)*(yp-cY)) ;
		  float alpha;
		  alpha = acos((yp-(float)cY)/r); 
		  if(x-cX <0 ) alpha =  2*3.14159265 - alpha;
		  //if(x-cX <0 && alpha >=  3.14159265/2) alpha =  3.14159265/2 - alpha;
		  int sector = (int)(alpha/alpha0);
		  //sector = 2 ;
		  if(r<= maxRadius && r >=minRadius){
		    Uint32 pc = colors.at(sector);
		     putpixel(surface, (int)x, (int)y, pc);
		  }else{
		     putpixel(surface,(int)x,(int)y,bgpc);
		  }
                }

        }

        if ( SDL_MUSTLOCK(surface) ) {
                SDL_UnlockSurface(surface);
        }
}



void fillCubicRadiant(SDL_Surface* surface , const Uint32 pc , uint offX , uint offY, const int R){
        /* Lock the screen for direct access to the pixels */
        uint b1 = 256 ;
        uint b2 = 65536 ;
        uint red = pc / b2 ;
        uint rem = pc % b2 ;
        uint green = rem/b1 ;
        uint blue = rem % b1 ;
        uint nr = 0 ;
        uint ng = 0 ;
        uint nb = 0 ;
        float factor = 0.0f ;
        if ( SDL_MUSTLOCK(surface) ) {
                if ( SDL_LockSurface(surface) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }

        Uint32 clr = 0 ;
        for( float x = offX-R ; x <= offX+R  ; x++){
                for( float y = offY-R ; y <= offY +R  ; y++){
                        float r = sqrt((x-offX)*(x-offX) + (y-offY)*(y-offY));
                        factor = (1-(float)r*r*r/((float)R*R*R)) ;
                        nr = (uint)(red * factor) ;
                        ng = (uint)(green*factor) ;
                        nb = (uint)(blue * factor) ;
                        if(r <= R ){

                                clr = (Uint32)(nb+ng*b1+nr*b2) ;
                                putpixel(surface, (int)x, (int)y, clr);
                        }



                }

        }

        if ( SDL_MUSTLOCK(surface) ) {
                SDL_UnlockSurface(surface);
        }

}





void fillQuadricRadiant(SDL_Surface* surface , const Uint32 pc , uint offX , uint offY, const int R){
        /* Lock the screen for direct access to the pixels */
        uint b1 = 256 ;
        uint b2 = 65536 ;
        uint red = pc / b2 ;
        uint rem = pc % b2 ;
        uint green = rem/b1 ;
        uint blue = rem % b1 ;
        uint nr = 0 ;
        uint ng = 0 ;
        uint nb = 0 ;
        float factor = 0.0f ;
        if ( SDL_MUSTLOCK(surface) ) {
                if ( SDL_LockSurface(surface) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }

        Uint32 clr = 0 ;
        for( float x = offX-R ; x <= offX+R  ; x++){
                for( float y = offY-R ; y <= offY +R  ; y++){
                        float r = sqrt((x-offX)*(x-offX) + (y-offY)*(y-offY));
                        factor = (1-(float)r*r/((float)R*R)) ;
                        nr = (uint)(red * factor) ;
                        ng = (uint)(green*factor) ;
                        nb = (uint)(blue * factor) ;
                        if(r <= R ){

                                clr = (Uint32)(nb+ng*b1+nr*b2) ;
                                putpixel(surface, (int)x, (int)y, clr);
                        }



                }

        }

        if ( SDL_MUSTLOCK(surface) ) {
                SDL_UnlockSurface(surface);
        }

}


void fillLinearRadiant(SDL_Surface* surface , const Uint32 pc , uint offX , uint offY, const int R){
        /* Lock the screen for direct access to the pixels */
        uint b1 = 256 ;
        uint b2 = 65536 ;
        uint red = pc / b2 ;
        uint rem = pc % b2 ;
        uint green = rem/b1 ;
        uint blue = rem % b1 ;
        uint nr = 0 ;
        uint ng = 0 ;
        uint nb = 0 ;
        float factor = 0.0f ;
        if ( SDL_MUSTLOCK(surface) ) {
                if ( SDL_LockSurface(surface) < 0 ) {
                        fprintf(stderr, "Can't lock screen: %s\n", SDL_GetError());
                        return;
                }
        }


        Uint32 clr = 0 ;
        for( float x = offX-R ; x <= offX+R  ; x++){
                for( float y = offY-R ; y <= offY +R  ; y++){
                        float r = sqrt((x-offX)*(x-offX) + (y-offY)*(y-offY));
                        factor = (1-(float)r/((float)R)) ;
                        nr = (uint)(red * factor) ;
                        ng = (uint)(green*factor) ;
                        nb = (uint)(blue * factor) ;
                        if(r <= R ){

                                clr = (Uint32)(nb+ng*b1+nr*b2) ;
                                putpixel(surface, (int)x, (int)y, clr);
                        }



                }

        }

        if ( SDL_MUSTLOCK(surface) ) {
                SDL_UnlockSurface(surface);
        }

}
/*

extern "C" int main(int argc, char* argv[] ){

        ModelManager manager("Psycho Skin");
        nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));

        MYLOGVERB = LOG_INFO;  // suppress debug messages

        manager.addSubComponent(d);
        nub::soft_ref<EventLog> el(new EventLog(manager));
        manager.addSubComponent(el);
        d->setEventLog(el);
        // Parse command-line:
        if (manager.parseCommandLine(argc, argv,"<profile.psymap> <path-to-skin> <logfilename>", 0, 3)==false)
                return(1);
        manager.setOptionValString(&OPT_EventLogFileName,manager.getExtraArg(2).c_str() );
        manager.start();
        d->clearScreen() ;
        SDL_Surface* blankSurface = getABlankSurface(200,200) ;
        //if(blank ==  0) cout<<"====================================null "<<endl ;
        Uint32 white = d->getUint32color(PixRGB<byte>(255, 255, 255));
        Uint32 yellow = d->getUint32color(PixRGB<byte>(255, 255, 0));
        //fillOval(blankSurface,white,0,0,80,80) ;
        //fillOval(blankSurface,white,0,80,80,80) ;

        fillLinearRadiant(blankSurface,white,50,50,40) ;
        fillLinearRadiant(blankSurface,white,110,80,2) ;
        fillCubicRadiant(blankSurface,white,110,110,2) ;
        fillQuadricRadiant(blankSurface,white,110,150,2) ;
        fillOval(blankSurface,white,110,180,20,20) ;
        fillCubicRadiant(blankSurface,yellow,80,120,15) ;

        //fillRectangle(blankSurface,white,150,100,40,40) ;
        SDL_Rect offset;
        d->clearScreen() ;
        //Give the offsets to the rectangle
        offset.x = (d->getWidth() - blankSurface->w) /2;
        offset.y = (d-> getHeight() - blankSurface->h) /2;
        d->displaySDLSurfacePatch(blankSurface , &offset,NULL , -2,false, true);
        d->waitForKey() ;
        d->displayText("HI");
        d->waitForKey() ;
        manager.stop();

}
*/
