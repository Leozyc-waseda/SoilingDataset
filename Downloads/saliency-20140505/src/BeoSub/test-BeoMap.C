/*!@file BeoSub/test-BeoMap.C Test frame grabbing and X display */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoMap.C $
// $Id: test-BeoMap.C 9412 2008-03-10 23:10:15Z farhan $
//
#include "BeoSub/BeoMap.H"
#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Raster/Raster.H"
#include "SIFT/VisualObject.H"
#include "SIFT/VisualObjectMatch.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define SCALE 2
#define RUNTIME 80

#define CUT_THRESHOLD 1.2

// these are gains to decide cutting positions.
//#define MAP_CUT_CONST_RIGHT 1.2
//#define MAP_CUT_CONST_LEFT 0.8


#define SIZE_THRESH 200


void makePanorama(const char* nam1);

//#define NAVG 20

/*! This simple executable tests video frame grabbing through the
  video4linux driver (see V4Lgrabber.H) or the IEEE1394 (firewire)
  grabber (see IEEE1394grabber.H). Selection of the grabber type is
  made via the --fg-type=XX command-line option. */


int globalcounter(0);
int pw(0);
int ph(0);

Image< PixRGB<byte> > im3;
bool cut = false;
//
// this structure is for putting the coordinates
// of an input image stitched within a map.
//
/*
typedef struct{
  int leftedge;
  int rightedge;
  int upperedge;
  int loweredge;
}mapedge;

mapedge medge;
*/
int main(const int argc, const char **argv)
{

  // instantiate a model manager:
  ModelManager manager("Frame Grabber Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  // make an object for window frame.
  XWindow win;
  XWindow win2(Dims(2000,2000));


  BeoMap bm(1.2,200,true);



  // let's get all our ModelComponent instances started:
  manager.start();

  Timer tim;// uint64 t[NAVG]; int frame = 0;

  int count = 0;
  char* buf = new char[9999];
  char* buf2 = new char[9999];

  //ImageSet< PixRGB<byte> > images;
  // int input(0);
  bool first=true;
  Timer tim2(1000000);

  while(1) {

    count++;
    tim.reset();

    //sprintf(buf, "./images/inputframe%d.png", 1+count*SCALE);
    sprintf(buf, "/lab/beobot/take/saliency/images/frame%d.png", 441+count*SCALE);
    //sprintf(buf, "./images/frame%d.png", 251+count*SCALE);



    Image< PixRGB<byte> > ima= Raster::ReadRGB(buf);;

    if(first) {
      Raster::WriteRGB(ima, "./resultimg/result.png");
      first=false;
    }
    else {
      //makePanorama(buf);
      bm.makePanorama(buf, "./resultimg/result.png");
    }
    win.drawImage(ima);
    sprintf(buf2, "./resultimg/result.png");
    Image< PixRGB<byte> > ima2= Raster::ReadRGB(buf2);;


    win2.drawImage(ima2);

    //sleep(1);

    if(count > RUNTIME)break;
  }
  uint64 t = tim2.get();
  LINFO("The total run time is  %.3fms", float(t) * 0.001F);


    // stop all our ModelComponents
  manager.stop();

  return 0;
}


/*
function that that given the current position in the map and target position, tells you where to turn

*/
void findTargetDir(Point2D<int> current, Point2D<int> previous, Point2D<int> target)
{
  //check if the target falls on the same line as the submarine
  if(current.i == previous.i)//vertical line
  {
    if(target.i == current.i) //they fall on the same line
      {
        if(current.j > previous.j && target.j > current.j)
          {
            printf("target is straight ahead\n");
          }
        else
          {
            printf("target is behind\n");
          }
      }
    else
      {
        if((current.j > previous.j && target.i < current.i)|| (current.j < previous.j && target.i > current.i))
          {
            printf("target is on the right\n");
          }
        else if((current.j > previous.j && target.i > current.i)||(current.j < previous.j && target.i < current.i))
          {
            printf("target is on the left\n");
          }

      }

  }
  else
    {
      //get the slope
      float slope = (current.j-previous.j)/(current.i - previous.i);
      float slopeTarget =  (current.j-target.j)/(current.i - target.i);
      if(slope == slopeTarget) // target fall on the same line
        {
          if((target.i - previous.i)/(current.i-previous.i) < 1 )
            {
              printf("target is behind\n");

            }
          else
            {
              printf("target is straight ahead\n");

            }

        }
      else
        {


        }


    }



}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */


/*void makePanorama(const char* nam1)
{


  ImageSet< PixRGB<byte> > images;


  Image< PixRGB<byte> > im1 = Raster::ReadRGB(nam1);
  images.push_back(im1);
  rutz::shared_ptr<VisualObject> vo1(new VisualObject(nam1, "", im1));
  LINFO("keypoint extractions completed for input image (%d keypoints)", vo1->numKeypoints());


  std::vector<SIFTaffine> affines;
  SIFTaffine comboaff; // default constructor is identity
  affines.push_back(comboaff);
  int minx = 0, miny = 0, maxx = im1.getWidth()-1, maxy = im1.getHeight()-1;
  bool modMinX = false,modMaxX = false,modMinY = false,modMaxY = false;
  for (int i = 1; i < 2; i ++)
  {
      const char *nam2 = "./resultimg/result.png";
      Image< PixRGB<byte> > im2 = Raster::ReadRGB(nam2);
      images.push_back(im2);
      // LINFO("just before initializing im2");
      rutz::shared_ptr<VisualObject> vo2;
      //  LINFO("asd");
      LINFO("Pass #%d\n", globalcounter);


      if(!cut){
         rutz::shared_ptr<VisualObject> votemp( new VisualObject(nam2, "", im2));
        vo2 = votemp;
      }
      else{
        rutz::shared_ptr<VisualObject> votemp( new VisualObject(nam2, "", im3));
        vo2 = votemp;
      }





      LINFO("keypoint extractions completed for map (%d keypoints)", vo2->numKeypoints());


       maxx = im2.getWidth()-1, maxy = im2.getHeight()-1;
      // compute the matching keypoints:
      //Timer tim(1000000);
      VisualObjectMatch match(vo1, vo2, VOMA_SIMPLE);
      LINFO("%d keypoints matched at pass %d", match.size(), globalcounter);


      //uint64 t = tim.get();

      //      LINFO("Found %u matches between %s and %s in %.3fms",
      //           match.size(), nam1, nam2, float(t) * 0.001F);

      // let's prune the matches:
      //uint np =
      match.prune();
      //LINFO("Pruned %u outlier matches.", np);

      // show our final affine transform:
      SIFTaffine aff = match.getSIFTaffine();
      std::cerr<<aff;

      // compose with the previous affines and store:
      comboaff = comboaff.compose(aff);
      affines.push_back(comboaff);

      // update panorama boundaries, using the inverse combo aff to
      // find the locations of the four corners of our image in the
      // panorama:
      if (comboaff.isInversible() == false) LFATAL("Oooops, singular affine!");
      SIFTaffine iaff = comboaff.inverse();
      const float ww = float(im2.getWidth() - 1);
      const float hh = float(im2.getHeight() - 1);
      float xx, yy; int x, y;

      iaff.transform(0.0F, 0.0F, xx, yy); x = int(xx); y = int(yy);
      if (x < minx) {minx = x; modMinX = true;}
      if (x > maxx) {maxx = x; modMaxX = true;}
      if (y < miny) {miny = y; modMinY = true;}
      if (y > maxy) {maxy = y; modMaxY = true;}

      iaff.transform(ww, 0.0F, xx, yy); x = int(xx); y = int(yy);
      if (x < minx) {minx = x; modMinX = true;}
      if (x > maxx) {maxx = x; modMaxX = true;}
      if (y < miny) {miny = y; modMinY = true;}
      if (y > maxy) {maxy = y; modMaxY = true;}


      iaff.transform(0.0F, hh, xx, yy); x = int(xx); y = int(yy);
      if (x < minx) {minx = x; modMinX = true;}
      if (x > maxx) {maxx = x; modMaxX = true;}
      if (y < miny) {miny = y; modMinY = true;}
      if (y > maxy) {maxy = y; modMaxY = true;}


      iaff.transform(ww, hh, xx, yy); x = int(xx); y = int(yy);
      if (x < minx) {minx = x; modMinX = true;}
      if (x > maxx) {maxx = x; modMaxX = true;}
      if (y < miny) {miny = y; modMinY = true;}
      if (y > maxy) {maxy = y; modMaxY = true;}

      //LINFO("modMinX %d, modMaxX %d, modMinY %d, modMaxY %d",modMinX, modMaxX, modMinY, modMaxY);
      // get ready for next pair:
      im1 = im2; vo1 = vo2; nam1 = nam2;
    }

  // all right, allocate the panorama:
  //LINFO("x = [%d .. %d], y = [%d .. %d]", minx, maxx, miny, maxy);
  int w = maxx - minx + 1, h = maxy - miny + 1;


  if(globalcounter==0){
    pw=w;
    ph=h;
  }

  //
  // if the map size is within acceptable range.
  //
  if(w <= pw && h <= ph)
    {

      //LINFO("Allocating %dx%d panorama...", w, h);
  if (w < 2 || h < 2) LFATAL("Oooops, panorama too small!");
  Image< PixRGB<byte> > pano(w, h, ZEROS);
  Image< PixRGB<byte> >::iterator p = pano.beginw();

  int minStitchedX=0, maxStitchedX=0, minStitchedY=0, maxStitchedY=0, counterStitching = 0;
  int minStitchedX2=0, maxStitchedX2=0, minStitchedY2=0, maxStitchedY2=0, counterStitching2 = 0;
  // let's stitch the images into the panorama. This code is similar
  // to that in VisualObjectMatch::getTransfTestImage() but modified
  // for a large panorama and many images:


  for (int j = 0; j < h; j ++)
    for (int i = 0; i < w; i ++)
      {
        // compute the value that should go into the current panorama
        // pixel based on all the images and affines; this is very
        // wasteful and may be optimized later:
        PixRGB<int> val(0); uint n = 0U;
        //LINFO("images.size is %d",images.size());
        for (uint k = 0; k < images.size(); k ++)
          {
            // get transformed coordinates for image k:
            float u, v;
            affines[k].transform(float(i + minx), float(j + miny), u, v);
            //            LINFO("K is %d, i+minx is %d, i+miny is %d, u is %f, v is %f",k,i+minx,i+miny,u,v);
            // if we are within bounds of image k, accumulate the pix value:
            if (images[k].coordsOk(u, v))
              {
                val += PixRGB<int>(images[k].getValInterp(u, v));
                //the if-else statement below gathers the min/max coordinates on the resulting image the input image is.
                if(k == 0 && counterStitching == 0)
                  {
                    minStitchedX = maxStitchedX = i;
                    minStitchedY = maxStitchedY = j;
                    counterStitching++;
                  }
                else if(k==0)
                  {
                    if(minStitchedX > i)
                      minStitchedX = i;
                    if(maxStitchedX < i)
                      maxStitchedX = i;
                    if(minStitchedY > j)
                      minStitchedY = j;
                    if(maxStitchedY < j)
                      maxStitchedY = j;
                    counterStitching++;
                  }
                if(counterStitching2 == 0)
                  {
                    minStitchedX2 = maxStitchedX2 = i;
                    minStitchedY2 = maxStitchedY2 = j;
                    counterStitching2++;
                  }
                else
                  {
                    if(minStitchedX2 > i)
                      minStitchedX2 = i;
                    if(maxStitchedX2 < i)
                      maxStitchedX2 = i;
                    if(minStitchedY2 > j)
                      minStitchedY2 = j;
                    if(maxStitchedY2 < j)
                      maxStitchedY2 = j;
                    counterStitching2++;
                  }

                ++ n;
              }
          }

        if (n > 0) *p = PixRGB<byte>(val / n);

        ++ p;
      }

  LINFO("Pixel coordinates of the area Stitched: xmin %d, xmax %d, ymin %d, ymax %d",minStitchedX, maxStitchedX, minStitchedY, maxStitchedY);
  LINFO("Pixel coordinates of the area Stitched: xmin2 %d, xmax2 %d, ymin2 %d, ymax2 %d",minStitchedX2, maxStitchedX2, minStitchedY2, maxStitchedY2);


  //if(maxStitchedX2 + 10 < w &&maxStitchedY2+10 < h){
  Image< PixRGB<byte> > pano2((maxStitchedX2+10<w)?maxStitchedX2+10:w, (maxStitchedY2+10<h)?maxStitchedY2+10:h, ZEROS);
  for(int i = 0; i < pano2.getWidth(); i++)
    for(int j = 0; j<pano2.getHeight(); j++)
      {
        pano2.setVal(i,j,pano.getVal(i,j));

      }
  //  }

    // w = maxStitchedX2+1;
    // h = maxStitchedY2+1;
  if(  (pano2.getHeight()*pano2.getWidth() / (float)((maxStitchedX - minStitchedX+1)*(maxStitchedY - minStitchedY+1)) >= CUT_THRESHOLD )
    {
      //  LINFO("GO CUT THE MAP!\n");

      //
      // Simple map cutting algorithm.
      //
      // Coded on Feb/8/2006.


      //LINFO("===================================================\n");

        // calculate cutting position.

        medge.rightedge = maxStitchedX + SIZE_THRESH;
        medge.leftedge = minStitchedX - SIZE_THRESH;
        medge.upperedge = minStitchedY - SIZE_THRESH;
        medge.loweredge = maxStitchedY + SIZE_THRESH;

        if( (pano2.getWidth() - medge.rightedge) < 20){
          medge.rightedge = -1; // -1 means dont cut a map.
        }

        if(medge.leftedge < 20){
          medge.leftedge = -1; // -1 means dont cut a map.
        }

        if( (medge.upperedge) <= 20){
          medge.upperedge = -1; // -1 means dont cut a map.
        }

        if( (pano2.getHeight() - medge.loweredge) < 20){
          medge.loweredge = -1; // -1 means dont cut a map.
        }


        LINFO("minX %d , maxX %d, minY %d, maxY %d", minStitchedX, maxStitchedX, minStitchedY, maxStitchedY);


                if(medge.upperedge > 0) LINFO("UPPER-SIDE :  %d \n", medge.upperedge);
        else LINFO("UPPER-SIDE : No cut");


                if(medge.rightedge > 0) LINFO("RIGHT-SIDE :  %d \n", medge.rightedge);
                else LINFO("RIGHT-SIDE : No cut");

                if(medge.leftedge > 0) LINFO("LEFT-SIDE :  %d \n", medge.leftedge);
                else LINFO("LEFT-SIDE : No cut");



                if(medge.loweredge > 0) LINFO("LOWER-SIDE :  %d \n", medge.loweredge);
                else LINFO("LOWER-SIDE : No cut");

                LINFO("Map size : (Xsize, Ysize) = (%d, %d)\n", w, h);

        //        LINFO("===================================================\n");

                int widthtmp(pano2.getWidth()); int hswitch(0);


        ///////////////////////////////////////////////////////////////////////////////
        //////// X-Direction cut.

        if((medge.rightedge != -1) && (medge.leftedge != -1)){
          im3.resize(widthtmp = (medge.rightedge+1) - medge.leftedge, pano2.getHeight());
          //LINFO("size of im3 is %d,%d",im3.getWidth(), im3.getHeight());
          for(int i = 0; i<im3.getWidth();i++)
            for(int j = 0; j <im3.getHeight();j++)
              {
                im3.setVal(i,j,pano2.getVal(i+medge.leftedge, j));
              }
          cut = true;
          hswitch = 1;
        }
        else if((medge.rightedge != -1) && (medge.leftedge==-1))
          {
            im3.resize(widthtmp = (medge.rightedge+1), pano2.getHeight());
            //  LINFO("size of im3 is %d,%d",im3.getWidth(), im3.getHeight());
            for(int i = 0; i<im3.getWidth();i++)
              for(int j = 0; j <im3.getHeight();j++)
                {
                  im3.setVal(i,j,pano2.getVal(i, j));
                }
            cut = true;
            hswitch = 2;
          }
        else if((medge.rightedge == -1) && (medge.leftedge!=-1))
          {
            im3.resize(widthtmp = (pano2.getWidth()-medge.leftedge-1), pano2.getHeight());
            //LINFO("size of im3 is %d,%d",im3.getWidth(), im3.getHeight());
            for(int i = 0; i<im3.getWidth();i++)
              for(int j = 0; j <im3.getHeight();j++)
                {
                  im3.setVal(i,j,pano2.getVal(i+medge.leftedge, j));
                }
            cut = true;
            hswitch = 3;
          }
        else{
          cut = false;
          hswitch = 0;
          widthtmp = pano2.getWidth();
        }

        ///////////////////////////////////////////////////////////////////////////////
        //////// Y-Direction cut.

        if((medge.upperedge != -1) && (medge.loweredge != -1)){// upper lower both cut.
          im3.resize(widthtmp, medge.loweredge - medge.upperedge );
          //LINFO("size of im3 is %d,%d",im3.getWidth(), im3.getHeight());
          for(int i = 0; i<im3.getWidth();i++)
            for(int j = 0; j <im3.getHeight();j++)
              {
                if(hswitch == 0)// left right no cut
                  im3.setVal(i,j,pano2.getVal(i, j+medge.upperedge));
                if(hswitch == 1)// left right both cut
                  im3.setVal(i,j,pano2.getVal(i+medge.leftedge, j+medge.upperedge));
                if(hswitch == 2)// left no cut, right cut
                  im3.setVal(i,j,pano2.getVal(i, j+medge.upperedge));
                if(hswitch == 3)// left cut right no cut.
                  im3.setVal(i,j,pano2.getVal(i+medge.leftedge, j+medge.upperedge));
              }
          cut = true;
        }
        else if((medge.upperedge != -1) && (medge.loweredge==-1)){//  upper cut lower no cut
            im3.resize(widthtmp, pano2.getHeight() - medge.upperedge);
            //LINFO("size of im3 is %d,%d",im3.getWidth(), im3.getHeight());
            for(int i = 0; i<im3.getWidth();i++)
              for(int j = 0; j <im3.getHeight();j++)
                {
                  //  im3.setVal(i,j,pano.getVal(i, j));
                  if(hswitch == 0)// left right no cut
                    im3.setVal(i,j,pano2.getVal(i, j+medge.upperedge));
                  if(hswitch == 1)// left right both cut
                    im3.setVal(i,j,pano2.getVal(i+medge.leftedge, j+medge.upperedge));
                  if(hswitch == 2)// left no cut, right cut
                    im3.setVal(i,j,pano2.getVal(i, j+medge.upperedge));
                  if(hswitch == 3)// left cut right no cut.
                    im3.setVal(i,j,pano2.getVal(i+medge.leftedge, j+medge.upperedge));
                }
            cut = true;
          }
        else if((medge.upperedge == -1) && (medge.loweredge !=-1)){// upper no cut lower cut
            im3.resize(widthtmp, medge.loweredge+1);
            //LINFO("size of im3 is %d,%d",im3.getWidth(), im3.getHeight());
            for(int i = 0; i<im3.getWidth();i++)
              for(int j = 0; j <im3.getHeight();j++)
                {
                  //          im3.setVal(i,j,pano.getVal(i+medge.leftedge, j));
                                    //  im3.setVal(i,j,pano.getVal(i, j));
                  if(hswitch == 0)// left right no cut
                    im3.setVal(i,j,pano2.getVal(i, j));
                  if(hswitch == 1)// left right both cut
                    im3.setVal(i,j,pano2.getVal(i+medge.leftedge, j));
                  if(hswitch == 2)// left no cut, right cut
                    im3.setVal(i,j,pano2.getVal(i, j));
                  if(hswitch == 3)// left cut right no cut.
                    im3.setVal(i,j,pano2.getVal(i+medge.leftedge, j));
                }
            cut = true;
          }
        else{
          cut = false;
        }



        ///////////

    }
  else
    {
      cut = false;
    }


  //  pano.resize(maxStitchedX2+1, maxStitchedY2+1,false);
  // save final panorama:
  //pano = rescale(pano, maxStitchedX2+1, maxStitchedY2+1);

  Raster::WriteRGB(pano2, "./resultimg/result.png");
  // LINFO("Done.");

  // to avid making a map extremely big.
  pw = (int)(pano2.getWidth()*1.5);//w*1.5);
  ph = (int)(pano2.getHeight()*1.5);//h*1.5);
  globalcounter++;
    }
}
*/
