/*!@file AppMedia/app-comparefixationmaps.C Create small images from originals */

//////////////////////////////////////////////////////////////////////////
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
//////////////////////////////////////////////////////////////////////////
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
//////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////
//
// Primary maintainer for this file: David Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppEye/app-comparefixationmaps.C $

#include "Image/Image.H"
#include "Image/ShapeOps.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Image/Convolutions.H"
#include "Image/Kernels.H"
#include "GUI/XWindow.H"

#include <vector>
#include <fstream>
#include <algorithm>

#define RANDITER 10000
#define DISPLAY 0
#define STD9 1.4364

//input should be two arguments to text files of eye positions' x y\n'
int main(int argc, char** argv)
{

const float blur1 = 1.5F*12.7F;
const float blur2 = 1.5F*10.19F;

 srand(time(0));

//load our files
std::vector<Point2D<int> > randPoints;
std::vector<uint> gind;
uint count = 0;
for (int ii=1; ii < argc-1; ii++){
    std::ifstream *itsRandFile = new std::ifstream(argv[ii]);
    if (itsRandFile->is_open() == false)
        LFATAL("Cannot open '%s' for reading", argv[ii]);
    else {
      while (!itsRandFile->eof())
        {
          Point2D<int> pr;
          (*itsRandFile) >> pr.i >> pr.j;
          randPoints.push_back(pr);
          count++;
        }
      randPoints.pop_back();    // why???
      gind.push_back(count - 1);  // offset -1 for ???
      itsRandFile->close();
      delete itsRandFile;
    }
}

//setup a preview window
 Dims dispdims(640*2,480);
 XWindow *prevwin;
 if (DISPLAY)
   prevwin = new XWindow(dispdims, 0, 0, "Preview"); //preview window

//open an output file
 std::ofstream *itsOutFile = new std::ofstream(argv[argc-1]);//our output file
 if (itsOutFile->is_open() == false)
   LFATAL("Cannot open '%s' for reading",argv[argc-1]);

//now we have read our files and know where our groups start and stop.
//g through NUMITER and create images taking KL between them
float data[RANDITER];
for (uint ii = 0; ii < RANDITER; ii++){
    //suffle the data
    std::vector< Point2D<int> > rshuffle = randPoints;
    if (ii < RANDITER-1) //calculate original groups last
        std::random_shuffle(rshuffle.begin(),rshuffle.end());

    Image<float> im1(640,480,ZEROS);
    Image<float> im2(640,480,ZEROS);
    //image1
    for (uint jj=0; jj < gind[0]; jj++){
        im1.setVal(rshuffle[jj].i,
                  rshuffle[jj].j,
                   im1.getVal(rshuffle[jj].i,rshuffle[jj].j)+1.0F);
    }
    //image2
    for (uint jj=gind[0]; jj < randPoints.size(); jj++){
        im2.setVal(rshuffle[jj].i,
                  rshuffle[jj].j,
                   im2.getVal(rshuffle[jj].i,rshuffle[jj].j)+1.0F);
    }


    //now lets run a truncating filter over this
    //image1
    im1 += .00000001;
        //Image<float> kernel = binomialKernel(9);
        //for (uint kk = 0; kk < floor(blur1/STD9); kk++)
        //im1 = sepFilter(im1,kernel,kernel,CONV_BOUNDARY_CLEAN);
        //Image<float> kernel = gaussian<float>(0.0F,blur1,0,.1F);
        //im1 = sepFilter(im1,kernel,kernel,CONV_BOUNDARY_CLEAN);
    Image<float> filter1 = gaussian2D<float>(blur1);
    filter1 /= sum(filter1);
    im1 = convolveHmax(im1,filter1);
    im1 /= sum(im1);

    //image 2
    im2 += .00000001;
        //for (uint kk = 0; kk < ceil(blur2/STD9); kk++)
        //im2 = sepFilter(im2,kernel,kernel,CONV_BOUNDARY_CLEAN);
    Image<float> filter2 = gaussian2D<float>(blur2);
    filter2 /= sum(filter2);
    im2 = convolveHmax(im2,filter2);
    im2 /= sum(im2);

    //lets display the images
    if (DISPLAY)
    {
        prevwin->drawImage(im1,0,0);
        prevwin->drawImage(im2,640,0);
    }

    //output KL
    data[ii] = .05F * (sum(im1 * log(im1/im2)) + sum(im2 * log(im2/im1)));
    (*itsOutFile) << sformat("%2.6f",data[ii]) << std::endl;
    LINFO("%2.6f ",data[ii]);

}
 itsOutFile->close();



}//END MAIN

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
