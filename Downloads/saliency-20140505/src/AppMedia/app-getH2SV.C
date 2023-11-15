/*!@file AppMedia/app-getH2SV.C get and raster the H2SV components of an image*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-getH2SV.C $
// $Id: app-getH2SV.C 12962 2010-03-06 02:13:53Z irock $
//

/*! This app is mainly useful to view the H2SV components to see what
  they look like from an image.
*/

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include <string>

using namespace std;

#define H2SV_TYPE PixH2SV2

int main(int argc, char** argv)
{
  LINFO("Get H2SV");
  string filename = argv[1];
  LINFO("Reading %s",filename.c_str());
  Image<PixRGB<byte> >   input    = Raster::ReadRGB(filename);
  Image<PixRGB<float> >  finput   = input;

  Image<H2SV_TYPE<float> > convert =
    static_cast<Image<H2SV_TYPE<float> > >(finput);

  Image<float> foutput;
  foutput.resize(convert.getWidth(),convert.getHeight());

  string affix[4] = {"H1","H2","S","V"};
  string dot      = ".";
  string type     = "png";

  for(int i = 0; i < 4; i++)
    {
      Image<float>::iterator ifoutput             = foutput.beginw();
      Image<H2SV_TYPE<float> >::iterator iconvert = convert.beginw();

      float avgVal = 0.0;
      int numVals = 0;
      std::vector<float> vals(0);

      while(iconvert != convert.endw())
        {
          vals.push_back(iconvert->p[i]);
          avgVal += iconvert->p[i];
          numVals++;
          *ifoutput = iconvert->p[i] * 255.0F;
          ++ifoutput; ++iconvert;
        }

      avgVal /= numVals;

      float stdDev = 0.0;
      for(uint j = 0; j < vals.size(); j++)
        {
          float temp = abs(vals[j] - avgVal);
          stdDev += temp * temp;
        }

            stdDev /= numVals - 1;
          stdDev = sqrt(stdDev);
          LINFO("Average %s val: %f",affix[i].c_str(), avgVal);
          LINFO("Std Dev %s val: %f",affix[i].c_str(), stdDev);
          string outputname = filename + dot + affix[i] + dot + type;
          LINFO("Writing %s",outputname.c_str());
          Raster::WriteGray(foutput,outputname);
          }
      // convert back to check conversion
  finput = static_cast<Image<PixRGB<float> > >(convert);

  input = finput;
  string check      = "check";
  string outputname = filename + dot + check + dot + type;
  Raster::WriteRGB(input,outputname);
}
