/*!@file AppMedia/test-lowDec.C
 */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/test-lowDec.C $
// $Id: test-lowDec.C 15468 2013-04-18 02:18:18Z itti $
//

#include "Component/ModelManager.H"
#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Neuro/StdBrain.H"
#include "Raster/PngWriter.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"
#include "Util/log.H"

#include <cstdio>

int main(const int argc, const char** argv)
{

  if (argc <= 1)
    {
      LINFO("usage: %s imagefile [sampling factor]",argv[0]);
      return -1;
    }

  int sub = 2;
  if (argc > 2) sscanf(argv[2],"%d",&sub);

  LINFO("subsampling factor is %d",sub);

  // load image
  Image<PixRGB <byte> > rgbimg = Raster::ReadRGB(argv[1]);
  Image<float> img = luminance(rgbimg);
  LINFO("image dimensions: %d x %d",img.getWidth(), img.getHeight());


  // process with old method
  Image<float> lowPx = lowPass5x(img);
  Image<float> oldresultX = decX(lowPx,sub);
  Image<float> lowP = lowPass5y(oldresultX);
  Image<float> oldresult = decY(lowP,sub);

  // process with new method
  Image<float> newresultX = lowPass5xDecX(img,sub);
  Image<float> newresult = lowPass5yDecY(newresultX,sub);


  LINFO("old dims = %d x %d",oldresult.getWidth(),oldresult.getHeight());
  LINFO("new dims = %d x %d",newresult.getWidth(),newresult.getHeight());

  // compare the results
  Image<float> diff = oldresult - newresult;
  float min, max;
  getMinMax(diff, min,max);

  // output the verdict
  LINFO("diff.min = %g; diff.max = %g",min,max);

  // save them images
  PngWriter::writeGray(Image<byte>(newresult), "new.png");
  PngWriter::writeGray(Image<byte>(oldresult), "old.png");
  PngWriter::writeGray(Image<byte>(diff), "diff.png");
  PngWriter::writeGray(Image<byte>(lowP), "lowp.png");

  // stop time for repeated execution
  const int N = 100;
  Timer timer;

  for (int i = 0; i < N; ++i) decY(lowPass5y(decX(lowPass5x(img),sub)),sub);
  int oldtime = (int)timer.getReset();
  for (int i = 0; i < N; ++i) lowPass5yDecY(lowPass5xDecX(img,sub),sub);

  int newtime = (int)timer.get();

  LINFO("old time = %i ms; new time = %i ms for %i iterations",
        oldtime,newtime,N);

  /*
  ModelManager mgr;
  mgr.allowOptions(OPTEXP_CORE);
  nub::soft_ref<StdBrain>  brain(new StdBrain(mgr));
  mgr.addSubComponent(brain);
  mgr.exportOptions(MC_RECURSE);
  mgr.setOptionValString(&OPT_RawVisualCortexType,"Std");
  mgr.start();

  timer.reset();

  for (int i = 0; i < N; ++i)
    {
      brain->input(rgbimg);
      brain->reset();
    }
  int tbrain = (int)timer.get();

  mgr.stop();

  LINFO("time for %i brain calls is %i ms",N,tbrain);
  */

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
