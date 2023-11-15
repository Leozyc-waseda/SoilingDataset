/*!@file AppMedia/test-KalmanFilter.C - a test program for KalmanFilter.[CH]
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/test-KalmanFilter.C $
// $Id: test-KalmanFilter.C 14376 2011-01-11 02:44:34Z pez $
//

#include "Image/Image.H"
#include "Image/KalmanFilter.H"
#include "Util/Types.H"
#include "Util/log.H"
#include <cmath>
#include <fstream>

int main(const int argc, const char** argv)
{
  if (argc < 3)
    LFATAL("usage: %s inputfile.txt outputfile.txt",argv[0]);

  std::ifstream is(argv[1]);
  std::ofstream os(argv[2]);

  float pNoise = 1.0F;
  float mNoise = 1.0F;

  if (argc > 3) sscanf(argv[3],"%g",&pNoise);
  if (argc > 4) sscanf(argv[4],"%g",&mNoise);

  KalmanFilter KF;

  while(is.good()) {
      float z;
      is >> z;

                if (!KF.isInitialized()) {
                        KF.init(z,pNoise,mNoise);
                        KF.getStateVector().getVal(0,0);
                }
                else {
                        KF.getEstimate();

                        if (z < 0.0F)
                                KF.update();
                        else
                                KF.update(z);
                }

                Image<float> x = KF.getStateVector();
                Image<float> P = KF.getCovariances();

                //os << z << ' ' << pred;

                os << z << ": ";

                for (int i = 0; i < 3; ++i)
                        os << ' ' << x.getVal(0,i);

                os << " ---- ";

                for (int i = 0; i < 3; ++i)
                        os << ' ' << sqrt(P.getVal(i,i));

                os << '\n';
        }

  os.close();
  is.close();
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
