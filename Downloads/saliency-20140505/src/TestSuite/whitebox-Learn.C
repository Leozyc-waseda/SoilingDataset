/*!@file TestSuite/whitebox-Learn.C Whitebox tests for neural networks and other learners. */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/whitebox-Learn.C $
// $Id: whitebox-Learn.C 10002 2008-07-29 17:18:48Z icore $
//

#ifndef TESTSUITE_WHITEBOX_LEARN_C_DEFINED
#define TESTSUITE_WHITEBOX_LEARN_C_DEFINED

#include "Learn/BackpropNetwork.H"
#include "TestSuite/TestSuite.H"

static void Learn_xx_backprop_nnet_xx_xor_xx_1(TestSuite& suite)
{
  // inputs for the xor problem (one data sample per column)
  const float X_[] =
    {
      0.5,  0.5, -0.5, -0.5,
      0.5, -0.5, -0.5,  0.5,
    };

  // outputs for the xor problem (one set of outputs per row)
  const float D_[] =
    {
      1, 0, 1, 0,
      0.2, 0.8, 0.2, 0.8
    };

  const Image<float> X(&X_[0], 4, 2);
  const Image<float> D(&D_[0], 4, 2);

  int nsuccess = 0;
  const int ntotal = 20;

  const float eta = 0.5f;
  const float alph = 0.5f;
  const int iters = 1000;

  // we have to do this as a loop and check that we succeed most of
  // the time; unfortunately backprop occasionally gets stuck in a
  // local minimum so we can't guarantee that the network will find
  // the optimal solution 100% of the time

  for (int i = 0; i < ntotal; ++i)
    {
      BackpropNetwork n;

      double E, C;

      n.train(X, D, 2, eta, alph, iters, &E, &C);

      if (E < 0.1 && C > 0.9)
        ++nsuccess;
    }

  REQUIRE_GTE(nsuccess, int(0.25*ntotal));
}

///////////////////////////////////////////////////////////////////////
//
// main
//
///////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  TestSuite suite;

  suite.ADD_TEST(Learn_xx_backprop_nnet_xx_xor_xx_1);

  suite.parseAndRun(argc, argv);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // TESTSUITE_WHITEBOX_LEARN_C_DEFINED
