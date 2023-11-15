/*!@file TestSuite/whitebox-LevelSpec.C Test the LevelSpec class */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/whitebox-LevelSpec.C $
// $Id: whitebox-LevelSpec.C 8630 2007-07-25 20:33:41Z rjpeters $
//

#include "Util/log.H"

#include "Image/LevelSpec.H"
#include "TestSuite/TestSuite.H"

///////////////////////////////////////////////////////////////////////
//
// Test functions
//
// note that these funny _xx_'s are just used as a hierarchical separator
// (since double-underscore __ is not legal in C/C++ identifiers); the _xx_ is
// expected to be replaced by something prettier, like "--", by the test
// driver script.
//
///////////////////////////////////////////////////////////////////////

static void LevelSpec_xx_maxIndex_xx_1(TestSuite& suite)
{
  uint sml = 4, delta_min = 3, delta_max = 4, level_min = 2, level_max = 4;

  LevelSpec spec(level_min, level_max, delta_min, delta_max, sml);

  REQUIRE_EQ(spec.maxIndex(), uint(6));
}

static void LevelSpec_xx_maxDepth_xx_1(TestSuite& suite)
{
  uint sml = 4, delta_min = 3, delta_max = 4, level_min = 2, level_max = 4;

  LevelSpec spec(level_min, level_max, delta_min, delta_max, sml);

  REQUIRE_EQ(spec.maxDepth(), uint(9));
}

static void LevelSpec_xx_clevOK_xx_1(TestSuite& suite)
{
  uint sml = 4, delta_min = 3, delta_max = 4, level_min = 2, level_max = 4;

  LevelSpec spec(level_min, level_max, delta_min, delta_max, sml);

  REQUIRE(!spec.clevOK(1));
  REQUIRE(spec.clevOK(2));
  REQUIRE(spec.clevOK(3));
  REQUIRE(spec.clevOK(4));
  REQUIRE(!spec.clevOK(5));
}

static void LevelSpec_xx_slevOK_xx_1(TestSuite& suite)
{
  uint sml = 4, delta_min = 3, delta_max = 4, level_min = 2, level_max = 4;

  LevelSpec spec(level_min, level_max, delta_min, delta_max, sml);

  REQUIRE(!spec.slevOK(4));
  REQUIRE(spec.slevOK(5));
  REQUIRE(spec.slevOK(6));
  REQUIRE(spec.slevOK(7));
  REQUIRE(spec.slevOK(8));
  REQUIRE(!spec.slevOK(9));
}

static void LevelSpec_xx_delOK_xx_1(TestSuite& suite)
{
  uint sml = 4, delta_min = 3, delta_max = 4, level_min = 2, level_max = 4;

  LevelSpec spec(level_min, level_max, delta_min, delta_max, sml);

  REQUIRE(!spec.delOK(2));
  REQUIRE(spec.delOK(3));
  REQUIRE(spec.delOK(4));
  REQUIRE(!spec.delOK(5));
}

static void LevelSpec_xx_csOK_xx_1(TestSuite& suite)
{
  uint sml = 4, delta_min = 3, delta_max = 4, level_min = 2, level_max = 4;

  LevelSpec spec(level_min, level_max, delta_min, delta_max, sml);

  const bool n = false;
  const bool YY = true;
  bool expect_csok[10][10] =
    {
      //  (surround level)
      //   0  1  2  3  4  5  6  7  8  9
          {n, n, n, n, n, n, n, n, n, n}, // 0 (center level)
          {n, n, n, n, n, n, n, n, n, n}, // 1
          {n, n, n, n, n, YY,YY,n, n, n}, // 2
          {n, n, n, n, n, n, YY,YY,n, n}, // 3
          {n, n, n, n, n, n, n, YY,YY,n}, // 4
          {n, n, n, n, n, n, n, n, n, n}, // 5
          {n, n, n, n, n, n, n, n, n, n}, // 6
          {n, n, n, n, n, n, n, n, n, n}, // 7
          {n, n, n, n, n, n, n, n, n, n}, // 8
          {n, n, n, n, n, n, n, n, n, n}, // 9
    };

  for (uint c = 0; c < 10; ++c)
    for (uint s = 0; s < 10; ++s)
      REQUIRE_EQ(spec.csOK(c, s), expect_csok[c][s]);
}

static void LevelSpec_xx_convert_xx_1(TestSuite& suite)
{
  uint sml = 4, delta_min = 3, delta_max = 4, level_min = 2, level_max = 4;

  LevelSpec spec(level_min, level_max, delta_min, delta_max, sml);

  struct CSI { uint center; uint surround; uint index; };

  CSI expect_csi[6] =
    {
      { 2, 5, 0 },
      { 3, 6, 1 },
      { 4, 7, 2 },
      { 2, 6, 3 },
      { 3, 7, 4 },
      { 4, 8, 5 },
    };

  for (uint i = 0; i < 6; ++i)
    {
      REQUIRE_EQ(spec.csToIndex(expect_csi[i].center,
                                expect_csi[i].surround),
                 expect_csi[i].index);

      uint c = 0, s = 0;
      spec.indexToCS(expect_csi[i].index, c, s);
      REQUIRE_EQ(c, expect_csi[i].center);
      REQUIRE_EQ(s, expect_csi[i].surround);
    }
}

///////////////////////////////////////////////////////////////////////
//
// main
//
///////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  TestSuite suite;

  suite.ADD_TEST(LevelSpec_xx_maxIndex_xx_1);
  suite.ADD_TEST(LevelSpec_xx_maxDepth_xx_1);
  suite.ADD_TEST(LevelSpec_xx_clevOK_xx_1);
  suite.ADD_TEST(LevelSpec_xx_slevOK_xx_1);
  suite.ADD_TEST(LevelSpec_xx_delOK_xx_1);
  suite.ADD_TEST(LevelSpec_xx_csOK_xx_1);
  suite.ADD_TEST(LevelSpec_xx_convert_xx_1);

  suite.parseAndRun(argc, argv);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
