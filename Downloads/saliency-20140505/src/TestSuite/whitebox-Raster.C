/*!@file TestSuite/whitebox-Raster.C Test Raster class */

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
// Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/whitebox-Raster.C $
// $Id: whitebox-Raster.C 8630 2007-07-25 20:33:41Z rjpeters $
//

#include "Image/ColorOps.H"
#include "Image/IO.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/Transforms.H" // for makeBinary()
#include "Raster/GenericFrame.H"
#include "Raster/PfmParser.H"
#include "Raster/PfmWriter.H"
#include "Raster/Raster.H"
#include "TestSuite/TestSuite.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "rutz/rand.h"

#include <algorithm> // for std::generate()
#include <deque> // for pfm tags
#include <limits> // for std::numeric_limits<>
#include <string>
#include <unistd.h> // for getpid(), unlink()

namespace
{

  class TempFile
  {
    static int theCounter;
    std::string itsName;
  public:
    TempFile(const char* ext = "")
      :
      itsName(sformat("test-Raster-%d-tmp-%d.%s",
                      int(getpid()), theCounter++, ext))
    {}

    ~TempFile()
    {
      unlink(itsName.c_str());
    }

    const char* name() const { return itsName.c_str(); }
  };

  int TempFile::theCounter = 1;

  template <class T>
  struct Randomizer
  {
    T top;

    Randomizer(T val = std::numeric_limits<T>::max()) : top(val) {}

    T operator()() const
    {
      return T( randomDouble() * double(top) );;
    }
  };

  template <class T>
  struct Randomizer<PixRGB<T> >
  {
    Randomizer<T> randomT;

    PixRGB<T> operator()() const
    {
      return PixRGB<T>(randomT(), randomT(), randomT());
    }
  };

  template <>
  struct Randomizer<float>
  {
    float operator()() const
    {
      return float(255.0f*randomDouble());
    }
  };

  template <class T>
  Image<T> randImage(int w, int h)
  {
    Image<T> result(w, h, NO_INIT);

    std::generate(result.beginw(), result.endw(), Randomizer<T>());

    return result;
  }

  template <class T>
  Image<T> randImage()
  {
    Randomizer<int> r(500);

    return randImage<T>(r()+10, r()+10);
  }
}

///////////////////////////////////////////////////////////////////////
//
// Test functions
//
///////////////////////////////////////////////////////////////////////

static void Raster_xx_img_gray_write_bw_read_gray_xx_1(TestSuite& suite)
{
  /* img gray */ Image<byte> i1 = randImage<byte>();

  /* write bw */ TempFile tf("pbm"); Raster::WriteGray(i1, tf.name());

  /* read gray */ Image<byte> i2 = Raster::ReadGray(tf.name());

  REQUIRE_EQ(i2, makeBinary(i1, byte(127), byte(0), byte(255)));
}

static void Raster_xx_img_gray_write_gray_read_gray_xx_1(TestSuite& suite)
{
  /* img gray */ Image<byte> i1 = randImage<byte>();

  /* write gray */ TempFile tf("pgm"); Raster::WriteGray(i1, tf.name());

  /* read gray */ Image<byte> i2 = Raster::ReadGray(tf.name());

  REQUIRE_EQ(i2, i1);
}

static void Raster_xx_img_gray_write_gray_read_rgb_xx_1(TestSuite& suite)
{
  /* img gray */ Image<byte> i1 = randImage<byte>();

  /* write gray */ TempFile tf("pgm"); Raster::WriteGray(i1, tf.name());

  /* read rgb */ Image<PixRGB<byte> > i2 = Raster::ReadRGB(tf.name());

  Image<byte> r, g, b; getComponents(i2, r, g, b);
  REQUIRE_EQ(r, i1);
  REQUIRE_EQ(g, i1);
  REQUIRE_EQ(b, i1);
}

static void Raster_xx_img_gray_write_rgb_read_gray_xx_1(TestSuite& suite)
{
  /* img gray */ Image<byte> i1 = randImage<byte>();

  /* write rgb */ TempFile tf("ppm"); Raster::WriteRGB(i1, tf.name());

  /* read gray */ Image<byte> i2 = Raster::ReadGray(tf.name());

  REQUIRE_EQ(i2, i1);
}

static void Raster_xx_img_gray_write_rgb_read_rgb_xx_1(TestSuite& suite)
{
  /* img gray */ Image<byte> i1 = randImage<byte>();

  /* write rgb */ TempFile tf("ppm"); Raster::WriteRGB(i1, tf.name());

  /* read rgb */ Image<PixRGB<byte> > i2 = Raster::ReadRGB(tf.name());

  Image<byte> r, g, b; getComponents(i2, r, g, b);
  REQUIRE_EQ(r, i1);
  REQUIRE_EQ(g, i1);
  REQUIRE_EQ(b, i1);
}

static void Raster_xx_img_rgb_write_gray_read_gray_xx_1(TestSuite& suite)
{
  /* img rgb */ Image<PixRGB<byte> > i1 = randImage<PixRGB<byte> >();

  /* write gray */ TempFile tf("pgm"); Raster::WriteGray(luminance(i1), tf.name());

  /* read gray */ Image<byte> i2 = Raster::ReadGray(tf.name());

  REQUIRE_EQ(i2, luminance(i1));
}

static void Raster_xx_img_rgb_write_gray_read_rgb_xx_1(TestSuite& suite)
{
  /* img rgb */ Image<PixRGB<byte> > i1 = randImage<PixRGB<byte> >();

  /* write gray */ TempFile tf("pgm"); Raster::WriteGray(luminance(i1), tf.name());

  /* read rgb */ Image<PixRGB<byte> > i2 = Raster::ReadRGB(tf.name());

  REQUIRE_EQ(luminance(i2), luminance(i1));
}

static void Raster_xx_img_rgb_write_rgb_read_gray_xx_1(TestSuite& suite)
{
  /* img rgb */ Image<PixRGB<byte> > i1 = randImage<PixRGB<byte> >();

  /* write rgb */ TempFile tf("ppm"); Raster::WriteRGB(i1, tf.name());

  /* read gray */ Image<byte> i2 = Raster::ReadGray(tf.name());

  REQUIRE_EQ(i2, luminance(i1));
}

static void Raster_xx_img_rgb_write_rgb_read_rgb_xx_1(TestSuite& suite)
{
  /* img rgb */ Image<PixRGB<byte> > i1 = randImage<PixRGB<byte> >();

  /* write rgb */ TempFile tf("ppm"); Raster::WriteRGB(i1, tf.name());

  /* read rgb */ Image<PixRGB<byte> > i2 = Raster::ReadRGB(tf.name());

  REQUIRE_EQ(i2, i1);
}

static void Raster_xx_img_float_write_float_read_float_xx_1(TestSuite& suite)
{
  /* img float */ Image<float> i1 = randImage<float>();

  /* write float */ TempFile tf("pfm"); Raster::WriteFloat(i1, 0, tf.name());

  /* read float */ Image<float> i2 = Raster::ReadFloat(tf.name());

  REQUIRE_EQ(i2, i1);
}

static void Raster_xx_pfm_tags_xx_1(TestSuite& suite)
{
  const Image<float> i1 = randImage<float>();

  std::deque<std::string> tagNames;
  std::deque<std::string> tagValues;

  tagNames.push_back("test tag name #1");
  tagValues.push_back("test tag value #1");

  tagNames.push_back("test tag name #2");
  tagValues.push_back("test tag value #2");

  TempFile tf("pfm");

  PfmWriter::writeFloat(i1, tf.name(), tagNames, tagValues);

  PfmParser parser(tf.name());

  REQUIRE_EQ(parser.getTagCount(), uint(2));

  std::string name, value;

  REQUIRE_EQ(parser.getTag(0, name, value), true);
  REQUIRE_EQ(name, tagNames[0]);
  REQUIRE_EQ(value, tagValues[0]);

  REQUIRE_EQ(parser.getTag(1, name, value), true);
  REQUIRE_EQ(name, tagNames[1]);
  REQUIRE_EQ(value, tagValues[1]);

  REQUIRE_EQ(parser.getComments(), std::string());

  const Image<float> i2 = parser.getFrame().asFloat();

  REQUIRE_EQ(i1, i2);
}


///////////////////////////////////////////////////////////////////////
//
// main
//
///////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  initRandomNumbers();

  TestSuite suite;

  suite.ADD_TEST(Raster_xx_img_gray_write_bw_read_gray_xx_1);
  suite.ADD_TEST(Raster_xx_img_gray_write_gray_read_gray_xx_1);
  suite.ADD_TEST(Raster_xx_img_gray_write_gray_read_rgb_xx_1);
  suite.ADD_TEST(Raster_xx_img_gray_write_rgb_read_gray_xx_1);
  suite.ADD_TEST(Raster_xx_img_gray_write_rgb_read_rgb_xx_1);
  suite.ADD_TEST(Raster_xx_img_rgb_write_gray_read_gray_xx_1);
  suite.ADD_TEST(Raster_xx_img_rgb_write_gray_read_rgb_xx_1);
  suite.ADD_TEST(Raster_xx_img_rgb_write_rgb_read_gray_xx_1);
  suite.ADD_TEST(Raster_xx_img_rgb_write_rgb_read_rgb_xx_1);
  suite.ADD_TEST(Raster_xx_img_float_write_float_read_float_xx_1);
  suite.ADD_TEST(Raster_xx_pfm_tags_xx_1);

  suite.parseAndRun(argc, argv);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
