/*!@file TestSuite/whitebox-Pixels.C Test PixRGB and other Pixels class */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/whitebox-Pixels.C $
// $Id: whitebox-Pixels.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Image/Pixels.H"
#include "TestSuite/TestSuite.H"
#include "Util/fpe.H"
#include "Util/log.H"

#include <cstdlib>
#include <cstring>
#include <typeinfo>

///////////////////////////////////////////////////////////////////////
//
// Test functions
//
// note that these funny _xx_'s are just used as a hierarchical
// separator (since e.g. double-underscore __ is reserved for
// implementation-defined identifiers in C/C++); the _xx_ is expected
// to be replaced by something prettier, like "--", by the test driver
// script.
//
///////////////////////////////////////////////////////////////////////

//! Shorthand for checking the values of any 3-element pixel:
#define REQUIRE_PIX3_EQ(pp,x1,x2,x3)            \
  REQUIRE_EQ((pp)[0], (x1));                    \
  REQUIRE_EQ((pp)[1], (x2));                    \
  REQUIRE_EQ((pp)[2], (x3));

//! Shorthand for checking the values of any 3-element pixel:
#define REQUIRE_PIX3_EQ2(pixel1,pixel2)         \
  REQUIRE_EQ((pixel1)[0], (pixel2)[0]);         \
  REQUIRE_EQ((pixel1)[1], (pixel2)[1]);         \
  REQUIRE_EQ((pixel1)[2], (pixel2)[2]);

//! Shorthand for checking the values of any 4-element pixel:
#define REQUIRE_PIX4_EQ(pp,x1,x2,x3,x4)         \
  REQUIRE_EQ((pp)[0], (x1));                    \
  REQUIRE_EQ((pp)[1], (x2));                    \
  REQUIRE_EQ((pp)[2], (x3));                    \
  REQUIRE_EQ((pp)[3], (x4));

//! Shorthand for checking the values of any 4-element pixel:
#define REQUIRE_PIX4_EQ2(pixel1,pixel2)         \
  REQUIRE_EQ((pixel1)[0], (pixel2)[0]);         \
  REQUIRE_EQ((pixel1)[1], (pixel2)[1]);         \
  REQUIRE_EQ((pixel1)[2], (pixel2)[2]);         \
  REQUIRE_EQ((pixel1)[3], (pixel2)[3]);

//---------------------------------------------------------------------
//
// Constructors & Destructors
//
//---------------------------------------------------------------------

static void Pixels_xx_constructors_xx_1(TestSuite& suite)
{
  PixRGB<int32> p1; // zero-initialization in default constructor
  REQUIRE_PIX3_EQ(p1, 0, 0, 0);

  p1.setRed(255); p1.setGreen(128); p1.setBlue(0); // setter functions
  REQUIRE_PIX3_EQ(p1, 255, 128, 0);

  PixRGB<byte> p2(1, 18, 57); // initialization of 3 individual elements
  REQUIRE_PIX3_EQ(p2, 1, 18, 57);

  PixRGB<int16> p3(-1256); // initialization of all elements to a single value
  REQUIRE_PIX3_EQ(p3, -1256, -1256, -1256);

  PixRGB<byte> p4(p2); // copy-initialization
  REQUIRE_PIX3_EQ(p4, 1, 18, 57);
}

static void Pixels_xx_conversions_xx_1(TestSuite& suite)
{
  PixRGB<byte> p1(12, 34, 56);

  PixRGB<int16> p2(p1);
  REQUIRE_PIX3_EQ(p2, 12, 34, 56);

  PixRGB<int32> p3(p1);
  REQUIRE_PIX3_EQ(p3, 12, 34, 56);

  PixRGB<float> p4(p1);
  REQUIRE_PIX3_EQ(p4, 12.0f, 34.0f, 56.0f);

  PixRGB<float> p5(-23.0f, 128.2f, 257.9f);
  PixRGB<byte> p6(p5);
  REQUIRE_PIX3_EQ(p6, 0, 128, 255);   // clamped conversion

  PixRGB<byte> p7(p5);
  REQUIRE_PIX3_EQ(p7, 0, 128, 255);   // clamped conversion

  PixRGB<double> p8(p5);   // this is to check correct use of numeric_limits:
  PixRGB<float> p9(p8);    // float to double and back to float
  REQUIRE_PIX3_EQ(p9, -23.0f, 128.2f, 257.9f);

  PixRGB<byte> p10(1.6f); // check that float->PixRGB<byte> conversion
                          // works, AND doesn't produce a warning if
                          // we're using "-Wall -Werror"
  REQUIRE_PIX3_EQ(p10, 1, 1, 1);

  PixRGB<byte> p11(1000.0f); // check that float->PixRGB<byte>
                             // conversion does proper clamping
  REQUIRE_PIX3_EQ(p11, 255, 255, 255);

  PixRGB<byte> p12(-1000.0f); // check that float->PixRGB<byte>
                              // conversion does proper clamping
  REQUIRE_PIX3_EQ(p12, 0, 0, 0);

}

static void Pixels_xx_conversions_xx_2(TestSuite& suite)
{
  // check all pix-pix conversions among PixRGB<byte>/PixRGB<int>/PixRGB<float>

  REQUIRE(typeid(PixRGB<byte>()  + PixRGB<byte> ()) == typeid(PixRGB<int>));
  REQUIRE(typeid(PixRGB<byte>()  + PixRGB<int>  ()) == typeid(PixRGB<int>));
  REQUIRE(typeid(PixRGB<byte>()  + PixRGB<float>()) == typeid(PixRGB<float>));

  REQUIRE(typeid(PixRGB<int>()   + PixRGB<byte> ()) == typeid(PixRGB<int>));
  REQUIRE(typeid(PixRGB<int>()   + PixRGB<int>  ()) == typeid(PixRGB<int>));
  REQUIRE(typeid(PixRGB<int>()   + PixRGB<float>()) == typeid(PixRGB<float>));

  REQUIRE(typeid(PixRGB<float>() + PixRGB<byte> ()) == typeid(PixRGB<float>));
  REQUIRE(typeid(PixRGB<float>() + PixRGB<int>  ()) == typeid(PixRGB<float>));
  REQUIRE(typeid(PixRGB<float>() + PixRGB<float>()) == typeid(PixRGB<float>));
}

static void Pixels_xx_conversions_xx_3(TestSuite& suite)
{
  // check all pix-scalar conversions among
  // PixRGB<byte>/PixRGB<int>/PixRGB<float> and byte/int/float

  REQUIRE(typeid(PixRGB<byte>()  + byte ()) == typeid(PixRGB<int>));
  REQUIRE(typeid(PixRGB<byte>()  + int  ()) == typeid(PixRGB<int>));
  REQUIRE(typeid(PixRGB<byte>()  + float()) == typeid(PixRGB<float>));

  REQUIRE(typeid(PixRGB<int>()   + byte ()) == typeid(PixRGB<int>));
  REQUIRE(typeid(PixRGB<int>()   + int  ()) == typeid(PixRGB<int>));
  REQUIRE(typeid(PixRGB<int>()   + float()) == typeid(PixRGB<float>));

  REQUIRE(typeid(PixRGB<float>() + byte ()) == typeid(PixRGB<float>));
  REQUIRE(typeid(PixRGB<float>() + int  ()) == typeid(PixRGB<float>));
  REQUIRE(typeid(PixRGB<float>() + float()) == typeid(PixRGB<float>));
}

static void Pixels_xx_conversions_xx_4(TestSuite& suite)
{
  const PixRGB<byte> rgb1  (  0,   0,   0);
  const PixRGB<byte> rgb2  (127, 127, 127);
  const PixRGB<byte> rgb3  (255, 255, 255);
  const PixRGB<byte> rgb4  (255,   0,   0);
  const PixRGB<byte> rgb5  (  0, 255,   0);
  const PixRGB<byte> rgb6  (  0,   0, 255);
  const PixRGB<byte> rgb7  (255, 127,  63);
  const PixRGB<byte> rgb8  (255,  63, 127);
  const PixRGB<byte> rgb9  (127, 255,  63);
  const PixRGB<byte> rgb10 ( 63, 255, 127);
  const PixRGB<byte> rgb11 (127,  63, 255);
  const PixRGB<byte> rgb12 ( 63, 127, 255);

  // test conversions rgb-->jpegyuv
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb1),    0, 128, 128);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb2),  127, 128, 128);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb3),  255, 128, 128);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb4),   76,  84, 255);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb5),  150,  43,  21);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb6),   29, 255, 107);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb7),  158,  74, 197);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb8),  128, 127, 218);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb9),  195,  53,  79);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb10), 183,  96,  42);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb11), 104, 213, 144);
  REQUIRE_PIX3_EQ(PixJpegYUV<byte>(rgb12), 122, 202,  85);

  // test conversions rgb-->videoyuv
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb1),   16, 128, 128);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb2),  125, 128, 128);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb3),  235, 128, 128);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb4),   81,  90, 240);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb5),  145,  54,  34);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb6),   41, 240, 110);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb7),  152,  81, 189);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb8),  126, 128, 208);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb9),  183,  63,  85);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb10), 173, 100,  53);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb11), 105, 203, 142);
  REQUIRE_PIX3_EQ(PixVideoYUV<byte>(rgb12), 121, 194,  91);

  // test round-trip conversions rgb-->jpegyuv-->rgb
  {
    PixRGB<byte> rgb2yuv2rgb1 ((PixJpegYUV<double>(rgb1))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb1 , rgb1 );
    PixRGB<byte> rgb2yuv2rgb2 ((PixJpegYUV<double>(rgb2))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb2 , rgb2 );
    PixRGB<byte> rgb2yuv2rgb3 ((PixJpegYUV<double>(rgb3))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb3 , rgb3 );
    PixRGB<byte> rgb2yuv2rgb4 ((PixJpegYUV<double>(rgb4))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb4 , rgb4 );
    PixRGB<byte> rgb2yuv2rgb5 ((PixJpegYUV<double>(rgb5))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb5 , rgb5 );
    PixRGB<byte> rgb2yuv2rgb6 ((PixJpegYUV<double>(rgb6))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb6 , rgb6 );
    PixRGB<byte> rgb2yuv2rgb7 ((PixJpegYUV<double>(rgb7))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb7 , rgb7 );
    PixRGB<byte> rgb2yuv2rgb8 ((PixJpegYUV<double>(rgb8))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb8 , rgb8 );
    PixRGB<byte> rgb2yuv2rgb9 ((PixJpegYUV<double>(rgb9))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb9 , rgb9 );
    PixRGB<byte> rgb2yuv2rgb10((PixJpegYUV<double>(rgb10)));  REQUIRE_PIX3_EQ2(rgb2yuv2rgb10, rgb10);
    PixRGB<byte> rgb2yuv2rgb11((PixJpegYUV<double>(rgb11)));  REQUIRE_PIX3_EQ2(rgb2yuv2rgb11, rgb11);
    PixRGB<byte> rgb2yuv2rgb12((PixJpegYUV<double>(rgb12)));  REQUIRE_PIX3_EQ2(rgb2yuv2rgb12, rgb12);
  }

  // test round-trip conversions rgb-->videoyuv-->rgb
  {
    PixRGB<byte> rgb2yuv2rgb1 ((PixVideoYUV<double>(rgb1))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb1 , rgb1 );
    PixRGB<byte> rgb2yuv2rgb2 ((PixVideoYUV<double>(rgb2))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb2 , rgb2 );
    PixRGB<byte> rgb2yuv2rgb3 ((PixVideoYUV<double>(rgb3))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb3 , rgb3 );
    PixRGB<byte> rgb2yuv2rgb4 ((PixVideoYUV<double>(rgb4))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb4 , rgb4 );
    PixRGB<byte> rgb2yuv2rgb5 ((PixVideoYUV<double>(rgb5))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb5 , rgb5 );
    PixRGB<byte> rgb2yuv2rgb6 ((PixVideoYUV<double>(rgb6))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb6 , rgb6 );
    PixRGB<byte> rgb2yuv2rgb7 ((PixVideoYUV<double>(rgb7))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb7 , rgb7 );
    PixRGB<byte> rgb2yuv2rgb8 ((PixVideoYUV<double>(rgb8))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb8 , rgb8 );
    PixRGB<byte> rgb2yuv2rgb9 ((PixVideoYUV<double>(rgb9))) ;  REQUIRE_PIX3_EQ2(rgb2yuv2rgb9 , rgb9 );
    PixRGB<byte> rgb2yuv2rgb10((PixVideoYUV<double>(rgb10)));  REQUIRE_PIX3_EQ2(rgb2yuv2rgb10, rgb10);
    PixRGB<byte> rgb2yuv2rgb11((PixVideoYUV<double>(rgb11)));  REQUIRE_PIX3_EQ2(rgb2yuv2rgb11, rgb11);
    PixRGB<byte> rgb2yuv2rgb12((PixVideoYUV<double>(rgb12)));  REQUIRE_PIX3_EQ2(rgb2yuv2rgb12, rgb12);
  }
}

static void Pixels_xx_conversions_xx_5(TestSuite& suite)
{
  const PixRGB<byte> rgb1  (  0,   0,   0);
  const PixRGB<byte> rgb2  (127, 127, 127);
  const PixRGB<byte> rgb3  (255, 255, 255);
  const PixRGB<byte> rgb4  (255,   0,   0);
  const PixRGB<byte> rgb5  (  0, 255,   0);
  const PixRGB<byte> rgb6  (  0,   0, 255);
  const PixRGB<byte> rgb7  (255, 127,  63);
  const PixRGB<byte> rgb8  (255,  63, 127);
  const PixRGB<byte> rgb9  (127, 255,  63);
  const PixRGB<byte> rgb10 ( 63, 255, 127);
  const PixRGB<byte> rgb11 (127,  63, 255);
  const PixRGB<byte> rgb12 ( 63, 127, 255);

  // test conversions rgb-->hsv
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb1)),    0,   0,   0);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb2)),    0,   0, 127);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb3)),    0,   0, 255);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb4)),    0, 100, 255);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb5)),  120, 100, 255);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb6)),  240, 100, 255);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb7)),   20,  75, 255);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb8)),  340,  75, 255);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb9)),  100,  75, 255);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb10)), 140,  75, 255);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb11)), 260,  75, 255);
  REQUIRE_PIX3_EQ(PixHSV<int>(PixHSV<double>(rgb12)), 220,  75, 255);

  // test conversions rgb-->h2sv2 (use floating point but not double precision to keep somewhat sane)
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb1), 0.500000,0.500000,0.000000,0.000000);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb2), 0.500000,0.500000,0.000000,0.4980392158029999793988906731101451441645622253417968750f);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb3), 0.500000,0.500000,0.000000,1.000000);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb4), 0.666666686535000052593602504202863201498985290527343750f,1.000000,1.000000,1.000000);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb5), 0.666666686535000052593602504202863201498985290527343750f,0.000000,1.000000,1.000000);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb6), 0.000000,0.500000,1.000000,1.000000);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb7),
                  0.777777791023000042436308376636588945984840393066406250f,
                  0.833333313465118408203125000000f,
                  0.75294119119600000367142911272821947932243347167968750f,
                  1.000000);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb8),
                  0.555555582046999951728594169253483414649963378906250f,
                  0.916666686535000052593602504202863201498985290527343750f,
                  0.75294119119600000367142911272821947932243347167968750f,
                  1.000000);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb9),
                  0.777777791023000042436308376636588945984840393066406250f,
                  0.1666666716340000076179705956747056916356086730957031250f,
                  0.75294119119600000367142911272821947932243347167968750f,
                  1.000000);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb10),
                  0.555555582046999951728594169253483414649963378906250f,
                  0.0833333358169000004700421868619741871953010559082031250f,
                  0.75294119119600000367142911272821947932243347167968750f,
                  1.000000);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb11),
                  0.11111111193900000126966176594578428193926811218261718750f,
                  0.583333313464999947406397495797136798501014709472656250f,
                  0.75294119119600000367142911272821947932243347167968750f,
                  1.000000);
  REQUIRE_PIX4_EQ(PixH2SV2<float>(rgb12),
                  0.11111111193900000126966176594578428193926811218261718750f,
                  0.4166666567330000181534899184043752029538154602050781250f,
                  0.75294119119600000367142911272821947932243347167968750f,
                  1.000000);

  // test round-trip conversions rgb-->hsv-->rgb
  {
    PixRGB<byte> rgb2hsv2rgb1 ((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb1))))) ;  REQUIRE_PIX3_EQ2(rgb2hsv2rgb1 , rgb1 );
    PixRGB<byte> rgb2hsv2rgb2 ((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb2))))) ;  REQUIRE_PIX3_EQ2(rgb2hsv2rgb2 , rgb2 );
    PixRGB<byte> rgb2hsv2rgb3 ((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb3))))) ;  REQUIRE_PIX3_EQ2(rgb2hsv2rgb3 , rgb3 );
    PixRGB<byte> rgb2hsv2rgb4 ((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb4))))) ;  REQUIRE_PIX3_EQ2(rgb2hsv2rgb4 , rgb4 );
    PixRGB<byte> rgb2hsv2rgb5 ((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb5))))) ;  REQUIRE_PIX3_EQ2(rgb2hsv2rgb5 , rgb5 );
    PixRGB<byte> rgb2hsv2rgb6 ((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb6))))) ;  REQUIRE_PIX3_EQ2(rgb2hsv2rgb6 , rgb6 );
    PixRGB<byte> rgb2hsv2rgb7 ((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb7))))) ;  REQUIRE_PIX3_EQ2(rgb2hsv2rgb7 , rgb7 );
    PixRGB<byte> rgb2hsv2rgb8 ((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb8))))) ;  REQUIRE_PIX3_EQ2(rgb2hsv2rgb8 , rgb8 );
    PixRGB<byte> rgb2hsv2rgb9 ((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb9))))) ;  REQUIRE_PIX3_EQ2(rgb2hsv2rgb9 , rgb9 );
    PixRGB<byte> rgb2hsv2rgb10((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb10)))));  REQUIRE_PIX3_EQ2(rgb2hsv2rgb10, rgb10);
    PixRGB<byte> rgb2hsv2rgb11((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb11)))));  REQUIRE_PIX3_EQ2(rgb2hsv2rgb11, rgb11);
    PixRGB<byte> rgb2hsv2rgb12((PixRGB<double>(PixHSV<int>(PixHSV<double>(rgb12)))));  REQUIRE_PIX3_EQ2(rgb2hsv2rgb12, rgb12);
  }
}

static void Pixels_xx_promotions_xx_1(TestSuite& suite)
{
  // check all pix-pix conversions among PixRGB<byte>/PixRGB<int>/PixRGB<float>

  REQUIRE(typeid(promote_trait<PixRGB<byte> , PixRGB<byte>        >::TP) == typeid(PixRGB<int>));
  REQUIRE(typeid(promote_trait<PixRGB<byte> , PixRGB<int16>       >::TP) == typeid(PixRGB<int>));
  REQUIRE(typeid(promote_trait<PixRGB<byte> , PixRGB<int32>       >::TP) == typeid(PixRGB<int>));
  REQUIRE(typeid(promote_trait<PixRGB<byte> , PixRGB<float>       >::TP) == typeid(PixRGB<float>));
  REQUIRE(typeid(promote_trait<PixRGB<byte> , PixRGB<double>      >::TP) == typeid(PixRGB<double>));
  REQUIRE(typeid(promote_trait<PixRGB<byte> , PixRGB<long double> >::TP) == typeid(PixRGB<long double>));

  REQUIRE(typeid(promote_trait<PixRGB<int16> , PixRGB<byte>        >::TP) == typeid(PixRGB<int>));
  REQUIRE(typeid(promote_trait<PixRGB<int16> , PixRGB<int16>       >::TP) == typeid(PixRGB<int>));
  REQUIRE(typeid(promote_trait<PixRGB<int16> , PixRGB<int32>       >::TP) == typeid(PixRGB<int>));
  REQUIRE(typeid(promote_trait<PixRGB<int16> , PixRGB<float>       >::TP) == typeid(PixRGB<float>));
  REQUIRE(typeid(promote_trait<PixRGB<int16> , PixRGB<double>      >::TP) == typeid(PixRGB<double>));
  REQUIRE(typeid(promote_trait<PixRGB<int16> , PixRGB<long double> >::TP) == typeid(PixRGB<long double>));

  REQUIRE(typeid(promote_trait<PixRGB<int32> , PixRGB<byte>        >::TP) == typeid(PixRGB<int>));
  REQUIRE(typeid(promote_trait<PixRGB<int32> , PixRGB<int16>       >::TP) == typeid(PixRGB<int>));
  REQUIRE(typeid(promote_trait<PixRGB<int32> , PixRGB<int32>       >::TP) == typeid(PixRGB<int>));
  REQUIRE(typeid(promote_trait<PixRGB<int32> , PixRGB<float>       >::TP) == typeid(PixRGB<float>));
  REQUIRE(typeid(promote_trait<PixRGB<int32> , PixRGB<double>      >::TP) == typeid(PixRGB<double>));
  REQUIRE(typeid(promote_trait<PixRGB<int32> , PixRGB<long double> >::TP) == typeid(PixRGB<long double>));

  REQUIRE(typeid(promote_trait<PixRGB<float> , PixRGB<byte>        >::TP) == typeid(PixRGB<float>));
  REQUIRE(typeid(promote_trait<PixRGB<float> , PixRGB<int16>       >::TP) == typeid(PixRGB<float>));
  REQUIRE(typeid(promote_trait<PixRGB<float> , PixRGB<int32>       >::TP) == typeid(PixRGB<float>));
  REQUIRE(typeid(promote_trait<PixRGB<float> , PixRGB<float>       >::TP) == typeid(PixRGB<float>));
  REQUIRE(typeid(promote_trait<PixRGB<float> , PixRGB<double>      >::TP) == typeid(PixRGB<double>));
  REQUIRE(typeid(promote_trait<PixRGB<float> , PixRGB<long double> >::TP) == typeid(PixRGB<long double>));

  REQUIRE(typeid(promote_trait<PixRGB<double> , PixRGB<byte>        >::TP) == typeid(PixRGB<double>));
  REQUIRE(typeid(promote_trait<PixRGB<double> , PixRGB<int16>       >::TP) == typeid(PixRGB<double>));
  REQUIRE(typeid(promote_trait<PixRGB<double> , PixRGB<int32>       >::TP) == typeid(PixRGB<double>));
  REQUIRE(typeid(promote_trait<PixRGB<double> , PixRGB<float>       >::TP) == typeid(PixRGB<double>));
  REQUIRE(typeid(promote_trait<PixRGB<double> , PixRGB<double>      >::TP) == typeid(PixRGB<double>));
  REQUIRE(typeid(promote_trait<PixRGB<double> , PixRGB<long double> >::TP) == typeid(PixRGB<long double>));

  REQUIRE(typeid(promote_trait<PixRGB<long double> , PixRGB<byte>        >::TP) == typeid(PixRGB<long double>));
  REQUIRE(typeid(promote_trait<PixRGB<long double> , PixRGB<int16>       >::TP) == typeid(PixRGB<long double>));
  REQUIRE(typeid(promote_trait<PixRGB<long double> , PixRGB<int32>       >::TP) == typeid(PixRGB<long double>));
  REQUIRE(typeid(promote_trait<PixRGB<long double> , PixRGB<float>       >::TP) == typeid(PixRGB<long double>));
  REQUIRE(typeid(promote_trait<PixRGB<long double> , PixRGB<double>      >::TP) == typeid(PixRGB<long double>));
  REQUIRE(typeid(promote_trait<PixRGB<long double> , PixRGB<long double> >::TP) == typeid(PixRGB<long double>));
}

static void Pixels_xx_conversions_xx_6(TestSuite& suite)
{
  PixRGB<float> p1(255.0f, 128.0f, 0.0f);
  PixH2SV2<float> p2(p1);

  REQUIRE(fabs(p2.H1() - 0.83399f) < 0.00001f);
  REQUIRE(fabs(p2.H2() - 0.74901f) < 0.00001f);
  REQUIRE(fabs(p2.S() - 1.0f) < 0.00001f);
  REQUIRE(fabs(p2.V() - 1.0f) < 0.00001f);
}

static void Pixels_xx_conversions_xx_7(TestSuite& suite)
{
  const PixRGB<byte> rgb1  (  0,   0,   0);
  const PixRGB<byte> rgb2  (127, 127, 127);
  const PixRGB<byte> rgb3  (255, 255, 255);
  const PixRGB<byte> rgb4  (255,   0,   0);
  const PixRGB<byte> rgb5  (  0, 255,   0);
  const PixRGB<byte> rgb6  (  0,   0, 255);
  const PixRGB<byte> rgb7  (255, 127,  63);
  const PixRGB<byte> rgb8  (255,  63, 127);
  const PixRGB<byte> rgb9  (127, 255,  63);
  const PixRGB<byte> rgb10 ( 63, 255, 127);
  const PixRGB<byte> rgb11 (127,  63, 255);
  const PixRGB<byte> rgb12 ( 63, 127, 255);

  // test conversions rgb-->dkl
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb1) * 127.0F + 128.0F),    5, 128, 130);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb2) * 127.0F + 128.0F),   77, 125, 132);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb3) * 127.0F + 128.0F),  257, 127, 127);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb4) * 127.0F + 128.0F),   56, 206, 109);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb5) * 127.0F + 128.0F),  174,  81,  64);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb6) * 127.0F + 128.0F),   36,  95, 215);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb7) * 127.0F + 128.0F),  106, 191,  93);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb8) * 127.0F + 128.0F),   71, 195, 133);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb9) * 127.0F + 128.0F),  189, 101,  61);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb10) * 127.0F + 128.0F), 185,  73,  89);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb11) * 127.0F + 128.0F),  56, 115, 207);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb12) * 127.0F + 128.0F),  86,  84, 195);

  // let's now enforce a bit more accuracy:
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb1) * 10000.0F), -9684,    17,   210);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb2) * 10000.0F), -3989,  -163,   316);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb3) * 10000.0F), 10162,   -24,   -28);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb4) * 10000.0F), -5652,  6193, -1454);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb5) * 10000.0F),  3637, -3656, -5026);
  REQUIRE_PIX3_EQ(PixDKL<int>(PixDKL<float>(rgb6) * 10000.0F), -7190, -2526,  6873);
}

static void Pixels_xx_operators_xx_1(TestSuite& suite)
{
  {
    PixRGB<byte> p1(12, 34, 56);
    REQUIRE_PIX3_EQ(p1 * 10, 120, 340, 560); // promoted to int
    p1 *= 10;
    REQUIRE_PIX3_EQ(p1, 120, 255, 255); // clamped to byte range
  }

  {
    PixRGB<byte> p1(120, 255, 255);
    REQUIRE_PIX3_EQ(p1 - 200, -80, 55, 55) // promoted to int
    p1 -= 200;
    REQUIRE_PIX3_EQ(p1, 0, 55, 55); // clamped to byte range
  }

  {
    PixRGB<byte> p1(0, 55, 55);
    REQUIRE_PIX3_EQ(p1 / 5, 0, 11, 11); // promoted to int
    REQUIRE_PIX3_EQ(p1 / -5, 0, -11, -11); // promoted to int
    p1 /= 5;
    REQUIRE_PIX3_EQ(p1, 0, 11, 11); // clamped to byte range
  }

  {
    PixRGB<byte> p1(0, 11, 11);
    REQUIRE_PIX3_EQ(p1 + 251, 251, 262, 262); // promoted to int
    p1 += 251;
    REQUIRE_PIX3_EQ(p1, 251, 255, 255); // clamped to byte range
  }

  {
    PixRGB<byte> p1(251, 255, 255);
    PixRGB<byte> p2(12, 34, 56);
    REQUIRE_PIX3_EQ(p1 / p2, 20, 7, 4); // (p1/p2) promoted to int
    p1 /= p2;
    REQUIRE_PIX3_EQ(p1, 20, 7, 4); // (p1/=p2) clamped to byte range
  }

  {
    PixRGB<byte> p1(20, 7, 4);
    REQUIRE_PIX3_EQ(p1 * p1, 400, 49, 16); // (p1*p1) promoted to int
    p1 *= p1;
    REQUIRE_PIX3_EQ(p1, 255, 49, 16); // (p1*=p1) clamped to byte range
  }

  {
    PixRGB<byte> p1(255, 49, 16);
    PixRGB<byte> p2(12, 34, 56);
    REQUIRE_PIX3_EQ(p2 - p1, -243, -15, 40); // (p2-1) promoted to int
    p2 -= p1;
    REQUIRE_PIX3_EQ(p2, 0, 0, 40); // (p2-=p1) clamped to byte range
  }

  {
    PixRGB<byte> p1(255, 49, 16);
    PixRGB<byte> p2(0, 220, 40);
    REQUIRE_PIX3_EQ(p1 + p2, 255, 269, 56); // (p1+p2) promoted to int
    p1 += p2;
    REQUIRE_PIX3_EQ(p1, 255, 255, 56); // (p1+=p2) clamped to byte range
  }

  {
    PixJpegYUV<int> p1(10, 20, 30);
    PixJpegYUV<float> p2(1, 2, 3);

    REQUIRE_PIX3_EQ(p1 * p2, 10.0f, 40.0f, 90.0f); // promoted to float
    p1 *= p2;
    REQUIRE_PIX3_EQ(p1, 10, 40, 90); // clamped back to int range
  }
}

static void Pixels_xx_operators_xx_2(TestSuite& suite)
{
  PixRGB<byte> p1(12, 34, 56);

  // here the result of p1 * float will be promoted to PixRGB<float>
  // and then assigned to p2:
  PixRGB<float> p2 = p1 * 100.0f;
  REQUIRE_PIX3_EQ(p2, 1200.0f, 3400.0f, 5600.0f);

  p2 = p1 / -2.0f;
  REQUIRE_PIX3_EQ(p2, -6.0f, -17.0f, -28.0f);

  // now, p1 * int will be promoted to PixRGB<int> then
  // range-converted to PixRGB<byte> then assigned to p3:
  PixRGB<byte> p3( p1 * 10 );
  REQUIRE_PIX3_EQ(p3, 120, 255, 255);

  // here, p1 * int will be PixRGB<int>, idem after division, then
  // clamped convert to PixRGB<byte> -- so, no overflow should occur:
  p3 = PixRGB<byte>( (p1 * 1000) / 1000 );
  REQUIRE_PIX3_EQ(p3, 12, 34, 56);

  //       PixRGB<int> -> PixRGB<float> -> same ->   same,  will clamp to byte
  p3 = PixRGB<byte>( ( ( (p1 * 1000)   -   5000.0f )   + 5000 ) / 1000.0f );
  REQUIRE_PIX3_EQ(p3, 12, 34, 56);
}

static void Pixels_xx_binary_layout_xx_1(TestSuite& suite)
{
  // Here we're checking that a PixRGB<byte> array has the expected
  // binary layout, i.e. is the same as a simple byte array with a
  // sequence of r, g, b values, in that order. We want to test this
  // because certain places in the code assume that PixRGB<byte> has
  // this layout, e.g. in PnmParser we slurp a series of raw bytes out
  // of the file into a PixRGB<byte> array.

  PixRGB<byte> pixels[2] =
    {
      PixRGB<byte>(55, 66, 77),
      PixRGB<byte>(111, 122, 133)
    };

  byte* mem = reinterpret_cast<byte*>(&pixels);

  REQUIRE_EQ(mem[0], 55);
  REQUIRE_EQ(mem[1], 66);
  REQUIRE_EQ(mem[2], 77);
  REQUIRE_EQ(mem[3], 111);
  REQUIRE_EQ(mem[4], 122);
  REQUIRE_EQ(mem[5], 133);

  mem[0] = 5;
  mem[1] = 6;
  mem[2] = 7;
  mem[3] = 11;
  mem[4] = 12;
  mem[5] = 13;

  REQUIRE_EQ(pixels[0].red(),   5);
  REQUIRE_EQ(pixels[0].green(), 6);
  REQUIRE_EQ(pixels[0].blue(),  7);
  REQUIRE_EQ(pixels[1].red(),   11);
  REQUIRE_EQ(pixels[1].green(), 12);
  REQUIRE_EQ(pixels[1].blue(),  13);
}

namespace
{
  const PixRGB<double> p1(-1.0, 2.4, 0.6);
  const PixRGB<double> p2(4.0, 9.0, 64.0);
  const PixRGB<double> p3(14.1, 0.3, 7.0);
  const PixRGB<double> p4(-1.0, 0.8, -0.3);
  const PixRGB<double> p5(-0.9, 0.8, -0.3);
  const PixRGB<double> p6(0.2, 0.4, 0.6);

  double coeff[4] = { 1.3, 1.5, 1.9, 2.1 } ;
}

static void Pixels_xx_math_functions_xx_1(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(maxmerge(p1,p3)*100.0), 1410, 240, 700);
}
static void Pixels_xx_math_functions_xx_2(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(minmerge(p2,p3)*100.0), 400, 30, 700);
}
static void Pixels_xx_math_functions_xx_3(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(abs(p1)*100.0), 100, 240, 60);
}
static void Pixels_xx_math_functions_xx_4(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(sqrt(p2)*100.0), 200, 300, 800);
}
static void Pixels_xx_math_functions_xx_5(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(floor(p1), -1.0, 2.0, 0.0);
}
static void Pixels_xx_math_functions_xx_6(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(ceil(p1), -1.0, 3.0, 1.0);
}
static void Pixels_xx_math_functions_xx_7(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(round(p1), -1.0, 2.0, 1.0);
}
static void Pixels_xx_math_functions_xx_8(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(log(p2)*100.0), 138, 219, 415);
}
static void Pixels_xx_math_functions_xx_9(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(log10(p2)*100.0), 60, 95, 180);
}
static void Pixels_xx_math_functions_xx_10(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(exp(p1)*100.0), 36, 1102, 182);
}
static void Pixels_xx_math_functions_xx_11(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(sin(p1)*100.0), -84, 67, 56);
}
static void Pixels_xx_math_functions_xx_12(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(cos(p1)*100.0), 54, -73, 82);
}
static void Pixels_xx_math_functions_xx_13(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(tan(p1)*100.0), -155, -91, 68);
}
static void Pixels_xx_math_functions_xx_14(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(atan(p1)*100.0), -78, 117, 54);
}
static void Pixels_xx_math_functions_xx_15(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(pow(p1, 3.0)*100.0), -100, 1382, 21);
}
static void Pixels_xx_math_functions_xx_16(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(pow(p3, p1)*100.0), 7, 5, 321);
}
static void Pixels_xx_math_functions_xx_17(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(erf(p1)*100.0), -84, 99, 60);
}
static void Pixels_xx_math_functions_xx_18(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(erfc(p1)*100.0), 184, 0, 39);
}
static void Pixels_xx_math_functions_xx_19(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(sec(p1)*100.0), 185, -135, 121);
}
static void Pixels_xx_math_functions_xx_20(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(cosec(p1)*100.0), -118, 148, 177);
}
static void Pixels_xx_math_functions_xx_21(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(cotan(p1)*100.0), -64, -109, 146);
}
static void Pixels_xx_math_functions_xx_22(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(asin(p4)*100.0), -157, 92, -30);
}
static void Pixels_xx_math_functions_xx_23(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(acos(p4)*100.0), 314, 64, 187);
}
static void Pixels_xx_math_functions_xx_24(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(asec(p2)*100.0), 131, 145, 155);
}
static void Pixels_xx_math_functions_xx_25(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(acosec(p2)*100.0), 25, 11, 1);
}
static void Pixels_xx_math_functions_xx_26(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(acotan(p1)*100.0), -78, 39, 103);
}
static void Pixels_xx_math_functions_xx_27(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(sinh(p1)*100.0), -117, 546, 63);
}
static void Pixels_xx_math_functions_xx_28(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(cosh(p1)*100.0), 154, 555, 118);
}
static void Pixels_xx_math_functions_xx_29(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(tanh(p1)*100.0), -76, 98, 53);
}
static void Pixels_xx_math_functions_xx_30(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(sech(p1)*100.0), 64, 17, 84);
}
static void Pixels_xx_math_functions_xx_31(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(cosech(p1)*100.0), -85, 18, 157);
}
static void Pixels_xx_math_functions_xx_32(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(cotanh(p1)*100.0), -131, 101, 186);
}
static void Pixels_xx_math_functions_xx_33(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(asinh(p1)*100.0), -88, 160, 56);
}
static void Pixels_xx_math_functions_xx_34(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(acosh(p2)*100.0), 206, 288, 485);
}
static void Pixels_xx_math_functions_xx_35(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(atanh(p5)*100.0), -147, 109, -30);
}
static void Pixels_xx_math_functions_xx_36(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(asech(p6)*100.0), 229, 156, 109);
}
static void Pixels_xx_math_functions_xx_37(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(acosech(p2)*100.0), 24, 11, 1);
}
static void Pixels_xx_math_functions_xx_38(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(acotanh(p2)*100.0), 25, 11, 1);
}
static void Pixels_xx_math_functions_xx_39(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(sign(p1)), -1, 1, 1);
}
static void Pixels_xx_math_functions_xx_40(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(logN(p2, 3.0)*100.0), 126, 200, 378);
}
static void Pixels_xx_math_functions_xx_41(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(logsig(p2, 3.0, 5.0)*100.0), 99, 99, 100);
}
static void Pixels_xx_math_functions_xx_42(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(tansig(p1)*100.0), 838, 100, 130);
}
static void Pixels_xx_math_functions_xx_43(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(saturate(p3, 6.6)*100.0), 660, 30, 660);
}
static void Pixels_xx_math_functions_xx_44(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(poly(p3, &coeff[0], 4)*100.0), 628695, 197, 82520);
}
static void Pixels_xx_math_functions_xx_45(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(gauss(p1, p3, p2)*100.0), 0, 4, 0);
}
static void Pixels_xx_math_functions_xx_46(TestSuite& suite)
{
  REQUIRE_PIX3_EQ(PixRGB<int>(gauss(p1, 1.0, 1.0)*100.0), 5, 14, 36);
}
static void Pixels_xx_math_functions_xx_47(TestSuite& suite)
{
  REQUIRE_EQ(int(max(p1)*100.0), 240);
}
static void Pixels_xx_math_functions_xx_48(TestSuite& suite)
{
  REQUIRE_EQ(int(min(p1)*100.0), -100);
}
static void Pixels_xx_math_functions_xx_49(TestSuite& suite)
{
  REQUIRE_EQ(isFinite(p1), true);
}
static void Pixels_xx_math_functions_xx_50(TestSuite& suite)
{
  REQUIRE_EQ(int(0.5 + sum(p1)*100.0), 200);
}
static void Pixels_xx_math_functions_xx_51(TestSuite& suite)
{
  REQUIRE_EQ(int(mean(p1)*100.0), 66);
}

///////////////////////////////////////////////////////////////////////
//
// main
//
///////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  TestSuite suite;

  suite.ADD_TEST(Pixels_xx_constructors_xx_1);
  suite.ADD_TEST(Pixels_xx_conversions_xx_1);
  suite.ADD_TEST(Pixels_xx_conversions_xx_2);
  suite.ADD_TEST(Pixels_xx_conversions_xx_3);
  suite.ADD_TEST(Pixels_xx_conversions_xx_4);
  suite.ADD_TEST(Pixels_xx_conversions_xx_5);
  suite.ADD_TEST(Pixels_xx_conversions_xx_6);
  suite.ADD_TEST(Pixels_xx_conversions_xx_7);
  suite.ADD_TEST(Pixels_xx_promotions_xx_1);
  suite.ADD_TEST(Pixels_xx_operators_xx_1);
  suite.ADD_TEST(Pixels_xx_operators_xx_2);
  suite.ADD_TEST(Pixels_xx_binary_layout_xx_1);

  suite.ADD_TEST(Pixels_xx_math_functions_xx_1);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_2);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_3);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_4);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_5);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_6);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_7);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_8);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_9);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_10);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_11);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_12);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_13);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_14);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_15);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_16);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_17);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_18);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_19);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_20);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_21);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_22);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_23);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_24);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_25);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_26);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_27);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_28);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_29);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_30);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_31);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_32);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_33);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_34);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_35);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_36);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_37);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_38);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_39);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_40);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_41);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_42);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_43);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_44);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_45);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_46);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_47);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_48);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_49);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_50);
  suite.ADD_TEST(Pixels_xx_math_functions_xx_51);

  suite.parseAndRun(argc, argv);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
