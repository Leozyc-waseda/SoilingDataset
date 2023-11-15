/*!@file TestSuite/whitebox-Image.C Test Image class */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/whitebox-Image.C $
// $Id: whitebox-Image.C 15310 2012-06-01 02:29:24Z itti $
//

#include "Util/log.H"

#include "Image/ColorOps.H"
#include "Image/Convolver.H"
#include "Image/Coords.H"    // for Dims operators
#include "Image/CutPaste.H"  // for crop() etc.
#include "Image/DrawOps.H"   // for warp3D() etc.
#include "Image/FilterOps.H" // for lowPass9() etc.
#include "Image/FourierEngine.H"
#include "Image/Hash.H"      // for sha1byte(), sha256byte()
#include "Image/IO.H"
#include "Image/Image.H"
#include "Image/Kernels.H"   // for binomialKernel()
#include "Image/Layout.H"
#include "Image/LinearAlgebra.H" // for svd(), naiveUnstablePseudoInv()
#include "Image/MathOps.H"   // for quadEnergy() etc.
#include "Image/MatrixOps.H" // for matrixMult(), matrixInv(), etc
#include "Image/Pixels.H"
#include "Image/Range.H"
#include "Image/ShapeOps.H"  // for zoomXY() etc.
#include "Image/Transforms.H" // for zoomXY() etc.
#include "Image/Point2D.H"
#include "TestSuite/TestSuite.H"
#include "rutz/rand.h"

#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>

namespace
{
  struct Counter
  {
    Counter(int cc = 0, int ss = 1) : c(cc), s(ss) {}

    int operator()() { const int r = c; c+=s; return r; }

    int c;
    int s;
  };

  template <class T, class F>
  Image<T> generateImage(const Dims& d, F func)
  {
    Image<T> result(d, NO_INIT);

    fill(result, func);

    return result;
  }

  template <class T, class F>
  inline Image<T> generateImage(int w, int h, F func)
  { return generateImage<T>(Dims(w, h), func); }
}

struct InstanceCounter
{
  static int numCreated;
  static int numDestroyed;

  static void reset() { numCreated = numDestroyed = 0; }

  InstanceCounter() { ++numCreated; }

  ~InstanceCounter() { ++numDestroyed; }

  InstanceCounter(const InstanceCounter&) { ++numCreated; }

  InstanceCounter& operator=(const InstanceCounter&) { return *this; }
};

int InstanceCounter::numCreated = 0;
int InstanceCounter::numDestroyed = 0;

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

//---------------------------------------------------------------------
//
// Rectangle
//
//---------------------------------------------------------------------

static void Rectangle_xx_getOverlap_xx_1(TestSuite& suite)
{
  // assume the original range is [0,10[, now if we merge that with a
  // range of [lo,hi[ then we expect to get either a range of
  // [newlo,newhi[ if valid is true, or else a null range if valid is
  // false:

  struct TestRange { int lo; int hi; int newlo; int newhi; bool valid; };

  const TestRange offsets[] =
    {
      // 'a' contained within original range
      // 'b' contained within second range
      // 'X' contained within overlap of the two ranges

      //     -1         0         1         2
      // 54321098765432101234567890123456789012345
      //                aaaaaaaaaa                 <----- original range

      /* bbbbbbbbbb                                */ { -15, -5, 0,  0, false },
      /*      bbbbbbbbbb                           */ { -10,  0, 0,  0, false },
      /*               b                           */ {  -1,  0, 0,  0, false },
      /*                X                          */ {   0,  1, 0,  1, true },
      /*           bbbbbXXXXX                      */ {  -5,  5, 0,  5, true },
      /*                XXXXX                      */ {   0,  5, 0,  5, true },
      /*                XXXXXXXXXX                 */ {   0, 10, 0, 10, true },
      /*                  XXXXXX                   */ {   2,  8, 2,  8, true },
      /*                     XXXXX                 */ {   5, 10, 5, 10, true },
      /*                     XXXXXbbbbb            */ {   5, 15, 5, 10, true },
      /*                         X                 */ {   9, 10, 9, 10, true },
      /*                          b                */ {  10, 11, 0,  0, false },
      /*                          bbbbbbbbbb       */ {  10, 20, 0,  0, false },
      /*                               bbbbbbbbbb  */ {  15, 25, 0,  0, false }
    };

  const int left = 10;
  const int top = -5;

  const Rectangle r1(Point2D<int>(left, top), Dims(10, 10));

  for (size_t xoff = 0; xoff < sizeof(offsets) / sizeof(offsets[0]); ++xoff)
    for (size_t yoff = 0; yoff < sizeof(offsets) / sizeof(offsets[0]); ++yoff)
      {
        const Rectangle r2(Point2D<int>(left + offsets[xoff].lo,
                                   top  + offsets[yoff].lo),
                           Dims(offsets[xoff].hi - offsets[xoff].lo,
                                offsets[yoff].hi - offsets[yoff].lo));

        const Rectangle r12 = r1.getOverlap(r2);
        const Rectangle r21 = r2.getOverlap(r1);

        REQUIRE_EQ_USERTYPE(r12, r21);

        if (!offsets[xoff].valid || !offsets[yoff].valid)
          REQUIRE(r12.isValid() == false);
        else
          {
            const Rectangle r3(Point2D<int>(left + offsets[xoff].newlo,
                                       top  + offsets[yoff].newlo),
                               Dims(offsets[xoff].newhi - offsets[xoff].newlo,
                                    offsets[yoff].newhi - offsets[yoff].newlo));

            REQUIRE_EQ_USERTYPE(r12, r3);
          }
      }
}

static void Rectangle_xx_constrain_xx_1(TestSuite& suite)
{
  REQUIRE_EQ_USERTYPE(constrainRect
                      (Rectangle(Point2D<int>(-2, -3), Dims(5, 15)),
                       Rectangle(Point2D<int>(0, 0), Dims(10, 10)),
                       7, 9),
                      Rectangle(Point2D<int>(0, 0), Dims(7, 9)));

  REQUIRE_EQ_USERTYPE(constrainRect
                      (Rectangle(Point2D<int>(-2, -3), Dims(5, 15)),
                       Rectangle(Point2D<int>(0, 0), Dims(10, 10)),
                       7, 12),
                      Rectangle(Point2D<int>(0, 0), Dims(7, 10)));
}

//---------------------------------------------------------------------
//
// Constructors & Destructors
//
//---------------------------------------------------------------------

static void Image_xx_construct_from_array_xx_1(TestSuite& suite)
{
  byte array[6] = { 3, 5, 7, 9, 11, 13 };

  Image<byte> a(&array[0], 3, 2);

  REQUIRE_EQ(a.getWidth(), 3);
  REQUIRE_EQ(a.getHeight(), 2);
  REQUIRE_EQ_USERTYPE(a.getDims(), Dims(3,2));
  REQUIRE_EQ_USERTYPE(a.getBounds(), Rectangle::tlbrI(0, 0, 1, 2));
  REQUIRE_EQ((int)a.getVal(2,0), 7);
  REQUIRE_EQ((int)a.getVal(0,1), 9);
}

static void Image_xx_construct_UDT_xx_1(TestSuite& suite)
{
  InstanceCounter::reset();

  {
    // We are testing that, even though we request NO_INIT, InstanceCounter
    // objects should still get constructed+destructed since they don't
    // satisfy TypeTraits<T>::isTrivial.
    Image<InstanceCounter> a(5, 5, NO_INIT);
  }

  // might create more than 25 objects if there were copies
  REQUIRE(InstanceCounter::numCreated >= 25);
  REQUIRE_EQ(InstanceCounter::numCreated, InstanceCounter::numDestroyed);
}

static void Image_xx_construct_and_clear_xx_1(TestSuite& suite)
{
  Image<float> a(3, 4, ZEROS);

  REQUIRE_EQ(a.getWidth(), 3);
  REQUIRE_EQ(a.getHeight(), 4);
  REQUIRE_EQ_USERTYPE(a.getDims(), Dims(3,4));
  REQUIRE_EQ_USERTYPE(a.getBounds(), Rectangle::tlbrI(0, 0, 3, 2));
  REQUIRE_EQ(a.getVal(0,0), 0.0f);
  REQUIRE_EQ(a.getVal(2,3), 0.0f);
}

static void Image_xx_default_construct_xx_1(TestSuite& suite)
{
  Image<byte> a;

  REQUIRE_EQ(a.getWidth(), 0);
  REQUIRE_EQ(a.getHeight(), 0);
  REQUIRE_EQ_USERTYPE(a.getDims(), Dims());
  REQUIRE_EQ_USERTYPE(a.getBounds(), Rectangle::tlbrI(0, 0, -1, -1));
  REQUIRE(a.begin() == a.end());
}

static void Image_xx_swap_xx_1(TestSuite& suite)
{
  Image<byte> a(3, 3, NO_INIT); a.setVal(2,2,100);
  Image<byte> b(4, 4, NO_INIT); b.setVal(3,3,50);

  a.swap(b);

  REQUIRE_EQ(a.getHeight(), 4);
  REQUIRE_EQ(a.getWidth(), 4);
  REQUIRE_EQ(b.getHeight(), 3);
  REQUIRE_EQ(b.getWidth(), 3);
  REQUIRE_EQ((int)a.getVal(3,3), 50);
  REQUIRE_EQ((int)b.getVal(2,2), 100);
}

static void Image_xx_copy_xx_1(TestSuite& suite)
{
  Image<float> a(2, 2, NO_INIT);
  a.setVal(0,0, 3.0);

  Image<float> b = a;

  REQUIRE_EQ(b.getVal(0,0), 3.0f);
}

static void Image_xx_copy_on_write_xx_1(TestSuite& suite)
{
  Image<float> a(3, 3, NO_INIT);
  a.setVal(1,1,2.5);

  Image<float> b = a;
  b.setVal(1,1,5.5);

  REQUIRE_EQ(a.getVal(1,1), 2.5f);
  REQUIRE(!a.hasSameData(b));
}

static void Image_xx_copy_on_write_xx_2(TestSuite& suite)
{
  Image<byte> a(3, 3, NO_INIT);

  Image<byte> b( a );

  // make sure that we didn't do an expensive data copy; i.e. check that the
  // ref-counting actually worked:
  REQUIRE(a.hasSameData(b));
}

static void Image_xx_copy_on_write_xx_3(TestSuite& suite)
{
  Image<byte> a(3, 3, NO_INIT);

  Image<byte> b;

  b = a;

  // like the previous test, but with assignment instead of copy-construct
  REQUIRE(a.hasSameData(b));
}

static void Image_xx_assignment_to_self_xx_1(TestSuite& suite)
{
  Image<byte> a(4, 4, NO_INIT); a.setVal(1,1,50);

  a = a;

  REQUIRE_EQ(a.refCount(), long(1));
  REQUIRE_EQ((int)a.getVal(1,1), 50);
}

static void Image_xx_reshape_xx_1(TestSuite& suite)
{
  const byte dat[] =
    {
      1, 5, 9, 13,
      2, 6, 10, 14,
      3, 7, 11, 15,
      4, 8, 12, 16
    };

  Image<byte> a(&dat[0], 4, 4);

  REQUIRE_EQ(reshape(a, Dims(16,1)), Image<byte>(&dat[0], 16, 1));
}

static void Image_xx_indexing_xx_1(TestSuite& suite)
{
  short array[9] = { 12570, -9, 11,
                     257,    0, 4999,
                     5353, -1701, 1 };

  Image<short> a(array, 3, 3);

  REQUIRE_EQ(a[0],             12570);
  REQUIRE_EQ(a.getVal(1, 1),   0);
  REQUIRE_EQ(a[Point2D<int>(2,1)], 4999);
  REQUIRE_EQ(a[Point2D<int>(1,2)], -1701);

  a[Point2D<int>(0,2)] = 12345;

  REQUIRE_EQ(a[6], 12345);
}

static void Image_xx_type_convert_xx_1(TestSuite& suite)
{
  float array[4] = { -10.9, 3.2, 254.7, 267.3 };

  Image<float> f(array, 4, 1);

  Image<byte> b = f;

  REQUIRE_EQ((int)b.getVal(0), 0); // clamped
  REQUIRE_EQ((int)b.getVal(1), 3);
  REQUIRE_EQ((int)b.getVal(2), 254);
  REQUIRE_EQ((int)b.getVal(3), 255); // clamped
}

static void Image_xx_type_convert_xx_2(TestSuite& suite)
{
  byte array[4] = { 0, 52, 107, 255 };

  Image<byte> b(array, 4, 1);

  Image<float> f = b;

  REQUIRE_EQ(f.getVal(0), 0.0f);
  REQUIRE_EQ(f.getVal(1), 52.0f);
  REQUIRE_EQ(f.getVal(2), 107.0f);
  REQUIRE_EQ(f.getVal(3), 255.0f);
}

static void Image_xx_type_convert_xx_3(TestSuite& suite)
{
  byte array[4] = { 0, 52, 107, 255 };

  Image<byte> b(array, 4, 1);

  // promotions:   (int)    (int)    and promote to float
  Image<float> f = (b * 5 - 100);

  REQUIRE_EQ(f.getVal(0), -100.0f);
  REQUIRE_EQ(f.getVal(1), 160.0f);
  REQUIRE_EQ(f.getVal(2), 435.0f);
  REQUIRE_EQ(f.getVal(3), 1175.0f);

  // promotions:   (int)    and cast back to byte with clamping
  Image<byte> bb = b * 2;

  REQUIRE_EQ(bb.getVal(0), 0);
  REQUIRE_EQ(bb.getVal(1), 104);
  REQUIRE_EQ(bb.getVal(2), 214);
  REQUIRE_EQ(bb.getVal(3), 255);

  //  (float)
  f = (b - 100.0F) / 2.0F;
  REQUIRE_EQ(f.getVal(0), -50.0f);
  REQUIRE_EQ(f.getVal(1), -24.0f);
  REQUIRE_EQ(f.getVal(2), 3.5f);
  REQUIRE_EQ(f.getVal(3), 77.5f);

  //  (double) and cast back to float with clamping
  f = (b - 100.0) / 2.0;
  REQUIRE_EQ(f.getVal(0), -50.0f);
  REQUIRE_EQ(f.getVal(1), -24.0f);
  REQUIRE_EQ(f.getVal(2), 3.5f);
  REQUIRE_EQ(f.getVal(3), 77.5f);
}

static void Image_xx_attach_detach_xx_1(TestSuite& suite)
{
  Image<float> a;

  float array[4] = { 2.0, 4.0, 6.0, 8.0 };

  a.attach(&array[0], 2, 2);

  REQUIRE_EQ(a.getVal(1,0), 4.0f);
  REQUIRE_EQ(a.getVal(0,1), 6.0f);

  // Make sure that if we modify the Image, that we "write through" to the
  // underlying float array
  a.setVal(1,1,3.5);

  REQUIRE_EQ(array[3], 3.5f);

  // Make sure that if we make a copy of the attach'ed array, that we DON'T
  // write through to the original float array
  Image<float> b = a.deepcopy();

  REQUIRE_EQ(b.getVal(1,1), 3.5f);

  b.setVal(0,0, -1.0f);

  REQUIRE_EQ(array[0], 2.0f);

  a.detach();
}

static void Image_xx_destruct_xx_1(TestSuite& suite)
{
  Image<float> a(5, 5, NO_INIT);

  REQUIRE_EQ(a.refCount(), long(1));

  {
    Image<float> b = a;

    REQUIRE_EQ(a.refCount(), long(2)); // make sure the count went up...
  }

  REQUIRE_EQ(a.refCount(), long(1)); // ... and back down again here
}

//---------------------------------------------------------------------
//
// Iterators
//
//---------------------------------------------------------------------

static void Image_xx_begin_end_xx_1(TestSuite& suite)
{
  Image<float> a(135, 1, NO_INIT);

  REQUIRE_EQ((int)(a.end() - a.begin()), 135);
  REQUIRE_EQ(*(a.begin()), a.getVal(0,0));
}

static void Image_xx_beginw_endw_xx_1(TestSuite& suite)
{
  Image<byte> b(20, 17, NO_INIT);

  b.setVal(0,0,1);

  Image<byte> c = b;

  REQUIRE(b.hasSameData(c));

  Image<byte>::iterator itr = c.beginw(), stop = c.endw();

  REQUIRE(!b.hasSameData(c));

  REQUIRE_EQ((int)(stop - itr), 20*17);

  REQUIRE_EQ(int(c.getVal(0)), 1);

  *itr = 49;

  REQUIRE_EQ(int(c.getVal(0)), 49);
  REQUIRE_EQ(int(b.getVal(0)), 1);
}

//---------------------------------------------------------------------
//
// Access functions
//
//---------------------------------------------------------------------

static void Image_xx_getMean_xx_1(TestSuite& suite)
{
  Image<float> img = generateImage<float>(100,100,Counter(0));

  REQUIRE_EQ(mean(img), 4999.5);
}

//---------------------------------------------------------------------
//
// Basic image manipulations
//
//---------------------------------------------------------------------

static void Image_xx_clear_xx_1(TestSuite& suite)
{
  Image<float> img(100, 100, NO_INIT);

  img.clear(1.5F);

  REQUIRE_EQ(mean(img), 1.5);
  REQUIRE_EQ(img.getVal(0,0), 1.5f);
  REQUIRE_EQ(img.getVal(99,99), 1.5f);
}

static void Image_xx_plus_eq_scalar_xx_1(TestSuite& suite)
{
  Image<float> img = generateImage<float>(100,100,Counter(0));

  REQUIRE_EQ(mean(img), 4999.5);

  img += 1.0F;

  REQUIRE_EQ(mean(img), 5000.5);
  REQUIRE_EQ(img.getVal(0), 1.0f);
  REQUIRE_EQ(img.getVal(img.getSize()-1), float(100*100));

  byte array[4] = { 0, 52, 107, 255 };
  Image<byte> b(array, 4, 1);

  b += 100;

  REQUIRE_EQ(b.getVal(0), 100);
  REQUIRE_EQ(b.getVal(1), 152);
  REQUIRE_EQ(b.getVal(2), 207);
  REQUIRE_EQ(b.getVal(3), 255);   // clamp
}

static void Image_xx_minus_eq_scalar_xx_1(TestSuite& suite)
{
  Image<float> img = generateImage<float>(100,100,Counter(0));

  REQUIRE_EQ(mean(img), 4999.5F);

  img -= 1.0F;

  REQUIRE_EQ(mean(img), 4998.5F);
  REQUIRE_EQ(img.getVal(0), -1.0f);
  REQUIRE_EQ(img.getVal(img.getSize()-1), float(100*100-2));

  byte array[4] = { 0, 52, 107, 255 };
  Image<byte> b(array, 4, 1);

  b -= 100;

  REQUIRE_EQ(b.getVal(0), 0);  // clamp
  REQUIRE_EQ(b.getVal(1), 0);  // clamp
  REQUIRE_EQ(b.getVal(2), 7);
  REQUIRE_EQ(b.getVal(3), 155);
}

static void Image_xx_mul_eq_scalar_xx_1(TestSuite& suite)
{
  Image<float> img = generateImage<float>(100,100,Counter(0));

  REQUIRE_EQ(mean(img), 4999.5);

  img *= 2.0f;

  REQUIRE_EQ(mean(img), 9999.0);
  REQUIRE_EQ(img.getVal(1), 2.0f);
  REQUIRE_EQ(img.getVal(img.getSize()-1), float((100*100-1)*2));

  byte array[4] = { 0, 52, 107, 255 };
  Image<byte> b(array, 4, 1);

  b *= 3;

  REQUIRE_EQ(b.getVal(0), 0);
  REQUIRE_EQ(b.getVal(1), 156);
  REQUIRE_EQ(b.getVal(2), 255);  // clamp
  REQUIRE_EQ(b.getVal(3), 255);  // clamp
}

static void Image_xx_div_eq_scalar_xx_1(TestSuite& suite)
{
  Image<float> img = generateImage<float>(100,100,Counter(0));

  REQUIRE_EQ(mean(img), 4999.5);

  img /= 2.0f;

  REQUIRE_EQ(mean(img), 2499.75);
  REQUIRE_EQ(img.getVal(1), 0.5f);
  REQUIRE_EQ(img.getVal(img.getSize()-1), float((100*100-1)/2.0));


  byte array[4] = { 0, 52, 107, 255 };
  Image<byte> b(array, 4, 1);

  b /= 3;

  REQUIRE_EQ(b.getVal(0), 0);
  REQUIRE_EQ(b.getVal(1), 17);
  REQUIRE_EQ(b.getVal(2), 35);
  REQUIRE_EQ(b.getVal(3), 85);
}

static void Image_xx_lshift_eq_scalar_xx_1(TestSuite& suite)
{
  byte barray[4] = { 0, 52, 107, 255 };
  Image<byte> b(barray, 4, 1);

  b <<= 1;

  REQUIRE_EQ(b.getVal(0), 0);
  REQUIRE_EQ(b.getVal(1), 104);
  REQUIRE_EQ(b.getVal(2), 214);
  REQUIRE_EQ(b.getVal(3), 254);

  int iarray[4] = { -37, -1, 297, 65535 };
  Image<int> i(iarray, 4, 1);

  i <<= 3;

  REQUIRE_EQ(i.getVal(0), -296);
  REQUIRE_EQ(i.getVal(1), -8);
  REQUIRE_EQ(i.getVal(2), 2376);
  REQUIRE_EQ(i.getVal(3), 524280);
}

static void Image_xx_rshift_eq_scalar_xx_1(TestSuite& suite)
{
  byte barray[4] = { 0, 52, 107, 255 };
  Image<byte> b(barray, 4, 1);

  b >>= 3;

  REQUIRE_EQ(b.getVal(0), 0);
  REQUIRE_EQ(b.getVal(1), 6);
  REQUIRE_EQ(b.getVal(2), 13);
  REQUIRE_EQ(b.getVal(3), 31);

  int iarray[4] = { -37, -1, 297, 65535 };
  Image<int> i(iarray, 4, 1);

  i >>= 3;

  REQUIRE_EQ(i.getVal(0), -5);
  REQUIRE_EQ(i.getVal(1), -1);
  REQUIRE_EQ(i.getVal(2), 37);
  REQUIRE_EQ(i.getVal(3), 8191);
}

static void Image_xx_plus_eq_array_xx_1(TestSuite& suite)
{
  Image<float> img1 = generateImage<float>(1000, 1000, Counter(0));

  Image<float> img2 = generateImage<float>(1000, 1000, Counter(1));

  REQUIRE_EQ(mean(img1), 499999.5);

  img1 += img2;

  REQUIRE_EQ(mean(img1), 1000000.0);
  REQUIRE_EQ(img1.getVal(0), 1.0f);
  REQUIRE_EQ(img1.getVal(img1.getSize()-1), float(1000*1000 - 1 + 1000*1000));

  byte array[4] = { 0, 52, 107, 255 }, array2[4] = { 201, 23, 56, 5 };
  Image<byte> b(array, 4, 1), bb(array2, 4, 1);

  b += bb;

  REQUIRE_EQ(b.getVal(0), 201);
  REQUIRE_EQ(b.getVal(1), 75);
  REQUIRE_EQ(b.getVal(2), 163);
  REQUIRE_EQ(b.getVal(3), 255); // clamp
}

static void Image_xx_minus_eq_array_xx_1(TestSuite& suite)
{
  Image<float> img1 = generateImage<float>(1000, 1000, Counter(1));

  Image<float> img2 = generateImage<float>(1000, 1000, Counter(0));

  REQUIRE_EQ(mean(img1), 500000.5);

  img1 -= img2;

  REQUIRE_EQ(mean(img1), 1.0);
  REQUIRE_EQ(img1.getVal(0), 1.0f);
  REQUIRE_EQ(img1.getVal(img1.getSize()-1), 1.0f);

  byte array[4] = { 0, 52, 107, 255 }, array2[4] = { 201, 23, 56, 5 };
  Image<byte> b(array, 4, 1), bb(array2, 4, 1);

  b -= bb;

  REQUIRE_EQ(b.getVal(0), 0); // clamp
  REQUIRE_EQ(b.getVal(1), 29);
  REQUIRE_EQ(b.getVal(2), 51);
  REQUIRE_EQ(b.getVal(3), 250);
}

static void Image_xx_mul_eq_array_xx_1(TestSuite& suite)
{
  Image<float> img1 = generateImage<float>(1000, 1000, Counter(0));

  Image<float> img2(1000, 1000, NO_INIT); img2.clear(2.0f);

  REQUIRE_EQ(mean(img1), 499999.5);

  img1 *= img2;

  REQUIRE_EQ(mean(img1), 999999.0);
  REQUIRE_EQ(img1.getVal(1), 2.0f);
  REQUIRE_EQ(img1.getVal(img1.getSize()-1), float((1000*1000-1)*2));

  byte array[4] = { 0, 52, 107, 255 }, array2[4] = { 201, 3, 56, 5 };
  Image<byte> b(array, 4, 1), bb(array2, 4, 1);

  b *= bb;

  REQUIRE_EQ(b.getVal(0), 0);
  REQUIRE_EQ(b.getVal(1), 156);
  REQUIRE_EQ(b.getVal(2), 255); // clamp
  REQUIRE_EQ(b.getVal(3), 255); // clamp
}

static void Image_xx_div_eq_array_xx_1(TestSuite& suite)
{
  Image<float> img1 = generateImage<float>(1000,1000,Counter(0));

  Image<float> img2(1000, 1000, NO_INIT); img2.clear(2.0f);

  REQUIRE_EQ(mean(img1), 499999.5);

  img1 /= img2;

  REQUIRE_EQ(mean(img1), 249999.75);
  REQUIRE_EQ(img1.getVal(1), 0.5f);
  REQUIRE_EQ(img1.getVal(img1.getSize()-1), 499999.5f);

  byte array[4] = { 0, 52, 107, 255 }, array2[4] = { 201, 3, 56, 5 };
  Image<byte> b(array, 4, 1), bb(array2, 4, 1);

  b /= bb;

  REQUIRE_EQ(b.getVal(0), 0);
  REQUIRE_EQ(b.getVal(1), 17);
  REQUIRE_EQ(b.getVal(2), 1);
  REQUIRE_EQ(b.getVal(3), 51);
}

static void Image_xx_plus_scalar_xx_1(TestSuite& suite)
{
  Image<float> img = generateImage<float>(200,500,Counter(0));

  REQUIRE_EQ(mean(img), 49999.5f);

  Image<float> img2 = img + 1.0f;

  REQUIRE_EQ(mean(img2), 50000.5f);
  REQUIRE_EQ(img2.getVal(0), 1.0f);
  REQUIRE_EQ(img2.getVal(img2.getSize()-1), float(200*500));
}

static void Image_xx_minus_scalar_xx_1(TestSuite& suite)
{
  Image<float> img = generateImage<float>(200,500,Counter(0));

  REQUIRE_EQ(mean(img), 49999.5F);

  Image<float> img2 = img - 1.0F;

  REQUIRE_EQ(mean(img2), 49998.5F);
  REQUIRE_EQ(img2.getVal(0), -1.0f);
  REQUIRE_EQ(img2.getVal(img2.getSize()-1), float(200*500-2));
}

static void Image_xx_mul_scalar_xx_1(TestSuite& suite)
{
  Image<float> img = generateImage<float>(200,500,Counter(0));

  REQUIRE_EQ(mean(img), 49999.5f);

  Image<float> img2 = img * 2.0f;

  REQUIRE_EQ(mean(img2), 99999.0f);
  REQUIRE_EQ(img2.getVal(1), 2.0f);
  REQUIRE_EQ(img2.getVal(img2.getSize()-1), float((200*500-1)*2));
}

static void Image_xx_div_scalar_xx_1(TestSuite& suite)
{
  Image<float> img = generateImage<float>(200,500,Counter(0));

  REQUIRE_EQ(mean(img), 49999.5f);

  Image<float> img2 = img / 2.0f;

  REQUIRE_EQ(mean(img2), 24999.75f);
  REQUIRE_EQ(img2.getVal(1), 0.5f);
  REQUIRE_EQ(img2.getVal(img2.getSize()-1), float((200*500-1)/2.0));
}

static void Image_xx_lshift_scalar_xx_1(TestSuite& suite)
{
  byte barray[4] = { 0, 52, 107, 255 };
  const Image<byte> b1(barray, 4, 1);

  const Image<byte> b = (b1 << 2);

  REQUIRE_EQ(b.getVal(0), 0);
  REQUIRE_EQ(b.getVal(1), 208);
  REQUIRE_EQ(b.getVal(2), 172);
  REQUIRE_EQ(b.getVal(3), 252);

  int iarray[4] = { -37, -1, 297, 65535 };
  const Image<int> i1(iarray, 4, 1);

  const Image<int> i = (i1 << 2);

  REQUIRE_EQ(i.getVal(0), -148);
  REQUIRE_EQ(i.getVal(1), -4);
  REQUIRE_EQ(i.getVal(2), 1188);
  REQUIRE_EQ(i.getVal(3), 262140);
}

static void Image_xx_rshift_scalar_xx_1(TestSuite& suite)
{
  byte barray[4] = { 0, 52, 107, 255 };
  const Image<byte> b1(barray, 4, 1);

  const Image<byte> b = (b1 >> 2);

  REQUIRE_EQ(b.getVal(0), 0);
  REQUIRE_EQ(b.getVal(1), 13);
  REQUIRE_EQ(b.getVal(2), 26);
  REQUIRE_EQ(b.getVal(3), 63);

  int iarray[4] = { -37, -1, 297, 65535 };
  const Image<int> i1(iarray, 4, 1);

  const Image<int> i = (i1 >> 2);

  REQUIRE_EQ(i.getVal(0), -10);
  REQUIRE_EQ(i.getVal(1), -1);
  REQUIRE_EQ(i.getVal(2), 74);
  REQUIRE_EQ(i.getVal(3), 16383);
}

static void Image_xx_plus_array_xx_1(TestSuite& suite)
{
  Image<float> img1 = generateImage<float>(100, 100, Counter(0));

  Image<float> img2 = generateImage<float>(100, 100, Counter(1));

  Image<float> img3 = img1 + img2;

  REQUIRE_EQ(mean(img3), 10000.0f);
  REQUIRE_EQ(img3.getVal(0), 1.0f);
  REQUIRE_EQ(img3.getVal(img3.getSize()-1), float(100*100 - 1 + 100*100));
}

static void Image_xx_minus_array_xx_1(TestSuite& suite)
{
  Image<float> img1 = generateImage<float>(100, 100, Counter(1));

  Image<float> img2 = generateImage<float>(100, 100, Counter(0));

  Image<float> img3 = img1 - img2;

  REQUIRE_EQ(mean(img3), 1.0f);
  REQUIRE_EQ(img3.getVal(0), 1.0f);
  REQUIRE_EQ(img3.getVal(img3.getSize()-1), 1.0f);
}

static void Image_xx_minus_array_xx_2(TestSuite& suite)
{
  const byte a1data[] = { 0, 1, 2, 3 };
  const byte a2data[] = { 3, 2, 1, 0 };

  // make sure that the promotions happen as we expect: the byte
  // values should get converted to signed int as a part of the
  // subtraction operation (otherwise 0u-3u would be undefined)
  const int diffdata[] = { -3, -1, 1, 3 };

  Image<byte> a1(&a1data[0], 4, 1);
  Image<byte> a2(&a2data[0], 4, 1);
  Image<int> diff(&diffdata[0], 4, 1);

  REQUIRE_EQ(a1 - a2, diff);
}

static void Image_xx_mul_array_xx_1(TestSuite& suite)
{
  Image<float> img1 = generateImage<float>(100, 100, Counter(0));

  Image<float> img2(100, 100, NO_INIT); img2.clear(2.0f);

  Image<float> img3 = img1 * img2;

  REQUIRE_EQ(mean(img3), 9999.0f);
  REQUIRE_EQ(img3.getVal(1), 2.0f);
  REQUIRE_EQ(img3.getVal(img3.getSize()-1), float((100*100-1)*2));
}

static void Image_xx_div_array_xx_1(TestSuite& suite)
{
  Image<float> img1 = generateImage<float>(100, 100, Counter(0));

  Image<float> img2(100, 100, NO_INIT); img2.clear(2.0f);

  Image<float> img3 = img1 / img2;

  REQUIRE_EQ(mean(img3), 2499.75);
  REQUIRE_EQ(img3.getVal(1), 0.5f);
  REQUIRE_EQ(img3.getVal(img3.getSize()-1), 4999.5f);
}

static void Image_xx_row_ops_xx_1(TestSuite& suite)
{
  const float Mdat[]  = { 2.0f, 4.0f, 6.0f, 8.0f,
                          6.0f, 12.0f, 18.0f, 24.0f };

  const Image<float> M(&Mdat[0], 4, 2);

  {
    const float dat[] = { 4.0f, 8.0f, 12.0f, 16.0f };
    REQUIRE_EQFP(meanRow(M), Image<float>(&dat[0], 4, 1), 1e-5);
  }

  const float vdat[] = { 1.0f, 2.0f, 3.0f, 4.0f };

  const Image<float> v(&vdat[0], 4, 1);

  {
    const float dat[] = { 3.0f, 6.0f, 9.0f, 12.0f,
                          7.0f, 14.0f, 21.0f, 28.0f };
    REQUIRE_EQFP(addRow(M, v), Image<float>(&dat[0], 4, 2), 1e-5);
  }

  {
    const float dat[] = { 1.0f, 2.0f, 3.0f, 4.0f,
                          5.0f, 10.0f, 15.0f, 20.0f };
    REQUIRE_EQFP(subtractRow(M, v), Image<float>(&dat[0], 4, 2), 1e-5);
  }

  {
    const float dat[] = { 2.0f, 8.0f, 18.0f, 32.0f,
                          6.0f, 24.0f, 54.0f, 96.0f };
    REQUIRE_EQFP(multiplyRow(M, v), Image<float>(&dat[0], 4, 2), 1e-5);
  }

  {
    const float dat[] = { 2.0f, 2.0f, 2.0f, 2.0f,
                          6.0f, 6.0f, 6.0f, 6.0f };
    REQUIRE_EQFP(divideRow(M, v), Image<float>(&dat[0], 4, 2), 1e-5);
  }

}

//---------------------------------------------------------------------
//
// Drawing functions
//
//---------------------------------------------------------------------

static void Image_xx_emptyArea_xx_1(TestSuite& suite)
{
  Image<byte> img(100, 100, NO_INIT);

  // fill with [0..2]
  Image<byte>::iterator itr = img.beginw(), stop = img.endw();
  int val = 0;
  while (itr != stop) *itr++ = (val++) % 3;

  REQUIRE_EQ(emptyArea(img), 3334);
}

//---------------------------------------------------------------------
//
// MathOps
//
//---------------------------------------------------------------------

static void Image_xx_mean_xx_1(TestSuite& suite)
{
  Image<byte> img = generateImage<byte>(10, 10, Counter(0));

  REQUIRE_EQ(mean(img), 49.5);
}

static void Image_xx_sum_xx_1(TestSuite& suite)
{
  Image<byte> img = generateImage<byte>(10, 10, Counter(0));

  REQUIRE_EQ(sum(img), 4950.0);
}

static void Image_xx_sum_xx_2(TestSuite& suite)
{
  // Empty image
  Image<byte> img;

  REQUIRE_EQ(sum(img), 0.0);
}

static void Image_xx_rangeOf_xx_1(TestSuite& suite)
{
  Image<byte> img(1,1, NO_INIT); img.setVal(0, 3);

  Range<byte> rng = rangeOf(img);
  REQUIRE_EQ(rng.min(), 3);
  REQUIRE_EQ(rng.max(), 3);
}

static void Image_xx_rangeOf_xx_2(TestSuite& suite)
{
  float data[] = { 1.f, -32.f, 14.f, -17.f, 21.f, -9.f };

  Image<float> img(data, 6, 1);

  Range<float> rng = rangeOf(img);
  REQUIRE_EQ(rng.min(), -32.f);
  REQUIRE_EQ(rng.max(), 21.f);
}

static void Image_xx_remapRange_xx_1(TestSuite& suite)
{
  float data[] = { 0.f, 1.f, 2.f, 3.f, 4.f };

  Image<float> img(data, 5, 1);

  Image<float> scaled = remapRange(img,
                                   Range<float>(0.f, 10.f),
                                   Range<float>(1.f, 2.f));

  float expected_data[] = { 1.0f, 1.1f, 1.2f, 1.3f, 1.4f };

  Image<float> expected(expected_data, 5, 1);

  REQUIRE_EQ(scaled, expected);
}

static void Image_xx_squared_xx_1(TestSuite& suite)
{
  Image<float> arr = generateImage<float>(1,11,Counter(0));

  float expected_data[] = {
    0.f, 1.f, 4.f, 9.f, 16.f, 25.f, 36.f, 49.f, 64.f, 81.f, 100.f };

  Image<float> expected(expected_data, 1, 11);

  REQUIRE_EQ(squared(arr), expected);
}

static void Image_xx_toPower_xx_1(TestSuite& suite)
{
  Image<float> arr = generateImage<float>(1,11,Counter(0));

  float expected_data[] = {
    0.f, 1.f, 8.f, 27.f, 64.f, 125.f, 216.f, 343.f, 512.f, 729.f, 1000.f };

  Image<float> expected(expected_data, 1, 11);

  REQUIRE_EQ(toPower(arr, 3.0), expected);
}

static void Image_xx_toPower_xx_2(TestSuite& suite)
{
  Image<byte> arr = generateImage<float>(1,11,Counter(0));

  byte expected_data[] = { 0, 1, 2, 5, 8, 11, 14, 18, 22, 27, 31 };

  Image<byte> expected(expected_data, 1, 11);
  Image<byte> result = toPower(arr, 1.5);

  REQUIRE_EQ(result, expected);
}

static void Image_xx_quadEnergy_xx_1(TestSuite& suite)
{
  byte a_data[4] = { 3, 5, 7, 9 };
  byte b_data[4] = { 2, 4, 6, 8 };

  Image<byte> a(a_data, 2, 2);
  Image<byte> b(b_data, 2, 2);

  Image<byte> res = quadEnergy(a, b);

  REQUIRE_EQ((int)res.getVal(0), 3); // 3.60555
  REQUIRE_EQ((int)res.getVal(1), 6); // 6.40312
  REQUIRE_EQ((int)res.getVal(2), 9); // 9.21954
  REQUIRE_EQ((int)res.getVal(3), 12); // 12.04159
}

static void Image_xx_overlay_xx_1(TestSuite& suite)
{
  float top_data[4] =    { 1.0, 2.0, 3.0, 4.0 };
  float bottom_data[4] = { 5.0, 10.0, 15.0, 20.0 };

  Image<float> top(top_data, 2, 2);
  Image<float> bottom(bottom_data, 2, 2);

  Image<float> res = overlay(top, bottom, 75.0);

  REQUIRE_EQ(res.getVal(0), 4.0f);
  REQUIRE_EQ(res.getVal(1), 8.0f);
  REQUIRE_EQ(res.getVal(2), 12.0f);
  REQUIRE_EQ(res.getVal(3), 16.0f);
}

static void Image_xx_exp_xx_1(TestSuite& suite)
{
  const double d[4] = { -1.0, 0.0, 1.0, 2.5 };
  const double e[4] = { 0.3679, 1.0, 2.7183, 12.1825 };

  Image<double> in(&d[0], 4, 1);
  Image<double> out(&e[0], 4, 1);

  REQUIRE_EQFP(exp(in), out, 1e-3);
}

static void Image_xx_log_xx_1(TestSuite& suite)
{
  const double d[4] = { 0.1, 1.0, 2.7183, 10000.0 };
  const double l[4] = { -2.3026, 0.0, 1.0, 9.2103 };

  Image<double> in(&d[0], 4, 1);
  Image<double> out(&l[0], 4, 1);

  REQUIRE_EQFP(log(in), out, 1e-3);
}

static void Image_xx_log10_xx_1(TestSuite& suite)
{
  const double d[4] = { 0.1, 1.0, 2.7183, 10000.0 };
  const double l[4] = { -1.0, 0.0, 0.4343, 4 };

  Image<double> in(&d[0], 4, 1);
  Image<double> out(&l[0], 4, 1);

  REQUIRE_EQFP(log10(in), out, 1e-3);
}

static void Image_xx_getMaskedMinMax_xx_1(TestSuite& suite)
{
  const byte d[4] = { 1, 2, 3, 4 };
  const byte m[4] = { 0, 1, 1, 0 };

  Image<byte> img(&d[0], 4, 1);
  Image<byte> msk(&m[0], 4, 1);

  byte mini, maxi, mino, maxo;
  getMaskedMinMax(img, msk, mini, maxi, mino, maxo);

  REQUIRE_EQ(mini, byte(2));
  REQUIRE_EQ(maxi, byte(3));
  REQUIRE_EQ(mino, byte(1));
  REQUIRE_EQ(maxo, byte(4));
}

static void Image_xx_getMaskedMinMaxAvg_xx_1(TestSuite& suite)
{
  const float d[4] = { 1.0F, 2.0F, 3.0F, 4.0F };
  const byte m[4] = { 0, 1, 1, 0 };

  Image<float> img(&d[0], 4, 1);
  Image<byte> msk(&m[0], 4, 1);

  float mini, maxi, avg;
  getMaskedMinMaxAvg(img, msk, mini, maxi, avg);

  REQUIRE_EQ(mini, 2.0F);
  REQUIRE_EQ(maxi, 3.0F);
  REQUIRE_EQ(avg, 2.5F);
}

//---------------------------------------------------------------------
//
// MatrixOps
//
//---------------------------------------------------------------------

static void Image_xx_vmMult_xx_1(TestSuite& suite)
{
  // test data generated in matlab (same as from
  // Image_xx_matrixMult_xx_3())
  const float a[] =
    {
      9.5012929e-01,  4.8598247e-01,  4.5646767e-01,  4.4470336e-01,  9.2181297e-01,  4.0570621e-01,  4.1027021e-01,
      2.3113851e-01,  8.9129897e-01,  1.8503643e-02,  6.1543235e-01,  7.3820725e-01,  9.3546970e-01,  8.9364953e-01,
      6.0684258e-01,  7.6209683e-01,  8.2140716e-01,  7.9193704e-01,  1.7626614e-01,  9.1690444e-01,  5.7891305e-02
    };

  const float b_data[] =
    {
      3.5286813e-01,  2.7218792e-01,  4.1864947e-01,  6.8127716e-01,
      8.1316650e-01,  1.9881427e-01,  8.4622142e-01,  3.7948102e-01,
      9.8613007e-03,  1.5273927e-02,  5.2515250e-01,  8.3179602e-01,
      1.3889088e-01,  7.4678568e-01,  2.0264736e-01,  5.0281288e-01,
      2.0276522e-01,  4.4509643e-01,  6.7213747e-01,  7.0947139e-01,
      1.9872174e-01,  9.3181458e-01,  8.3811845e-01,  4.2889237e-01,
      6.0379248e-01,  4.6599434e-01,  1.9639514e-02,  3.0461737e-01
    };

  const float c[] =
    {
      1.3119739e+00,  1.6738263e+00,  2.1065254e+00,  2.3879927e+00,
      1.7671561e+00,  2.3166881e+00,  2.2831973e+00,  2.0177129e+00,
      1.2048438e+00,  1.8804617e+00,  2.3788915e+00,  2.3200124e+00
    };

  const Image<float> b(b_data, 4, 7);

  // this should use dgemv from lapack if lapack is available

  typedef Image<float> VEC;

  REQUIRE_EQFP(vmMult(VEC(a+7*0, 7, 1), b), VEC(c+4*0, 4, 1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(a+7*1, 7, 1), b), VEC(c+4*1, 4, 1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(a+7*2, 7, 1), b), VEC(c+4*2, 4, 1), 1e-5);

  // and now repeat using matrixMult(), since vmMult() is just a
  // special case of matrixMult()

  REQUIRE_EQFP(matrixMult(VEC(a+7*0, 7, 1), b), VEC(c+4*0, 4, 1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(a+7*1, 7, 1), b), VEC(c+4*1, 4, 1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(a+7*2, 7, 1), b), VEC(c+4*2, 4, 1), 1e-5);
}

static void Image_xx_vmMult_xx_2(TestSuite& suite)
{
  // test data generated in matlab (same as first row from
  // Image_xx_matrixMult_xx_4())
  const double d[] =
    {
      1.8965375e-01,  8.1797434e-01,  6.2131013e-01,  2.9872301e-01,  6.4052650e-01,  7.6795039e-01,  9.3338011e-01,  1.2862575e-02,  6.9266939e-01,  2.7310247e-01,  9.9429549e-01,  2.3788031e-01,
      1.9343116e-01,  6.6022756e-01,  7.9482108e-01,  6.6144258e-01,  2.0906940e-01,  9.7084494e-01,  6.8333232e-01,  3.8396729e-01,  8.4079061e-02,  2.5476930e-01,  4.3979086e-01,  6.4583125e-01,
      6.8222322e-01,  3.4197062e-01,  9.5684345e-01,  2.8440859e-01,  3.7981837e-01,  9.9008259e-01,  2.1255986e-01,  6.8311597e-01,  4.5435515e-01,  8.6560348e-01,  3.4004795e-01,  9.6688742e-01,
      3.0276440e-01,  2.8972590e-01,  5.2259035e-01,  4.6922429e-01,  7.8332865e-01,  7.8886169e-01,  8.3923824e-01,  9.2842462e-02,  4.4182830e-01,  2.3235037e-01,  3.1421731e-01,  6.6493121e-01,
      5.4167385e-01,  3.4119357e-01,  8.8014221e-01,  6.4781123e-02,  6.8084575e-01,  4.3865853e-01,  6.2878460e-01,  3.5338324e-02,  3.5325046e-01,  8.0487174e-01,  3.6507839e-01,  8.7038103e-01,
      1.5087298e-01,  5.3407902e-01,  1.7295614e-01,  9.8833494e-01,  4.6109513e-01,  4.9831130e-01,  1.3377275e-01,  6.1239548e-01,  1.5360636e-01,  9.0839754e-01,  3.9323955e-01,  9.9273048e-03,
      6.9789848e-01,  7.2711322e-01,  9.7974690e-01,  5.8279168e-01,  5.6782871e-01,  2.1396333e-01,  2.0713273e-01,  6.0854036e-01,  6.7564465e-01,  2.3189432e-01,  5.9152520e-01,  1.3700989e-01,
      3.7837300e-01,  3.0929016e-01,  2.7144726e-01,  4.2349626e-01,  7.9421065e-01,  6.4349229e-01,  6.0719894e-01,  1.5759818e-02,  6.9921333e-01,  2.3931256e-01,  1.1974662e-01,  8.1875583e-01,
      8.6001160e-01,  8.3849604e-01,  2.5232935e-01,  5.1551175e-01,  5.9182593e-02,  3.2003558e-01,  6.2988785e-01,  1.6354934e-02,  7.2750913e-01,  4.9754484e-02,  3.8128797e-02,  4.3016605e-01,
      8.5365513e-01,  5.6807246e-01,  8.7574190e-01,  3.3395148e-01,  6.0286909e-01,  9.6009860e-01,  3.7047683e-01,  1.9007459e-01,  4.7838438e-01,  7.8384075e-02,  4.5859795e-01,  8.9032172e-01,
      5.9356291e-01,  3.7041356e-01,  7.3730599e-01,  4.3290660e-01,  5.0268804e-02,  7.2663177e-01,  5.7514778e-01,  5.8691847e-01,  5.5484199e-01,  6.4081541e-01,  8.6986735e-01,  7.3490821e-01,
      4.9655245e-01,  7.0273991e-01,  1.3651874e-01,  2.2594987e-01,  4.1537486e-01,  4.1195321e-01,  4.5142483e-01,  5.7581090e-02,  1.2104711e-01,  1.9088657e-01,  9.3423652e-01,  6.8732359e-01,
      8.9976918e-01,  5.4657115e-01,  1.1756687e-02,  5.7980687e-01,  3.0499868e-01,  7.4456578e-01,  4.3895325e-02,  3.6756804e-01,  4.5075394e-01,  8.4386950e-01,  2.6444917e-01,  3.4611197e-01,
      8.2162916e-01,  4.4488020e-01,  8.9389797e-01,  7.6036501e-01,  8.7436717e-01,  2.6794725e-01,  2.7185123e-02,  6.3145116e-01,  7.1588295e-01,  1.7390025e-01,  1.6030034e-01,  1.6603474e-01,
      6.4491038e-01,  6.9456724e-01,  1.9913807e-01,  5.2982312e-01,  1.5009499e-02,  4.3992431e-01,  3.1268505e-01,  7.1763442e-01,  8.9284161e-01,  1.7079281e-01,  8.7285526e-01,  1.5561258e-01
    };

  const double e_data[] =
    {
      1.9111631e-01,  4.9162489e-02,  7.5366963e-01,  5.4878213e-01,  4.2525316e-01,  4.8496372e-01,  5.4851281e-01,  8.3881007e-02,  2.9198392e-01,  7.5455138e-01,  4.7404145e-01,  6.1011358e-01,  7.6943640e-01,  7.1168466e-02,  4.0180434e-01,  6.6405187e-01,  1.6627012e-01,  7.5947939e-01,
      4.2245153e-01,  6.9318045e-01,  7.9387177e-01,  9.3158335e-01,  5.9466337e-01,  1.1461282e-01,  2.6176957e-01,  9.4546279e-01,  8.5796351e-01,  7.9112320e-01,  9.0898935e-01,  7.0149260e-01,  4.4416162e-01,  3.1428029e-01,  3.0768794e-01,  7.2406171e-01,  3.9390601e-01,  9.4975928e-01,
      8.5597571e-01,  6.5010641e-01,  9.1995721e-01,  3.3519743e-01,  5.6573857e-01,  6.6485557e-01,  5.9734485e-01,  9.1594246e-01,  3.3575514e-01,  8.1495207e-01,  5.9624714e-01,  9.2196203e-02,  6.2062012e-01,  6.0838366e-01,  4.1156796e-01,  2.8163360e-01,  5.2075748e-01,  5.5793851e-01,
      4.9024999e-01,  9.8298778e-01,  8.4472150e-01,  6.5553106e-01,  7.1654240e-01,  3.6537389e-01,  4.9277997e-02,  6.0198742e-01,  6.8020385e-01,  6.7000386e-01,  3.2895530e-01,  4.2488914e-01,  9.5168928e-01,  1.7502018e-01,  2.8593923e-01,  2.6181868e-01,  7.1812397e-01,  1.4233016e-02,
      8.1593477e-01,  5.5267324e-01,  3.6775288e-01,  3.9190421e-01,  5.1131145e-01,  1.4004446e-01,  5.7105749e-01,  2.5356058e-01,  5.3444421e-02,  2.0087641e-01,  4.7819443e-01,  3.7557666e-01,  6.4000966e-01,  6.2102743e-01,  3.9412761e-01,  7.0847140e-01,  5.6918952e-01,  5.9617708e-01,
      4.6076983e-01,  4.0007352e-01,  6.2080133e-01,  6.2731479e-01,  7.7640121e-01,  5.6677280e-01,  7.0085723e-01,  8.7345081e-01,  3.5665554e-01,  2.7308816e-01,  5.9717078e-01,  1.6615408e-01,  2.4732763e-01,  2.4595993e-01,  5.0301449e-01,  7.8385902e-01,  4.6080617e-01,  8.1620571e-01,
      4.5735438e-01,  1.9878852e-01,  7.3127726e-01,  6.9908014e-01,  4.8934548e-01,  8.2300831e-01,  9.6228826e-01,  5.1340071e-01,  4.9830460e-01,  6.2623464e-01,  1.6144875e-01,  8.3315146e-01,  3.5270199e-01,  5.8735822e-01,  7.2197985e-01,  9.8615781e-01,  4.4530705e-01,  9.7709235e-01,
      4.5068888e-01,  6.2520102e-01,  1.9389318e-01,  3.9718395e-01,  1.8590445e-01,  6.7394863e-01,  7.5051823e-01,  7.3265065e-01,  4.3444054e-01,  5.3685169e-01,  8.2947425e-01,  8.3863970e-01,  1.8786048e-01,  5.0605345e-01,  3.0620855e-01,  4.7334271e-01,  8.7744606e-02,  2.2190808e-01,
      4.1221906e-01,  7.3336280e-01,  9.0481233e-01,  4.1362890e-01,  7.0063541e-01,  9.9944730e-01,  7.3999305e-01,  4.2222659e-01,  5.6245842e-01,  5.9504051e-02,  9.5612241e-01,  4.5161403e-01,  4.9064436e-01,  4.6477892e-01,  1.1216371e-01,  9.0281883e-01,  4.4348322e-01,  7.0368367e-01,
      9.0160982e-01,  3.7588548e-01,  5.6920575e-01,  6.5521295e-01,  9.8270880e-01,  9.6163641e-01,  4.3187339e-01,  9.6137000e-01,  6.1662113e-01,  8.8961759e-02,  5.9554800e-01,  9.5660138e-01,  4.0927433e-01,  5.4141893e-01,  4.4328996e-01,  4.5105876e-01,  3.6629985e-01,  5.2206092e-01,
      5.5839392e-03,  9.8764629e-03,  6.3178993e-01,  8.3758510e-01,  8.0663775e-01,  5.8862165e-02,  6.3426596e-01,  7.2059239e-02,  1.1333998e-01,  2.7130817e-01,  2.8748213e-02,  1.4715324e-01,  4.6352558e-01,  9.4232657e-01,  4.6676255e-01,  8.0451681e-01,  3.0253382e-01,  9.3289706e-01,
      2.9740568e-01,  4.1985781e-01,  2.3441296e-01,  3.7160803e-01,  7.0356766e-01,  3.6031117e-01,  8.0302634e-01,  5.5340797e-01,  8.9825174e-01,  4.0907232e-01,  8.1211782e-01,  8.6993293e-01,  6.1094355e-01,  3.4175909e-01,  1.4668875e-02,  8.2886448e-01,  8.5184470e-01,  7.1335444e-01
    };

  const double f[] =
    {
      2.9772969e+00,  2.8490411e+00,  4.4797086e+00,  4.0472153e+00,  4.2385898e+00,  3.1089057e+00,  3.9680287e+00,  3.6182944e+00,  2.8321589e+00,  2.8590936e+00,  3.2751040e+00,  2.9576420e+00,  3.1684692e+00,  3.3303050e+00,  2.6519174e+00,  4.6993431e+00,  2.9377072e+00,  4.9226116e+00,
      2.8828908e+00,  2.9468518e+00,  3.8639357e+00,  3.5524610e+00,  3.7833252e+00,  2.9278979e+00,  3.5016651e+00,  3.9692976e+00,  3.0388124e+00,  3.1115468e+00,  3.2611307e+00,  2.9748704e+00,  2.9958628e+00,  2.5970407e+00,  2.3159149e+00,  3.8482168e+00,  2.8993006e+00,  3.9599320e+00,
      3.6616983e+00,  3.3154566e+00,  4.2936709e+00,  3.7892124e+00,  4.5560450e+00,  4.0146890e+00,  4.3795135e+00,  4.5824720e+00,  3.4797825e+00,  3.1939967e+00,  4.4170725e+00,  3.8467199e+00,  3.5581161e+00,  3.0685804e+00,  2.4724131e+00,  4.4983547e+00,  3.2227638e+00,  4.5532449e+00,
      2.8770496e+00,  2.6837391e+00,  3.6312439e+00,  3.1895117e+00,  3.6330218e+00,  2.9321011e+00,  3.6152253e+00,  3.2475376e+00,  2.6380256e+00,  2.5503276e+00,  3.0408538e+00,  2.8843431e+00,  3.0061753e+00,  2.5875682e+00,  2.2089052e+00,  4.0333769e+00,  2.9073422e+00,  3.9545291e+00,
      3.2261417e+00,  2.5285158e+00,  3.7453372e+00,  3.2509103e+00,  3.9258666e+00,  3.2580602e+00,  3.7827043e+00,  3.5485848e+00,  2.8301733e+00,  2.6563408e+00,  3.3658067e+00,  3.3009714e+00,  3.1525772e+00,  2.8519437e+00,  2.2367581e+00,  4.0317027e+00,  2.9054567e+00,  4.2438312e+00,
      2.7175339e+00,  2.7874049e+00,  3.1339862e+00,  3.1080394e+00,  3.3142479e+00,  2.5340986e+00,  2.3388486e+00,  3.3125004e+00,  2.4677117e+00,  2.1814506e+00,  2.7400965e+00,  2.7884337e+00,  2.6173022e+00,  2.1913543e+00,  1.8988358e+00,  2.8064566e+00,  2.1689501e+00,  2.6368829e+00,
      3.0274399e+00,  3.2152336e+00,  4.2571178e+00,  3.4918421e+00,  3.6468027e+00,  3.0499596e+00,  3.4026987e+00,  3.5039376e+00,  2.7403580e+00,  3.1970256e+00,  3.6175464e+00,  2.9387411e+00,  3.4106452e+00,  2.8549384e+00,  2.1980184e+00,  3.7847895e+00,  2.5788601e+00,  3.8722196e+00,
      2.6204453e+00,  2.6003722e+00,  3.3132585e+00,  2.8606140e+00,  3.4053166e+00,  2.7714573e+00,  3.3263859e+00,  2.9014815e+00,  2.6262746e+00,  2.1894414e+00,  3.1162571e+00,  2.8516770e+00,  2.8813169e+00,  2.2267207e+00,  1.8123421e+00,  3.8500517e+00,  2.7820085e+00,  3.6468303e+00,
      1.9514074e+00,  2.3236933e+00,  3.4771615e+00,  2.8717049e+00,  2.8585138e+00,  2.5205856e+00,  2.6733132e+00,  2.6320762e+00,  2.6747614e+00,  2.5998615e+00,  2.9002340e+00,  2.7250883e+00,  2.6814182e+00,  1.6101946e+00,  1.6266525e+00,  3.3724282e+00,  2.1568087e+00,  3.3935804e+00,
      3.0410726e+00,  3.0017518e+00,  4.2835098e+00,  3.6473755e+00,  4.1143910e+00,  3.1462353e+00,  4.0665241e+00,  3.7358847e+00,  3.0687749e+00,  3.2723858e+00,  3.8724654e+00,  3.1290047e+00,  3.5834522e+00,  2.7560633e+00,  2.3354593e+00,  4.5214364e+00,  3.1668170e+00,  4.6689449e+00,
      3.0465649e+00,  2.9554987e+00,  4.3779882e+00,  3.9635060e+00,  4.4176241e+00,  3.7532293e+00,  4.2454293e+00,  4.0283616e+00,  3.3024941e+00,  3.1422887e+00,  3.7717235e+00,  3.6158122e+00,  3.4025077e+00,  3.2034486e+00,  2.5230896e+00,  4.5324723e+00,  2.9804889e+00,  4.5961247e+00,
      1.8121819e+00,  1.8008390e+00,  2.9678931e+00,  3.0937592e+00,  3.1415898e+00,  1.8038475e+00,  2.8690504e+00,  2.3885329e+00,  2.2755915e+00,  2.2616361e+00,  2.4096396e+00,  2.5260285e+00,  2.5221974e+00,  2.3072328e+00,  1.7953609e+00,  3.5424877e+00,  2.2167963e+00,  3.7083622e+00,
      2.5259039e+00,  2.5013631e+00,  3.4267798e+00,  3.2397377e+00,  3.5555266e+00,  2.8694980e+00,  2.8299854e+00,  3.1837414e+00,  2.7081506e+00,  2.3140245e+00,  3.2457792e+00,  3.1137285e+00,  2.8586286e+00,  1.9626517e+00,  1.8924617e+00,  3.4525684e+00,  2.2397491e+00,  3.3826311e+00,
      3.1189663e+00,  3.3296225e+00,  3.9541879e+00,  3.0499924e+00,  3.3688889e+00,  2.9958105e+00,  3.1656017e+00,  3.2712432e+00,  2.5462913e+00,  2.9841602e+00,  3.6107629e+00,  2.8016577e+00,  3.4604375e+00,  2.4543084e+00,  1.9793549e+00,  3.4650897e+00,  2.5831721e+00,  3.3252931e+00,
      2.1014542e+00,  2.6514890e+00,  3.8076366e+00,  3.4706349e+00,  3.4221974e+00,  2.8748121e+00,  3.2500480e+00,  2.9766863e+00,  2.6829611e+00,  2.6263997e+00,  3.2525714e+00,  2.8956778e+00,  2.8036752e+00,  2.5255539e+00,  1.9645498e+00,  3.8438016e+00,  2.1336250e+00,  3.7436370e+00
    };

  const Image<double> e(e_data, 18, 12);

  typedef Image<double> VEC;

  // this should use dgemv from lapack if lapack is available

  REQUIRE_EQFP(vmMult(VEC(d+12* 0, 12,1), e), VEC(f+18* 0, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12* 1, 12,1), e), VEC(f+18* 1, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12* 2, 12,1), e), VEC(f+18* 2, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12* 3, 12,1), e), VEC(f+18* 3, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12* 4, 12,1), e), VEC(f+18* 4, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12* 5, 12,1), e), VEC(f+18* 5, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12* 6, 12,1), e), VEC(f+18* 6, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12* 7, 12,1), e), VEC(f+18* 7, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12* 8, 12,1), e), VEC(f+18* 8, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12* 9, 12,1), e), VEC(f+18* 9, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12*10, 12,1), e), VEC(f+18*10, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12*11, 12,1), e), VEC(f+18*11, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12*12, 12,1), e), VEC(f+18*12, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12*13, 12,1), e), VEC(f+18*13, 18,1), 1e-5);
  REQUIRE_EQFP(vmMult(VEC(d+12*14, 12,1), e), VEC(f+18*14, 18,1), 1e-5);

  // and now repeat using matrixMult(), since vmMult() is just a
  // special case of matrixMult()

  REQUIRE_EQFP(matrixMult(VEC(d+12* 0, 12,1), e), VEC(f+18* 0, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12* 1, 12,1), e), VEC(f+18* 1, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12* 2, 12,1), e), VEC(f+18* 2, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12* 3, 12,1), e), VEC(f+18* 3, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12* 4, 12,1), e), VEC(f+18* 4, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12* 5, 12,1), e), VEC(f+18* 5, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12* 6, 12,1), e), VEC(f+18* 6, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12* 7, 12,1), e), VEC(f+18* 7, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12* 8, 12,1), e), VEC(f+18* 8, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12* 9, 12,1), e), VEC(f+18* 9, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12*10, 12,1), e), VEC(f+18*10, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12*11, 12,1), e), VEC(f+18*11, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12*12, 12,1), e), VEC(f+18*12, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12*13, 12,1), e), VEC(f+18*13, 18,1), 1e-5);
  REQUIRE_EQFP(matrixMult(VEC(d+12*14, 12,1), e), VEC(f+18*14, 18,1), 1e-5);
}

static void Image_xx_matrixMult_xx_1(TestSuite& suite)
{
  Image<float> a = generateImage<float>(5, 3, Counter(0));
  Image<float> b = generateImage<float>(2, 5, Counter(0));

  float expected_data[] = {
     60.0F,    70.0F,
    160.0F,   195.0F,
    260.0F,   320.0F };

  Image<float> expected(expected_data, 2, 3);

  // this should use sgemm from lapack if lapack is available

  REQUIRE_EQ(matrixMult(a, b), expected);
}

static void Image_xx_matrixMult_xx_2(TestSuite& suite)
{
  Image<byte> a = generateImage<byte>(5, 3, Counter(0));
  Image<byte> b = generateImage<byte>(2, 5, Counter(0));

  byte expected_data[] = {
     60,    70,
    160,   195,
    255,   255 };  // last two values clamped

  Image<byte> expected(expected_data, 2, 3);

  // this should NOT use anything from lapack; we use lapack only for
  // float and double

  Image<byte> result = matrixMult(a, b); // will clamp

  REQUIRE_EQ(result, expected);
}

static void Image_xx_matrixMult_xx_3(TestSuite& suite)
{
  // test data generated in matlab
  const float a_data[] =
    {
      9.5012929e-01,  4.8598247e-01,  4.5646767e-01,  4.4470336e-01,  9.2181297e-01,  4.0570621e-01,  4.1027021e-01,
      2.3113851e-01,  8.9129897e-01,  1.8503643e-02,  6.1543235e-01,  7.3820725e-01,  9.3546970e-01,  8.9364953e-01,
      6.0684258e-01,  7.6209683e-01,  8.2140716e-01,  7.9193704e-01,  1.7626614e-01,  9.1690444e-01,  5.7891305e-02
    };

  const float b_data[] =
    {
      3.5286813e-01,  2.7218792e-01,  4.1864947e-01,  6.8127716e-01,
      8.1316650e-01,  1.9881427e-01,  8.4622142e-01,  3.7948102e-01,
      9.8613007e-03,  1.5273927e-02,  5.2515250e-01,  8.3179602e-01,
      1.3889088e-01,  7.4678568e-01,  2.0264736e-01,  5.0281288e-01,
      2.0276522e-01,  4.4509643e-01,  6.7213747e-01,  7.0947139e-01,
      1.9872174e-01,  9.3181458e-01,  8.3811845e-01,  4.2889237e-01,
      6.0379248e-01,  4.6599434e-01,  1.9639514e-02,  3.0461737e-01
    };

  const float c_data[] =
    {
      1.3119739e+00,  1.6738263e+00,  2.1065254e+00,  2.3879927e+00,
      1.7671561e+00,  2.3166881e+00,  2.2831973e+00,  2.0177129e+00,
      1.2048438e+00,  1.8804617e+00,  2.3788915e+00,  2.3200124e+00
    };

  const Image<float> a(a_data, 7, 3);
  const Image<float> b(b_data, 4, 7);
  const Image<float> c(c_data, 4, 3);

  // this should use dgemm from lapack if lapack is available

  REQUIRE_EQFP(matrixMult(a, b), c, 1e-5);
}

static void Image_xx_matrixMult_xx_4(TestSuite& suite)
{
  // test data generated in matlab
  const double d_data[] =
    {
      1.8965375e-01,  8.1797434e-01,  6.2131013e-01,  2.9872301e-01,  6.4052650e-01,  7.6795039e-01,  9.3338011e-01,  1.2862575e-02,  6.9266939e-01,  2.7310247e-01,  9.9429549e-01,  2.3788031e-01,
      1.9343116e-01,  6.6022756e-01,  7.9482108e-01,  6.6144258e-01,  2.0906940e-01,  9.7084494e-01,  6.8333232e-01,  3.8396729e-01,  8.4079061e-02,  2.5476930e-01,  4.3979086e-01,  6.4583125e-01,
      6.8222322e-01,  3.4197062e-01,  9.5684345e-01,  2.8440859e-01,  3.7981837e-01,  9.9008259e-01,  2.1255986e-01,  6.8311597e-01,  4.5435515e-01,  8.6560348e-01,  3.4004795e-01,  9.6688742e-01,
      3.0276440e-01,  2.8972590e-01,  5.2259035e-01,  4.6922429e-01,  7.8332865e-01,  7.8886169e-01,  8.3923824e-01,  9.2842462e-02,  4.4182830e-01,  2.3235037e-01,  3.1421731e-01,  6.6493121e-01,
      5.4167385e-01,  3.4119357e-01,  8.8014221e-01,  6.4781123e-02,  6.8084575e-01,  4.3865853e-01,  6.2878460e-01,  3.5338324e-02,  3.5325046e-01,  8.0487174e-01,  3.6507839e-01,  8.7038103e-01,
      1.5087298e-01,  5.3407902e-01,  1.7295614e-01,  9.8833494e-01,  4.6109513e-01,  4.9831130e-01,  1.3377275e-01,  6.1239548e-01,  1.5360636e-01,  9.0839754e-01,  3.9323955e-01,  9.9273048e-03,
      6.9789848e-01,  7.2711322e-01,  9.7974690e-01,  5.8279168e-01,  5.6782871e-01,  2.1396333e-01,  2.0713273e-01,  6.0854036e-01,  6.7564465e-01,  2.3189432e-01,  5.9152520e-01,  1.3700989e-01,
      3.7837300e-01,  3.0929016e-01,  2.7144726e-01,  4.2349626e-01,  7.9421065e-01,  6.4349229e-01,  6.0719894e-01,  1.5759818e-02,  6.9921333e-01,  2.3931256e-01,  1.1974662e-01,  8.1875583e-01,
      8.6001160e-01,  8.3849604e-01,  2.5232935e-01,  5.1551175e-01,  5.9182593e-02,  3.2003558e-01,  6.2988785e-01,  1.6354934e-02,  7.2750913e-01,  4.9754484e-02,  3.8128797e-02,  4.3016605e-01,
      8.5365513e-01,  5.6807246e-01,  8.7574190e-01,  3.3395148e-01,  6.0286909e-01,  9.6009860e-01,  3.7047683e-01,  1.9007459e-01,  4.7838438e-01,  7.8384075e-02,  4.5859795e-01,  8.9032172e-01,
      5.9356291e-01,  3.7041356e-01,  7.3730599e-01,  4.3290660e-01,  5.0268804e-02,  7.2663177e-01,  5.7514778e-01,  5.8691847e-01,  5.5484199e-01,  6.4081541e-01,  8.6986735e-01,  7.3490821e-01,
      4.9655245e-01,  7.0273991e-01,  1.3651874e-01,  2.2594987e-01,  4.1537486e-01,  4.1195321e-01,  4.5142483e-01,  5.7581090e-02,  1.2104711e-01,  1.9088657e-01,  9.3423652e-01,  6.8732359e-01,
      8.9976918e-01,  5.4657115e-01,  1.1756687e-02,  5.7980687e-01,  3.0499868e-01,  7.4456578e-01,  4.3895325e-02,  3.6756804e-01,  4.5075394e-01,  8.4386950e-01,  2.6444917e-01,  3.4611197e-01,
      8.2162916e-01,  4.4488020e-01,  8.9389797e-01,  7.6036501e-01,  8.7436717e-01,  2.6794725e-01,  2.7185123e-02,  6.3145116e-01,  7.1588295e-01,  1.7390025e-01,  1.6030034e-01,  1.6603474e-01,
      6.4491038e-01,  6.9456724e-01,  1.9913807e-01,  5.2982312e-01,  1.5009499e-02,  4.3992431e-01,  3.1268505e-01,  7.1763442e-01,  8.9284161e-01,  1.7079281e-01,  8.7285526e-01,  1.5561258e-01
    };

  const double e_data[] =
    {
      1.9111631e-01,  4.9162489e-02,  7.5366963e-01,  5.4878213e-01,  4.2525316e-01,  4.8496372e-01,  5.4851281e-01,  8.3881007e-02,  2.9198392e-01,  7.5455138e-01,  4.7404145e-01,  6.1011358e-01,  7.6943640e-01,  7.1168466e-02,  4.0180434e-01,  6.6405187e-01,  1.6627012e-01,  7.5947939e-01,
      4.2245153e-01,  6.9318045e-01,  7.9387177e-01,  9.3158335e-01,  5.9466337e-01,  1.1461282e-01,  2.6176957e-01,  9.4546279e-01,  8.5796351e-01,  7.9112320e-01,  9.0898935e-01,  7.0149260e-01,  4.4416162e-01,  3.1428029e-01,  3.0768794e-01,  7.2406171e-01,  3.9390601e-01,  9.4975928e-01,
      8.5597571e-01,  6.5010641e-01,  9.1995721e-01,  3.3519743e-01,  5.6573857e-01,  6.6485557e-01,  5.9734485e-01,  9.1594246e-01,  3.3575514e-01,  8.1495207e-01,  5.9624714e-01,  9.2196203e-02,  6.2062012e-01,  6.0838366e-01,  4.1156796e-01,  2.8163360e-01,  5.2075748e-01,  5.5793851e-01,
      4.9024999e-01,  9.8298778e-01,  8.4472150e-01,  6.5553106e-01,  7.1654240e-01,  3.6537389e-01,  4.9277997e-02,  6.0198742e-01,  6.8020385e-01,  6.7000386e-01,  3.2895530e-01,  4.2488914e-01,  9.5168928e-01,  1.7502018e-01,  2.8593923e-01,  2.6181868e-01,  7.1812397e-01,  1.4233016e-02,
      8.1593477e-01,  5.5267324e-01,  3.6775288e-01,  3.9190421e-01,  5.1131145e-01,  1.4004446e-01,  5.7105749e-01,  2.5356058e-01,  5.3444421e-02,  2.0087641e-01,  4.7819443e-01,  3.7557666e-01,  6.4000966e-01,  6.2102743e-01,  3.9412761e-01,  7.0847140e-01,  5.6918952e-01,  5.9617708e-01,
      4.6076983e-01,  4.0007352e-01,  6.2080133e-01,  6.2731479e-01,  7.7640121e-01,  5.6677280e-01,  7.0085723e-01,  8.7345081e-01,  3.5665554e-01,  2.7308816e-01,  5.9717078e-01,  1.6615408e-01,  2.4732763e-01,  2.4595993e-01,  5.0301449e-01,  7.8385902e-01,  4.6080617e-01,  8.1620571e-01,
      4.5735438e-01,  1.9878852e-01,  7.3127726e-01,  6.9908014e-01,  4.8934548e-01,  8.2300831e-01,  9.6228826e-01,  5.1340071e-01,  4.9830460e-01,  6.2623464e-01,  1.6144875e-01,  8.3315146e-01,  3.5270199e-01,  5.8735822e-01,  7.2197985e-01,  9.8615781e-01,  4.4530705e-01,  9.7709235e-01,
      4.5068888e-01,  6.2520102e-01,  1.9389318e-01,  3.9718395e-01,  1.8590445e-01,  6.7394863e-01,  7.5051823e-01,  7.3265065e-01,  4.3444054e-01,  5.3685169e-01,  8.2947425e-01,  8.3863970e-01,  1.8786048e-01,  5.0605345e-01,  3.0620855e-01,  4.7334271e-01,  8.7744606e-02,  2.2190808e-01,
      4.1221906e-01,  7.3336280e-01,  9.0481233e-01,  4.1362890e-01,  7.0063541e-01,  9.9944730e-01,  7.3999305e-01,  4.2222659e-01,  5.6245842e-01,  5.9504051e-02,  9.5612241e-01,  4.5161403e-01,  4.9064436e-01,  4.6477892e-01,  1.1216371e-01,  9.0281883e-01,  4.4348322e-01,  7.0368367e-01,
      9.0160982e-01,  3.7588548e-01,  5.6920575e-01,  6.5521295e-01,  9.8270880e-01,  9.6163641e-01,  4.3187339e-01,  9.6137000e-01,  6.1662113e-01,  8.8961759e-02,  5.9554800e-01,  9.5660138e-01,  4.0927433e-01,  5.4141893e-01,  4.4328996e-01,  4.5105876e-01,  3.6629985e-01,  5.2206092e-01,
      5.5839392e-03,  9.8764629e-03,  6.3178993e-01,  8.3758510e-01,  8.0663775e-01,  5.8862165e-02,  6.3426596e-01,  7.2059239e-02,  1.1333998e-01,  2.7130817e-01,  2.8748213e-02,  1.4715324e-01,  4.6352558e-01,  9.4232657e-01,  4.6676255e-01,  8.0451681e-01,  3.0253382e-01,  9.3289706e-01,
      2.9740568e-01,  4.1985781e-01,  2.3441296e-01,  3.7160803e-01,  7.0356766e-01,  3.6031117e-01,  8.0302634e-01,  5.5340797e-01,  8.9825174e-01,  4.0907232e-01,  8.1211782e-01,  8.6993293e-01,  6.1094355e-01,  3.4175909e-01,  1.4668875e-02,  8.2886448e-01,  8.5184470e-01,  7.1335444e-01
    };

  const double f_data[] =
    {
      2.9772969e+00,  2.8490411e+00,  4.4797086e+00,  4.0472153e+00,  4.2385898e+00,  3.1089057e+00,  3.9680287e+00,  3.6182944e+00,  2.8321589e+00,  2.8590936e+00,  3.2751040e+00,  2.9576420e+00,  3.1684692e+00,  3.3303050e+00,  2.6519174e+00,  4.6993431e+00,  2.9377072e+00,  4.9226116e+00,
      2.8828908e+00,  2.9468518e+00,  3.8639357e+00,  3.5524610e+00,  3.7833252e+00,  2.9278979e+00,  3.5016651e+00,  3.9692976e+00,  3.0388124e+00,  3.1115468e+00,  3.2611307e+00,  2.9748704e+00,  2.9958628e+00,  2.5970407e+00,  2.3159149e+00,  3.8482168e+00,  2.8993006e+00,  3.9599320e+00,
      3.6616983e+00,  3.3154566e+00,  4.2936709e+00,  3.7892124e+00,  4.5560450e+00,  4.0146890e+00,  4.3795135e+00,  4.5824720e+00,  3.4797825e+00,  3.1939967e+00,  4.4170725e+00,  3.8467199e+00,  3.5581161e+00,  3.0685804e+00,  2.4724131e+00,  4.4983547e+00,  3.2227638e+00,  4.5532449e+00,
      2.8770496e+00,  2.6837391e+00,  3.6312439e+00,  3.1895117e+00,  3.6330218e+00,  2.9321011e+00,  3.6152253e+00,  3.2475376e+00,  2.6380256e+00,  2.5503276e+00,  3.0408538e+00,  2.8843431e+00,  3.0061753e+00,  2.5875682e+00,  2.2089052e+00,  4.0333769e+00,  2.9073422e+00,  3.9545291e+00,
      3.2261417e+00,  2.5285158e+00,  3.7453372e+00,  3.2509103e+00,  3.9258666e+00,  3.2580602e+00,  3.7827043e+00,  3.5485848e+00,  2.8301733e+00,  2.6563408e+00,  3.3658067e+00,  3.3009714e+00,  3.1525772e+00,  2.8519437e+00,  2.2367581e+00,  4.0317027e+00,  2.9054567e+00,  4.2438312e+00,
      2.7175339e+00,  2.7874049e+00,  3.1339862e+00,  3.1080394e+00,  3.3142479e+00,  2.5340986e+00,  2.3388486e+00,  3.3125004e+00,  2.4677117e+00,  2.1814506e+00,  2.7400965e+00,  2.7884337e+00,  2.6173022e+00,  2.1913543e+00,  1.8988358e+00,  2.8064566e+00,  2.1689501e+00,  2.6368829e+00,
      3.0274399e+00,  3.2152336e+00,  4.2571178e+00,  3.4918421e+00,  3.6468027e+00,  3.0499596e+00,  3.4026987e+00,  3.5039376e+00,  2.7403580e+00,  3.1970256e+00,  3.6175464e+00,  2.9387411e+00,  3.4106452e+00,  2.8549384e+00,  2.1980184e+00,  3.7847895e+00,  2.5788601e+00,  3.8722196e+00,
      2.6204453e+00,  2.6003722e+00,  3.3132585e+00,  2.8606140e+00,  3.4053166e+00,  2.7714573e+00,  3.3263859e+00,  2.9014815e+00,  2.6262746e+00,  2.1894414e+00,  3.1162571e+00,  2.8516770e+00,  2.8813169e+00,  2.2267207e+00,  1.8123421e+00,  3.8500517e+00,  2.7820085e+00,  3.6468303e+00,
      1.9514074e+00,  2.3236933e+00,  3.4771615e+00,  2.8717049e+00,  2.8585138e+00,  2.5205856e+00,  2.6733132e+00,  2.6320762e+00,  2.6747614e+00,  2.5998615e+00,  2.9002340e+00,  2.7250883e+00,  2.6814182e+00,  1.6101946e+00,  1.6266525e+00,  3.3724282e+00,  2.1568087e+00,  3.3935804e+00,
      3.0410726e+00,  3.0017518e+00,  4.2835098e+00,  3.6473755e+00,  4.1143910e+00,  3.1462353e+00,  4.0665241e+00,  3.7358847e+00,  3.0687749e+00,  3.2723858e+00,  3.8724654e+00,  3.1290047e+00,  3.5834522e+00,  2.7560633e+00,  2.3354593e+00,  4.5214364e+00,  3.1668170e+00,  4.6689449e+00,
      3.0465649e+00,  2.9554987e+00,  4.3779882e+00,  3.9635060e+00,  4.4176241e+00,  3.7532293e+00,  4.2454293e+00,  4.0283616e+00,  3.3024941e+00,  3.1422887e+00,  3.7717235e+00,  3.6158122e+00,  3.4025077e+00,  3.2034486e+00,  2.5230896e+00,  4.5324723e+00,  2.9804889e+00,  4.5961247e+00,
      1.8121819e+00,  1.8008390e+00,  2.9678931e+00,  3.0937592e+00,  3.1415898e+00,  1.8038475e+00,  2.8690504e+00,  2.3885329e+00,  2.2755915e+00,  2.2616361e+00,  2.4096396e+00,  2.5260285e+00,  2.5221974e+00,  2.3072328e+00,  1.7953609e+00,  3.5424877e+00,  2.2167963e+00,  3.7083622e+00,
      2.5259039e+00,  2.5013631e+00,  3.4267798e+00,  3.2397377e+00,  3.5555266e+00,  2.8694980e+00,  2.8299854e+00,  3.1837414e+00,  2.7081506e+00,  2.3140245e+00,  3.2457792e+00,  3.1137285e+00,  2.8586286e+00,  1.9626517e+00,  1.8924617e+00,  3.4525684e+00,  2.2397491e+00,  3.3826311e+00,
      3.1189663e+00,  3.3296225e+00,  3.9541879e+00,  3.0499924e+00,  3.3688889e+00,  2.9958105e+00,  3.1656017e+00,  3.2712432e+00,  2.5462913e+00,  2.9841602e+00,  3.6107629e+00,  2.8016577e+00,  3.4604375e+00,  2.4543084e+00,  1.9793549e+00,  3.4650897e+00,  2.5831721e+00,  3.3252931e+00,
      2.1014542e+00,  2.6514890e+00,  3.8076366e+00,  3.4706349e+00,  3.4221974e+00,  2.8748121e+00,  3.2500480e+00,  2.9766863e+00,  2.6829611e+00,  2.6263997e+00,  3.2525714e+00,  2.8956778e+00,  2.8036752e+00,  2.5255539e+00,  1.9645498e+00,  3.8438016e+00,  2.1336250e+00,  3.7436370e+00
    };

  const Image<double> d(d_data, 12, 15);
  const Image<double> e(e_data, 18, 12);
  const Image<double> f(f_data, 18, 15);

  // this should use dgemm from lapack if lapack is available

  REQUIRE_EQFP(matrixMult(d, e), f, 1e-5);
}

static void Image_xx_matrixInv_xx_1(TestSuite& suite)
{
  float orig_data[] = {
    1,     7,     5,     9,
    6,     5,     5,     5,
    10,    20,    30,    40,
    1,     9,     7,     6 };

  Image<float> a(orig_data, 4, 4);

  float expected_data[] = { // from matlab inv() function
   0.01529051987768,  0.20489296636086, -0.01529051987768, -0.09174311926605,
   0.15290519877676,  0.04892966360856, -0.05290519877676,  0.08256880733945,
  -0.36391437308868, -0.07645259938838,  0.06391437308869,  0.18348623853211,
   0.19266055045872, -0.01834862385321,  0.00733944954128, -0.15596330275229 };

  Image<float> expected(expected_data, 4, 4);

  REQUIRE_EQFP(matrixInv(a), expected, 1.0e-5F);
}

static void Image_xx_matrixInv_xx_2(TestSuite& suite)
{
  /* test data generated in matlab with:
     sz=7;
     [y, idx] = sort(rand(sz*sz, 1));
     disp(reshape(idx, sz, sz));
     disp(inv(reshape(idx, sz, sz)));
  */

  const float orig_data7[] =
    {
      43,    16,    44,    34,    36,    46,    17,
      23,    31,     6,    26,    49,    28,    27,
      11,     2,     5,     4,     9,    37,    42,
      45,     1,    30,    40,    19,    25,    14,
      10,    33,     7,    39,    47,    35,    20,
      24,    18,     8,    12,    21,    13,    48,
      32,    15,    41,    38,    29,    22,     3
    };

  const float expected_data7[] =
    {
        0.0863664417165848,  -0.0342027308610078,  -0.0690509290541315,   0.0268123823662261,   0.0226507977720774,   0.0387060132448695, -0.110298233327886,
         0.184717788211439,   -0.158457450073631,   -0.144594411698772,  -0.0120193825757529,      0.1237223361442,    0.114187504191344, -0.192020508088664,
       -0.0376325996019747,   0.0173327100498285,   0.0376865299596207,  -0.0313288524314864,  -0.0330709668386202,  -0.0120148878751662, 0.0885582174684409,
       -0.0275145031181991,   -0.027622575353721, 0.000948089575441405,   0.0248025225647299,   0.0326718611234568,  0.00303449174780607,0.00913606104042284,
        -0.148202066322019,     0.16289455708109,    0.109236859441512,  -0.0133361690172031,   -0.112867758761273,  -0.0943874893011545,  0.169331672554365,
        0.0686841039544181,  -0.0444316914804608,  -0.0223629756868543,  0.00298105055676686,   0.0396401960952903,  0.00752975701431266, -0.074898694930778,
       -0.0530652065294748,   0.0213069824245555,   0.0404957714365029, -0.00485087186723845,  -0.0217336543989586,-0.000840911607400576, 0.0563155454935637
    };

  Image<float> a7(orig_data7, 7, 7);
  Image<float> expected7(expected_data7, 7, 7);

  REQUIRE_EQFP(matrixInv(a7), expected7, 1.0e-5F);
}

static void Image_xx_matrixInv_xx_3(TestSuite& suite)
{
  /* test data generated in matlab with:
     sz=14;
     [y, idx] = sort(rand(sz*sz, 1));
     disp(reshape(idx, sz, sz));
     disp(inv(reshape(idx, sz, sz)));
  */

  const float orig_data14[] =
    {
        6,  160,  195,  153,    5,   32,  104,   20,  109,   66,  196,  128,  122,   21,
      191,   90,   19,  141,   24,   65,  108,   79,   58,  132,  168,   37,   30,  150,
      163,   45,  145,  107,  177,   87,   44,  190,   71,   29,   35,  169,  123,   69,
      114,   48,  133,   22,  156,  161,   38,   88,  142,   23,  179,   93,   11,  189,
       59,  126,   72,   82,   27,  184,  136,   34,  116,  164,  158,    9,  187,  131,
       84,  110,   57,  151,    1,  129,   86,   49,  139,   46,  165,   67,   81,  112,
       54,  121,  167,  134,  152,  182,  157,  180,   89,   95,   14,  174,  127,  125,
      100,  176,  102,   85,  103,  173,  147,   17,  130,   51,   75,   42,   73,   77,
      140,  143,   97,   55,  137,  124,   36,   60,   13,   83,    4,  115,   91,   53,
       78,   39,  166,   47,  154,  181,   74,  146,  144,   26,   64,   94,  172,   62,
       68,  135,  155,  185,   28,  138,  178,   33,    3,   40,   63,  193,   16,  194,
      183,   15,   80,   12,    7,  148,   99,  113,   98,   61,    8,   50,  105,   41,
      101,    2,  186,   31,   76,  162,  149,  192,   25,  117,   92,  175,   70,  170,
      118,   96,  159,  120,  119,  171,   56,  106,   18,   43,  188,  111,   10,   52
    };

  const float expected_data14[] =
    {
      7.53563392062889e-05,     0.000461707861660145,      0.00372449797232745,     0.000976014170193408,      0.00110971554872798,     -0.00275736817253676,      -0.0030838794331058,     0.000644139519418342,    -0.000373763475101289,     -0.00242814669384377,      0.00151417461528592,      0.00405285048664069,     -0.00191325733529331,     0.000437202785289814,
      -0.00205231848684335,        0.161112139816245,       -0.233529578918986,      -0.0998048351837866,       -0.195973606570308,       0.0976133973055097,      -0.0626092469087169,      -0.0102011444468062,        0.155621382432984,        0.319331712618087,     -0.00981464989785451,        -0.11723015374426,       0.0959682935636109,      -0.0662733175762567,
       0.00617429918685153,       0.0838497332111019,        -0.12088634449044,      -0.0370600916526799,      -0.0908447091405074,        0.024365092863746,      -0.0189883694308366,      -0.0212361627308311,       0.0694863490875842,        0.158816697415575,      0.00394414867624216,      -0.0414965239062269,       0.0272880174719444,      -0.0271786546398205,
       0.00229235617311322,      -0.0430798220069746,       0.0655717066455695,       0.0328171454409947,       0.0595232785391925,      -0.0360331650829408,        0.027782699369807,     -0.00631120804640155,      -0.0487762021179046,      -0.0927695791891363,      0.00753820406482533,       0.0405404686468824,      -0.0415702670292879,        0.023412757449553,
     -0.000773962977241737,      -0.0833884802715835,        0.124835722682279,       0.0495325285143792,        0.100808690170754,      -0.0494642249314272,         0.02960502027195,       0.0132651266524234,      -0.0791908401186713,       -0.165204993489785,      0.00235110811386487,       0.0530602105499828,      -0.0456738197005483,       0.0333464397653038,
      -0.00227263966037385,      -0.0419536956512115,        0.052276980269038,       0.0233534263813399,       0.0473461935710808,      -0.0190575175852659,       0.0163974953052667,      0.00129245475276669,      -0.0347738950301648,      -0.0765962563150846,      0.00221586597406838,       0.0295047899075718,      -0.0232165419375637,       0.0192687944398009,
      -0.00243408179849765,        -0.05146565904427,        0.078601083198067,       0.0209591798695574,       0.0575525797581148,      -0.0223995920940085,      0.00891341551471089,       0.0208503321070358,      -0.0501011810263884,      -0.0986858023152499,    -0.000313241141926206,       0.0260014317223912,      -0.0146014966781001,       0.0170531835993851,
      -0.00316165731675941,        0.154417515924666,       -0.220502866905236,      -0.0948364444416008,       -0.185737315868655,       0.0913629714277201,      -0.0553673808491966,      -0.0132747302628695,         0.14207612195838,        0.303615391802225,      -0.0105016926531228,        -0.10917933265103,       0.0905415501630788,      -0.0605385696865733,
       0.00431962296105473,      -0.0744114741281338,        0.104167579186192,       0.0513392034700611,       0.0873595209074504,      -0.0434430099208895,       0.0361884599841365,      0.00366132423521931,      -0.0708723088378299,       -0.151086293442918,       0.0012921697396745,       0.0586021550281484,      -0.0481225928114504,       0.0278352130701018,
       0.00603080726907753,       -0.117028827890343,        0.166291502211568,       0.0799026577873399,        0.146724081508712,      -0.0782854508777231,       0.0584770985886571,     0.000484062857531725,       -0.111273515210444,       -0.243621145463788,      0.00432315988844692,       0.0931472700367292,      -0.0776689408741061,       0.0500031364844012,
      -0.00179427989626322,      -0.0129877241534507,       0.0200096511411447,     0.000785980729038293,       0.0118751820514703,      0.00317334498048273,       -0.006027133506779,      0.00933642019990867,     -0.00945936975257917,      -0.0208671802272393,     -0.00348575622888212,     -0.00151864268549087,      0.00525120510517503,      0.00424911581372456,
     -6.62200107920678e-06,       -0.153126572929898,        0.214532637971622,       0.0821362342482833,        0.169416047801648,      -0.0662596632338817,       0.0470724122126177,       0.0214654729105009,       -0.129491939723347,       -0.293422791051627,        0.001314817279614,       0.0941185088799686,      -0.0677714609075475,       0.0537656264717411,
      -0.00257306548966421,       0.0351531875139322,      -0.0476630134419376,      -0.0283892345334294,       -0.041552450054926,       0.0259333876386372,      -0.0235836020163034,     0.000149894936715443,       0.0362535177337894,       0.0783828511291872,     -0.00096775604298647,      -0.0321788332441728,       0.0268588346486472,      -0.0178747861295432,
      -0.00172665693499422,        0.103580989074918,       -0.146449167610552,       -0.056330386951719,       -0.118211250058327,       0.0532027357703076,      -0.0369148821556138,      -0.0144037069277378,       0.0937462253549283,        0.202232551557968,     0.000264533062638735,      -0.0698776598085848,       0.0530908950765443,      -0.0426499130339133
    };

  Image<float> a14(orig_data14, 14, 14);
  Image<float> expected14(expected_data14, 14, 14);

  REQUIRE_EQFP(matrixInv(a14), expected14, 1.0e-5F);
}

static void Image_xx_matrixDet_xx_1(TestSuite& suite)
{
  float orig_data[] = {
    1,     7,     5,     9,
    6,     5,     5,     5,
    10,    20,    30,    40,
    1,     9,     7,     6 };

  Image<float> a(orig_data, 4, 4);

  float expected = 3269.999511718750000F;

  REQUIRE_EQ(matrixDet(a), expected);
}

static void Image_xx_matrixDet_xx_2(TestSuite& suite)
{
  Image<float> a(10, 10, ZEROS);
  float expected = 0.0F;

  REQUIRE_EQ(matrixDet(a), expected);
}

static void Image_xx_matrixDet_xx_3(TestSuite& suite)
{
  Image<byte> a = eye<byte>(17); // identity matrix
  float expected = 1.0F;

  REQUIRE_EQ(matrixDet(a), expected);
}

namespace
{
  Image<double> svdIdent(const Image<double>& M,
                         Image<double>& PI,
                         const SvdAlgo algo,
                         const SvdFlag flags = 0)
  {
    Image<double> U;
    Image<double> S;
    Image<double> V;

    svd(M, U, S, V, algo, flags);

    Image<double> US = matrixMult(U, S);
    Image<double> MM = matrixMult(US, transpose(V));

    PI = matrixMult(V, matrixMult(naiveUnstablePseudoInv(S), transpose(U)));

    return MM;
  }
}

static void Image_xx_svd_gsl_xx_1(TestSuite& suite)
{
  rutz::urand gen(time((time_t*)0) + getpid());

  Image<double> x(50, 9000, NO_INIT);

  for (Image<double>::iterator itr = x.beginw(), stop = x.endw();
       itr != stop;
       ++itr)
    {
      *itr = gen.fdraw();
    }

  Image<double> xpi;
  Image<double> xx(svdIdent(x, xpi, SVD_GSL, SVD_TALL));

  // we're checking two identities here: first, that we can
  // reconstruct the original matrix 'x' from its SVD decomposition,
  // and second, that we get the same pseudoinverse whether we build
  // it directly with naiveUnstablePseudoInv(), or whether we
  // construct it from the SVD decomposition

  const double gsl_err = RMSerr(x, xx);
  const double gsl_pi_err = RMSerr(xpi, naiveUnstablePseudoInv(x));

  REQUIRE_LT(gsl_err, 1e-10);
  REQUIRE_LT(gsl_pi_err, 1e-10);
}

static void Image_xx_svd_lapack_xx_1(TestSuite& suite)
{
  rutz::urand gen(time((time_t*)0) + getpid());

  Image<double> x(50, 9000, NO_INIT);

  for (Image<double>::iterator itr = x.beginw(), stop = x.endw();
       itr != stop;
       ++itr)
    {
      *itr = gen.fdraw();
    }

  Image<double> xpi;
  Image<double> xx(svdIdent(x, xpi, SVD_LAPACK));

  const double lapack_err = RMSerr(x, xx);
  const double lapack_pi_err = RMSerr(xpi, naiveUnstablePseudoInv(x));

  REQUIRE_LT(lapack_err, 1e-10);
  REQUIRE_LT(lapack_pi_err, 1e-10);
}

static void Image_xx_svd_lapack_xx_2(TestSuite& suite)
{
  // like the previous test, but here using float instead of double

  rutz::urand_frange gen(time((time_t*)0) + getpid());

  Image<float> x(50, 9000, NO_INIT);
  fill(x, gen);

  Image<float> U;
  Image<float> S;
  Image<float> V;

  svdf(x, U, S, V, SVD_LAPACK);

  Image<float> US = matrixMult(U, S);
  Image<float> xx = matrixMult(US, transpose(V));

  const double lapack_err = RMSerr(x, xx);
  const double rsq        = corrcoef(x, xx);

  REQUIRE_LT(lapack_err, 1e-5);
  REQUIRE_GT(rsq, 0.99);
}

static void Image_xx_svd_full_xx_1(TestSuite& suite)
{
  /* test data from matlab:

     A=rand(9,7);
     [U,S,V]=svd(A);
     format long g;
     disp(A)
     disp(U)
     disp(S)
     disp(V)
   */
  const double Adata[] = {
         0.950129285147175,   0.444703364353194,   0.410270206990945,   0.603792479193819,   0.846221417824324,   0.502812883996251,   0.150872976149765,
         0.231138513574288,   0.615432348100095,   0.893649530913534,    0.27218792496996,   0.525152496305172,   0.709471392703387,   0.697898481859863,
         0.606842583541787,   0.791937037427035,  0.0578913047842686,   0.198814267761062,   0.202647357650387,   0.428892365340997,   0.378373000512671,
           0.4859824687093,   0.921812970744803,      0.352868132217,  0.0152739270290363,   0.672137468474288,   0.304617366869394,    0.86001160488682,
         0.891298966148902,   0.738207245810665,   0.813166497303758,   0.746785676564429,   0.838118445052387,   0.189653747547175,   0.853655130662768,
         0.762096833027395,   0.176266144494618, 0.00986130066092356,   0.445096432287947,  0.0196395138648175,   0.193431156405215,   0.593562912539682,
         0.456467665168341,   0.405706213062095,    0.13889088195695,   0.931814578461665,   0.681277161282135,   0.682223223591384,   0.496552449703103,
        0.0185036432482244,   0.935469699107605,   0.202765218560273,   0.465994341675424,   0.379481018027998,   0.302764400776609,    0.89976917516961,
         0.821407164295253,   0.916904439913408,    0.19872174266149,   0.418649467727506,   0.831796017609607,   0.541673853898088,   0.821629160735343
  };

  const double Udata[] = {
        -0.334369457483348,   0.553753246384211,   0.173722104758991,   0.216107949399242,    0.33819994715599,   0.107950179334316,  -0.168945273189193,   0.457659744648732,  -0.375026665540279,
        -0.329665554632197,  -0.289771377227679,   0.634134705802461,  -0.018028983235094,  0.0467322256708354,  -0.615149159205802,  0.0907953598669529,  0.0925800794220671,    0.08656814784916,
        -0.246139085952845, -0.0548593578229483,   -0.41050144345874,    0.11040680321451,   0.332992352163831,   -0.37647619850007,  -0.618601139886217,  -0.340483123953779,  0.0670115015894877,
        -0.338502200326128,  -0.400181962186731,   -0.13227390062104,   0.362805829219031,   0.129457018880407,   0.171533541715475,   0.398807623266607,  -0.305062586901122,  -0.527117278260459,
         -0.44489875430374,   0.131261679408881,   0.240577903720863,   0.271982040280096,  -0.592256457043096,   0.318449189950086,  -0.285471727381645,  -0.272222800411694,   0.212618330011173,
        -0.198783947890269,   0.257075663136555,  -0.465947906013995, -0.0378515144915132,  -0.542621813915509,  -0.497996499745613,   0.245305115243475,   0.187677943372127,  -0.194987949697117,
        -0.324145914503961,   0.328343166397085,  0.0761178782792188,   -0.71190747904549,    0.14722746538287,  0.0786683148078264,   0.192639295168155,  -0.450336123723707, -0.0820783064210453,
         -0.29677060429998,  -0.504820108786232,  -0.140000844979403,  -0.466845088254261,  -0.137738046651579,   0.252315401154192,  -0.322375638216978,   0.451927826068197,  -0.169527726112523,
        -0.416840705481074, -0.0162676625612421,  -0.285143189435573,  0.0948209656476317,   0.264461771501813,   0.139706911019148,   0.372550868730716,   0.237938856853049,   0.671427163909875
  };

  const double Sdata[] = {
          4.33980802951066,                  0,                  0,                  0,                  0,                  0,                  0,
                         0,    1.1612093946393,                  0,                  0,                  0,                  0,                  0,
                         0,                  0,  0.879337316628834,                  0,                  0,                  0,                  0,
                         0,                  0,                  0,  0.760487539411249,                  0,                  0,                  0,
                         0,                  0,                  0,                  0,  0.658580414390557,                  0,                  0,
                         0,                  0,                  0,                  0,                  0,  0.509093207477467,                  0,
                         0,                  0,                  0,                  0,                  0,                  0,  0.320075663338178,
                         0,                  0,                  0,                  0,                  0,                  0,                  0,
                         0,                  0,                  0,                  0,                  0,                  0,                  0
  };

  const double Vdata[] = {
        -0.403622982097106,   0.578252330821083,  -0.391767249118321,   0.529050685580787, -0.0947464346071052,  -0.245674262341264,-0.00194801953388936,
         -0.46392416666584,  -0.478943519182328,  -0.279266979790604,  0.0820353792216147,   0.307808958667134,   0.142898419581493,  -0.596239306165418,
        -0.257442552523379,  -0.109255973768973,   0.777953044405826,   0.332765835361449,  -0.298225513057705,  -0.241240252591135,    -0.2422954967662,
        -0.318285152564917,   0.443339154898195,  0.0596398081141739,  -0.659948317149671,  -0.326435253493529,   0.178830452220086,  -0.352868193530783,
        -0.412560890051571,   0.146384150301362,   0.297908475746163,   0.109867117049205,    0.34346695774658,   0.648160416413319,   0.411659413522294,
        -0.292710011070547,  0.0554502483926221,   0.149523329998174,  -0.365108802679842,   0.562068682918466,  -0.631639863837263,   0.201728365526942,
        -0.445415661076658,  -0.450830149565138,  -0.220576164554481,  -0.147429525356922,  -0.516611625256527, -0.0998306836562218,   0.501112780763699
  };

  Image<double> A(&Adata[0], 7, 9);
  Image<double> Ux(&Udata[0], 9, 9);
  Image<double> Sx(&Sdata[0], 7, 9);
  Image<double> Vx(&Vdata[0], 7, 7);

#ifdef HAVE_LAPACK // SVD_FULL is only supported with LAPACK (and not with GSL)
  Image<double> Ulapack, Slapack, Vlapack;
  svd(A, Ulapack, Slapack, Vlapack, SVD_LAPACK, SVD_FULL);

  REQUIRE_EQFP(Ulapack, Ux, 1e-10);
  REQUIRE_EQFP(Slapack, Sx, 1e-10);
  REQUIRE_EQFP(Vlapack, Vx, 1e-10);
#endif
}

//---------------------------------------------------------------------
//
// ShapeOps
//
//---------------------------------------------------------------------

static void Image_xx_rescale_xx_1(TestSuite& suite)
{
  float orig_data[] =
    {
      0.0, 0.5,
      0.5, 1.0
    };

  Image<float> orig(orig_data, 2, 2);

  Image<float> scaled = rescale(orig, 2, 2);

  REQUIRE_EQ(orig, scaled);
}

static void Image_xx_rescale_xx_2(TestSuite& suite)
{
  // test rescaling with dest size > src size

  float orig_data[] =
    {
      0.0, 0.5,
      0.5, 1.0
    };

  float scaled_data[] =
    {
      0.000, 0.125, 0.375, 0.500,
      0.125, 0.250, 0.500, 0.625,
      0.375, 0.500, 0.750, 0.875,
      0.500, 0.625, 0.875, 1.000
    };

  Image<float> orig(orig_data, 2, 2);
  Image<float> expected(scaled_data, 4, 4);

  Image<float> actual = rescale(orig, 4, 4);

  REQUIRE_EQ(actual, expected);
}

static void Image_xx_rescale_xx_3(TestSuite& suite)
{
  // test rescaling with dest size < src size

  float orig_data[] =
    {
      0.0, 0.25, 0.5, 0.75,
      0.25, 0.5, 0.75, 1.0,
      0.5, 0.75, 1.0, 1.25,
      0.75, 1.0, 1.25, 1.5
    };

  float scaled_data[] =
    {
      0.25, 0.75,
      0.75, 1.25
    };

  Image<float> orig(orig_data, 4, 4);
  Image<float> expected(scaled_data, 2, 2);

  Image<float> actual = rescale(orig, 2, 2);

  REQUIRE_EQ(actual, expected);
}

static void Image_xx_rescale_xx_4(TestSuite& suite)
{
  // test rescaling rgb with dest size > src size

  typedef PixRGB<byte> PB;

  const PixRGB<byte> orig_data_rgb[] =
    {
      PB(0, 1, 2),    PB(33, 34, 35),
      PB(29, 30, 31), PB(64, 65, 66)
    };

  const PixRGB<byte> scaled_data_rgb[] =
    {
      PB(0, 1, 2),    PB(8, 9, 10),   PB(24, 25, 26), PB(33, 34, 35),
      PB(7, 8, 9),    PB(15, 16, 17), PB(32, 33, 34), PB(40, 41, 42),
      PB(21, 22, 23), PB(30, 31, 32), PB(47, 48, 49), PB(56, 57, 58),
      PB(29, 30, 31), PB(37, 38, 39), PB(55, 56, 57), PB(64, 65, 66)
    };

  Image<PixRGB<byte> > orig_rgb(orig_data_rgb, 2, 2);
  Image<PixRGB<byte> > expected_rgb(scaled_data_rgb, 4, 4);

  Image<PixRGB<byte> > actual_rgb = rescale(orig_rgb, 4, 4);

  REQUIRE_EQ(actual_rgb, expected_rgb);
}

static void Image_xx_rescale_xx_5(TestSuite& suite)
{
  // test rescaling rgb with dest size < src size

  typedef PixRGB<byte> PB;

  const PixRGB<byte> orig_data_rgb[] =
    {
      PB(0, 1, 2),    PB(25, 26, 27),    PB(50, 51, 52),    PB(75, 76, 77),
      PB(25, 26, 27), PB(50, 51, 52),    PB(75, 76, 77),    PB(100, 101, 102),
      PB(50, 51, 52), PB(75, 76, 77),    PB(100, 101, 102), PB(125, 126, 127),
      PB(75, 76, 77), PB(100, 101, 102), PB(125, 126, 127), PB(150, 151, 152)
    };

  const PixRGB<byte> scaled_data_rgb[] =
    {
      PB(25, 26, 27), PB(75, 76, 77),
      PB(75, 76, 77), PB(125, 126, 127)
    };

  Image<PixRGB<byte> > orig_rgb(orig_data_rgb, 4, 4);
  Image<PixRGB<byte> > expected_rgb(scaled_data_rgb, 2, 2);

  Image<PixRGB<byte> > actual_rgb = rescale(orig_rgb, 2, 2);

  REQUIRE_EQ(actual_rgb, expected_rgb);
}

static void Image_xx_zoomXY_xx_1(TestSuite& suite)
{
  byte orig_data[] =
    {
      1, 3, 5, 7, 9
    };

  byte zoomed_data[] =
    {
      1, 3, 5, 7, 9,
      1, 3, 5, 7, 9
    };

  Image<byte> orig(orig_data, 5, 1);
  Image<byte> expected(zoomed_data, 5, 2);

  Image<byte> actual = zoomXY(orig, 1, 2);

  REQUIRE_EQ(actual, expected);
}

static void Image_xx_zoomXY_xx_2(TestSuite& suite)
{
  float orig_data[] =
    {
      0.2, 0.4, 0.6, 0.8
    };

  float zoomed_data[] =
    {
      0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8
    };

  Image<float> orig(orig_data, 1, 4);
  Image<float> expected(zoomed_data, 2, 4);

  Image<float> actual = zoomXY(orig, 2, 1);

  REQUIRE_EQ(actual, expected);
}

static void Image_xx_zoomXY_xx_3(TestSuite& suite)
{
  byte orig_data[] =
    {
      1, 2,
      3, 4
    };

  byte zoomed_data[] =
    {
      1,1,1, 2,2,2,
      1,1,1, 2,2,2,
      1,1,1, 2,2,2,
      1,1,1, 2,2,2,
      3,3,3, 4,4,4,
      3,3,3, 4,4,4,
      3,3,3, 4,4,4,
      3,3,3, 4,4,4
    };

  Image<byte> orig(orig_data, 2, 2);
  Image<byte> expected(zoomed_data, 6, 8);

  Image<byte> actual = zoomXY(orig, 3, 4);

  REQUIRE_EQ(actual, expected);
}

static void Image_xx_zoomXY_xx_4(TestSuite& suite)
{
  Image<byte> orig(5,5, NO_INIT);
  Image<byte> zoomed00 = zoomXY(orig, 0, 0);
  Image<byte> zoomed01 = zoomXY(orig, 0, 1);
  Image<byte> zoomed10 = zoomXY(orig, 1, 0);
  Image<byte> zoomed11 = zoomXY(orig, 1, 1);

  REQUIRE_EQ(zoomed00.getWidth(), 0);
  REQUIRE_EQ(zoomed00.getHeight(), 0);

  REQUIRE_EQ(zoomed01.getWidth(), 0);
  REQUIRE_EQ(zoomed01.getHeight(), 0);

  REQUIRE_EQ(zoomed10.getWidth(), 0);
  REQUIRE_EQ(zoomed10.getHeight(), 0);

  REQUIRE_EQ(zoomed11, orig);
}

static void Image_xx_concatX_xx_1(TestSuite& suite)
{
  const Image<byte> left = generateImage<byte>(6, 6, Counter(1));
  const Image<byte> right = generateImage<byte>(3, 6, Counter(100));

  const byte expected_data[] =
    {
      1,  2,  3,  4,  5,  6,  100, 101, 102,
      7,  8,  9,  10, 11, 12, 103, 104, 105,
      13, 14, 15, 16, 17, 18, 106, 107, 108,
      19, 20, 21, 22, 23, 24, 109, 110, 111,
      25, 26, 27, 28, 29, 30, 112, 113, 114,
      31, 32, 33, 34, 35, 36, 115, 116, 117
    };

  REQUIRE_EQ(concatX(left, right),
             Image<byte>(expected_data, 9, 6));

  const Image<byte> empty;

  REQUIRE_EQ(concatX(left, empty), left);
  REQUIRE_EQ(concatX(empty, right), right);
  REQUIRE_EQ(concatX(empty, empty), empty);
}

static void Image_xx_concatY_xx_1(TestSuite& suite)
{
  const Image<byte> top = generateImage<byte>(4, 3, Counter(1));
  const Image<byte> bottom = generateImage<byte>(4, 4, Counter(100));

  const byte expected_data[] =
    {
      1,   2,   3,   4,
      5,   6,   7,   8,
      9,   10,  11,  12,
      100, 101, 102, 103,
      104, 105, 106, 107,
      108, 109, 110, 111,
      112, 113, 114, 115
    };

  REQUIRE_EQ(concatY(top, bottom),
             Image<byte>(expected_data, 4, 7));

  const Image<byte> empty;

  REQUIRE_EQ(concatY(top, empty), top);
  REQUIRE_EQ(concatY(empty, bottom), bottom);
  REQUIRE_EQ(concatY(empty, empty), empty);
}

static void Image_xx_concatLooseX_xx_1(TestSuite& suite)
{
  const Image<byte> left = generateImage<byte>(4, 4, Counter(1));
  const Image<byte> right = generateImage<byte>(2, 2, Counter(100));

  const byte expected_data[] =
    {
      1,  2,  3,  4,  100, 101,
      5,  6,  7,  8,  102, 103,
      9,  10, 11, 12, 0,   0,
      13, 14, 15, 16, 0,   0
    };

  REQUIRE_EQ(concatLooseX(left, right),
             Image<byte>(expected_data, 6, 4));

  const Image<byte> empty;

  REQUIRE_EQ(concatLooseX(empty, empty), empty);
}

static void Image_xx_concatLooseY_xx_1(TestSuite& suite)
{
  const Image<byte> top = generateImage<byte>(5, 5, Counter(1));
  const Image<byte> bottom = generateImage<byte>(3, 3, Counter(100));

  const byte expected_data[] =
    {
      1,   2,   3,   4,  5,
      6,   7,   8,   9,  10,
      11,  12,  13,  14, 15,
      16,  17,  18,  19, 20,
      21,  22,  23,  24, 25,
      100, 101, 102, 0,  0,
      103, 104, 105, 0,  0,
      106, 107, 108, 0,  0
    };

  REQUIRE_EQ(concatLooseY(top, bottom),
             Image<byte>(expected_data, 5, 8));

  const Image<byte> empty;

  REQUIRE_EQ(concatLooseY(empty, empty), empty);
}

static void Image_xx_crop_xx_1(TestSuite& suite)
{
  /*
  byte orig_data[] =
    {
       1,  2,  3,  4,  5,  6,  7,
       8,  9, 10, 11, 12, 13, 14,
      15, 16, 17, 18, 19, 20, 21,
      22, 23, 24, 25, 26, 27, 28,
      29, 30, 31, 32, 33, 34, 35,
      36, 37, 38, 39, 40, 41, 42,
      43, 44, 45, 46, 47, 48, 49
    };
  */

  const Image<byte> orig = generateImage<byte>(7, 7, Counter(1));

  // Sub-image == full image
  REQUIRE_EQ(orig, crop(orig, Point2D<int>(0,0), Dims(7, 7)));
  REQUIRE_EQ(orig, crop(orig, Rectangle::tlbrI(0, 0, 6, 6)));

  // Sub-image == empty image
  Image<byte> empty;
  REQUIRE_EQ(empty, crop(orig, Point2D<int>(3,3), Dims(0, 0)));

  // Sub-image == upper all
  byte upper_rows[] =
    {
      1, 2, 3, 4, 5, 6, 7,
      8, 9, 10, 11, 12, 13, 14
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(0,0), Dims(7, 2)),
             Image<byte>(upper_rows, 7, 2));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(0, 0, 1, 6)),
             Image<byte>(upper_rows, 7, 2));

  // Sub-image == upper left
  byte upper_left[] =
    {
      1, 2, 3, 4,
      8, 9, 10, 11,
      15, 16, 17, 18
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(0,0), Dims(4, 3)),
             Image<byte>(upper_left, 4, 3));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(0, 0, 2, 3)),
             Image<byte>(upper_left, 4, 3));

  // Sub-image == upper center
  byte upper_center[] =
    {
      2, 3, 4, 5, 6,
      9, 10, 11, 12, 13,
      16, 17, 18, 19, 20,
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(1,0), Dims(5, 3)),
             Image<byte>(upper_center, 5, 3));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(0, 1, 2, 5)),
             Image<byte>(upper_center, 5, 3));

  // Sub-image == upper right
  byte upper_right[] =
    {
      4, 5, 6, 7,
      11, 12, 13, 14,
      18, 19, 20, 21
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(3,0), Dims(4, 3)),
             Image<byte>(upper_right, 4, 3));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(0, 3, 2, 6)),
             Image<byte>(upper_right, 4, 3));

  // Sub-image == middle all
  byte middle_all[] =
    {
      22, 23, 24, 25, 26, 27, 28,
      29, 30, 31, 32, 33, 34, 35,
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(0,3), Dims(7, 2)),
             Image<byte>(middle_all, 7, 2));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(3, 0, 4, 6)),
             Image<byte>(middle_all, 7, 2));

  // Sub-image == middle left
  byte middle_left[] =
    {
      15, 16, 17,
      22, 23, 24,
      29, 30, 31,
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(0,2), Dims(3, 3)),
             Image<byte>(middle_left, 3, 3));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(2, 0, 4, 2)),
             Image<byte>(middle_left, 3, 3));

  // Sub-image == middle center
  byte middle_center[] =
    {
      25
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(3,3), Dims(1, 1)),
             Image<byte>(middle_center, 1, 1));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(3, 3, 3, 3)),
             Image<byte>(middle_center, 1, 1));

  // Sub-image == middle right
  byte middle_right[] =
    {
      13, 14,
      20, 21,
      27, 28,
      34, 35,
      41, 42,
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(5,1), Dims(2, 5)),
             Image<byte>(middle_right, 2, 5));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(1, 5, 5, 6)),
             Image<byte>(middle_right, 2, 5));

  // Sub-image == lower all
  byte lower_all[] =
    {
      43, 44, 45, 46, 47, 48, 49
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(0,6), Dims(7, 1)),
             Image<byte>(lower_all, 7, 1));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(6, 0, 6, 6)),
             Image<byte>(lower_all, 7, 1));

  // Sub-image == lower left
  byte lower_left[] =
    {
      22, 23, 24,
      29, 30, 31,
      36, 37, 38,
      43, 44, 45,
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(0,3), Dims(3, 4)),
             Image<byte>(lower_left, 3, 4));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(3, 0, 6, 2)),
             Image<byte>(lower_left, 3, 4));

  // Sub-image == lower center
  byte lower_center[] =
    {
      38, 39, 40,
      45, 46, 47,
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(2,5), Dims(3, 2)),
             Image<byte>(lower_center, 3, 2));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(5, 2, 6, 4)),
             Image<byte>(lower_center, 3, 2));

  // Sub-image == lower right
  byte lower_right[] =
    {
      19, 20, 21,
      26, 27, 28,
      33, 34, 35,
      40, 41, 42,
      47, 48, 49
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(4,2), Dims(3, 5)),
             Image<byte>(lower_right, 3, 5));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(2, 4, 6, 6)),
             Image<byte>(lower_right, 3, 5));

  // Sub-image == all left
  byte all_left[] =
    {
      1, 2, 3, 4,
      8, 9, 10, 11,
      15, 16, 17, 18,
      22, 23, 24, 25,
      29, 30, 31, 32,
      36, 37, 38, 39,
      43, 44, 45, 46,
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(0, 0), Dims(4, 7)),
             Image<byte>(all_left, 4, 7));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(0, 0, 6, 3)),
             Image<byte>(all_left, 4, 7));

  // Sub-image == all center
  byte all_center[] =
    {
      5, 6,
      12, 13,
      19, 20,
      26, 27,
      33, 34,
      40, 41,
      47, 48,
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(4, 0), Dims(2, 7)),
             Image<byte>(all_center, 2, 7));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(0, 4, 6, 5)),
             Image<byte>(all_center, 2, 7));

  // Sub-image == all right
  byte all_right[] =
    {
      3, 4, 5, 6, 7,
      10, 11, 12, 13, 14,
      17, 18, 19, 20, 21,
      24, 25, 26, 27, 28,
      31, 32, 33, 34, 35,
      38, 39, 40, 41, 42,
      45, 46, 47, 48, 49
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(2, 0), Dims(5, 7)),
             Image<byte>(all_right, 5, 7));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(0, 2, 6, 6)),
             Image<byte>(all_right, 5, 7));

  // Sub-image == interior chunk 1
  byte interior_chunk_1[] =
    {
       9, 10, 11, 12,
      16, 17, 18, 19,
      23, 24, 25, 26,
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(1, 1), Dims(4, 3)),
             Image<byte>(interior_chunk_1, 4, 3));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(1, 1, 3, 4)),
             Image<byte>(interior_chunk_1, 4, 3));

  // Sub-image == interior chunk 1
  byte interior_chunk_2[] =
    {
      33, 34,
      40, 41,
    };

  REQUIRE_EQ(crop(orig, Point2D<int>(4, 4), Dims(2, 2)),
             Image<byte>(interior_chunk_2, 2, 2));
  REQUIRE_EQ(crop(orig, Rectangle::tlbrI(4, 4, 5, 5)),
             Image<byte>(interior_chunk_2, 2, 2));
}

static void Image_xx_interpolate_xx_1(TestSuite& suite)
{
  // To avoid bugs in involving transposing width/height values, the test
  // arrays here should be non-square.
  const float orig_data[] =
    {
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0
    };

  const Image<float> orig(orig_data, 6, 5);

  const float expected_data[] =
    {
      1.00, 1.00, 1.00, 2.00, 3.00, 3.00, 3.00, 2.00, 1.00, 1.00, 1.00, 0.75,
      1.00, 1.00, 1.00, 2.00, 3.00, 3.00, 3.00, 2.00, 1.00, 1.00, 1.00, 0.75,
      1.00, 1.00, 1.00, 2.00, 3.00, 3.00, 3.00, 2.00, 1.00, 1.00, 1.00, 0.75,
      2.00, 2.00, 2.00, 2.50, 3.00, 3.00, 3.00, 2.50, 2.00, 2.00, 2.00, 1.50,
      3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 2.25,
      2.00, 2.00, 2.00, 2.50, 3.00, 3.00, 3.00, 2.50, 2.00, 2.00, 2.00, 1.50,
      1.00, 1.00, 1.00, 2.00, 3.00, 3.00, 3.00, 2.00, 1.00, 1.00, 1.00, 0.75,
      1.00, 1.00, 1.00, 2.00, 3.00, 3.00, 3.00, 2.00, 1.00, 1.00, 1.00, 0.75,
      1.00, 1.00, 1.00, 2.00, 3.00, 3.00, 3.00, 2.00, 1.00, 1.00, 1.00, 0.75,
      0.75, 0.75, 0.75, 1.50, 2.25, 2.25, 2.25, 1.50, 0.75, 0.75, 0.75, 0.50
    };

  const Image<float> expected(expected_data, 12, 10);

  const Image<float> result = interpolate(orig);

  REQUIRE_EQ(result, expected);
}

static void Image_xx_decY_xx_1(TestSuite& suite)
{
  const Image<byte> src = generateImage<byte>(5, 50, Counter(0));

  const byte expected_data[] =
    {
      0, 1, 2, 3, 4,
      65, 66, 67, 68, 69,
      130, 131, 132, 133, 134
    };

  const Image<byte> expected(expected_data, 5, 3);

  REQUIRE_EQ(decY(src, 13), expected);
  REQUIRE_EQ(decXY(src, 1, 13), expected);
}

static void Image_xx_blurAndDecY_xx_1(TestSuite& suite)
{
  rutz::urand_frange gen(0.0, 256.0, time((time_t*)0) + getpid());

  Image<float> src(400, 7500, NO_INIT);

  {
    gen = fill(src, gen);

    const int reduction_factor = 13;

    Image<float> yFilter(1, reduction_factor, NO_INIT);
    yFilter.clear(1.0f/reduction_factor);

    const Image<float> ff1 =
      decY(sepFilter(src, Image<float>(), yFilter,
                     CONV_BOUNDARY_CLEAN),
           reduction_factor);

    const Image<float> ff2 = blurAndDecY(src, reduction_factor);

    REQUIRE_LTE(RMSerr(ff1, ff2), 0.0001);
    REQUIRE_GTE(corrcoef(ff1, ff2), 0.9999);
  }

  {
    gen = fill(src, gen);

    const int reduction_factor = 19;

    Image<float> yFilter(1, reduction_factor, NO_INIT);
    yFilter.clear(1.0f/reduction_factor);

    const Image<float> ff1 =
      decY(sepFilter(src, Image<float>(), yFilter,
                     CONV_BOUNDARY_CLEAN),
           reduction_factor);

    const Image<float> ff2 = blurAndDecY(src, reduction_factor);

    REQUIRE_LTE(RMSerr(ff1, ff2), 0.0001);
    REQUIRE_GTE(corrcoef(ff1, ff2), 0.9999);
  }

  {
    gen = fill(src, gen);

    const int reduction_factor = 25;

    Image<float> yFilter(1, reduction_factor, NO_INIT);
    yFilter.clear(1.0f/reduction_factor);

    const Image<float> ff1 =
      decY(sepFilter(src, Image<float>(), yFilter,
                     CONV_BOUNDARY_CLEAN),
           reduction_factor);

    const Image<float> ff2 = blurAndDecY(src, reduction_factor);

    REQUIRE_LTE(RMSerr(ff1, ff2), 0.0001);
    REQUIRE_GTE(corrcoef(ff1, ff2), 0.9999);
  }

}

//---------------------------------------------------------------------
//
// FilterOps
//
//---------------------------------------------------------------------

static void Image_xx_lowPass3x_xx_1(TestSuite& suite)
{
  // To avoid bugs in involving transposing width/height values, the test
  // arrays here should be non-square.
  const float orig_data[] =
    {
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0
    };

  const Image<float> orig(orig_data, 6, 5);

  const float expected_data[] =
    {
      1.0, 1.5, 2.5, 2.5, 1.5, 1.0,
      1.0, 1.5, 2.5, 2.5, 1.5, 1.0,
      3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
      1.0, 1.5, 2.5, 2.5, 1.5, 1.0,
      1.0, 1.5, 2.5, 2.5, 1.5, 1.0
    };

  const Image<float> expected(expected_data, 6, 5);

  const Image<float> result = lowPass3x(orig);

  REQUIRE_EQ(result, expected);
}

static void Image_xx_lowPass3y_xx_1(TestSuite& suite)
{
  // To avoid bugs in involving transposing width/height values, the test
  // arrays here should be non-square.
  const float orig_data[] =
    {
      1.0, 1.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 1.0, 1.0,
      3.0, 3.0, 3.0, 3.0, 3.0,
      3.0, 3.0, 3.0, 3.0, 3.0,
      1.0, 1.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 1.0, 1.0
    };

  const Image<float> orig(orig_data, 5, 6);

  const float expected_data[] =
    {
      1.0, 1.0, 3.0, 1.0, 1.0,
      1.5, 1.5, 3.0, 1.5, 1.5,
      2.5, 2.5, 3.0, 2.5, 2.5,
      2.5, 2.5, 3.0, 2.5, 2.5,
      1.5, 1.5, 3.0, 1.5, 1.5,
      1.0, 1.0, 3.0, 1.0, 1.0
    };

  const Image<float> expected(expected_data, 5, 6);

  const Image<float> result = lowPass3y(orig);

  REQUIRE_EQ(result, expected);
}

static void Image_xx_lowPass3_xx_1(TestSuite& suite)
{
  const float orig_data[] =
    {
       6.0, 18.0,  6.0, 18.0,  6.0,
       6.0, 18.0,  6.0, 18.0,  6.0,
      18.0,  6.0, 18.0,  6.0, 18.0,
      18.0,  6.0, 18.0,  6.0, 18.0,
       6.0, 18.0,  6.0, 18.0,  6.0,
       6.0, 18.0,  6.0, 18.0,  6.0
    };

  const Image<float> orig(orig_data, 5, 6);

  const float expected_data[] =
    {
      10.0, 12.0, 12.0, 12.0, 10.0,
      11.0, 12.0, 12.0, 12.0, 11.0,
      13.0, 12.0, 12.0, 12.0, 13.0,
      13.0, 12.0, 12.0, 12.0, 13.0,
      11.0, 12.0, 12.0, 12.0, 11.0,
      10.0, 12.0, 12.0, 12.0, 10.0
    };

  const Image<float> expected(expected_data, 5, 6);

  REQUIRE_EQ(lowPass3(orig), expected);
  REQUIRE_EQ(lowPass3x(lowPass3y(orig)), expected);
  REQUIRE_EQ(lowPass3y(lowPass3x(orig)), expected);
}

static void Image_xx_lowPass5x_xx_1(TestSuite& suite)
{
  const float orig_data[] =
    {
      10, 20, 30, 20, 10, 50, 50, 50, 10, 20, 30, 20, 10,
      20, 30, 40, 30, 20, 50, 50, 50, 20, 30, 40, 30, 20,
      30, 40, 50, 40, 30, 50, 50, 50, 30, 40, 50, 40, 30,
      20, 30, 40, 30, 20, 50, 50, 50, 20, 30, 40, 30, 20,
      10, 20, 30, 20, 10, 50, 50, 50, 10, 20, 30, 20, 10,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      10, 20, 30, 20, 10, 50, 50, 50, 10, 20, 30, 20, 10,
      20, 30, 40, 30, 20, 50, 50, 50, 20, 30, 40, 30, 20,
      30, 40, 50, 40, 30, 50, 50, 50, 30, 40, 50, 40, 30,
      20, 30, 40, 30, 20, 50, 50, 50, 20, 30, 40, 30, 20,
      10, 20, 30, 20, 10, 50, 50, 50, 10, 20, 30, 20, 10,
    };

  const Image<float> orig(orig_data, 13, 11);

  // Just work with the integer portions of the results
  const byte expected_data[] =
    {
      15, 20, 22, 21, 26, 38, 45, 38, 26, 21, 22, 20, 15,
      25, 30, 32, 31, 33, 41, 46, 41, 33, 31, 32, 30, 25,
      35, 40, 42, 40, 40, 44, 47, 44, 40, 40, 42, 40, 35,
      25, 30, 32, 31, 33, 41, 46, 41, 33, 31, 32, 30, 25,
      15, 20, 22, 21, 26, 38, 45, 38, 26, 21, 22, 20, 15,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      15, 20, 22, 21, 26, 38, 45, 38, 26, 21, 22, 20, 15,
      25, 30, 32, 31, 33, 41, 46, 41, 33, 31, 32, 30, 25,
      35, 40, 42, 40, 40, 44, 47, 44, 40, 40, 42, 40, 35,
      25, 30, 32, 31, 33, 41, 46, 41, 33, 31, 32, 30, 25,
      15, 20, 22, 21, 26, 38, 45, 38, 26, 21, 22, 20, 15,
    };

  const Image<byte> expected(expected_data, 13, 11);

  REQUIRE_EQ(Image<byte>(lowPass5x(orig)), expected);
}

static void Image_xx_lowPass5y_xx_1(TestSuite& suite)
{
  const float orig_data[] =
    {
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20, 10,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30, 20,
      30, 40, 50, 40, 30, 50, 30, 40, 50, 40, 30,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30, 20,
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20, 10,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20, 10,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30, 20,
      30, 40, 50, 40, 30, 50, 30, 40, 50, 40, 30,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30, 20,
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20, 10,
    };

  const Image<float> orig(orig_data, 11, 13);

  // Just work with the integer portions of the results
  const byte expected_data[] =
    {
      15, 25, 35, 25, 15, 50, 15, 25, 35, 25, 15,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30, 20,
      22, 32, 42, 32, 22, 50, 22, 32, 42, 32, 22,
      21, 31, 40, 31, 21, 50, 21, 31, 40, 31, 21,
      26, 33, 40, 33, 26, 50, 26, 33, 40, 33, 26,
      38, 41, 44, 41, 38, 50, 38, 41, 44, 41, 38,
      45, 46, 47, 46, 45, 50, 45, 46, 47, 46, 45,
      38, 41, 44, 41, 38, 50, 38, 41, 44, 41, 38,
      26, 33, 40, 33, 26, 50, 26, 33, 40, 33, 26,
      21, 31, 40, 31, 21, 50, 21, 31, 40, 31, 21,
      22, 32, 42, 32, 22, 50, 22, 32, 42, 32, 22,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30, 20,
      15, 25, 35, 25, 15, 50, 15, 25, 35, 25, 15,
    };

  const Image<byte> expected(expected_data, 11, 13);

  REQUIRE_EQ(Image<byte>(lowPass5y(orig)), expected);
}

static void Image_xx_lowPass5_xx_1(TestSuite& suite)
{
  const float orig_data[] =
    {
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20, 10,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30, 20,
      30, 40, 50, 40, 30, 50, 30, 40, 50, 40, 30,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30, 20,
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20, 10,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20, 10,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30, 20,
      30, 40, 50, 40, 30, 50, 30, 40, 50, 40, 30,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30, 20,
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20, 10,
    };

  const Image<float> orig(orig_data, 11, 13);

  // Just work with the integer portions of the results
  const byte expected_data[] =
    {
      20, 25, 27, 26, 27, 29, 27, 26, 27, 25, 20,
      25, 30, 32, 31, 31, 32, 31, 31, 32, 30, 25,
      27, 32, 35, 33, 33, 34, 33, 33, 35, 32, 27,
      26, 31, 33, 32, 32, 33, 32, 32, 33, 31, 26,
      30, 33, 34, 34, 34, 36, 34, 34, 34, 33, 30,
      39, 41, 42, 41, 42, 42, 42, 41, 42, 41, 39,
      45, 46, 46, 46, 46, 47, 46, 46, 46, 46, 45,
      39, 41, 42, 41, 42, 42, 42, 41, 42, 41, 39,
      30, 33, 34, 34, 34, 36, 34, 34, 34, 33, 30,
      26, 31, 33, 32, 32, 33, 32, 32, 33, 31, 26,
      27, 32, 35, 33, 33, 34, 33, 33, 35, 32, 27,
      25, 30, 32, 31, 31, 32, 31, 31, 32, 30, 25,
      20, 25, 27, 26, 27, 29, 27, 26, 27, 25, 20,
    };

  const Image<byte> expected(expected_data, 11, 13);

  REQUIRE_EQ(Image<byte>(lowPass5(orig)), expected);
  REQUIRE_EQ(Image<byte>(lowPass5x(lowPass5y(orig))), expected);
  REQUIRE_EQ(Image<byte>(lowPass5y(lowPass5x(orig))), expected);
}

static void Image_xx_binomialKernel_xx_1(TestSuite& suite)
{
  float b0[] = { 1 };
  float b1[] = { 1,  1 };
  float b2[] = { 1,  2,  1 };
  float b3[] = { 1,  3,  3,  1 };
  float b4[] = { 1,  4,  6,  4,  1 };
  float b5[] = { 1,  5, 10, 10,  5,  1 };
  float b6[] = { 1,  6, 15, 20, 15,  6,  1 };
  float b7[] = { 1,  7, 21, 35, 35, 21,  7,  1 };
  float b8[] = { 1,  8, 28, 56, 70, 56, 28,  8,  1 };

  REQUIRE_EQ(Image<float>(b0, 1, 1), binomialKernel(1)*1.0f);
  REQUIRE_EQ(Image<float>(b1, 2, 1), binomialKernel(2)*2.0f);
  REQUIRE_EQ(Image<float>(b2, 3, 1), binomialKernel(3)*4.0f);
  REQUIRE_EQ(Image<float>(b3, 4, 1), binomialKernel(4)*8.0f);
  REQUIRE_EQ(Image<float>(b4, 5, 1), binomialKernel(5)*16.0f);
  REQUIRE_EQ(Image<float>(b5, 6, 1), binomialKernel(6)*32.0f);
  REQUIRE_EQ(Image<float>(b6, 7, 1), binomialKernel(7)*64.0f);
  REQUIRE_EQ(Image<float>(b7, 8, 1), binomialKernel(8)*128.0f);
  REQUIRE_EQ(Image<float>(b8, 9, 1), binomialKernel(9)*256.0f);
}

static void Image_xx_sepFilter_xx_1(TestSuite& suite)
{
  // check that an even-sized filter is equivalent to a one-larger
  // odd-sized filter with an extra 0 at the end, when applied to an
  // image that is larger than the filter size

  const Image<float> src = generateImage<float>(12, 12, Counter(0));

  const Image<float> kern8 = binomialKernel(8);
  const Image<float> kern9 = hcat(kern8, Image<float>(1,1,ZEROS)).render();

  const Image<float> f8x = sepFilter(src, kern8, Image<float>(), CONV_BOUNDARY_ZERO);
  const Image<float> f9x = sepFilter(src, kern9, Image<float>(), CONV_BOUNDARY_ZERO);

  const Image<float> f8y = sepFilter(src, Image<float>(), kern8, CONV_BOUNDARY_ZERO);
  const Image<float> f9y = sepFilter(src, Image<float>(), kern9, CONV_BOUNDARY_ZERO);

  REQUIRE_LT(rangeOf(abs(f8x-f9x)).max(), 0.5f);
  REQUIRE_LT(fabs(mean(f8x-f9x)), 0.005f);
  REQUIRE_EQFP(f8x, f9x, 0.005f);

  REQUIRE_LT(rangeOf(abs(f8y-f9y)).max(), 0.5f);
  REQUIRE_LT(fabs(mean(f8y-f9y)), 0.005f);
  REQUIRE_EQFP(f8y, f9y, 0.005f);
}

static void Image_xx_sepFilter_xx_2(TestSuite& suite)
{
  // check that CONV_BOUNDARY_ZERO and CONV_BOUNDARY_CLEAN give
  // identical results away from the boundaries

  const Image<float> src = generateImage<float>(15, 15, Counter(0));

  const Image<float> kern9 = binomialKernel(9);

  const Image<float> f9x0 = sepFilter(src, kern9, Image<float>(),
                                      CONV_BOUNDARY_ZERO);
  const Image<float> f9y0 = sepFilter(src, Image<float>(), kern9,
                                      CONV_BOUNDARY_ZERO);

  const Image<float> f9xc = sepFilter(src, kern9, Image<float>(),
                                      CONV_BOUNDARY_CLEAN);
  const Image<float> f9yc = sepFilter(src, Image<float>(), kern9,
                                      CONV_BOUNDARY_CLEAN);

  const Image<float> cutf9x0 = crop(f9x0, Point2D<int>(4,0), Dims(7,15));
  const Image<float> cutf9y0 = crop(f9y0, Point2D<int>(0,4), Dims(15,7));
  const Image<float> cutf9xc = crop(f9xc, Point2D<int>(4,0), Dims(7,15));
  const Image<float> cutf9yc = crop(f9yc, Point2D<int>(0,4), Dims(15,7));

  REQUIRE_EQFP(cutf9x0, cutf9xc, 0.0001f);
  REQUIRE_EQFP(cutf9y0, cutf9yc, 0.0001f);
}

static void Image_xx_sepFilter_xx_3(TestSuite& suite)
{
  const Image<float> src = generateImage<float>(50, 50, Counter(0));

  const Image<float> kern = binomialKernel(9);

  const Image<float> f1 = sepFilter(src, kern, kern,
                                    CONV_BOUNDARY_CLEAN);

  const Image<float> f2 = lowPass9(src);

  REQUIRE_LT(rangeOf(abs(f2-f1)).max(), 0.5f);
  REQUIRE_LT(fabs(mean(f2-f1)), 0.005f);
}

static void Image_xx_sepFilter_xx_4(TestSuite& suite)
{
  // check that an even-sized filter is equivalent to a one-larger
  // odd-sized filter with an extra 0 at the end, when applied to an
  // image that is larger than the filter size

  const Image<float> src = generateImage<float>(12, 12, Counter(0));

  const Image<float> kern8 = binomialKernel(8);
  const Image<float> kern9 = hcat(kern8, Image<float>(1,1,ZEROS)).render();

  const Image<float> f8x = sepFilter(src, kern8, Image<float>(), CONV_BOUNDARY_CLEAN);
  const Image<float> f9x = sepFilter(src, kern9, Image<float>(), CONV_BOUNDARY_CLEAN);

  const Image<float> f8y = sepFilter(src, Image<float>(), kern8, CONV_BOUNDARY_CLEAN);
  const Image<float> f9y = sepFilter(src, Image<float>(), kern9, CONV_BOUNDARY_CLEAN);

  REQUIRE_LT(rangeOf(abs(f8x-f9x)).max(), 0.5f);
  REQUIRE_LT(fabs(mean(f8x-f9x)), 0.005f);
  REQUIRE_EQFP(f8x, f9x, 0.005f);

  REQUIRE_LT(rangeOf(abs(f8y-f9y)).max(), 0.5f);
  REQUIRE_LT(fabs(mean(f8y-f9y)), 0.005f);
  REQUIRE_EQFP(f8y, f9y, 0.005f);
}

static void Image_xx_sepFilter_xx_5(TestSuite& suite)
{
  // check that an even-sized filter is equivalent to a one-larger
  // odd-sized filter with an extra 0 at the end, when applied to an
  // image that is smaller than the filter size

  const Image<float> src = generateImage<float>(7, 7, Counter(0));

  const Image<float> kern8 = binomialKernel(8);
  const Image<float> kern9 = hcat(kern8, Image<float>(1,1,ZEROS)).render();

  const Image<float> f8x = sepFilter(src, kern8, Image<float>(), CONV_BOUNDARY_CLEAN);
  const Image<float> f9x = sepFilter(src, kern9, Image<float>(), CONV_BOUNDARY_CLEAN);

  const Image<float> f8y = sepFilter(src, Image<float>(), kern8, CONV_BOUNDARY_CLEAN);
  const Image<float> f9y = sepFilter(src, Image<float>(), kern9, CONV_BOUNDARY_CLEAN);

  REQUIRE_LT(rangeOf(abs(f8x-f9x)).max(), 0.5f);
  REQUIRE_LT(fabs(mean(f8x-f9x)), 0.005f);
  REQUIRE_EQFP(f8x, f9x, 0.005f);

  REQUIRE_LT(rangeOf(abs(f8y-f9y)).max(), 0.5f);
  REQUIRE_LT(fabs(mean(f8y-f9y)), 0.005f);
  REQUIRE_EQFP(f8y, f9y, 0.005f);
}

static void Image_xx_orientedFilter_xx_1(TestSuite& suite)
{
  const float orig_data[] =
    {
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30,
      30, 40, 50, 40, 30, 50, 30, 40, 50, 40,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30,
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30,
      30, 40, 50, 40, 30, 50, 30, 40, 50, 40,
      20, 30, 40, 30, 20, 50, 20, 30, 40, 30,
      10, 20, 30, 20, 10, 50, 10, 20, 30, 20,
    };

  const Image<float> src(orig_data, 10, 13);

  const float expected_data[] = {
    233902.047,250623.719,253935.844,239278.641,238924.609,258473.656,261919.859,262513.344,281796.719,303053.625,
    225843.578,234913.422,234116.266,218271.859,208632.438,218174.469,226722.812,236762.672,256971.500,275891.656,
    200403.828,205929.266,202567.281,189035.500,177270.094,174477.828,178821.438,195386.891,218143.078,234475.688,
    102263.820,113528.578,115940.648,110839.195,107660.242,102110.594,93207.430,102330.367,122657.445,134862.062,
    91459.391,85742.062,91527.344,95810.195,98878.625,109111.586,109412.320,98036.555,97433.875,108059.281,
    251815.953,229960.188,218164.219,213880.000,209260.328,213255.656,222066.203,224363.047,229832.062,243398.000,
    320321.594,293108.562,272337.875,264862.750,260077.484,255934.797,261658.047,272536.219,284596.844,298885.781,
    272513.438,247958.188,225470.578,219610.375,220951.125,213255.656,210765.562,221831.625,233799.156,243067.531,
    127430.023,116273.438,100976.875,97668.055,109103.297,109111.586,99769.367,101574.281,107191.016,108095.719,
    94058.680,107256.539,110298.391,98822.922,92682.031,102110.594,108085.828,113659.148,125567.055,135643.500,
    198810.375,204759.812,202499.297,189836.312,177871.188,174477.828,178199.688,194368.016,217465.688,234422.422,
    238918.094,245396.422,241252.156,230916.797,225684.375,218174.484,209885.812,225632.281,255207.125,275071.062,
    253072.625,265567.594,263707.219,256049.375,260779.906,258473.656,240346.828,247880.875,279652.781,301962.344
  };
  const Image<float> expected(expected_data, 10, 13);

  const float expected_datare[] = {
    -73954.562,31536.285,305258.156,327229.906,186689.266,827527.375,105085.578,39506.438,-210786.000,-287240.656,
    -373156.281,-483831.938,-390970.969,-33109.398,156572.422,745248.562,368956.469,528518.500,504409.688,138411.406,
    -258415.766,-616607.812,-928103.875,-692339.125,-358547.625,-188067.750,159617.281,525646.688,892320.188,732832.938,
    225334.047,87022.234,-246115.750,-412436.156,-361673.344,-906619.812,-277650.219,-190245.594,108113.734,333193.406,
    182812.406,358083.469,398477.781,110242.641,-35650.812,-589856.312,-172324.484,-371650.031,-465816.969,-175818.984,
    220974.734,623095.688,877500.375,923877.562,751232.688,400494.281,-45188.574,-480158.656,-801298.312,-932475.562,
    -724454.875,-360562.125,88808.508,517125.500,822848.375,933500.000,822848.375,517125.500,88808.508,-360562.125,
    -842592.438,-932475.562,-801298.312,-480158.656,-45188.574,400494.281,751232.688,923877.562,877500.375,623095.688,
    293.893,-175818.984,-465816.969,-371650.031,-172324.484,-589856.312,-35650.812,110242.641,398477.781,358083.469,
    337541.312,333193.406,108113.734,-190245.594,-277650.219,-906619.812,-361673.344,-412436.156,-246115.750,87022.234,
    433558.719,732832.938,892320.188,525646.688,159617.281,-188067.750,-358547.625,-692339.125,-928103.875,-616607.812,
    -89531.617,138411.406,504409.688,528518.500,368956.469,745248.562,156572.422,-33109.398,-390970.969,-483831.938,
    -182930.812,-287240.656,-210786.000,39506.438,105085.578,827527.375,186689.266,327229.906,305258.156,31536.285
  };
  const Image<float> expected_re(expected_datare, 10, 13);

  const float expected_dataim[] = {
    171428.141,372065.875,469605.625,179855.891,2001.617,-431996.188,-154317.562,-371304.156,-518923.156,-238579.891,
    13488.465,282167.750,636279.750,559120.562,338987.656,562162.688,57434.102,-185419.000,-550709.625,-542728.500,
    -496923.812,-421313.438,-100227.195,279958.500,430297.094,914359.188,536874.562,530476.938,274202.312,-143757.016,
    -297745.062,-553298.375,-705079.562,-378956.969,-92843.711,222402.156,249675.641,526800.375,738932.812,450215.625,
    37901.371,-105847.977,-393608.219,-356754.969,-183264.594,-723527.312,-71841.250,36108.480,311008.906,329416.500,
    906968.812,695107.188,318457.688,-133688.016,-554140.562,-843223.938,-932405.625,-800543.500,-478897.875,-43720.777,
    588716.688,861055.875,929266.000,777176.625,440843.281,0.000,-440843.281,-777176.625,-929266.000,-861055.875,
    -401821.156,43720.777,478897.875,800543.500,932405.625,843223.938,554140.562,133688.016,-318457.688,-695107.188,
    -186699.766,-329416.500,-311008.906,-36108.480,71841.250,723527.312,183264.594,356754.969,393608.219,105847.977,
    -159666.562,-450215.625,-738932.812,-526800.375,-249675.641,-222402.156,92843.711,378956.969,705079.562,553298.375,
    354596.719,143757.016,-274202.312,-530476.938,-536874.562,-914359.188,-430297.094,-279958.500,100227.195,421313.438,
    362507.438,542728.500,550709.625,185419.000,-57434.102,-562162.688,-338987.656,-559120.562,-636279.750,-282167.750,
    37325.633,238579.891,518923.156,371304.156,154317.562,431996.188,-2001.617,-179855.891,-469605.625,-372065.875
  };
  const Image<float> expected_im(expected_dataim, 10, 13);

  const float expected_dataarg[] = {
    -43051.074,-47969.469,-52887.859,-57806.250,-62724.645,-67643.031,-72561.422,-77479.812,-82398.203,-87316.602,
    -31777.238,-36695.629,-41614.023,-46532.410,-51450.801,-56369.195,-61287.586,-66205.977,-71124.367,-76042.758,
    -20503.398,-25421.791,-30340.184,-35258.570,-40176.965,-45095.352,-50013.746,-54932.137,-59850.527,-64768.922,
    -9229.562,-14147.953,-19066.344,-23984.734,-28903.125,-33821.516,-38739.906,-43658.301,-48576.688,-53495.078,
    2044.277,-2874.114,-7792.504,-12710.896,-17629.285,-22547.676,-27466.068,-32384.461,-37302.848,-42221.242,
    13318.115,8399.725,3481.334,-1437.057,-6355.448,-11273.838,-16192.230,-21110.621,-26029.012,-30947.402,
    24591.953,19673.562,14755.173,9836.781,4918.391,0.000,-4918.391,-9836.781,-14755.173,-19673.562,
    35865.793,30947.402,26029.012,21110.621,16192.230,11273.838,6355.448,1437.057,-3481.334,-8399.725,
    47139.629,42221.242,37302.848,32384.461,27466.068,22547.676,17629.285,12710.896,7792.504,2874.114,
    58413.473,53495.078,48576.688,43658.301,38739.906,33821.516,28903.125,23984.734,19066.344,14147.953,
    69687.312,64768.922,59850.527,54932.137,50013.746,45095.352,40176.965,35258.570,30340.184,25421.791,
    80961.148,76042.758,71124.367,66205.977,61287.586,56369.195,51450.801,46532.410,41614.023,36695.629,
    92234.984,87316.602,82398.203,77479.812,72561.422,67643.031,62724.645,57806.250,52887.859,47969.469
  };
  const Image<float> expected_arg(expected_dataarg, 10, 13);

  const float expected_datacos[] = {
    -3961.144,844.571,5450.066,8763.521,9999.426,8864.782,5628.579,1058.019,-3763.364,-7692.572,
    -9993.474,-8638.313,-5235.283,-591.134,4193.155,7983.380,9880.998,9436.146,6754.281,2471.191,
    -4613.744,-8256.666,-9942.194,-9270.744,-6401.494,-2014.652,2849.800,7038.654,9558.866,9812.975,
    6034.656,1553.691,-3295.604,-7363.617,-9685.949,-9712.050,-7435.731,-3396.636,1447.693,5948.821,
    9791.773,9589.809,7114.404,2952.401,-1909.524,-6318.761,-9230.021,-9953.134,-8316.676,-4708.597,
    2367.164,6674.833,9400.111,9896.921,8047.484,4290.244,-484.077,-5143.639,-8583.807,-9989.026,
    -7760.631,-3862.476,951.350,5539.641,8814.658,10000.000,8814.658,5539.641,951.350,-3862.476,
    -9026.164,-9989.026,-8583.807,-5143.639,-484.077,4290.244,8047.484,9896.921,9400.111,6674.833,
    15.741,-4708.597,-8316.676,-9953.134,-9230.021,-6318.761,-1909.524,2952.401,7114.404,9589.809,
    9039.672,5948.821,1447.693,-3396.636,-7435.731,-9712.050,-9685.949,-7363.617,-3295.604,1553.691,
    7740.738,9812.975,9558.866,7038.654,2849.800,-2014.652,-6401.494,-9270.744,-9942.194,-8256.666,
    -2397.740,2471.191,6754.281,9436.146,9880.998,7983.380,4193.155,-591.134,-5235.283,-8638.313,
    -9798.116,-7692.572,-3763.364,1058.019,5628.579,8864.782,9999.426,8763.521,5450.066,844.571
  };
  const Image<float> expected_cos(expected_datacos, 10, 13);

  const float expected_datasin[] = {
    9182.012,9964.271,8384.317,4816.708,107.210,-4627.704,-8265.536,-9943.872,-9264.831,-6389.392,
    361.234,5037.811,8520.082,9982.513,9078.405,6022.096,1538.139,-3310.463,-7374.259,-9689.852,
    -8872.056,-5641.584,-1073.671,3748.775,7682.505,9794.957,9585.335,7103.333,2937.357,-1924.973,
    -7973.890,-9878.565,-9441.345,-6765.881,-2486.441,2382.455,6686.546,9405.470,9894.654,8038.130,
    2030.068,-2834.708,-7027.464,-9554.231,-9815.993,-7750.694,-3847.951,967.019,5552.739,8822.082,
    9715.788,7446.248,3411.438,-1432.116,-5936.160,-9032.929,-9988.276,-8575.720,-5130.133,-468.353,
    6306.552,9223.951,9954.644,8325.405,4722.478,0.000,-4722.478,-8325.405,-9954.644,-9223.951,
    -4304.458,468.353,5130.133,8575.720,9988.276,9032.929,5936.160,1432.116,-3411.438,-7446.248,
    -9999.987,-8822.082,-5552.739,-967.019,3847.951,7750.694,9815.993,9554.231,7027.464,2834.708,
    -4276.020,-8038.130,-9894.654,-9405.470,-6686.546,-2382.455,2486.441,6765.881,9441.345,9878.565,
    6330.954,1924.973,-2937.357,-7103.333,-9585.335,-9794.957,-7682.505,-3748.775,1073.671,5641.584,
    9708.287,9689.852,7374.259,3310.463,-1538.139,-6022.096,-9078.405,-9982.513,-8520.082,-5037.811,
    1999.231,6389.392,9264.831,9943.872,8265.536,4627.704,-107.210,-4816.708,-8384.317,-9964.271
  };
  const Image<float> expected_sin(expected_datasin, 10, 13);

  const float expected_datare2[] = {
    -198870.672,-170569.031,-105781.297,3016.711,131301.156,228119.469,261414.922,243822.578,207083.750,169946.781,
    -225843.203,-226634.922,-194836.891,-115253.492,-939.464,110167.797,189945.500,234423.188,251465.484,250186.500,
    -170630.672,-195557.047,-202542.375,-175012.906,-111708.742,-26743.404,64441.207,145408.375,201620.875,231070.000,
    -37747.637,-59289.730,-82312.852,-101016.750,-107654.484,-92048.188,-50309.738,3362.449,48356.617,77090.258,
    61872.816,80620.445,90513.578,78322.539,40683.930,-6542.297,-45842.941,-74071.273,-94482.938,-107893.844,
    7200.699,75335.188,148189.641,200200.141,207049.547,167023.625,93622.852,4572.647,-78962.953,-142378.016,
    -156214.453,-72886.625,35119.691,146583.953,227419.734,255934.797,229471.484,160911.203,77425.305,3687.065,
    -237189.906,-184710.141,-104918.352,-5231.254,92054.281,167023.625,208386.156,211017.281,184240.484,147178.312,
    -124733.383,-115620.148,-99428.438,-76080.328,-46203.367,-6542.297,40836.289,80799.805,101851.312,107687.352,
    78096.062,72204.789,49448.051,5076.073,-49886.719,-92048.188,-108077.398,-104294.734,-92861.250,-79724.750,
    192877.625,204758.359,193294.750,144867.641,64690.156,-26743.402,-111793.133,-177760.344,-216907.656,-231758.359,
    168181.391,205336.094,231437.406,229415.047,189495.672,110167.805,-285.233,-113552.664,-198297.938,-246859.703,
    70314.711,121817.273,181537.141,235599.938,260322.391,228119.469,132569.672,9338.555,-94736.156,-163147.797
  };
  const Image<float> expected_re2(expected_datare2, 10, 13);

  const float expected_dataim2[] = {
    123128.484,183625.859,230854.344,239259.625,199612.047,121532.492,16256.098,-97282.133,-191117.000,-250917.500,
    413.703,61813.645,129803.758,185362.453,208630.312,188316.656,123789.906,33201.414,-52909.965,-116288.109,
    -105104.086,-64531.355,-3175.899,71448.609,137643.906,172416.078,166806.594,130508.391,83279.195,39818.387,
    -95042.117,-96816.664,-81650.656,-45617.367,1113.320,44200.730,78463.711,102275.109,112723.055,110656.523,
    67354.094,29189.809,-13584.793,-55183.090,-90121.031,-108915.273,-99345.258,-64223.145,-23797.771,5977.211,
    251712.969,217270.109,160110.766,75263.273,-30337.646,-132593.688,-201365.734,-224316.453,-215841.672,-197410.953,
    279647.906,283901.688,270063.906,220602.438,126176.727,0.000,-125728.977,-219962.672,-273862.531,-298863.031,
    134180.906,165424.984,199572.344,219548.062,200861.656,132593.688,31580.619,-68417.609,-143935.734,-193443.453,
    -26076.699,-12308.285,17615.771,61244.031,98837.125,108915.273,91029.242,61552.637,33409.941,9387.089,
    -52421.746,-79312.258,-98593.242,-98692.469,-78110.648,-44200.730,-1349.855,45177.543,84521.438,109741.164,
    48205.688,771.516,-60358.137,-122683.297,-165690.500,-172416.078,-138771.125,-78614.125,-15568.865,35241.121,
    169696.406,134374.469,68112.594,-26292.568,-122575.797,-188316.656,-209885.625,-194976.203,-160650.547,-121344.070,
    243108.203,235980.281,191274.062,100269.422,-15440.636,-121532.484,-200479.125,-247704.906,-263117.344,-254094.578
  };
  const Image<float> expected_im2(expected_dataim2, 10, 13);

  // let's check that the overall filter works:
  float k = 1.23F, theta = 23.57F, intensity = 18.67F;
  Image<float> orifilt = orientedFilter(src, k, theta, intensity);
  REQUIRE_EQFP(orifilt * 1000.0F, expected, 0.001F);

  // let's do a breakdown of the various steps, taking code from
  // Image_FilterOps.C:
  double kx = double(k) * cos((theta + 90.0) * M_PI / 180.0);
  double ky = double(k) * sin((theta + 90.0) * M_PI / 180.0);
  Image<float> re(src.getDims(), NO_INIT), im(src.getDims(), NO_INIT);
  Image<float> cosi(src.getDims(), NO_INIT), sinu(src.getDims(), NO_INIT),
    argu(src.getDims(), NO_INIT);

  Image<float>::const_iterator sptr = src.begin();
  Image<float>::iterator reptr = re.beginw(), imptr = im.beginw(),
    argptr = argu.beginw(), cosptr = cosi.beginw(), sinptr = sinu.beginw();
  int w2l = src.getWidth() / 2, w2r = src.getWidth() - w2l;
  int h2l = src.getHeight() / 2, h2r = src.getHeight() - h2l;
  int total = 0;
  for (int j = -h2l; j < h2r; j ++)
    for (int i = -w2l; i < w2r; i ++)
      {
        double arg = kx * double(i) + ky * double(j);
        float val = (*sptr++) * intensity;
        *reptr++ = float(val * cos(arg));
        *imptr++ = float(val * sin(arg));

        // additional testing not part of orientedFilter() proper:
        *argptr++ = arg;
        *cosptr++ = cos(arg);
        *sinptr++ = sin(arg);

        total ++;
      }

  REQUIRE_EQ(total, 10 * 13);

  REQUIRE_EQFP(re * 1000.0F, expected_re, 0.001F);
  REQUIRE_EQFP(im * 1000.0F, expected_im, 0.001F);
  REQUIRE_EQFP(argu * 10000.0F, expected_arg, 0.001F);
  REQUIRE_EQFP(sinu * 10000.0F, expected_sin, 0.001F);
  REQUIRE_EQFP(cosi * 10000.0F, expected_cos, 0.001F);

  re = ::lowPass9(re);
  im = ::lowPass9(im);

  REQUIRE_EQFP(re * 1000.0F, expected_re2, 0.001F);
  REQUIRE_EQFP(im * 1000.0F, expected_im2, 0.001F);

  Image<float> ener = quadEnergy(re, im);
  // this result should be the same as the one of orientedFilter():
  REQUIRE_EQFP(ener * 1000.0F, expected, 0.001F);
}

static void Image_xx_gradientmag_xx_1(TestSuite& suite)
{
  // To avoid bugs in involving transposing width/height values, the test
  // arrays here should be non-square.
  const float orig_data[] =
    {
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0
    };

  const Image<float> orig(orig_data, 6, 5);

  const float sq = sqrtf(8.0f);
  const float expected_data[] =
    {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, sq,  2.0, 2.0, sq,  0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, sq,  2.0, 2.0, sq,  0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };

  const Image<float> expected(expected_data, 6, 5);

  const Image<float> result = gradientmag(orig);

  REQUIRE_EQ(result, expected);
}

static void Image_xx_gradientori_xx_1(TestSuite& suite)
{
  // To avoid bugs in involving transposing width/height values, the test
  // arrays here should be non-square.
  const float orig_data[] =
    {
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0
    };

  const Image<float> orig(orig_data, 6, 5);

  const float o000 = 0.0F;
  const float o045 = float(M_PI / 4.0);
  const float o180 = float(M_PI);
  const float o135 = float(3.0 * M_PI / 4.0);

  const float expected_data[] =
    {
      0.0,   0.0,   0.0,   0.0,   0.0, 0.0,
      0.0,  o045,  o000,  o180,  o135, 0.0,
      0.0,  o000,  o000,  o000,  o000, 0.0,
      0.0, -o045,  o000,  o180, -o135, 0.0,
      0.0,   0.0,   0.0,   0.0,   0.0, 0.0
    };

  const Image<float> expected(expected_data, 6, 5);

  const Image<float> result = gradientori(orig);

  REQUIRE_EQ(result, expected);
}

static void Image_xx_gradient_xx_1(TestSuite& suite)
{
  // To avoid bugs in involving transposing width/height values, the test
  // arrays here should be non-square.
  const float orig_data[] =
    {
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
      1.0, 1.0, 3.0, 3.0, 1.0, 1.0
    };

  const Image<float> orig(orig_data, 6, 5);

  const float sq = sqrtf(8.0f);
  const float expected_dataM[] =
    {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, sq,  2.0, 2.0, sq,  0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, sq,  2.0, 2.0, sq,  0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };

  const Image<float> expectedM(expected_dataM, 6, 5);

  const float o000 = 0.0F;
  const float o045 = float(M_PI / 4.0);
  const float o180 = float(M_PI);
  const float o135 = float(3.0 * M_PI / 4.0);

  const float expected_dataO[] =
    {
      0.0,   0.0,   0.0,   0.0,   0.0, 0.0,
      0.0,  o045,  o000,  o180,  o135, 0.0,
      0.0,  o000,  o000,  o000,  o000, 0.0,
      0.0, -o045,  o000,  o180, -o135, 0.0,
      0.0,   0.0,   0.0,   0.0,   0.0, 0.0
    };

  const Image<float> expectedO(expected_dataO, 6, 5);

  Image<float> resultM, resultO;
  gradient(orig, resultM, resultO);

  REQUIRE_EQ(resultM, expectedM);
  REQUIRE_EQ(resultO, expectedO);
}

#ifdef HAVE_FFTW3_H

static void Image_xx_fft_xx_1(TestSuite& suite)
{
  // try the convolution with all 4 possible combination of even+odd
  // filter widths+heights -- this is to test the slightly tricky
  // zero-padding involved in fourier convolution
  for (int xfiltsize = 14; xfiltsize <= 15; ++xfiltsize)
    for (int yfiltsize = 16; yfiltsize <= 17; ++yfiltsize)
      {
        Image<float> src =
          generateImage<float>(Dims(200,200),
                               rutz::urand_frange(time((time_t*)0) + getpid()));

        Image<float> boxcar(xfiltsize, yfiltsize, NO_INIT);
        std::fill(boxcar.beginw(), boxcar.endw(),
                  1.0f/(boxcar.getSize()));

        Convolver fc(boxcar, src.getDims());

        const Image<float> conv1 = fc.spatialConvolve(src);

        const Image<float> conv2 = fc.fftConvolve(src);

        REQUIRE_GT(corrcoef(conv1, conv2), 0.999);
        REQUIRE_LT(RMSerr(conv1, conv2), 1.2e-7);
      }

  {
    Image<float> x(512, 512, ZEROS);
    FourierEngine<double> eng(x.getDims());
    // just test that this doesn't crash -- buggy fftw3 installs have
    // been known to crash for certain input sizes when fftw3 is built
    // with --enable-sse2:
    (void) eng.fft(x);
  }
}

#endif // HAVE_FFTW3_H

static void Image_xx_Digest_xx_1(TestSuite& suite)
{
  REQUIRE_EQ(Digest<16>::fromString("00112233445566778899aabbccddeeff").asString(),
             std::string("00112233445566778899aabbccddeeff"));

  bool caught;

  // string too short:
  try {
    caught = false;
    //Digest<16> d = 
      Digest<16>::fromString("00112233445566778899aabbccddeef");
  } catch (lfatal_exception& e) { caught = true; }
  REQUIRE(caught);

  // string too long:
  try {
    caught = false;
    // Digest<16> d = 
      Digest<16>::fromString("00112233445566778899aabbccddeefff");
  } catch (lfatal_exception& e) { caught = true; }
  REQUIRE(caught);

  // invalid hex chars in string:
  try {
    caught = false;
    //Digest<16> d = 
      Digest<16>::fromString("00112233445566778899aabbccddeefg");
  } catch (lfatal_exception& e) { caught = true; }
  REQUIRE(caught);
}

static void Image_xx_md5_xx_1(TestSuite& suite)
{
  // test data from http://www.cr0.net:8040/code/crypto/md5/; these
  // are the standard RFC 1321 test vectors
  const char* msg[] =
    {
      "",
      "a",
      "abc",
      "message digest",
      "abcdefghijklmnopqrstuvwxyz",
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
      "12345678901234567890123456789012345678901234567890123456789012345678901234567890"
    };

  const char* val[] =
    {
      "d41d8cd98f00b204e9800998ecf8427e",
      "0cc175b9c0f1b6a831c399e269772661",
      "900150983cd24fb0d6963f7d28e17f72",
      "f96b697d7cb7938d525a2f31aaf161d0",
      "c3fcd3d76192e4007dfb496cca67e13b",
      "d174ab98d277d9f5a5611c2c9f419d9f",
      "57edf4a22be3c955ac49da2e2107b67a"
    };

  const Image<byte> img0((const byte*)msg[0], strlen(msg[0]), 1);
  const Image<byte> img1((const byte*)msg[1], strlen(msg[1]), 1);
  const Image<byte> img2((const byte*)msg[2], strlen(msg[2]), 1);
  const Image<byte> img3((const byte*)msg[3], strlen(msg[3]), 1);
  const Image<byte> img4((const byte*)msg[4], strlen(msg[4]), 1);
  const Image<byte> img5((const byte*)msg[5], strlen(msg[5]), 1);
  const Image<byte> img6((const byte*)msg[6], strlen(msg[6]), 1);

  const Digest<16> digest0 = md5byte(&img0);
  const Digest<16> digest1 = md5byte(&img1);
  const Digest<16> digest2 = md5byte(&img2);
  const Digest<16> digest3 = md5byte(&img3);
  const Digest<16> digest4 = md5byte(&img4);
  const Digest<16> digest5 = md5byte(&img5);
  const Digest<16> digest6 = md5byte(&img6);

  REQUIRE_EQ(digest0.asString(), std::string(val[0]));
  REQUIRE_EQ(digest1.asString(), std::string(val[1]));
  REQUIRE_EQ(digest2.asString(), std::string(val[2]));
  REQUIRE_EQ(digest3.asString(), std::string(val[3]));
  REQUIRE_EQ(digest4.asString(), std::string(val[4]));
  REQUIRE_EQ(digest5.asString(), std::string(val[5]));
  REQUIRE_EQ(digest6.asString(), std::string(val[6]));
}

static void Image_xx_sha1_xx_1(TestSuite& suite)
{
  // test data from http://www.cr0.net:8040/code/crypto/sha1/; these
  // are the standard FIPS-180-1 test vectors
  const char* msg[] =
    {
      "abc",
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    };

  const char* val[] =
    {
      "a9993e364706816aba3e25717850c26c9cd0d89d",
      "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
      "34aa973cd4c4daa4f61eeb2bdbad27316534016f"
    };

  const Image<byte> img0((const byte*)msg[0], strlen(msg[0]), 1);
  const Image<byte> img1((const byte*)msg[1], strlen(msg[1]), 1);
  Image<byte> img2(Dims(1000000, 1), NO_INIT);
  img2.clear('a');

  const Digest<20> digest0 = sha1byte(&img0);
  const Digest<20> digest1 = sha1byte(&img1);
  const Digest<20> digest2 = sha1byte(&img2);

  REQUIRE_EQ(digest0.asString(), std::string(val[0]));
  REQUIRE_EQ(digest1.asString(), std::string(val[1]));
  REQUIRE_EQ(digest2.asString(), std::string(val[2]));
}

static void Image_xx_sha256_xx_1(TestSuite& suite)
{
  // test data from http://www.cr0.net:8040/code/crypto/sha2/; these
  // are the standard FIPS-180-1 test vectors
  const char* msg[] =
    {
      "abc",
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    };

  const char* val[] =
    {
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    };

  const Image<byte> img0((const byte*)msg[0], strlen(msg[0]), 1);
  const Image<byte> img1((const byte*)msg[1], strlen(msg[1]), 1);
  Image<byte> img2(1000000, 1, NO_INIT);
  img2.clear(byte('a'));

  const Digest<32> digest0 = sha256byte(&img0);
  const Digest<32> digest1 = sha256byte(&img1);
  const Digest<32> digest2 = sha256byte(&img2);

  REQUIRE_EQ(digest0.asString(), std::string(val[0]));
  REQUIRE_EQ(digest1.asString(), std::string(val[1]));
  REQUIRE_EQ(digest2.asString(), std::string(val[2]));
}

//---------------------------------------------------------------------
//
// DrawOps
//
//---------------------------------------------------------------------

static void Image_xx_warp3D_xx_1(TestSuite& suite)
{
  const Image<PixRGB<byte> > ima =
    generateImage<byte>(Dims(11, 11), Counter(5, 2));

  const float z_[] = {
    25.1, 10.2,   .3,   .4,   .5,   .6,   .7,   .8,   .9,   .0,   .1,
    10.2,  5.3,  1.4,   .5,   .6,   .7,   .8,   .9,  3.0,   .1,   .2,
     5.3,  1.4,   .5,   .6,   .7,   .8,   .9,  4.0,  8.1,  3.2,   .3,
     5.4,  2.5,  6.6,   .7,   .8,   .9,  9.0,   .1,  3.2,   .3,   .4,
     4.5,  1.6, 12.7,  2.8,   .9,   .0,   .1,   .2,   .3,   .4,   .5,
     3.6,   .7,  6.8,   .9,  2.0,  9.1, 15.2,   .3, 18.4, 24.5, 15.6,
     2.7,   .8,   .9,   .0,   .1,  1.2,   .3,  9.4,   .5,   .6,   .7,
     1.8,   .9,   .0,   .1,  9.2,   .3,  2.4,   .5,   .6,   .7,   .8,
     1.9,   .0,   .1,  9.2,   .3,   .4,   .5,  1.6,   .7,   .8,   .9,
    15.0,  3.1,  5.2,  6.3,  8.4,  9.5, 11.6, 12.7, 14.8, 15.9, 17.0,
    10.1,  2.2,  3.3,  4.4,  5.5,  6.6,  7.7,  8.8,  9.9, 10.0, 11.1,
  };

  Image<float> z(&z_[0], 11, 11);

  const byte expected_w1_[] = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   5,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  29,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  29,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  29,   0,   0,   0,   0,   0,   0, 133,   0,   0,   0,
    0,   0,   0,  29,   0,   0,   0,   0,   0,   0, 157,   0,   0,   0,
    0,   0,   0,  29,   0,   0,   0,   0,   0,   0, 157,   0,   0,   0,
    0,   0,   0,  29,   0,   0,   0,   0,   0, 133, 157,   0,   0,   0,
    0,   0,   0,  29,   0,   0,   0,   0,   0, 133, 157,   0,   0,   0,
    0,   0,   0,  29,   0,   0,   0,   0,   0, 133, 157, 135,   0,   0,
    0,   0,   0,  29,   0,   0,   0,   0,   0, 131, 157, 135,   0,   0,
    0,   0,   0,  29,   0,   0,   0,   0,   0, 155, 157, 135,   0,   0,
    0,   0,   0,  29,   0,   0,   0, 127,   0, 155, 157, 135,   0,   0,
    0,   0,   0,  29,   0,   0,   0, 151,   0, 155, 223, 157,   0,   0,
    0, 203,   0,  97,   0,   0,   0, 151,   0, 221, 245, 157,   0,   0,
    0, 227,   0, 121,   0,   0, 127, 151, 219, 245, 245, 157,   0,   0,
    0, 227,   0, 121,   0,   0, 127, 217, 243, 245, 245, 157,   0,   0,
    0, 227,   0, 121,   0,   0, 215, 241, 243, 243, 245, 157,   0,   0,
    0, 227,   0, 121,   0, 215, 239, 239, 241, 243, 245, 157,   0,   0,
    0, 227, 205, 187,  99, 213, 237, 239, 241, 243, 245, 157,   0,   0,
    0, 227, 227, 209, 211, 237, 237, 239, 175, 243, 155, 157,   0,   0,
    0, 227, 227, 211, 235, 235, 237, 149, 175, 153, 155, 157,   0,   0,
    0,   0, 227, 209, 233, 235, 147, 151, 175, 153, 155, 157,   0,   0,
    0,   0, 227, 231, 233, 191, 169, 173, 175, 153, 155, 157,   0,   0,
    0, 181, 227, 229, 189, 191, 171, 173, 175, 153, 155, 157,  69,   0,
    0, 183, 227, 187, 189, 191, 193, 195, 175, 153, 155, 157,   0,   0,
    0,   0, 183, 185, 189, 191, 193, 173, 197, 199, 201,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
  };
  const Image<byte> expected_w1(&expected_w1_[0], 14, 43);

  const byte expected_w2_[] = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0, 203, 225,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0, 205, 227,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0, 227, 227,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0, 205, 227, 231, 233, 235, 237, 239, 241, 223,   0,   0,
    0,   0,   0,   0, 181,   0, 187, 189,   0,   0,   0, 133,   0, 243, 245,   0,
    0,   5,   0, 159,   0, 183, 185, 167, 189,   0, 133, 157,   0,   0,   0,   0,
    0,  29,   0, 137, 161, 163, 167, 169, 191,   0, 131, 157, 135,   0,   0,   0,
    0,  29,  93,  97, 121, 141, 165, 125, 169, 193, 195, 157, 135,   0,   0,   0,
    0,  49,  73,  95,  75, 121, 123, 149, 151, 153, 173, 197, 199, 201,   0,   0,
    0,  51,  29,  73,  77,   0, 101,  83, 149, 131, 153, 155, 177, 179,   0,   0,
    0,   9,  51,  53,  77,  79,  83, 103, 107, 129,   0, 155, 157,   0,   0,   0,
    0,   0,   9,  31,  55,  57,  81,  61,  85, 109, 111, 113,   0,   0,   0,   0,
    0,   0,   0,  11,  13,  15,  39,  41,  43,   0,  67,  91,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,  17,  19,   0,  45,  69,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,  21,  23,  47,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  25,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  };
  const Image<byte> expected_w2(&expected_w2_[0], 16, 32);

  float pitch, yaw;
  Dims d;

  pitch = -25.0f;
  yaw = -15.0f;
  d = Dims();
  const Image<byte> w1 = luminance(warp3D(ima, z, pitch, yaw, 25.1f, d));

  pitch = +75.0f;
  yaw = +25.0f;
  d = Dims();
  const Image<byte> w2 = luminance(warp3D(ima, z, pitch, yaw, 25.1f, d));

  REQUIRE_EQ(w2, expected_w2);
}

//---------------------------------------------------------------------
//
// Transforms
//
//---------------------------------------------------------------------

static void Image_xx_makeBinary_xx_1(TestSuite& suite)
{
  const byte data[6] = { 0, 1, 63, 64, 254, 255 };
  const byte bindata[6] = { 20, 20, 20, 10, 10, 10 };

  Image<byte> img(data, 6, 1);
  Image<byte> binimg(bindata, 6, 1);

  REQUIRE_EQ(makeBinary(img, byte(63), byte(20), byte(10)), binimg);
}

static void Image_xx_contour2D_xx_1(TestSuite& suite)
{
  byte orig_data[] =
    {
      0, 1, 0, 0, 0,
      1, 1, 1, 1, 0,
      0, 1, 1, 0, 0,
      0, 1, 0, 1, 0,
      0, 0, 0, 0, 0
    };

  byte contour_data[] =
    {
      0,   255, 0,   0,   0,
      255, 0,   255, 255, 0,
      0,   255, 255, 0,   0,
      0,   255, 0,   255, 0,
      0,   0,   0,   0,   0
    };

  Image<byte> orig(orig_data, 5, 5);

  Image<byte> contour(contour_data, 5, 5);

  Image<byte> c2d = contour2D(orig);

  REQUIRE_EQ(c2d, contour);
}

//--------------------------------------------------------------------
//
// Point2d
//
//-------------------------------------------------------------------

static void Point2D_xx_testConstructAdd_xx_1(TestSuite& suite)
{

    Point2D<int> pi(1,1);
    Point2D<float> pf(2.0,2.0);
    Point2D<double> pd(3.9,3.9);
    pi += 2;
    pf +=2.5;
    pd +=1.0;
    REQUIRE_EQ(pi.i,3);
    REQUIRE_EQ(pf.i,4.5);
    REQUIRE_EQ(pd.i,4.9);
}


static void Point2D_xx_testConvertType_xx_1(TestSuite& suite)
{

    Point2D<float> pf(1.0,1.0);
    Point2D<double>pdf(pf);
    Point2D<int>   adder(1,1);

    pdf = pdf + adder;

    REQUIRE_EQ(pdf.i,2.0);

    REQUIRE_EQ(sizeof(pdf+adder), sizeof(Point2D<double>));

    Point2D<int> p2(300, -450);
    Point2D<byte> p3(p2);
    REQUIRE_EQ(p3.i, byte(255));
    REQUIRE_EQ(p3.j, byte(0));
}




///////////////////////////////////////////////////////////////////////
//
// main
//
///////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  TestSuite suite;

  suite.ADD_TEST(Rectangle_xx_getOverlap_xx_1);
  suite.ADD_TEST(Rectangle_xx_constrain_xx_1);

  suite.ADD_TEST(Image_xx_construct_from_array_xx_1);
  suite.ADD_TEST(Image_xx_construct_UDT_xx_1);
  suite.ADD_TEST(Image_xx_construct_and_clear_xx_1);
  suite.ADD_TEST(Image_xx_default_construct_xx_1);
  suite.ADD_TEST(Image_xx_swap_xx_1);
  suite.ADD_TEST(Image_xx_copy_xx_1);
  suite.ADD_TEST(Image_xx_copy_on_write_xx_1);
  suite.ADD_TEST(Image_xx_copy_on_write_xx_2);
  suite.ADD_TEST(Image_xx_copy_on_write_xx_3);
  suite.ADD_TEST(Image_xx_assignment_to_self_xx_1);
  suite.ADD_TEST(Image_xx_reshape_xx_1);
  suite.ADD_TEST(Image_xx_indexing_xx_1);
  suite.ADD_TEST(Image_xx_type_convert_xx_1);
  suite.ADD_TEST(Image_xx_type_convert_xx_2);
  suite.ADD_TEST(Image_xx_type_convert_xx_3);
  suite.ADD_TEST(Image_xx_attach_detach_xx_1);
  suite.ADD_TEST(Image_xx_destruct_xx_1);
  suite.ADD_TEST(Image_xx_begin_end_xx_1);
  suite.ADD_TEST(Image_xx_beginw_endw_xx_1);
  suite.ADD_TEST(Image_xx_getMean_xx_1);
  suite.ADD_TEST(Image_xx_clear_xx_1);

  suite.ADD_TEST(Image_xx_plus_eq_scalar_xx_1);
  suite.ADD_TEST(Image_xx_minus_eq_scalar_xx_1);
  suite.ADD_TEST(Image_xx_mul_eq_scalar_xx_1);
  suite.ADD_TEST(Image_xx_div_eq_scalar_xx_1);
  suite.ADD_TEST(Image_xx_lshift_eq_scalar_xx_1);
  suite.ADD_TEST(Image_xx_rshift_eq_scalar_xx_1);

  suite.ADD_TEST(Image_xx_plus_eq_array_xx_1);
  suite.ADD_TEST(Image_xx_minus_eq_array_xx_1);
  suite.ADD_TEST(Image_xx_mul_eq_array_xx_1);
  suite.ADD_TEST(Image_xx_div_eq_array_xx_1);

  suite.ADD_TEST(Image_xx_plus_scalar_xx_1);
  suite.ADD_TEST(Image_xx_minus_scalar_xx_1);
  suite.ADD_TEST(Image_xx_mul_scalar_xx_1);
  suite.ADD_TEST(Image_xx_div_scalar_xx_1);
  suite.ADD_TEST(Image_xx_lshift_scalar_xx_1);
  suite.ADD_TEST(Image_xx_rshift_scalar_xx_1);

  suite.ADD_TEST(Image_xx_plus_array_xx_1);
  suite.ADD_TEST(Image_xx_minus_array_xx_1);
  suite.ADD_TEST(Image_xx_minus_array_xx_2);
  suite.ADD_TEST(Image_xx_mul_array_xx_1);
  suite.ADD_TEST(Image_xx_div_array_xx_1);

  suite.ADD_TEST(Image_xx_row_ops_xx_1);

  suite.ADD_TEST(Image_xx_emptyArea_xx_1);

  // MathOps
  suite.ADD_TEST(Image_xx_mean_xx_1);
  suite.ADD_TEST(Image_xx_sum_xx_1);
  suite.ADD_TEST(Image_xx_sum_xx_2);
  suite.ADD_TEST(Image_xx_rangeOf_xx_1);
  suite.ADD_TEST(Image_xx_rangeOf_xx_2);
  suite.ADD_TEST(Image_xx_remapRange_xx_1);
  suite.ADD_TEST(Image_xx_squared_xx_1);
  suite.ADD_TEST(Image_xx_toPower_xx_1);
  suite.ADD_TEST(Image_xx_toPower_xx_2);
  suite.ADD_TEST(Image_xx_quadEnergy_xx_1);
  suite.ADD_TEST(Image_xx_overlay_xx_1);
  suite.ADD_TEST(Image_xx_exp_xx_1);
  suite.ADD_TEST(Image_xx_log_xx_1);
  suite.ADD_TEST(Image_xx_log10_xx_1);
  suite.ADD_TEST(Image_xx_getMaskedMinMax_xx_1);
  suite.ADD_TEST(Image_xx_getMaskedMinMaxAvg_xx_1);

  // MatrixOps
  suite.ADD_TEST(Image_xx_vmMult_xx_1);
  suite.ADD_TEST(Image_xx_vmMult_xx_2);
  suite.ADD_TEST(Image_xx_matrixMult_xx_1);
  suite.ADD_TEST(Image_xx_matrixMult_xx_2);
  suite.ADD_TEST(Image_xx_matrixMult_xx_3);
  suite.ADD_TEST(Image_xx_matrixMult_xx_4);
  suite.ADD_TEST(Image_xx_matrixInv_xx_1);
  suite.ADD_TEST(Image_xx_matrixInv_xx_2);
  suite.ADD_TEST(Image_xx_matrixInv_xx_3);
  suite.ADD_TEST(Image_xx_matrixDet_xx_1);
  suite.ADD_TEST(Image_xx_matrixDet_xx_2);
  suite.ADD_TEST(Image_xx_matrixDet_xx_3);
  suite.ADD_TEST(Image_xx_svd_gsl_xx_1);
  suite.ADD_TEST(Image_xx_svd_lapack_xx_1);
  suite.ADD_TEST(Image_xx_svd_lapack_xx_2);
  suite.ADD_TEST(Image_xx_svd_full_xx_1);

  // ShapeOps
  suite.ADD_TEST(Image_xx_rescale_xx_1);
  suite.ADD_TEST(Image_xx_rescale_xx_2);
  suite.ADD_TEST(Image_xx_rescale_xx_3);
  suite.ADD_TEST(Image_xx_rescale_xx_4);
  suite.ADD_TEST(Image_xx_rescale_xx_5);
  suite.ADD_TEST(Image_xx_zoomXY_xx_1);
  suite.ADD_TEST(Image_xx_zoomXY_xx_2);
  suite.ADD_TEST(Image_xx_zoomXY_xx_3);
  suite.ADD_TEST(Image_xx_zoomXY_xx_4);
  suite.ADD_TEST(Image_xx_interpolate_xx_1);
  suite.ADD_TEST(Image_xx_decY_xx_1);
  suite.ADD_TEST(Image_xx_blurAndDecY_xx_1);

  // CutPaste
  suite.ADD_TEST(Image_xx_concatX_xx_1);
  suite.ADD_TEST(Image_xx_concatY_xx_1);
  suite.ADD_TEST(Image_xx_concatLooseX_xx_1);
  suite.ADD_TEST(Image_xx_concatLooseY_xx_1);
  suite.ADD_TEST(Image_xx_crop_xx_1);

  // FilterOps
  suite.ADD_TEST(Image_xx_lowPass3x_xx_1);
  suite.ADD_TEST(Image_xx_lowPass3y_xx_1);
  suite.ADD_TEST(Image_xx_lowPass3_xx_1);
  suite.ADD_TEST(Image_xx_lowPass5x_xx_1);
  suite.ADD_TEST(Image_xx_lowPass5y_xx_1);
  suite.ADD_TEST(Image_xx_lowPass5_xx_1);
  suite.ADD_TEST(Image_xx_binomialKernel_xx_1);
  suite.ADD_TEST(Image_xx_sepFilter_xx_1);
  suite.ADD_TEST(Image_xx_sepFilter_xx_2);
  suite.ADD_TEST(Image_xx_sepFilter_xx_3);
  suite.ADD_TEST(Image_xx_sepFilter_xx_4);
  suite.ADD_TEST(Image_xx_sepFilter_xx_5);
  suite.ADD_TEST(Image_xx_orientedFilter_xx_1);
  suite.ADD_TEST(Image_xx_gradientmag_xx_1);
  suite.ADD_TEST(Image_xx_gradientori_xx_1);
  suite.ADD_TEST(Image_xx_gradient_xx_1);

#ifdef HAVE_FFTW3_H
  suite.ADD_TEST(Image_xx_fft_xx_1);
#endif // HAVE_FFTW3_H

  suite.ADD_TEST(Image_xx_Digest_xx_1);
  suite.ADD_TEST(Image_xx_md5_xx_1);
  suite.ADD_TEST(Image_xx_sha1_xx_1);
  suite.ADD_TEST(Image_xx_sha256_xx_1);

  // DrawOps
  suite.ADD_TEST(Image_xx_warp3D_xx_1);

  // Transforms
  suite.ADD_TEST(Image_xx_makeBinary_xx_1);
  suite.ADD_TEST(Image_xx_contour2D_xx_1);

  // Point2d
  suite.ADD_TEST(Point2D_xx_testConstructAdd_xx_1);
  suite.ADD_TEST(Point2D_xx_testConvertType_xx_1);


  suite.parseAndRun(argc, argv);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
