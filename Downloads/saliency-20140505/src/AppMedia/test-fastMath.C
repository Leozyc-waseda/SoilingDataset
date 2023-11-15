/*!@file AppMedia/test-fastMath.C
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
// Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/test-fastMath.C $
// $Id: test-fastMath.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Util/log.H"
#include "Util/MathFunctions.H"
#include "Util/Timer.H"
#include <stdio.h>
#include <stdlib.h>

int main(const int argc, const char** argv)
{
  const int large_samples = (int)pow(2,8);
  const int loops         = (int)pow(2,8);
  const int ticks_per_sec = 1000000;
  LINFO("Using %d large samples",large_samples);
  LINFO("Using %d loops",loops);
  double DLA[large_samples + 1];
  float  FLA[large_samples + 1];
  int    ILA[large_samples + 1];

  {
    for(int i = 0; i < large_samples; i++)
    {
      DLA[i] = pow(rand(),10*rand());
      FLA[i] = (float)DLA[i];
      ILA[i] = static_cast<int>(round(DLA[i]));
    }
  }

  Timer tim(ticks_per_sec);
  LINFO("Ticks per second %d",tim.ticksPerSec());
  tim.reset();
  int t1,t2,t3;

  // ######################################################################
  {
    float store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += sqrt(FLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for float SQRT - %d mcs (result %f)",t3,store);
  }

  // ######################################################################
  {
    double store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0F;
        for(int i = 0; i < large_samples; i++)
        {
          store += sqrt(DLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for double SQRT - %d mcs (result %f)",t3,store);
  }

  LINFO("<<<NEXT>>>");

  // ######################################################################
  {
    LINFO("fastSqrt_2 : Log 2 Approx Fast Square Root - Float");
    const int samples = 7;

    float A[samples];
    float F[samples];
    float N[samples];

    A[0] = 1.0f;
    A[1] = 2.0f;
    A[2] = 8.0f;
    A[3] = 100.0f;
    A[4] = M_PI;
    A[5] = 100000;
    A[6] = 1.0f/3.0f;

    for(int i = 0; i < samples; i++)
    {
      F[i] = fastSqrt_2(A[i]);
      N[i] = sqrt(A[i]);
      LINFO("\t%d SQRT %f - Log 2 Fast %f Normal %f",i,A[i],F[i],N[i]);
    }

    float store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastSqrt_2(FLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for Log 2 fast float SQRT - %d mcs (result %f)",t3,store);
  }

  LINFO("<<<NEXT>>>");

  // ######################################################################
  {
    LINFO("fastSqrt_Bab : Log 2 Babylonian Approx Fast Square Root - Float");
    const int samples = 7;

    float A[samples];
    float F[samples];
    float N[samples];

    A[0] = 1.0f;
    A[1] = 2.0f;
    A[2] = 8.0f;
    A[3] = 100.0f;
    A[4] = M_PI;
    A[5] = 100000;
    A[6] = 1.0f/3.0f;

    for(int i = 0; i < samples; i++)
    {
      F[i] = fastSqrt_Bab(A[i]);
      N[i] = sqrt(A[i]);
      LINFO("\t%d SQRT %f - Log 2 Fast Babylonian %f Normal %f",
            i,A[i],F[i],N[i]);
    }

    float store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastSqrt_Bab(FLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for Log 2 fast Babylonian float SQRT - %d mcs (result %f)",t3,store);
  }

  LINFO("<<<NEXT>>>");

  // ######################################################################
  {
    LINFO("fastSqrt_Q3 : Quake 3 Fast Square Root - Float");
    const int samples = 7;

    float A[samples];
    float F[samples];
    float N[samples];

    A[0] = 1.0f;
    A[1] = 2.0f;
    A[2] = 8.0f;
    A[3] = 100.0f;
    A[4] = M_PI;
    A[5] = 100000;
    A[6] = 1.0f/3.0f;

    for(int i = 0; i < samples; i++)
    {
      F[i] = fastSqrt_Q3(A[i]);
      N[i] = sqrt(A[i]);
      LINFO("\t%d SQRT %f - Fast %f Normal %f",i,A[i],F[i],N[i]);
    }

    float store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastSqrt_Q3(FLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for fast float SQRT - %d mcs (result %f)",t3,store);
  }

  LINFO("<<<NEXT>>>");

  // ######################################################################
  {
    LINFO("fastSqrt_Bab_2 : Log 2 Babylonian 2 Approx Fast Square Root - Float");
    const int samples = 7;

    float A[samples];
    float F[samples];
    float N[samples];

    A[0] = 1.0f;
    A[1] = 2.0f;
    A[2] = 8.0f;
    A[3] = 100.0f;
    A[4] = M_PI;
    A[5] = 100000;
    A[6] = 1.0f/3.0f;

    for(int i = 0; i < samples; i++)
    {
      F[i] = fastSqrt_Bab_2(A[i]);
      N[i] = sqrt(A[i]);
      LINFO("\t%d SQRT %f - Log 2 Fast Babylonian %f Normal %f",
            i,A[i],F[i],N[i]);
    }

    float store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastSqrt_Bab_2(FLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for Log 2 fast Babylonian float SQRT - %d mcs (result %f)",t3,store);
  }

  LINFO("<<<NEXT>>>");

  // ######################################################################
  {
    LINFO("fastSqrt_2 : Log 2 Approx Fast Square Root - Double");
    const int samples = 7;

    double A[samples];
    double F[samples];
    double N[samples];

    A[0] = 1.0F;
    A[1] = 2.0F;
    A[2] = 8.0F;
    A[3] = 100.0F;
    A[4] = M_PI;
    A[5] = 100000;
    A[6] = 1.0F/3.0F;

    for(int i = 0; i < samples; i++)
    {
      F[i] = fastSqrt_2(A[i]);
      N[i] = sqrt(A[i]);
      LINFO("\t%d SQRT %f - Fast %f Normal %f",i,A[i],F[i],N[i]);
    }

    double store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0F;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastSqrt_2(DLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for fast double SQRT - %d mcs (result %f)",t3,store);
  }

  LINFO("<<<NEXT>>>");

  // ######################################################################
  {
    LINFO("fastSqrt_Bab : Log 2 Babylonian Approx Fast Square Root - Double");
    const int samples = 7;

    double A[samples];
    double F[samples];
    double N[samples];

    A[0] = 1.0f;
    A[1] = 2.0f;
    A[2] = 8.0f;
    A[3] = 100.0f;
    A[4] = M_PI;
    A[5] = 100000;
    A[6] = 1.0f/3.0f;

    for(int i = 0; i < samples; i++)
    {
      F[i] = fastSqrt_Bab(A[i]);
      N[i] = sqrt(A[i]);
      LINFO("\t%d SQRT %f - Log 2 Fast Babylonian %f Normal %f",
            i,A[i],F[i],N[i]);
    }

    double store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastSqrt_Bab(DLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for Log 2 fast Babylonian double SQRT - %d mcs (result %f)",t3,store);
  }

  LINFO("<<<NEXT>>>");

  // ######################################################################
  {
    LINFO("fastSqrt_Q3 : Quake 3 Fast Square Root - Double");
    const int samples = 7;

    double A[samples];
    double F[samples];
    double N[samples];

    A[0] = 1.0F;
    A[1] = 2.0F;
    A[2] = 8.0F;
    A[3] = 100.0F;
    A[4] = M_PI;
    A[5] = 100000;
    A[6] = 1.0F/3.0F;

    for(int i = 0; i < samples; i++)
    {
      F[i] = fastSqrt_Q3(A[i]);
      N[i] = sqrt(A[i]);
      LINFO("\t%d SQRT %f - Fast %f Normal %f",i,A[i],F[i],N[i]);
    }

    double store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0F;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastSqrt_Q3(DLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for fast double SQRT - %d mcs (result %f)",t3,store);
  }
  LINFO("<<<NEXT>>>");

  // ######################################################################
  {
    LINFO("fastSqrt_Bab_2 : Log 2 Babylonian 2 Approx Fast Square Root - double");
    const int samples = 7;

    double A[samples];
    double F[samples];
    double N[samples];

    A[0] = 1.0f;
    A[1] = 2.0f;
    A[2] = 8.0f;
    A[3] = 100.0f;
    A[4] = M_PI;
    A[5] = 100000;
    A[6] = 1.0f/3.0f;

    for(int i = 0; i < samples; i++)
    {
      F[i] = fastSqrt_Bab_2(A[i]);
      N[i] = sqrt(A[i]);
      LINFO("\t%d SQRT %f - Log 2 Fast Babylonian %f Normal %f",
            i,A[i],F[i],N[i]);
    }

    double store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastSqrt_Bab_2(DLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for Log 2 fast Babylonian float SQRT - %d mcs (result %f)",t3,store);
  }

  // ######################################################################
  {
    LINFO("fastLog : Fast Log - double");
    const int samples = 7;

    double A[samples];
    double F[samples];
    double N[samples];

    A[0] = 1.0f;
    A[1] = 2.0f;
    A[2] = 8.0f;
    A[3] = 100.0f;
    A[4] = M_PI;
    A[5] = 100000;
    A[6] = 1.0f/3.0f;

    for(int i = 0; i < samples; i++)
    {
      F[i] = fastLog(A[i]);
      N[i] = log(A[i]);
      LINFO("\t%d Log %f - Fast Log %f Normal %f",
            i,A[i],F[i],N[i]);
    }

    double store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastLog(DLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time for fastLog - %d mcs (result %f)",t3,store);
  }


  // ######################################################################
  LINFO("<<<NEXT - CHECKING IMPLEMENTATION>>>");
  {
    int store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0;
        for(int i = 0; i < large_samples; i++)
        {
          store += (int)sqrt(ILA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time Int SQRT BASELINE - %d mcs (result %d)",t3,store);
  }

  {
    int store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastSqrt(ILA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time Int SQRT - %d mcs (result %d)",t3,store);
  }

  {
    float store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += sqrt(FLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time Float SQRT BASELINE - %d mcs (result %f)",t3,store);
  }

  {
    float store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastSqrt(FLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time Float SQRT - %d mcs (result %f)",t3,store);
  }

  {
    double store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += log(DLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time Log BASELINE - %d mcs (result %f)",t3,store);
  }

  {
    double store;
    t1 = tim.get();
    for(int l = 0; l < loops; l++)
    {
      for(int k = 0; k < loops; k++)
      {
        store = 0.0f;
        for(int i = 0; i < large_samples; i++)
        {
          store += fastLog(DLA[i]);
        }
      }
    }
    t2 = tim.get();
    t3 = t2 - t1;
    LINFO(">>> Time FastLog - %d mcs (result %f)",t3,store);
  }
  return 1;
}

