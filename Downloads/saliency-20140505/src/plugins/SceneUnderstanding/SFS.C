/*!@file SceneUnderstanding/SFS.C  Shape from shading */


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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/SFS.C $
// $Id: SFS.C 13551 2010-06-10 21:56:32Z itti $
//

#ifndef SFS_C_DEFINED
#define SFS_C_DEFINED

#include "plugins/SceneUnderstanding/SFS.H"

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Layout.H"
#include "Simulation/SimEventQueue.H"
#include "Simulation/SimEvents.H"
#include "Media/MediaSimEvents.H"
#include "Channels/InputFrame.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Convolutions.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_SFS = {
  MOC_SORTPRI_3,   "SFS-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_SFSShowDebug =
  { MODOPT_ARG(bool), "SFSShowDebug", &MOC_SFS, OPTEXP_CORE,
    "Show debug img",
    "sfs-debug", '\0', "<true|false>", "false" };


// ######################################################################
SFS::SFS(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventInputFrame),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  itsShowDebug(&OPT_SFSShowDebug, this),
  itsInitialized(false),
  itsNumIter(10),
  itsPs(52.46),
  itsQs(11.73)

{
}

// ######################################################################
SFS::~SFS()
{

}

// ######################################################################
void SFS::onSimEventInputFrame(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventInputFrame>& e)
{

}

// ######################################################################
void SFS::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "SFS", FrameInfo("SFS", SRC_POS));
    }
}


// ######################################################################
void SFS::evolve()
{

}

// ######################################################################
void SFS::evolve(const Image<byte>& img)
{

  Image<float> Zn1(img.getDims(), NO_INIT);
  Image<float> Si1(img.getDims(), NO_INIT);

  /* assume the initial estimate zero at time n-1 */
  for(int i=0;i<img.getWidth();i++)
    for(int j=0;j<img.getHeight();j++)
    {
      Zn1.setVal(i,j,0.0);
      Si1.setVal(i,j, 0.01);
    }

  double Wn=0.0001*0.0001;

  for(int iter=0; iter<itsNumIter; iter++)
  {
    Image<float> Zn(img.getDims(), NO_INIT);
    Image<float> Si(img.getDims(), NO_INIT);

    for(int i=0;i<img.getWidth();i++)
      for(int j=0;j<img.getHeight();j++)
      {
        double p,q;
        if(j-1 < 0 || i-1 < 0) /* take care boundary */
          p = q = 0.0;
        else {
          p = Zn1.getVal(i,j) - Zn1.getVal(i,j-1);
          q = Zn1.getVal(i,j) - Zn1.getVal(i-1,j);
        }
        double pq = 1.0 + p*p + q*q;
        double PQs = 1.0 + itsPs*itsPs + itsQs*itsQs;
        double Eij = img.getVal(i,j)/255.0;
        double fZ = -1.0*(Eij - std::max(0.0,(1+p*itsPs+q*itsQs)/(sqrt(pq)*sqrt(PQs))));
        double dfZ = -1.0*( (itsPs+itsQs)/(sqrt(pq)*sqrt(PQs)) -
                            (p+q)*(1.0+p*itsPs+q*itsQs) /
                            (sqrt(pq*pq*pq)*sqrt(PQs)));
        double Y = fZ + dfZ*Zn1.getVal(i,j);
        double K = Si1.getVal(i,j)*dfZ/(Wn+dfZ*Si1.getVal(i,j)*dfZ);
        Si.setVal(i,j,  (1.0 - K*dfZ)*Si1.getVal(i,j));
        Zn.setVal(i,j, Zn1.getVal(i,j) + K*(Y-dfZ*Zn1.getVal(i,j)));
      }

    Image<float> height = Zn;
    inplaceNormalize(height, 0.0F, 100.0F);
    Dims dims;
    Image<PixRGB<byte> > dImg = img;
    Image<PixRGB<byte> > pImg = warp3D(dImg, height, 70.0, 40.0, 150.0F, dims);
    SHOWIMG(pImg);

    for(int i=0;i<img.getWidth();i++)
      for(int j=0;j<img.getHeight();j++)
      {
        Zn1.setVal(i,j, Zn.getVal(i,j));
        Si1.setVal(i,j, Si.getVal(i,j));
      }
  }

}

Layout<PixRGB<byte> > SFS::getDebugImage()
{

  Layout<PixRGB<byte> > disp;

  return disp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

