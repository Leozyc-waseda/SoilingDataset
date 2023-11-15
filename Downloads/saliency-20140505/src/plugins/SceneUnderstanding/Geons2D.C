/*!@file SceneUnderstanding/Geons2D.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Geons2D.C $
// $Id: Geons2D.C 13551 2010-06-10 21:56:32Z itti $
//

#ifndef Geons2D_C_DEFINED
#define Geons2D_C_DEFINED

#include "plugins/SceneUnderstanding/Geons2D.H"

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Convolutions.H"
#include "Image/fancynorm.H"
#include "Image/Point3D.H"
#include "Simulation/SimEventQueue.H"
#include "Neuro/EnvVisualCortex.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_Geons2D = {
  MOC_SORTPRI_3,   "Geons2D-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_Geons2DShowDebug =
  { MODOPT_ARG(bool), "Geons2DShowDebug", &MOC_Geons2D, OPTEXP_CORE,
    "Show debug img",
    "Geons2D-debug", '\0', "<true|false>", "false" };


// ######################################################################
Geons2D::Geons2D(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventGanglionOutput),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  itsShowDebug(&OPT_Geons2DShowDebug, this)

{

  itsMemStorage = cvCreateMemStorage(0);
}

// ######################################################################
Geons2D::~Geons2D()
{
  cvReleaseMemStorage(&itsMemStorage);

}

// ######################################################################
void Geons2D::onSimEventGanglionOutput(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventGanglionOutput>& e)
{
  itsGeons2DCellsInput = e->getCells()[0];
  evolve();

  q.post(rutz::make_shared(new SimEventGeons2DOutput(this, itsGeons2DState)));

}

// ######################################################################
void Geons2D::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "Geons2D", FrameInfo("Geons2D", SRC_POS));
    }
}


void Geons2D::setBias(const Image<float> &biasImg)
{
  //itsGeons2DCellsBias[0] = biasImg;

}

// ######################################################################
void Geons2D::evolve()
{

  Image<byte> in = itsGeons2DCellsInput;
  cvSmooth( img2ipl(in), img2ipl(in), CV_GAUSSIAN, 9, 9 ); // smooth it, otherwise a lot of false circles may be detected
  CvSeq* circles = cvHoughCircles( img2ipl(in),
      itsMemStorage,
      CV_HOUGH_GRADIENT,
      2, //2 times smaller resolution
      in.getHeight()/6, //min distance between centers of detected circles
      100, //higher thresh for canny
      10 ); //accum threshold

  itsGeons2DState.clear();
  for(int i = 0; i < circles->total; i++ )
  {
    float* p = (float*)cvGetSeqElem( circles, i );
    Geons2DState state(Point2D<int>(cvRound(p[0]),(int)cvRound(p[1])), cvRound(p[2]));
    itsGeons2DState.push_back(state);
  }

  cvClearMemStorage(itsMemStorage);
}

Layout<PixRGB<byte> > Geons2D::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;

  Image<float> in = itsGeons2DCellsInput;
  inplaceNormalize(in, 0.0F, 255.0F);

  Image<float> perc = in; //(in.getDims(), ZEROS);
  LINFO("%lu circles", itsGeons2DState.size());
  for(uint i=0; i<itsGeons2DState.size(); i++)
  {
    Geons2DState geons2DState = itsGeons2DState[i];
    drawCircle(perc, geons2DState.pos, (int)geons2DState.radius, 255.0F, 2);

    //perc.setVal(edgeState.pos, edgeState.prob);
  }

  inplaceNormalize(perc, 0.0F, 255.0F);


  Layout<PixRGB<byte> > disp;

  outDisp = hcat(toRGB(Image<byte>(in)), toRGB(Image<byte>(perc)));

  return outDisp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

