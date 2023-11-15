/*!@file SceneUnderstanding/LGN.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/LGN.C $
// $Id: LGN.C 14397 2011-01-14 20:33:47Z lior $
//

#ifndef LGN_C_DEFINED
#define LGN_C_DEFINED

#include "plugins/SceneUnderstanding/LGN.H"

#include "Image/DrawOps.H"
//#include "Image/OpenCVUtil.H"
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

const ModelOptionCateg MOC_LGN = {
  MOC_SORTPRI_3,   "LGN-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_LGNShowDebug =
  { MODOPT_ARG(bool), "LGNShowDebug", &MOC_LGN, OPTEXP_CORE,
    "Show debug img",
    "lgn-debug", '\0', "<true|false>", "false" };


//Define the inst function name
SIMMODULEINSTFUNC(LGN);

// ######################################################################
LGN::LGN(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventInputFrame),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  itsShowDebug(&OPT_LGNShowDebug, this),
  itsInitialized(false)

{
}

// ######################################################################
LGN::~LGN()
{

}

// ######################################################################
void LGN::init(Dims numCells)
{
  Image<float> img(numCells, ZEROS);
  for(int i=0; i<3; i++)
  {
    itsCellsInput.push_back(img);
    itsCellsMu.push_back(img);
    img.clear(1.0);
    itsCellsSig.push_back(img);
  }


  //NOTE: Should the LGN cells be initalizied randomly from the
  //using the learned sigma?
  //

  itsInitialized = true;

}

// ######################################################################
void LGN::onSimEventInputFrame(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventInputFrame>& e)
{
  // here is the inputs image:
  GenericFrame frame = e->frame();

  const Image<PixRGB<byte> > inimg = rescale(frame.asRgb(), 640, 480);
  itsCurrentImg = inimg;

  if (!itsInitialized)
    init(itsCurrentImg.getDims());


  rutz::shared_ptr<GenericFrame::MetaData> metaData;
  if (frame.hasMetaData(std::string("ObjectsData")))
    metaData = frame.getMetaData(std::string("ObjectsData"));

  //set the LGN input
  Image<float>  lum,rg,by;
  //getDKL(inimg, lum, rg, by);
  getLAB(inimg, lum, rg, by);

  //Normalize all values to the same level
  inplaceNormalize(lum, 0.0F, 255.0F);
  inplaceNormalize(rg, 0.0F, 255.0F);
  inplaceNormalize(by, 0.0F, 255.0F);

  Image<float> kernel = gaussian<float>(0.0F, 1.4, lum.getWidth(),1.0F);
  Image<float> kernel2 = gaussian<float>(0.0F, 1.5, lum.getWidth(),1.0F);
  Image<float> kernel3 = gaussian<float>(0.0F, 1.5, lum.getWidth(),1.0F);
  //// do the convolution:
  itsCellsInput[LUM] = sepFilter(lum, kernel, kernel, CONV_BOUNDARY_CLEAN);
  itsCellsInput[RG] = sepFilter(rg, kernel2, kernel2, CONV_BOUNDARY_CLEAN);
  itsCellsInput[BY] = sepFilter(by, kernel3, kernel3, CONV_BOUNDARY_CLEAN);

  itsCellsMu[LUM] = itsCellsInput[LUM];
  itsCellsMu[RG] = itsCellsInput[RG];
  itsCellsMu[BY] = itsCellsInput[BY];
  //evolve();

  //Output the cells
  q.post(rutz::make_shared(new SimEventLGNOutput(this, itsCellsMu, metaData)));

}

// ######################################################################
void LGN::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "LGN", FrameInfo("LGN", SRC_POS));
    }
}


void LGN::setBias(const ImageSet<float>& prior)
{

}

// ######################################################################
void LGN::evolve()
{
  float R = 2; //5; //Sensor noise variance
  float Q=0.1;  //Prcess noise variance

  //getchar();
  for(int i=0; i<3; i++)
  {

    //Update the gagnlion cells
    Image<float>::const_iterator inPtr = itsCellsInput[i].begin();
    Image<float>::const_iterator inStop = itsCellsInput[i].end();

    Image<float>::iterator muPtr = itsCellsMu[i].beginw();
    Image<float>::iterator sigPtr = itsCellsSig[i].beginw();

    //Kalman filtering  for each LGN cell
    while(inPtr != inStop)
    {
      //Predict
      float mu_hat = *muPtr; // + *gangPriorPtr;
      float sig_hat = *sigPtr + Q;

      //update
      float K = (sig_hat)/(sig_hat + R);
      //*muPtr = mu_hat + K * (*inPtr - mu_hat);
      *muPtr =*inPtr;
      *sigPtr = (1-K)*sig_hat;

      //Calculate surprise KL(P(M|D),P(M))
      //P(M|D) = N(*muPtr, * sigPtr);
      //P(M) = N(mu_hat, sig_hat);

      float surprise = (((*muPtr-mu_hat)*(*muPtr-mu_hat)) + (*sigPtr * *sigPtr) + (sig_hat*sig_hat));
      surprise = surprise / (2*sig_hat*sig_hat);
      surprise += log(sig_hat / *sigPtr);

      //if (surprise > 0.1)
      //  *outPtr = *inPtr;
      //else
      //*outPtr = surprise;

      ++inPtr;
      ++muPtr;
      ++sigPtr;
    }
  }

}

Layout<PixRGB<byte> > LGN::getDebugImage()
{
  //Display the results
  Image<float> lumPerc = itsCellsMu[0];
  Image<float> rgPerc = itsCellsMu[1];
  Image<float> byPerc = itsCellsMu[2];

  //Display result
  inplaceNormalize(lumPerc, 0.0F, 255.0F);
  inplaceNormalize(rgPerc, 0.0F, 255.0F);
  inplaceNormalize(byPerc, 0.0F, 255.0F);
  //inplaceNormalize(gangPercSig, 0.0F, 255.0F);
  //inplaceNormalize(gangOut, 0.0F, 255.0F);

  Layout<PixRGB<byte> > disp;
  disp = hcat(itsCurrentImg, toRGB(Image<byte>(lumPerc)));
  disp = hcat(disp, toRGB(Image<byte>(rgPerc)));
  disp = hcat(disp, toRGB(Image<byte>(byPerc)));

  return disp;

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

