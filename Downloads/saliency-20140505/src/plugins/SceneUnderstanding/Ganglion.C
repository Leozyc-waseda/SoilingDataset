/*!@file SceneUnderstanding/Ganglion.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Ganglion.C $
// $Id: Ganglion.C 13551 2010-06-10 21:56:32Z itti $
//

#ifndef GANGLION_C_DEFINED
#define GANGLION_C_DEFINED

#include "plugins/SceneUnderstanding/Ganglion.H"

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

const ModelOptionCateg MOC_GANGLION = {
  MOC_SORTPRI_3,   "Ganglion-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_GanglionShowDebug =
  { MODOPT_ARG(bool), "GanglionShowDebug", &MOC_GANGLION, OPTEXP_CORE,
    "Show debug img",
    "ganglion-debug", '\0', "<true|false>", "false" };
//#include "Neuro/NeuroSimEvents.H"


// ######################################################################
Ganglion::Ganglion(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventInputFrame),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  itsShowDebug(&OPT_GanglionShowDebug, this)

{
  init(Dims(320,240));
}

// ######################################################################
Ganglion::~Ganglion()
{

}

// ######################################################################
void Ganglion::init(Dims numCells)
{
  Image<float> img(numCells, ZEROS);
  itsGanglionCellsMu.push_back(img);
  itsGanglionCellsInput.push_back(img);
  itsGanglionCellsOutput.push_back(img);
  itsGanglionCellsOutput.push_back(img);

  img.clear(1.0);
  itsGanglionCellsSig.push_back(img);
  itsGanglionCellsPrior.push_back(img);


  //itsProbe.i = 100;
  //itsProbe.j = 220;
  itsProbe.i = -1;
  itsProbe.j = -1;

  //NOTE: Should the ganglion cells be initalizied randomly from the
  //using the learned sigma?

}

// ######################################################################
void Ganglion::onSimEventInputFrame(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventInputFrame>& e)
{
  // here is the inputs image:
  const Image<PixRGB<byte> > inimg = e->frame().asRgb();
  itsCurrentImg = inimg;

  //set the ganglion input
  Image<float> rg, by;
  getRGBY(inimg, rg, by, 25.0F);
  ////Image<byte> input = by + rg + luminance(inimg);
  ////TODO should return 3 seperate channels, for better discrimnation
  Image<float> in = rg; // + rg + luminance(inimg);
  inplaceNormalize(in, 0.0F, 255.0F);
  //Image<byte> input = in;
  Image<byte> input = luminance(inimg);

 // cvEqualizeHist(img2ipl(input),img2ipl(input));
  itsGanglionCellsInput[LUM] = input;

  //itsGanglionCellsInput[RG] = rg;
  //itsGanglionCellsInput[BY] = by;

  evolve();

  //Output the cells
  q.post(rutz::make_shared(new SimEventGanglionOutput(this, itsGanglionCellsOutput)));

  // Create a new Retina Image from the inputImage, and post it to the queue
  //q.post(rutz::make_shared(
  //      new SimEventRetinaImage(
  //        this,
  //        InputFrame(InputFrame::fromRgb(&inimg, q.now())),
  //        Rectangle(Point2D<int>(0,0), inimg.getDims()),
  //        Point2D<int>(0,0)
  //        )
  //      )
  //    );

}

// ######################################################################
void Ganglion::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "Ganglion", FrameInfo("Ganglion", SRC_POS));

  //    if (SeC<SimEventVisualCortexOutput> e = q.check<SimEventVisualCortexOutput>(this, SEQ_ANY))
  //      ofs->writeGray(e->vco(), "SMap", FrameInfo("SMap", SRC_POS));

    }
}


void Ganglion::setBias(const ImageSet<float>& prior)
{

  Point2D<int> maxPos; float maxVal;
  findMax(prior[0], maxPos, maxVal);
  LINFO("maxVal %f", maxVal);


  float filter_period = 100;
  float elongation = 2.0;
  float angle = 90;
  int size = -1;
  const double major_stddev = filter_period / 30.0;
  const double minor_stddev = major_stddev * elongation;

  Image<float> gabor = gaborFilter3(major_stddev, minor_stddev,
      filter_period, 90, 180 - angle + 90.0 , size);

   gabor -= mean(gabor);

  // normalize to unit sum-of-squares:
  gabor /= sum(squared(gabor));

  gabor *= 20;
  gabor += 1.0;

  const int w = itsGanglionCellsMu[0].getWidth(), h = itsGanglionCellsMu[0].getHeight();
  const int Nx = gabor.getWidth(), Ny = gabor.getHeight();
  Image<float> result = itsGanglionCellsMu[0]; //(w, h, ZEROS);
  //Image<float> result(w,h,ZEROS);

  Image<float>::const_iterator sptr = itsGanglionCellsMu[0].begin();
  Image<float>::iterator dptr = result.beginw();

  const int kkk = Nx * Ny - 1;
  const int Nx2 = (Nx - 1) / 2;
  const int Ny2 = (Ny - 1) / 2;

  const float* filter = gabor.getArrayPtr();
  // very inefficient implementation; one has to be crazy to use non
  // separable filters anyway...
  for (int j = 0; j < h; ++j) // LOOP rows of src
    {
      const int j_offset = j-Ny2;

      for (int i = 0; i < w; ++i) // LOOP columns of src
        {
          if (prior[0].getVal(i,j) > 1)
          {
            ////float sum = 0;
            const int i_offset = i-Nx2;
            const int kj_bgn = std::max(- j_offset, 0);
            const int kj_end = std::min(h - j_offset, Ny);


            //float filter_period = 100;
            //float elongation = 2.0;
            //float angle = 90; //(prior[1].getVal(i,j)+M_PI/2)*180/M_PI;
            //int size = -1;
            //const double major_stddev = filter_period / 30.0;
            //const double minor_stddev = major_stddev * elongation;

            //Image<float> gabor2 = gaborFilter3(major_stddev, minor_stddev,
            //    filter_period, 90, 180 - angle + 90.0 , size);

            //gabor2 -= mean(gabor2);

            //// normalize to unit sum-of-squares:
            //gabor2 /= sum(squared(gabor2));

            //gabor2 *= 20;
            //gabor2 += 1.0;


            for (int kj = kj_bgn; kj < kj_end; ++kj) // LOOP rows of filter
            {
              const int kjj = kj + j_offset; // row of src
              const int src_offset = w * kjj + i_offset;
              const int filt_offset = kkk - Nx*kj;

              const int ki_bgn = std::max(-i_offset, 0);
              const int ki_end = std::min(w-i_offset, Nx);

              for (int ki = ki_bgn; ki < ki_end; ++ki) // LOOP cols of filt
              {
                  dptr[ki + src_offset] += sptr[ki + src_offset] * filter[filt_offset - ki];
                //dptr[ki + src_offset] = sptr[ki + src_offset] * 3.0;
              }
            }
            //SHOWIMG(result);
          }

        }
    }
    LINFO("Result");
    SHOWIMG(result);
    itsGanglionCellsMu[0] = result;
  //Create a filtered output



  //float scale = 2;
  //Image<float> dog = dogFilter<float>(1.75F + 0.5F*scale, ori, (int)(1 + 1*scale));

  //// normalize to zero mean:
  //dog -= mean(dog);

  //// normalize to unit sum-of-squares:
  //dog /= sum(squared(dog));



  //SHOWIMG(itsGanglionCellsPrior[0]);
  ////if (maxVal > 1.0)
  //{
  //  tmp.clear(1.0F);
  //  maxPos.i = 67 ; maxPos.j = 95;

  //  //Rectangle rectPos(Point2D<int>(maxPos.i, maxPos.j), Dims(50,3));
  //  //Rectangle rectNeg(Point2D<int>(maxPos.i, maxPos.j-3), Dims(50,3));
  //  //drawFilledRectOR(tmp, rect, 255.0F, 1, prior[0].getVal(maxPos));
  //  //drawFilledRect(tmp, rectPos, 1.01F);
  //  //drawFilledRect(tmp, rectNeg, 0.95F);
  //  //drawCircle(tmp, maxPos, 10, 255.0F);
  //  //inplaceNormalize(tmp, 0.0F, 255.0F);


  //  itsGanglionCellsPrior[0] = tmp;

  //  //SHOWIMG(tmp);
  //  //SHOWIMG(itsGanglionCellsMu[0]);
  //  itsGanglionCellsMu[0] *= tmp;
  //  //SHOWIMG(itsGanglionCellsMu[0]);

  //}
 // else {
 //   itsGanglionCellsPrior[0].clear(0.0F);
 // }




  //SHOWIMG(tmp);


}

// ######################################################################
void Ganglion::setInput(const Image<PixRGB<byte> > &img)
{

  //Retina processing

  //histogram equlization


  //set the ganglion input
  //Image<float> rg, by;
  //getRGBY(img, rg, by, 25.0F);
  Image<byte> input = luminance(img);
  if (itsProbe.isValid())
  {
    printf("%i %i %i %i %i\n", input.getVal(itsProbe),
        input.getVal(itsProbe.i-1, itsProbe.j-1),
        input.getVal(itsProbe.i+1, itsProbe.j-1),
        input.getVal(itsProbe.i+1, itsProbe.j+1),
        input.getVal(itsProbe.i-1, itsProbe.j+1));
  }

 // cvEqualizeHist(img2ipl(input),img2ipl(input));
  itsGanglionCellsInput[LUM] = input;

  //itsGanglionCellsInput[RG] = rg;
  //itsGanglionCellsInput[BY] = by;


  evolve();
}

// ######################################################################
void Ganglion::evolve()
{
  float R = 3.5; //Sensor noise variance
  float Q=0.5;  //Prcess noise variance

  //getchar();
  for(int i=0; i<1; i++)
  {

    //Update the gagnlion cells
    Image<float>::const_iterator inPtr = itsGanglionCellsInput[i].begin();
    Image<float>::const_iterator inStop = itsGanglionCellsInput[i].end();

    Image<float>::iterator gangMuPtr = itsGanglionCellsMu[i].beginw();
    Image<float>::iterator gangPriorPtr = itsGanglionCellsPrior[i].beginw();
    Image<float>::iterator gangSigPtr = itsGanglionCellsSig[i].beginw();
    Image<float>::iterator outMuPtr = itsGanglionCellsOutput[0].beginw();
    Image<float>::iterator outSigPtr = itsGanglionCellsOutput[1].beginw();

    //Kalman filtering  for each ganglion cell
    while(inPtr != inStop)
    {
      //Predict
      float mu_hat = *gangMuPtr; // + *gangPriorPtr;
      float sig_hat = *gangSigPtr + Q;

      //update
      float K = (sig_hat)/(sig_hat + R);
      *gangMuPtr = mu_hat + K * (*inPtr - mu_hat);
      *gangSigPtr = (1-K)*sig_hat;

      //Calculate surprise KL(P(M|D),P(M))
      //P(M|D) = N(*gangMuPtr, * gangSigPtr);
      //P(M) = N(mu_hat, sig_hat);

      float surprise = (((*gangMuPtr-mu_hat)*(*gangMuPtr-mu_hat)) + (*gangSigPtr * *gangSigPtr) + (sig_hat*sig_hat));
      surprise = surprise / (2*sig_hat*sig_hat);
      surprise += log(sig_hat / *gangSigPtr);

      //if (surprise > 0.1)
      //  *outPtr = *inPtr;
      //else
      //*outPtr = surprise;
      *outMuPtr = *inPtr; //*gangMuPtr; TODO change back
      *outSigPtr = *gangSigPtr;

      ++inPtr;
      ++gangMuPtr;
      ++gangPriorPtr;
      ++gangSigPtr;
      ++outMuPtr;
      ++outSigPtr;
    }
  }

}

Layout<PixRGB<byte> > Ganglion::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;
  //Display the results
  Image<float> gangIn = itsGanglionCellsInput[0];
  Image<float> gangPercMu = itsGanglionCellsMu[0];

  //SHOWIMG(itsGanglionCellsSig[0]);


  //Display result
  inplaceNormalize(gangIn, 0.0F, 255.0F);


  if (itsProbe.isValid())
    drawCircle(gangIn, itsProbe, 6, 255.0F);
  inplaceNormalize(gangPercMu, 0.0F, 255.0F);
  //inplaceNormalize(gangPercSig, 0.0F, 255.0F);
  //inplaceNormalize(gangOut, 0.0F, 255.0F);

  Layout<PixRGB<byte> > disp;
  disp = hcat(itsCurrentImg, toRGB(Image<byte>(gangPercMu)));
  //disp = hcat(disp, toRGB(Image<byte>(gangPercSig)));
  //disp = hcat(disp, toRGB(Image<byte>(gangOut)));
  outDisp = vcat(outDisp, disp);

  return outDisp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

