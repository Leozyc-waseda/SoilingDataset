/*!@file Matlab/mexSaliencySkin.C This is the MEX file version of the Saliency program<br>
  This file compiles into a MEX file that can be called from within
  Matlab for extracting the saliency information from an image. The
  call from Matlab is:<BR> <TT>[num_sal, coords, times, salmap,
  modfunc, areas, labels] = Saliency(image, targets, max_time, foa_size,
  weights, normtype, smoothFactor, levels);</TT><P>

  Compared to mexSaliency.C, this version contains a HueChannel.

  The return values are:

  @param num_sal the number of salient spots found
  @param coords (2, num_sal) array that contains the x and y coordinates
  of the salient spots in (1,:) and (2,:), respectively
  @param times (1, num_sal) array that contains the evolution times
  @param salmap the saliency map (normalized between 0 and 1)
  @param modfunc (num_sal, height, width) array that contains the modulation
  functions for the salient spots, maxnormalized to 1.0
  @param areas (1, num_sal) array that contains the number of pixels for
  salient spot that have contributed to the modulation function
  @param labels {num_sal} cell array containing the label strings from the
  size info analysis

  The only <I>required</I> argument is:
  @param image the input image

  <I>Optional</I> paramters that have pre-assigned default values are:
  @param targets map of the same size as the input image, in which
  targets for the focus of attention are 255 and the rest is
  zero. Pass a scalar 0 (the default) if you have no targets to look for.
  @param max_time the maximum amount of (simulated) time that the saliency map
  should evolve in seconds <I>(default: 0.7)</I>
  @param foa_size the size of the focus of attention in pixels.
  The default is 1/12 of min(height, width) of the input image.
  Pass -1 if you want to use the default.
  @param weights Vector of length 3 containing the weights for the
  following channels (in this order): <TT>[wIntens, wOrient, wColor]</TT>
  <I>(default: [1.0 1.0 1.0])</I>
  @param normtype normalization type; see fancynorm.H
  <I>(default: 2 = VCXNORM_NORM)</I>
  @param smoothMethod method used for smoothing the shapeEstimator masks;
  see ShapeEstimatorModes.H; (default: 1 = Gaussian Smoothing)
  @param levels Vector of length 6 containing the following parameters:
  <TT>[sm_level, level_min, level_max, delta_min, delta_max, nborients]</TT>
  <I>(default: [4 2 4 3 4 4])</I>
*/

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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Matlab/mexSaliencySkin.C $
// $Id: mexSaliencySkin.C 15468 2013-04-18 02:18:18Z itti $
//

#define FPEXCEPTIONSDISABLED

#include "Channels/ChannelOpts.H"
#include "Channels/SkinHueChannel.H"
#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Image/fancynorm.H"
#include "Image/LevelSpec.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Matlab/mexConverts.H"
#include "Media/MediaSimEvents.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/NeuroSimEvents.H"
#include "Neuro/ShapeEstimator.H"
#include "Neuro/StdBrain.H"
#include "Neuro/VisualCortex.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Util/StringConversions.H"

#include <cmath>
#include <mex.h>
#include <matrix.h> // matlab
#include <sstream>
#include <string>
#include <vector>

// only used for debugging
//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif


// ##########################################################################
// ############  The main function
// ##########################################################################
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

#ifdef DEBUG
  LOG_FLAGS |= LOG_FULLTRACE;
#else
  LOG_FLAGS &= (~LOG_FULLTRACE);
#endif

  // #### check whether # of params is 1..8 and number of returned values <= 7
  if (nrhs < 1)  mexErrMsgTxt("At least one parameter required: image");
  if (nrhs > 8) mexErrMsgTxt("Too many parameters (max. 8).");
  if (nlhs > 7)  mexErrMsgTxt("Too many return values (max. 7).");

  // counter for parameters
  int p = 1;

  // #### first input: the image
  Image< PixRGB<byte> > img = mexArray2RGBImage<byte>(prhs[p-1]);
  p++;

  // #### targets
  Image<byte> targets;
  if ((nrhs >= p) && (mxGetNumberOfElements(prhs[p-1]) > 1))
    targets = mexArray2Image<byte>(prhs[p-1]);
  p++;

  // #### maximum evolution time
  double max_time = 0.7;
  if (nrhs >= p) max_time = (double)mxGetScalar(prhs[p-1]);
  p++;

  // #### foa_size
  //int foa_size = -1;
  //if (nrhs >= p) foa_size = (int)mxGetScalar(prhs[p-1]);
  p++;

  // #### weight for all the channels
  /*double wIntens = 1.0, wOrient = 1.0, wColor = 1.0;
  if (nrhs >= p)
    {
      std::vector<double> weights = mexArr2Vector<double>(prhs[p-1], 3);
      wIntens = weights[0]; wOrient  = weights[1];
      wColor  = weights[2];
    }*/
  p++;

  // #### norm_type
  int norm_type = VCXNORM_FANCY;
  if (nrhs >= p) norm_type = (int)mxGetScalar(prhs[p-1]);
  p++;


  // #### smoothing method for shapeEstimator
  int smoothMethod = 1;
  if (nrhs >= p) smoothMethod = (int)mxGetScalar(prhs[p-1]);
  p++;

  // #### which levels?
  int sml=4, level_min=2, level_max=4, delta_min=3, delta_max=4, nborients=4;
  if (nrhs >= p)
    {
         std::vector<int> lev = mexArr2Vector<int>(prhs[p-1], 6);
         sml = lev[0]; level_min = lev[1]; level_max = lev[2];
         delta_min = lev[3]; delta_max = lev[4]; nborients = lev[5];
    }
  p++;


  // #### make sure, all parameters are within range
  if (max_time < 0.0) mexErrMsgTxt("max_time must be positive.");
  if ((norm_type < VCXNORM_NONE))
    mexErrMsgTxt("norm_type can only have the following values:\n"
                 "0  -  no normalization\n"
                 "1  -  maximum normalization\n"
                 "2  -  fancy normalization (default)\n"
                 "3  -  fancy normalization - fast implementation\n"
                 "4  -  fancy normalization with one iteration");
  if ((smoothMethod < 0) || (smoothMethod >= NBSHAPEESTIMATORSMOOTHMETHODS))
    mexErrMsgTxt("smoothing methods can have the following values:\n"
                 "0 - no smoothing\n"
                 "1 - Gaussian smoothing (default)\n"
                 "2 - Chamfer smoothing");
  if (nborients <= 0) mexErrMsgTxt("nborients must be positive");
  if (level_min > level_max) mexErrMsgTxt("must have level_min <= level_max");
  if (delta_min > delta_max) mexErrMsgTxt("must have delta_min <= delta_max");
  if (sml < level_max) mexErrMsgTxt("must have sml >= level_max");


#ifdef DEBUG
  // This is for debugging only
  std::cout << "#### mexSaliency\n";
  std::cout << "max_time = " << max_time << "\n";
  std::cout << "norm_type = " << norm_type <<"\n";
  std::cout << "foa_size = " << foa_size <<"\n";
  std::cout << "wIntens = " << wIntens <<"\n";
  std::cout << "wOrient = " << wOrient <<"\n";
  std::cout << "wColor = " << wColor <<"\n";
  std::cout << "nborient = " << nborients <<"\n";
  std::cout << "level_min = " << level_min <<"\n";
  std::cout << "level_max = " << level_max <<"\n";
  std::cout << "delta_min = " << delta_min <<"\n";
  std::cout << "delta_max = " << delta_max <<"\n";
  std::cout << "sml = " << sml <<"\n";
  // end debugging
#endif


  // get a standard brain:
  ModelManager manager("Mex Attention Model");
  manager.allowOptions(OPTEXP_CORE);

  manager.setOptionValString(&OPT_UsingFPE,"false");

#ifdef DEBUG
  manager.setOptionValString(&OPT_DebugMode,"true");
#else
  manager.setOptionValString(&OPT_DebugMode,"false");
#endif

  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::soft_ref<StdBrain> brain(new StdBrain(manager));
  manager.addSubComponent(brain);
  manager.exportOptions(MC_RECURSE);

  // set a few custom defaults for our possible options:
  manager.setOptionValString(&OPT_NumOrientations,
                             convertToString(nborients));
  manager.setOptionValString(&OPT_LevelSpec,
                             convertToString(LevelSpec(level_min,level_max,
                                                       delta_min,delta_max,sml)));
  manager.setOptionValString(&OPT_MaxNormType,
                             convertToString(MaxNormType(norm_type)));
  //manager.setOptionValString(&OPT_UseRandom, "true");
  manager.setOptionValString(&OPT_UseRandom, "false");
  manager.setOptionValString(&OPT_IORtype,"ShapeEst");
  manager.setOptionValString(&OPT_ShapeEstimatorMode, "FeatureMap");
  manager.setOptionValString(&OPT_ShapeEstimatorSmoothMethod,
                             convertToString(ShapeEstimatorSmoothMethod
                                             (smoothMethod)));
  manager.setOptionValString(&OPT_RawVisualCortexChans,"OIC");

  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  // get started:
  manager.start();

  LFATAL("fixme");

  nub::soft_ref<VisualCortex> vc;///////// = brain->getVC();
  /*
  nub::soft_ref<SkinHueChannel> hue(new SkinHueChannel(manager));
  hue->exportOptions(MC_RECURSE);
  vc->addSubChan(hue, "hue");
  vc->setSubchanTotalWeight("hue", 5.0);
  hue->start();

  vc->setSubchanTotalWeight("color", wColor);
  vc->setSubchanTotalWeight("intensity", wIntens);
  vc->setSubchanTotalWeight("orientation", wOrient);
  */
  // let everyone (and in particular our TargetChecker) know about our target mask:
  seq->post(rutz::make_shared(new SimEventTargetMask(brain.get(), targets)));

  // post the image to the brain:
  seq->post(rutz::make_shared(new SimEventInputFrame(brain.get(), GenericFrame(img), 0)));

  Image<float> salmap;//////////// = vc->getOutput();
  inplaceNormalize(salmap, 0.0f, 1.0f);
  salmap = rescale(salmap,img.getDims());

  std::vector<Point2D<int> > coords;
  std::vector<double> times;
  std::vector<int> areas;
  ImageSet<float> s_vect;
  std::vector<std::string> labelstrings;

  int num_sal = 0;

  const bool forever = (targets.initialized() && (max_time == 0.0));

  // time-loop, evolve
  while (((seq->now().secs() < max_time)||forever))
    {
      (void) seq->evolve();

      // we have a winner
      if (SeC<SimEventWTAwinner> e = seq->check<SimEventWTAwinner>(0))
        {
          const Point2D<int> winner = e->winner().p;

          num_sal++;

          coords.push_back(winner);
          times.push_back(seq->now().secs());

          //only do the shape estimator stuff if necessary
          if (nlhs > 5)
            {
              Image<float> semask; std::string selabel; int searea;
              if (SeC<SimEventShapeEstimatorOutput>
                  ese = seq->check<SimEventShapeEstimatorOutput>(0))
                {
                  semask = ese->smoothMask();
                  selabel = ese->winningLabel();
                  searea = ese->objectArea();
                }

              if (!semask.initialized()) semask.resize(img.getDims(),ZEROS);
              s_vect.push_back(semask);

              areas.push_back(searea);

              std::ostringstream os;
              os << (seq->now().msecs()) << " ms - " << selabel;
              labelstrings.push_back(os.str());
            }
        } // end if (brain->gotCovertShift())
    } // end while

#ifdef DEBUG
  // This is for debugging only
  std::cout << "#### mexSaliency\n";
  std::cout << "x\ty\tt\n";
  for (int i = 0; i < num_sal; i++)
    std::cout << coords[i].i << "\t" << coords[i].j << "\t" << times[i] << "\n";
  // end debugging
#endif

  // get stopped:
  manager.stop();

  // ######## create the return variables
  p = 1;

  // #### number of salient spots
  if (nlhs >= p) plhs[p-1] = mxCreateDoubleScalar(num_sal);
  p++;

  // #### x and y coordinates of salient spot
  if (nlhs >= p) plhs[p-1] = Point2DVec2mexArr(coords);
  p++;

  // #### evolution times
  if (nlhs >= p) plhs[p-1] = Vector2mexArr(times);
  p++;

  // #### the saliency map
  if (nlhs >= p) plhs[p-1] = Image2mexArray(salmap);
  p++;

  // #### shape info
  mxArray* Labels = mxCreateCellArray(1, &num_sal);
  if (nlhs >= p)
    {
      for (int i = 0; i < num_sal; i++)
        {
          mxSetCell(Labels, i, mxCreateString(labelstrings[i].c_str()));
#ifdef DEBUG
          std::cout << labelstrings[i].c_str() << "\n";
#endif
        }
      plhs[p-1] = ImgVec2mexArr(s_vect);
    }
  p++;

  // #### area measures of shape info
  if (nlhs >= p) plhs[p-1] = Vector2mexArr(areas);
  p++;

  // #### label strings
  if (nlhs >= p) plhs[p-1] = Labels;
  p++;

 // #### done
  return;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
