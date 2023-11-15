/*!@file MBARI/MbariFrameSeries.C Customized output frame series class
  for use in MbariResultViewer */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/MbariFrameSeries.C $
// $Id: MbariFrameSeries.C 14376 2011-01-11 02:44:34Z pez $
//

#ifndef MBARI_MBARIFRAMESERIES_C_DEFINED
#define MBARI_MBARIFRAMESERIES_C_DEFINED

#include "MBARI/MbariFrameSeries.H"

#include "Component/GlobalOpts.H" // for OPT_TestMode
#include "Component/ModelOptionDef.H"
#include "Component/OptionManager.H"
#include "Image/CutPaste.H"   // for inplaceEmbed()
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"   // for rescale()
#include "Media/MediaOpts.H"
#include "Raster/Raster.H"
#include "Transport/RasterInputSeries.H"
#include "Transport/TransportOpts.H" // for OPT_OutputRasterFileFormat
#include "Util/FileUtil.H" // for splitPath()
#include "Util/TextLog.H"
#include "Util/log.H"
#include "Util/sformat.H"

#include <cstdio>

// maximum number of displays we cando before giving up
#define MAXDISP 20

const ModelOptionCateg MOC_MBARIRV = {
  MOC_SORTPRI_4, "MBARI Result Viewer Related Options" };

// Used by: InputMbariFrameSeries
static const ModelOptionDef OPT_InputMbariPreserveAspect =
  { MODOPT_FLAG, "InputMbariPreserveAspect", &MOC_MBARIRV, OPTEXP_MRV,
    "Preserve input frame aspect ratio if rescaling to fixed dims",
    "mbari-preserve-input-aspect", '\0', "", "false" };

// Used by: OutputMbariFrameSeries
static const ModelOptionDef OPT_OutputMbariShowFrames =
  { MODOPT_FLAG, "OutputMbariShowFrames", &MOC_MBARIRV, OPTEXP_MRV,
    "Show output frames",
    "mbari-display-output-frames", '\0', "", "true" };

// ######################################################################
// #################### InputMbariFrameSeries
// ######################################################################

InputMbariFrameSeries::InputMbariFrameSeries(OptionManager& mgr,
                                   const std::string& descrName,
                                   const std::string& tag) :
  ModelComponent(mgr, "Input "+descrName, "Input"+tag),
  itsDims(&OPT_InputFrameDims, this),
  itsPreserveAspect(&OPT_InputMbariPreserveAspect, this),
  itsRasterFileFormat(&OPT_InputRasterFileFormat, this),
  itsStem("")
{}

// ######################################################################
InputMbariFrameSeries::~InputMbariFrameSeries()
{}

// ######################################################################
Dims InputMbariFrameSeries::peekDims(const int fnum)
{
  // if we are doing resizing, our dims are the resized dims:
  if (itsDims.getVal().w() != 0 && itsDims.getVal().h() != 0)
    return itsDims.getVal();

  return Raster::getImageDims(sformat("%s%06d", itsStem.c_str(), fnum),
                              itsRasterFileFormat.getVal());
}

// ######################################################################
Image<PixRGB<byte> > InputMbariFrameSeries::readRGB(const int fnum)
{
  const Image< PixRGB<byte> > ima =
    Raster::ReadRGB(sformat("%s%06d", itsStem.c_str(), fnum),
                    itsRasterFileFormat.getVal());

  if (itsDims.getVal().isEmpty())
    return ima;

  if (itsPreserveAspect.getVal())
    {
      Image<PixRGB<byte> > res(itsDims.getVal(), ZEROS);
      PixRGB<byte> bg = PixRGB<byte>(64, 64, 64);
      inplaceEmbed(res, ima, res.getBounds(), bg, true);
      return res;
    }

  return rescale(ima, itsDims.getVal());
}

// ######################################################################
void InputMbariFrameSeries::setFileStem(const std::string& stem)
{
  itsStem = stem;
}

// ######################################################################
// #################### OutputMbariFrameSeries
// ######################################################################
OutputMbariFrameSeries::OutputMbariFrameSeries(OptionManager& mgr)
  :
  ModelComponent(mgr, "Output MBARI Frame Series",
                 "OutputMbariFrameSeries"),
  itsLogFile(&OPT_TextLogFile, this),
  itsTestMode(&OPT_TestMode, this),
  itsDims(&OPT_OutputFrameDims, this),
  itsPreserveAspect(&OPT_OutputPreserveAspect, this),
  itsShowFrames(&OPT_OutputMbariShowFrames, this),
  itsRasterFileFormat(&OPT_OutputRasterFileFormat, this),
  itsStem(""),
  itsDidDisplay(0)
{  }

// ######################################################################
OutputMbariFrameSeries::~OutputMbariFrameSeries()
{ }

// ######################################################################
void OutputMbariFrameSeries::setFileStem(const std::string& stem)
{
  if (started())
    LFATAL("Cannot change file stem while started");
  itsStem = stem;
}

// ######################################################################
std::string OutputMbariFrameSeries::getFileStem() const
{ return itsStem; }

// ######################################################################
void OutputMbariFrameSeries::
writeMbariRGB(const Image< PixRGB<byte> >& image,
              const std::string& otherstem,
              const int framenum)
{
  // figure out the file name to use:
  std::string fname(computeFileName(framenum, otherstem));

  // find out file format:
  const RasterFileFormat ff = itsRasterFileFormat.getVal();

  // resize the image as appropriate:
  Image< PixRGB<byte> > ima = OutputMbariFrameSeries::doResizeImage(image);

  // write the image:
  fname = Raster::WriteRGB(ima, fname, ff);
  textLog(itsLogFile.getVal(), "WriteRGB", fname);

  // do we want to show the image?
  if (okToDisplay())
    Raster::Display(fname.c_str());
}

// ######################################################################
void OutputMbariFrameSeries::
writeMbariGray(const Image<byte>& image,
               const std::string& otherstem,
               const int framenum)
{
  // figure out the file name to use:
  std::string fname(computeFileName(framenum, otherstem));

  // find out file format:
  const RasterFileFormat ff = itsRasterFileFormat.getVal();

  // resize the image as appropriate:
  Image<byte> ima = OutputMbariFrameSeries::doResizeImage(image);

  // write the image:
  fname = Raster::WriteGray(ima, fname, ff);
  textLog(itsLogFile.getVal(), "WriteGray", fname);

  // do we want to show the image?  do it only if we are not in test mode
  if (okToDisplay())
    Raster::Display(fname.c_str());
}

// ######################################################################
void OutputMbariFrameSeries::
writeMbariFloat(const Image<float>& image,
                const std::string& otherstem,
                int flags,
                const int framenum)
{
  // figure out the file name to use:
  std::string fname(computeFileName(framenum, otherstem));

  // find out file format:
  RasterFileFormat ff = itsRasterFileFormat.getVal();

  // resize the image as appropriate:
  Image<float> ima = OutputMbariFrameSeries::doResizeImage(image);

  // do special handling for FLOAT_NORM_PRESERVE -- in that case, we
  // want to turn off FLOAT_NORM_0_255, and save the image in
  // RASFMT_PFM format:
  if (flags & FLOAT_NORM_PRESERVE)
    {
      flags &= ~FLOAT_NORM_0_255;
      ff = RASFMT_PFM;
    }

  // write the image:
  fname = Raster::WriteFloat(ima, flags, fname, ff);
  textLog(itsLogFile.getVal(), "WriteFloat", fname);

  // do we want to show the image?  do it only if we are not in test mode
  if (okToDisplay())
    Raster::Display(fname.c_str());
}

// ######################################################################
std::string OutputMbariFrameSeries::
computeFileName(const int framenum, const std::string& otherstem) const
{
  // first split itsStem into path and file components:
  std::string path, file;
  splitPath(itsStem, path, file);

  // use an alternate stem
  std::string otherpath, otherfile;
  splitPath(otherstem, otherpath, otherfile);

  return sformat("%s%s%06d", path.c_str(), otherfile.c_str(), framenum);
}

// ######################################################################
template <class T>
Image<T> OutputMbariFrameSeries::doResizeImage(const Image<T>& input) const
{
  if (itsDims.getVal().isEmpty()) return input;
  if (itsPreserveAspect.getVal())
    {
      Image<T> res(itsDims.getVal(), ZEROS);
      T bg = T(); bg += 64;
      inplaceEmbed(res, input, res.getBounds(), bg, true);
      return res;
    }
  return rescale(input, itsDims.getVal());
}

// ######################################################################
bool OutputMbariFrameSeries::okToDisplay()
{
  if (itsTestMode.getVal() == false && itsShowFrames.getVal())
    {
      ++ itsDidDisplay;
      if (itsDidDisplay > MAXDISP)
        LERROR("**** TOO MANY WINDOWS! NOT DISPLAYING IMAGE...");
      else
        return true;
    }
  return false;
  // itsDidDisplay is reset to zero at each update()
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // MBARI_MBARIFRAMESERIES_C_DEFINED
