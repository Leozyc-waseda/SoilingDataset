/*!@file MBARI/MbariFrameSeries.H Customized output frame series class
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/MbariFrameSeries.H $
// $Id: MbariFrameSeries.H 6464 2006-04-13 18:01:55Z rjpeters $
//

#ifndef MBARI_MBARIFRAMESERIES_H_DEFINED
#define MBARI_MBARIFRAMESERIES_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Image/Dims.H"
#include "Raster/RasterFileFormat.H"
#include "Util/Types.H"

struct ModelOptionCateg;
template <class T> class Image;
template <class T> class PixRGB;

//! MbariResultViewer options
extern const ModelOptionCateg MOC_MBARIRV;

// ######################################################################
//! InputMbariFrameSeries manages a series of input frames
/*! NOTE: this class was split off from the standard InputFrameSeries
    classes, in order to support the different input paradigm used in
    MbariResultViewer and test-mbari, while allowing the standard
    InputFrameSeries's interface to be minimized. */
class InputMbariFrameSeries : public ModelComponent
{
public:
  //! Constructor
  InputMbariFrameSeries(OptionManager& mgr,
                        const std::string& descrName = "Frame Series",
                        const std::string& tagName = "FrameSeries");

  //! Destructor
  virtual ~InputMbariFrameSeries();

  //! Get dims of a frame on disk, without fully reading it
  /*! It is okay to call this before the model is started().  If input
      resizing is being done, this will return the resized dims.  All
      in all, this will return the size of whatever you will get when
      calling readRGB(). */
  Dims peekDims(const int fnum);

  //! Read a color frame
  Image< PixRGB<byte> > readRGB(const int fnum);

  //! Set the filename stem
  void setFileStem(const std::string& stem);

private:
  //! dims to which frames should be resized on read/write, or 0x0 for none
  OModelParam<Dims> itsDims;

  //! if true, aspect ratio of original image will be preserved when resizing
  OModelParam<bool> itsPreserveAspect;

  //! input file format
  OModelParam<RasterFileFormat> itsRasterFileFormat;

  //! filename stem;
  std::string itsStem;
};

// ######################################################################
//! OutputMbariFrameSeries manages a series of output frames
/*! NOTE: this class was split off from the standard OutputFrameSeries
    class, in order to support the different output paradigm used in
    MbariResultViewer and test-mbari, while allowing the standard
    OutputFrameSeries's interface to be minimized. */
class OutputMbariFrameSeries : public ModelComponent
{
public:
  //! Constructor
  OutputMbariFrameSeries(OptionManager& mgr);

  //! Destructor
  virtual ~OutputMbariFrameSeries();

  //! Set the file stem - for more info see ModelComponent
  void setFileStem(const std::string& stem);

  //! Get the file stem
  std::string getFileStem() const;

  //! Write a frame to disk with a specific frame number
  void writeMbariRGB(const Image< PixRGB<byte> >& image,
                     const std::string& otherstem,
                     const int framenum);

  //! Write a frame to disk with a specific frame number
  void writeMbariGray(const Image<byte>& image,
                      const std::string& otherstem,
                      const int framenum);

  //! Write a frame to disk with a specific frame number
  void writeMbariFloat(const Image<float>& image,
                       const std::string& otherstem,
                       const int flags,
                       const int framenum);

private:
  //! text log file name
  OModelParam<std::string> itsLogFile;

  //! in test mode, don't do any displays
  OModelParam<bool> itsTestMode;

  //! dims to which frames should be resized on read/write, or 0x0 for none
  OModelParam<Dims> itsDims;

  //! if true, aspect ratio of original image will be preserved when resizing
  OModelParam<bool> itsPreserveAspect;

  //! show the frames using xv
  OModelParam<bool> itsShowFrames;

  //! default file format; will be used if RASFMT_AUTO is specified at read/write
  OModelParam<RasterFileFormat> itsRasterFileFormat;

  std::string itsStem;        //!< filename stem
  int itsDidDisplay;          //!< did we show some frames since last update?

  //! resize an image to our dims
  template <class T>
  Image<T> doResizeImage(const Image<T>& input) const;

  //! compute filename corresponding to args and return as string
  /*! Use sformat() to generate otherstem if you need printf-style
      formatting. */
  std::string computeFileName(const int framenum,
                              const std::string& otherstem) const;

  //! have we displayed too many windows, or can we still display more?
  bool okToDisplay();
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // MBARI_MBARIFRAMESERIES_H_DEFINED
