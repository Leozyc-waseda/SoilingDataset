/*!@file BeoSub/BeoSubMotor.H Low-level driver for BeoSub motors */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubMotor.H $
// $Id: BeoSubMotor.H 4679 2005-06-24 04:59:53Z rjpeters $
//

// core of the code contributed by Harris Chiu, USC/ISI

#ifndef BEOSUBMOTOR_H_DEFINED
#define BEOSUBMOTOR_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Util/Types.H"

// motor channels:
//
// channel 0 is frame rate of the camera
// channel 1 is Submarine left right motion
// channel 2 is Submarine Camera Up Down motion
// channel 3 is Submarine forward backward motion motion
// channel 4 is Submarine Camera left right motion
// channel 5 is unused
// channel 6 is Submarine Up Down motion
// channel 7 is unused
// channel 8 is unused
#define BSM_LEFTRIGHT 1
#define BSM_CAMUPDOWN 2
#define BSM_FORBACKWARD 3
#define BSM_CAMLEFTRIGHT 4
#define BSM_UPDOWN 6

//! Low-level driver for the BeoSub motors
class BeoSubMotor : public ModelComponent
{
public:
  // Constructor
  BeoSubMotor(OptionManager& mgr,
              const std::string& descrName = "BeoSub Motor",
              const std::string& tagName = "BeoSub Motor",
              const int def_parallel_port_addr = 0x378);

  //! Destructor
  ~BeoSubMotor();

  //! Set a value
  /*! Set a value not less than a minimum and
    not larger than maximum to protect controller box from damage
    write the value to the parallel port
    @param value the value to set
    @param channel the channel to set the value to
    @param immed if true, immediately send all values to the
    interface, otherwise just store the value internally and it will
    be sent at the next writePort() or setValue() with immed set to
    true.
    @return true on success */
  bool setValue(const byte value, const int channel, const bool immed = true);

  //! Get the last set value on a given channel
  byte getValue(const int channel) const;

  //! Pulse a channel on for a bit
  /*!@param positive use a pulse value higher than rest if true, lower
    then rest otherwise */
  bool pulseValue(const int channel, const bool positive);

  //! Write out our internal data to the parallel port
  /*! @return true on success */
  bool WritePort();

  unsigned char chMin[9];
  unsigned char chMax[9];
  unsigned char chDefault[9];
protected:
  NModelParam<int> itsPort;  //!< base address of parallel port
  unsigned char strobelo;
  unsigned char strobehi;
  unsigned char ch[9];  //!< channel values
  void reset();
  void SendCh(const int value);
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
