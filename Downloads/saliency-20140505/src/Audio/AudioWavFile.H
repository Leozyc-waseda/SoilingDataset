/*!@file Audio/AudioWavFile.H File I/O on audio .wav files */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Audio/AudioWavFile.H $
// $Id: AudioWavFile.H 8605 2007-07-21 02:10:31Z itti $
//

#ifndef AUDIO_AUDIOWAVFILE_H_DEFINED
#define AUDIO_AUDIOWAVFILE_H_DEFINED

#include "Audio/AudioBuffer.H"
#include <string>
#include <vector>

//! Write an audio .wav file
/*! Write a data buffer to file as a basic uncompressed .wav file.
  @param fname name of the .wav file to be written.
  @param buf buffer containing the audio samples. */
template <class T>
void writeAudioWavFile(const std::string& fname,
                       const AudioBuffer<T>& buf);

//! Write an audio .wav file
/*! Write a series of data buffers to a single file as a basic
  uncompressed .wav file.
  @param fname name of the .wav file to be written.
  @param buf buffers containing the audio samples. */
template <class T>
void writeAudioWavFile(const std::string& fname,
                       const std::vector<AudioBuffer<T> >& buf);

//! Read an audio .wav file
/*! Read a data buffer from file as a basic uncompressed .wav
  file. Caveat: one needs to know the number of bits/sample beforehand
  so that one will call the correct template function... */
template <class T>
void readAudioWavFile(const std::string& fname, AudioBuffer<T>& buf);


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // DEVICES_AUDIOWAVFILE_H_DEFINED
