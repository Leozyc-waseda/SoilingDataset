/*!@file Audio/AudioBuffer.H A buffer of audio data */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Audio/AudioBuffer.H $
// $Id: AudioBuffer.H 8647 2007-07-30 19:40:34Z abondada $
//

#ifndef AUDIO_AUDIOBUFFER_H_DEFINED
#define AUDIO_AUDIOBUFFER_H_DEFINED

#include "Image/ArrayData.H"

//! AudioBuffer is a buffer of audio data
/*! AudioBuffer has copy-on-write/ref-counting semantics and automatic
  memory management, just like Image does. The template argument
  specifies the type of the audio data (typically, byte, or int16). */
template <class T>
class AudioBuffer {

public:
  // ############################################################
  /*! @name Constructors, destructors, assignment */
  //@{

  //! Constructor from raw data
  inline AudioBuffer(const T* data, const uint nsampl, const uint nchan,
                     const float freq);

  //! Allocate memory for future data
  inline AudioBuffer(const uint nsampl, const uint nchan, const float freq,
                     InitPolicy init);

  //! Copy constructor
  inline AudioBuffer(const AudioBuffer<T>& A);

  //! Assigment operator
  inline AudioBuffer<T>& operator=(const AudioBuffer<T>& A);

  //! Empty constructor
  inline AudioBuffer();

  //! Destructor
  inline ~AudioBuffer();

  //@}

  // ############################################################
  /*! @name Access functions */
  //@{

  //! Get number of samples
  inline uint nsamples() const;

  //! Get number of channels
  inline uint nchans() const;

  //! Get sampling frequency in samples/s
  inline float freq() const;

  //! Set sampling frequency in samples/s
  inline void setFreq(const float f);

  //! Get a sample
  inline const T& getVal(const uint sample, const uint channel);

  //! Get total data size in number of T's
  /*! This is nsamples * nchans */
  inline uint size() const;

  //! Get total data size in bytes
  /*! This is nsamples * nchans * sizeof(T) */
  inline uint sizeBytes() const;

  //! Get a read-only pointer to the raw data
  inline const T* getDataPtr() const;

  //! Get a read/write pointer to the raw data
  inline T* getDataPtr();

  //@}

private:
  ArrayHandle<T> itsHdl; // the data proper
  float itsFreq;  // sampling frequency
};

// ######################################################################
// #################### Inlined functions:
// ######################################################################

template <class T>
inline AudioBuffer<T>::AudioBuffer(const T* data, const uint nsampl,
                                   const uint nchan, const float freq) :
  itsHdl(new ArrayData<T>(Dims(nsampl, nchan), data)),
  itsFreq(freq)
{ }

// ######################################################################
template <class T>
inline AudioBuffer<T>::AudioBuffer(const uint nsampl, const uint nchan,
                                   const float freq, InitPolicy init) :
  itsHdl(new ArrayData<T>(Dims(nsampl, nchan), init)),
  itsFreq(freq)
{ }

// ######################################################################
template <class T>
inline AudioBuffer<T>::AudioBuffer(const AudioBuffer<T>& A) :
  itsHdl(A.itsHdl),
  itsFreq(A.itsFreq)
{ }

// ######################################################################
template <class T>
inline AudioBuffer<T>& AudioBuffer<T>::operator=(const AudioBuffer<T>& A)
{
  AudioBuffer<T> A_copy(A);
  itsHdl.swap(A_copy.itsHdl);
  itsFreq = A.itsFreq;
  return *this;
}

// ######################################################################
template <class T>
inline AudioBuffer<T>::AudioBuffer() :
  itsHdl(new ArrayData<T>()),
  itsFreq(0.0F)
{ }

// ######################################################################
template <class T>
inline AudioBuffer<T>::~AudioBuffer()
{ }

// ######################################################################
template <class T>
inline uint AudioBuffer<T>::nsamples() const
{ return itsHdl.get().w(); }

// ######################################################################
template <class T>
inline uint AudioBuffer<T>::nchans() const
{ return itsHdl.get().h(); }

// ######################################################################
template <class T>
inline float AudioBuffer<T>::freq() const
{ return itsFreq; }

// ######################################################################
template <class T>
inline void AudioBuffer<T>::setFreq(const float f)
{ itsFreq = f; }

// ######################################################################
template <class T>
inline const T& AudioBuffer<T>::getVal(const uint sample, const uint channel)
{
  ASSERT(sample < nsamples() && channel < nchans());
  return itsHdl.get().data()[sample + nsamples() * channel];
}

// ######################################################################
template <class T>
inline uint AudioBuffer<T>::size() const
{ return nsamples() * nchans(); }

// ######################################################################
template <class T>
inline uint AudioBuffer<T>::sizeBytes() const
{ return nsamples() * nchans() * sizeof(T); }

// ######################################################################
template <class T>
inline const T* AudioBuffer<T>::getDataPtr() const
{ return itsHdl.get().data(); }

// ######################################################################
template <class T>
inline T* AudioBuffer<T>::getDataPtr()
{ return itsHdl.uniq().dataw(); }


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // AUDIO_AUDIOBUFFER_H_DEFINED
