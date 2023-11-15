/*!@file Audio/AudioWavFile.C File I/O on audio .wav files */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Audio/AudioWavFile.C $
// $Id: AudioWavFile.C 8648 2007-07-30 19:41:42Z abondada $
//

#ifndef AUDIO_AUDIOWAVFILE_C_DEFINED
#define AUDIO_AUDIOWAVFILE_C_DEFINED

#include "Audio/AudioWavFile.H"
#include "Util/log.H"
#include <cstdio>

// ######################################################################
template <class T>
void writeAudioWavFile(const std::string& fname,
                       const AudioBuffer<T>& buf)
{
  std::vector<AudioBuffer<T> > vbuf;
  vbuf.push_back(buf);
  writeAudioWavFile(fname, vbuf);
}

// ######################################################################
template <class T>
void writeAudioWavFile(const std::string& fname,
                       const std::vector<AudioBuffer<T> >& buf)
{
  FILE *fil = fopen(fname.c_str(), "wb");
  if (fil == NULL) PLFATAL("Cannot write %s", fname.c_str());

  // Write a .wav file. Format was gleaned from
  // http://www.borg.com/~jglatt/tech/wave.htm and from
  // http://ccrma.stanford.edu/courses/422/projects/WaveFormat/
  uint bufsiz = 0;
  for (uint i = 0; i < buf.size(); ++ i) bufsiz += buf[i].sizeBytes();

  const uint fsiz = bufsiz + 36;
  const uint freq = uint(buf[0].freq());
  const uint brate = freq * sizeof(T) * buf[0].nchans(); // byte rate
  const byte bps = byte(sizeof(T) * 8);
  const uint blockalign = buf[0].nchans() * sizeof(T);

  LDEBUG("buffer: freq=%u, brate=%u, bps=%d",
         freq, brate, int(bps));

  const byte header[44] =
    {
      /* 0-3    */ 'R', 'I', 'F', 'F',

      /* 4-5    */ byte(fsiz & 0xFF), byte((fsiz >> 8) & 0xFF),
      /* 6-7    */ byte((fsiz >> 16) & 0xFF), byte((fsiz >> 24) & 0xFF),

      /* 8-21   */ 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ', 0x10, 0, 0, 0, 1, 0,
      /* 22-23  */ byte(buf[0].nchans()), 0,

      /* 24-25  */ byte(freq & 0xFF), byte((freq >> 8) & 0xFF),
      /* 26-27  */ byte((freq >> 16) & 0xFF), byte((freq >> 24) & 0xFF),

      /* 28-29  */ byte(brate & 0xFF), byte((brate >> 8) & 0xFF),
      /* 30-31  */ byte((brate >> 16) & 0xFF), byte((brate >> 24) & 0xFF),

      /* 32-39  */ byte(blockalign & 0xFF), byte((blockalign >> 8) & 0xFF),
      /* 34-39  */ bps, 0, 'd', 'a', 't', 'a',

      /* 40-41  */ byte(bufsiz & 0xFF), byte((bufsiz >> 8) & 0xFF),
      /* 42-43  */ byte((bufsiz >> 16) & 0xFF), byte((bufsiz >> 24) & 0xFF)
    };

  if (fwrite(header, 44, 1, fil) != 1)
    PLFATAL("Error writing header to %s", fname.c_str());

  for (uint i = 0; i < buf.size(); ++ i)
    if (fwrite(buf[i].getDataPtr(), buf[i].sizeBytes(), 1, fil) != 1)
      PLFATAL("Error writing data chunk %u to %s", i, fname.c_str());

  fclose(fil);

  LINFO("%d bytes written to %s", bufsiz, fname.c_str());
}

// ######################################################################
template <class T>
void readAudioWavFile(const std::string& fname, AudioBuffer<T>& buf)
{
  FILE *fil = fopen(fname.c_str(), "rb");
  if (fil == NULL) PLFATAL("Cannot read %s", fname.c_str());

  // Read a .wav file. Format was gleaned from
  // http://www.borg.com/~jglatt/tech/wave.htm and from
  // http://ccrma.stanford.edu/courses/422/projects/WaveFormat/
  byte header[44];
  if (fread(header, 44, 1, fil) != 1)
    PLFATAL("Error reading header from %s", fname.c_str());

  // decode header:
  if (header[0]!='R' || header[1]!='I' || header[2]!='F' || header[3]!='F')
    LFATAL("Wrong magic - %s is not a .wav file", fname.c_str());

  const uint nchan = header[22];
  const uint freq = header[24] | (header[25]<<8) |
    (header[26]<<16) | (header[27]<<24);
  const uint blockalign = header[32] + (header[33]<<8);
  const uint bps = header[34];
  const uint bufsiz =
    header[40] | (header[41]<<8) | (header[42]<<16) | (header[43]<<24);

  LDEBUG("%s: nchan=%u, freq=%u, blockalign=%u, bps=%u, bufsiz=%u",
         fname.c_str(), nchan, freq, blockalign, bps, bufsiz);

  if (bps != sizeof(T) * 8)
    LFATAL("Wrong bps: file=%u, buffer=%u", bps, uint(sizeof(T)));

  // initialize the AudioBuffer:
  AudioBuffer<T> b(bufsiz / (nchan * (bps / 8)), nchan, float(freq), NO_INIT);

  // read the audio data:
  if (fread(b.getDataPtr(), bufsiz, 1, fil) != 1)
    PLFATAL("Error reading data from %s", fname.c_str());

  // return buffer:
  fclose(fil);
  buf = b;
}

// template instantiations:
template void writeAudioWavFile(const std::string& fname,
                                const std::vector<AudioBuffer<byte> >& buf);
template void writeAudioWavFile(const std::string& fname,
                                const std::vector<AudioBuffer<uint16> >& buf);
template void writeAudioWavFile(const std::string& fname,
                                const std::vector<AudioBuffer<int16> >& buf);

template void writeAudioWavFile(const std::string& fname,
                                const AudioBuffer<byte>& buf);
template void writeAudioWavFile(const std::string& fname,
                                const AudioBuffer<uint16>& buf);
template void writeAudioWavFile(const std::string& fname,
                                const AudioBuffer<int16>& buf);

template void readAudioWavFile(const std::string& fname,
                               AudioBuffer<byte>& buf);
template void readAudioWavFile(const std::string& fname,
                               AudioBuffer<uint16>& buf);
template void readAudioWavFile(const std::string& fname,
                               AudioBuffer<int16>& buf);

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // DEVICES_AUDIOWAVFILE_C_DEFINED
