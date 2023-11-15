/*!@file AppMedia/app-wav-write.C wav file from text */

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
// Primary maintainer for this file: David J. Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-wav-write.C $

#include "Component/ModelManager.H"
#include "Audio/AudioBuffer.H"
#include "Audio/AudioWavFile.H"

#include "Util/StringConversions.H"
#include "Util/Types.H"

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Wav Writer");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "frequency <File1> ... <fileN>", 2, -1)==false)
    return(1);
  
  manager.start();
  const float frequency = fromStr<float>(manager.getExtraArg(0));
  const uint nfiles = manager.numExtraArgs(); 

  for (uint ii = 1; ii < nfiles; ++ii) {
    //gobble up our file
    float max = 0, min = 0;
    std::vector<float> signal;
    
    const std::string filename = manager.getExtraArg(ii);
    std::ifstream myfile(filename.c_str());
    if (myfile.is_open())
      {
        while (!myfile.eof() ) 
          {
            std::string line;
            getline (myfile, line);

            if (line != "")
              {
                const float val = fromStr<float>(line);
                max = (val > max) ? val : max;
                min = (val < min) ? val : min;            
                signal.push_back(val);
              }
          }
        myfile.close();
        
        //transfer to a C-style array
        const uint size = signal.size();
        int16 *buffer = new int16[size];
        buffer += size;
        for (uint jj = 0; jj < size; ++jj)
          {
            *buffer = int16( (signal.back() - min) / (max - min) * std::numeric_limits<int16>::max());
            signal.pop_back();
            --buffer;
          }

        AudioBuffer<int16> outputbuffer(buffer, size, 1, frequency);
        writeAudioWavFile(filename + ".wav", outputbuffer);
      }
    else
      LFATAL("Could not open '%s' for reading", 
             manager.getExtraArg(ii).c_str());    
  }
  
  // stop all our ModelComponents
  manager.stop();
  
  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
