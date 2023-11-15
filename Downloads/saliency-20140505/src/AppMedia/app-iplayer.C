/*!@file AppMedia/app-iplayer.C An awesome ilab media player,
   basically a ifstream and ofstream with some added key commands for
   forward, backward, pause, and to mark frame numbers.  */

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
// Primary maintainer for this file: David J. Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-iplayer.C $

#ifndef APPMEDIA_APP_STREAM_C_DEFINED
#define APPMEDIA_APP_STREAM_C_DEFINED

#define INIT_BUFF_LENGTH 100

#include "Component/ModelManager.H"
#include "Media/Streamer.H"
#include "Util/StringConversions.H"
#include "Devices/KeyBoard.H"
#include "Raster/Raster.H"

#include <iostream>
#include <fstream>
#include <deque>
#include <vector>

class iPlayer : public Streamer
{
public:
  iPlayer()
    :
    Streamer("iPlayer"),
    itsOutPrefix("iplayer-output"),
    itsKeyBoard(),
    itsBuffer(),
    itsFrameCount(-1),
    itsStart(), 
    itsEnd(),
    itsEdits(),
    itsOutFile(),
    itsBuffLength(INIT_BUFF_LENGTH)
  { };

  ~iPlayer()
  {
    if ( (itsStart.size() > 0) && (itsEnd.size() > 0) )
      {
        if (itsStart.size() < itsEnd.size())
          itsEnd.pop_back();
        else if (itsStart.size() > itsEnd.size())
          itsStart.pop_back();
        
        for (uint ii = 0; ii < itsStart.size(); ++ii)
          itsOutStream << itsStart[ii] << " " << itsEnd[ii] << std::endl;
        
        itsOutStream.close();
      }
  }
private:
  //handle any extra args
  virtual void handleExtraArgs(const ModelManager& mgr)
  { 
    itsOutFile = mgr.getExtraArg(0);
    itsOutStream.open(itsOutFile.c_str(), std::ios::out | std::ios::app);

    if (mgr.numExtraArgs() > 1)
      itsBuffLength = fromStr<uint>(mgr.getExtraArg(1));

    if (!itsOutStream.is_open())
      LFATAL("Could not open '%s' for output.", itsOutFile.c_str());

    LINFO("Key bindings: ");
    LINFO("<space>: pause");
    LINFO("<,>: backward");
    LINFO("<.>: forward");
    LINFO("<z>: mark start of sequence");
    LINFO("<x>: mark end of sequence");
    LINFO("<u>: undo add sequence start/end");
    LINFO(" ");
    LINFO(" ");
  };
  
  //process on every frame
  virtual void onFrame(const GenericFrame& input,
                       FrameOstream& ofs,
                       const int frameNum)
  {
    itsBuffer.push_back(input);
    if (itsBuffer.size() > itsBuffLength)
      itsBuffer.pop_front();
    

    ofs.writeFrame(itsBuffer.back(), itsOutPrefix,
                   FrameInfo("copy of input frame", SRC_POS));

    ++itsFrameCount;
    LINFO("showing frame: %d ", itsFrameCount);
    
    int key = itsKeyBoard.getKeyAsChar(false);
    switch (key)
      {
      case 32: //space
        {
          LINFO("entering edit mode");
          key = -1;
          uint itsEditPos = itsFrameCount; 
          uint itsBufPos = itsBuffer.size() - 1;
          
          while (key != 32)//keep in edit mode until space is again pressed
            {
              key = itsKeyBoard.getKeyAsChar(true);   
              switch (key)
                {
                case 27://escape key
                  LFATAL("Playback aborted by user, exiting.");
                  break;

                case 44: //,
                  {
                    LINFO("back one frame");
                    if (itsBufPos <= 0)
                      LINFO("cannot seek backward any further.");
                    else
                      {
                        --itsBufPos;
                        --itsEditPos;
                      }
                  }
                  break;
                  
                case 46: //. 
                  {
                    LINFO("forward one frame");
                    if ( itsBufPos >= (itsBuffer.size() - 1)  )
                      LINFO("cannot seek forward any further.");
                    else
                      {
                        ++itsBufPos;
                        ++itsEditPos;
                      }
                  }
                  break;
                  
                case 122://z
                  {
                    int diff = itsStart.size() - itsEnd.size();
                    if (diff > 0)
                      LINFO("You have attempted to add two clip onset markers "
                            "in a row, ignoring command");
                    else
                      {
                        LINFO("marked clip onset");
                        itsStart.push_back(itsEditPos);
                        itsEdits.push_back(ADD_START);
                      }
                  }
                  break;
                  
                case 120://x
                  {
                    int diff = itsStart.size() - itsEnd.size();
                    if (diff < 0) 
                      LINFO("You have attempted to add two clip offset markers "
                            "in a row, ignoring command");
                    else
                      {
                        LINFO("marked clip offset");
                        itsEnd.push_back(itsEditPos);
                        itsEdits.push_back(ADD_END);
                      }
                  }
                  break;
                  
                case 117://u
                  {
                    if (itsEdits.size() > 0)
                      {
                        COMMAND last = itsEdits.back();
                        LINFO("undoing last edit: '%s' ", 
                              last ? "mark end" : "mark start");
                        itsEdits.pop_back();
                        
                        if (last)
                          itsEnd.pop_back();
                        else
                          itsStart.pop_back();
                      }
                    else
                      LINFO("No more edits to undo");
                  }
                  break;
                }
              
              //now show a frame since we have recieved a command
              ofs.writeFrame(itsBuffer[itsBufPos], itsOutPrefix,
                             FrameInfo("copy of input frame", SRC_POS));
              LINFO("showing frame: %d ", itsEditPos);
            }
          LINFO("exiting edit mode");    
        }
        break;
        
      case 27://escape key
        LFATAL("Playback aborted by user, exiting.");
        break;
      }
  };

  enum COMMAND {ADD_START = 0, ADD_END = 1};//to hold our edit states  

  std::string itsOutPrefix; //our output prefix
  KeyBoard itsKeyBoard; //a hook to our keyboard
  std::deque<GenericFrame> itsBuffer; //hold our generic frames
  int itsFrameCount; //keep track of which frame we are on
  std::vector<uint> itsStart, itsEnd;
  std::vector<COMMAND> itsEdits;
  std::string itsOutFile;
  std::ofstream itsOutStream;
  uint itsBuffLength;
};

int main(const int argc, const char **argv)
{
  iPlayer s;
  return s.run(argc, argv, "<output file name>", 1, 2);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_STREAM_C_DEFINED
