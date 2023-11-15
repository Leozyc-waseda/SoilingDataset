/*!@file Apps/NeovisionII/a---ChipValidatorGui.C Simple app to validate chips */

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
// Primary maintainer for this file: Laurent Itti
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Apps/BorderWatch/BorderWatchGui.C $
// $Id: BorderWatchGui.C 13059 2010-03-26 08:14:32Z itti $
//

#include "QtUtil/Util.H" // for argv2qt()
#include "NeovisionII/ChipValidator/ChipValidatorQt.qt.H"
#include "Image/Image.H"
#include "Image/PixelsTypes.H"
#include "Raster/Raster.H"
#include "Util/sformat.H"

#include <QtGui/QApplication>

#include <sys/types.h>
#include <dirent.h> 
#include <sys/stat.h>

bool sortChips(const ChipData& d1, const ChipData& d2)
{
  return d1.file < d2.file;
}

//! load all chips from a directory
void loadChips(std::vector<ChipData>& vec, const std::string& path, const bool positive)
{
  DIR *dir = opendir(path.c_str());
  if (dir == NULL) PLFATAL("Cannot opendir '%s'", path.c_str());

  struct dirent *entry;
  while ( (entry = readdir(dir)) ) {
    if (entry->d_name[0] == '.') continue;
    ChipData cdata;
    std::string fullname = sformat("%s/%s", path.c_str(), entry->d_name);
    cdata.image = Raster::ReadRGB(fullname);
    cdata.file = entry->d_name;
    cdata.positive = positive;

    vec.push_back(cdata);
  }
  // Sort the chips by file name - useful if similar files are similarly named
  std::sort(vec.begin(),vec.end(),sortChips);
  if (closedir(dir)) PLFATAL("Error closing directory '%s'", path.c_str());
}

//! Chip Validator gui
/*! Directory structure assumes subdirs as follows:
  - tp/ true positives
  - tn/ true negatives */
int main(int argc, const char **argv)
{
  LOG_FLAGS &= (~LFATAL_XTRA_NOISY); LOG_FLAGS &= (~LFATAL_PRINTS_ABORT);

  if (argc != 3) LFATAL("USAGE: %s <indir> <outdir>", argv[0]);

  // make sure output directory does not exist:
  mode_t mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
  if (mkdir(argv[2], mode)) PLFATAL("Error creating output directory '%s'", argv[2]);
  if (mkdir(sformat("%s/tp", argv[2]).c_str(), mode)) PLFATAL("Error creating output directory '%s/tp/'", argv[2]);
  if (mkdir(sformat("%s/tn", argv[2]).c_str(), mode)) PLFATAL("Error creating output directory '%s/tn/'", argv[2]);

  // load all the input chips:
  std::vector<ChipData> chipvec;
  LINFO("Loading true positive chips from '%s/tp/' ...", argv[1]);
  loadChips(chipvec, sformat("%s/tp", argv[1]), true);
  LINFO("Loading true negative chips from '%s/tn/' ...", argv[1]);
  loadChips(chipvec, sformat("%s/tn", argv[1]), false);
  LINFO("  ... loaded %" ZU " chips in total", chipvec.size());

  // create a QApplication:
  LINFO("Starting GUI...");
  int qtargc = 1; const char* qtargv[1] = { "ChipValidatorGui" };
  QApplication a(qtargc, argv2qt(qtargc, qtargv));

  // and a widget:
  const Dims griddims(9, 5);
  ChipValidatorQt cqt(&a, chipvec, griddims);
  cqt.setWindowTitle("iLab USC -- Chip Validator GUI");
  cqt.show();

  // main loop for QApplication:
  const int ret = a.exec();

  // save all the chips:
  LINFO("Saving %" ZU " validated chips to '%s'", chipvec.size(), argv[2]);
  for (size_t i = 0; i < chipvec.size(); ++i)
    Raster::WriteRGB(chipvec[i].image,
		     sformat("%s/%s/%s", argv[2], chipvec[i].positive ? "tp" : "tn", chipvec[i].file.c_str()));
  LINFO("All Done.");

  // cleanup and exit:
  return ret;
}
