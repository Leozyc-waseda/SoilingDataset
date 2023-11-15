/*!@file AppNeuro/app-combineOptimalGains.C combine optimal gains files */

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
// Primary maintainer for this file: Vidhya Navalpakkam <navalpak@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/app-combineOptimalGains.C $
// $Id: app-combineOptimalGains.C 11159 2009-05-02 03:04:10Z itti $
//

#include "Channels/OptimalGains.H"
#include "Component/ModelManager.H"
#include "Component/ParamMap.H"
#include "Component/ModelOptionDef.H"
#include "Neuro/NeuroOpts.H"
#include "Util/sformat.H"

static const ModelOptionDef OPT_SaveTo =
  { MODOPT_ARG(std::string), "SaveTo", &MOC_DISPLAY, OPTEXP_SAVE,
    "Save output gains to file if name specified (rather than stdout)",
    "save-to", '\0', "", "" };

// ######################################################################
    /* combine params:
       1. for each i, find the mean salience s_iT, s_iD across all inputs
       2. find SNR_i = s_iT / s_iD
       3. find g_i = SNR_i / (sum(SNR_i) / n)
    */
rutz::shared_ptr<ParamMap>
combineParamMaps(std::vector<rutz::shared_ptr<ParamMap> > pmaps,
                 const uint indent)
{
  rutz::shared_ptr<ParamMap> outpmap(new ParamMap());
  const std::string id(indent, ' ');

  // take the first pmap as reference for the keys, and loop over keys:
  ParamMap::key_iterator
    itr = pmaps[0]->keys_begin(), stop = pmaps[0]->keys_end();

  while (itr != stop) {
    const std::string name = *itr;

    // find a submap or copy the subchanidx info:
    if (pmaps[0]->isLeaf(name)) {
      if (name.compare("subchanidx") == 0)
        outpmap->putIntParam(name, pmaps[0]->getIntParam(name));
    } else {
      // it's a subpmap, let's recurse through it: for that, we need
      // to extract the subpmaps from all our input pmaps. This will
      // LFATAL if there is inconsistency among our input pmaps:
      std::vector<rutz::shared_ptr<ParamMap> > subpmaps;
      for (uint i = 0; i < pmaps.size(); i ++)
        subpmaps.push_back(pmaps[i]->getSubpmap(name));

      // recurse:
      LDEBUG("%s%s:", id.c_str(), name.c_str());
      outpmap->putSubpmap(name, combineParamMaps(subpmaps, indent + 2));
    }
    ++itr;
  }

  // now compute the gain at our level:
  uint i = 0;
  double sumSNR = 0.0; std::vector<double> SNR;
  while (pmaps[0]->hasParam(sformat("salienceT(%d)", i)))
    {
      double sT = 0.0, sD = 0.0;
      for (uint j = 0; j < pmaps.size(); j ++)
        {
          sT += pmaps[j]->getDoubleParam(sformat("salienceT(%d)", i));
          sD += pmaps[j]->getDoubleParam(sformat("salienceD(%d)", i));
        }
      sT /= pmaps.size(); sD /= pmaps.size();

      // compute SNR:
      const double snr = (sT + OPTIGAIN_BG_FIRING) / (sD + OPTIGAIN_BG_FIRING);
      SNR.push_back(snr);
      sumSNR += snr;
      ++i;
    }
  sumSNR /= SNR.size();

  // find the optimal gains
  for (uint idx = 0; idx < SNR.size(); idx ++)
    {
      const double g = SNR[idx] / sumSNR;
      LDEBUG("%sgain(%d) = %f, SNR = %f", id.c_str(), idx, g, SNR[idx]);
      outpmap->putDoubleParam(sformat("gain(%d)", idx), g);
    }

  return outpmap;
}

// ######################################################################
//! Combine several stsd.pmap files into a gains.pmap file
/*! The stsd.pmap files should be obtained by running something like:

    ezvision --in=xmlfile:testfile.xml --out=display --pfc-type=OG
         --vc-type=Std --vc-chans=GNO --stsd-filename=stsd.pmap --nouse-older-version

    which will compute the salience of target and distractor and save
    those to a ParamMap. Once you have several of these (or just one),
    you can use the present program to generate an optimal gains
    ParamMap (it is written to stdout). Finally, you can use these
    gains for biased saliency computations using a PrefrontalCortexGS,
    e.g.:

   ezvision --in=image.png --out=display -X --pfc-type=GS --vc-type=Std --vc-chans=GNO
         --gains-filename=gains.pmap --nouse-older-version
*/
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Optimal Gains Combiner");

  OModelParam<std::string> saveTo(&OPT_SaveTo, &manager);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<stsd1.stsd> ... <stsdN.stsd> [--save-to=out.pmap]", 1, -1) == false)
    return(1);

  // do post-command-line configs:
  std::vector<rutz::shared_ptr<ParamMap> > pmaps;
  for (uint i = 0; i < manager.numExtraArgs(); ++i)
    {
      LINFO("Loading: %s", manager.getExtraArg(i).c_str());
      pmaps.push_back(ParamMap::loadPmapFile(manager.getExtraArg(i)));
    }

  rutz::shared_ptr<ParamMap> outpmap = combineParamMaps(pmaps, 0);

  // write output:
  if (saveTo.getVal().empty() == false) outpmap->format(saveTo.getVal());
  else outpmap->format(std::cout);

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
