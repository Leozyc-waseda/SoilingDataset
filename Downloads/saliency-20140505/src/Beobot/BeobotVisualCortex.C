/*!@file Beobot/BeobotVisualCortex.C Implementation of navigation algorithm */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotVisualCortex.C $
// $Id: BeobotVisualCortex.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Beobot/BeobotVisualCortex.H"

#include "Beobot/beobot-defs.H"
#include "Channels/Jet.H"
#include "Image/ColorOps.H" // for luminance(), getRGBY(), etc.
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H" // for downSize()
#include "Image/Transforms.H"
#include "Image/fancynorm.H" // for maxNormalize()
#include "Util/Assert.H"
#include "Util/Timer.H"

#include <cmath> // for sqrt()
#include <limits> // for numeric_limits<float>::max() instead of FLOATMAX


//######################################################################
BeobotVisualCortex::BeobotVisualCortex()
{ initialized = false; }

// ######################################################################
void BeobotVisualCortex::init(const int imgw, const int imgh,
                              const int lev_min, const int lev_max,
                              const int delta_min, const int delta_max,
                              const int smlev, const int nborient,
                              const MaxNormType normtype, const int jlev,
                              const int jdepth, const int nbneig,
                              nub::soft_ref<Beowulf> beow)
{
  iw = imgw; ih = imgh;
  lmin = lev_min; lmax = lev_max; dmin = delta_min; dmax = delta_max;
  sml = smlev; nori = nborient; nortyp = normtype; beo = beow;
  jetlevel = jlev; jetdepth = jdepth; nbneigh = nbneig;

  scene.resize(iw, ih, true);

  JetSpec *js = new JetSpec;
  js->addIndexRange(RG, RAW, jlev, jlev + jdepth - 1);
  js->addIndexRange(BY, RAW, jlev, jlev + jdepth - 1);
  js->addIndexRange(INTENS, RAW, jlev, jlev + jdepth - 1);
  js->addIndexRange(ORI, RAW, 0, nborient - 1);  // orientations
  js->addIndexRange(ORI, RAW, jlev, jlev + jdepth - 1);
  js->print();
  jetSpec = rutz::make_shared(js);

  jets.init(iw >> jetlevel, ih >> jetlevel, nbneigh);
  ImageSpring<Jet <float> >::iterator itr = jets.beginw(), stop = jets.endw();
  while (itr != stop) { itr->init(jetSpec); itr ++; }

  sminput.resize(iw >> sml, ih >> sml, true);
  initialized = true;
}

//#######################################################################
void BeobotVisualCortex::newVisualInput(Image< PixRGB<byte> >& newscene)
{ scene = newscene; iw = scene.getWidth(); ih = scene.getHeight(); }

// ######################################################################
Image< PixRGB<byte> >* BeobotVisualCortex::getScenePtr()
{ return &scene; }

// ######################################################################
void BeobotVisualCortex::process(const int frame)
{
  currframe = frame;
  sminput.clear(); nbcmap = 0; nbjmap = 0;

  if (beo.get())  // use parallel version (we are master)
    {
      masterProcess(frame);

      while (nbcmap != NBCMAP || nbjmap != NBJMAP) masterCollect();
    }
  else      // process everything on current CPU
    singleCPUprocess(frame);

  // finalize computation of saliency map input:
  sminput = maxNormalize(sminput, 0.0f, 9.0f, nortyp);

  // find most salient location:
  float maxval;
  findMax(sminput, winner, maxval);

  // rescale those coordinates to scale of original image:
  winner.i <<= sml; winner.i += 1 << (sml - 1);
  winner.j <<= sml; winner.j += 1 << (sml - 1);
}

// ######################################################################
void BeobotVisualCortex::processStart(const int frame)
{
  if (beo.isInvalid()) LFATAL("This is for parallel processing only!");
  currframe = frame;
  masterProcess(frame);
}

// ######################################################################
void BeobotVisualCortex::processEnd(const int frame)
{
  if (beo.isInvalid()) LFATAL("This is for parallel processing only!");
  sminput.clear(); nbcmap = 0; nbjmap = 0; currframe = frame; Timer tim;
  while((nbcmap != NBCMAP || nbjmap != NBJMAP) && tim.get() < 80)
    {
      masterCollect();
      //LINFO("frame %d: nc=%d nj=%d",frame,nbcmap,nbjmap);
    }

  // finalize computation of saliency map input:
  sminput = maxNormalize(sminput, 0.0f, 9.0f, nortyp);

  // find most salient location:
  float maxval;
  findMax(sminput, winner, maxval);

  // rescale those coordinates to scale of original image:
  winner.i <<= sml; winner.i += 1 << (sml - 1);
  winner.j <<= sml; winner.j += 1 << (sml - 1);
}

// ######################################################################
void BeobotVisualCortex::singleCPUprocess(const int frame)
{
  currframe = frame;

  Image<float> lumf = luminance(scene); // from Image_ColorOps.H

  Image<float> rgf, byf;
  getRGBY(scene, rgf, byf, byte(25)); // from Image_ColorOps.H

  // compute intensity:
  ImageSet<float> intensPyr = buildPyrGaussian(lumf, 0, sml + dmax + 1, 5);

  // compute colors:
  ImageSet<float> rgPyr = buildPyrGaussian(rgf, 0, sml + dmax + 1, 5);
  ImageSet<float> byPyr = buildPyrGaussian(byf, 0, sml + dmax + 1, 5);

  // compute orientations:
  ImageSet<float> oriPyr[nori];
  for (int i = 0; i < nori; i ++)
    {
      oriPyr[i] = buildPyrOriented(lumf, 0, sml + dmax + 1, 5,
                                   float(i) * 180.0f / float(nori));
    }

  // compute flicker:
  if (prevlum.initialized() == false) prevlum = lumf;
  prevlum -= lumf;
  ImageSet<float> flickPyr = buildPyrGaussian(prevlum, 0, sml + dmax + 1, 5);
  prevlum = lumf;

  // compute comspicuity maps and accumulate into sminput:
  Image<float> cmap;
  sminput.resize(intensPyr[sml].getDims(), true);
  computeCmap(intensPyr, cmap); sminput += cmap;
  computeCmap(rgPyr, cmap); sminput += cmap;
  computeCmap(byPyr, cmap); sminput += cmap;
  computeCmap(flickPyr, cmap); sminput += cmap;
  for (int i = 0; i < nori; i++)
    { computeCmap(oriPyr[i], cmap); sminput += cmap; }

  // fill-in the jets -- IGNORE FLICKER CHANNEL:
  ImageSpring< Jet<float> >::iterator jt = jets.beginw();
  Point2D<int> p; int step = 1 << jetlevel;
  int jmax = jets.getHeight() * step, imax = jets.getWidth() * step;
  for (p.j = 0; p.j < jmax; p.j += step)
    for (p.i = 0; p.i < imax; p.i += step)
      {
        for (int k = jetlevel; k < jetlevel + jetdepth; k ++)
          {
            float rgval = getPyrPixel(rgPyr, p, k); jt->setVal(rgval, RG, RAW, k);
            float byval = getPyrPixel(byPyr, p, k); jt->setVal(byval, BY, RAW, k);
            float ival = getPyrPixel(intensPyr, p, k); jt->setVal(ival, INTENS, RAW, k);
            for (int o = 0; o < nori; o ++)
              {
                float oval = getPyrPixel(oriPyr[o], p, k);
                jt->setVal(oval, ORI, RAW, o, k);
              }
          }
        jt ++;
      }

  nbcmap = NBCMAP; nbjmap = NBJMAP;
}

//######################################################################
void BeobotVisualCortex::masterProcess(const int frame)
{
  ASSERT(initialized);

  TCPmessage smsg;
  currframe = frame; nbcmap = 0;

  // compute luminance and send it off:
  Image<byte> lum = luminance(scene);
  smsg.reset(frame, BEO_LUMFLICK);
  smsg.addImage(lum);
  beo->send(0, smsg);  // send off to intensity/RG/BY/flick slave
  smsg.setAction(BEO_ORI0_45);
  beo->send(1, smsg);  // send off to ori0/ori45 slave
  smsg.setAction(BEO_ORI90_135);
  beo->send(2, smsg);  // send off to ori90/ori135 slave

  // compute RG and BY and send them off:
  Image<byte> r, g, b, y; getRGBY(scene, r, g, b, y, (byte)25);
  smsg.reset(frame, BEO_REDGREEN);
  smsg.addImage(r); smsg.addImage(g);
  beo->send(0, smsg);  // send off to intensity/RG/BY/flick slave
  smsg.reset(frame, BEO_BLUEYELLOW);
  smsg.addImage(b); smsg.addImage(y);
  beo->send(0, smsg);  // send off to intensity/RG/BY/flick slave
}

// ######################################################################
void BeobotVisualCortex::slaveProcess()
{
  TCPmessage rmsg; int rframe, raction, rnode = -1;  // receive from any node
  int nbrec = 0;
  while (beo->receive(rnode, rmsg, rframe, raction, 5)) // wait up to 5ms
    {
      //LINFO("GOT frame %d, action %d from node %d", rframe, raction, rnode);
      //Timer tim;
      switch(raction)
        {
        case BEO_INIT:       // ##############################
          {
            // ooops, someone wants to re-initialize us!
            // reinitialization of beowulf is handled automatically.
          }
          break;
        case BEO_LUMFLICK:     // ##############################
          {
            // get the luminance image out of the message:
            Image<byte> ima = rmsg.getElementByteIma();
            Image<float> fima = ima;  // convert to float

            // compute intensity maps and send to collector:
            computeFeature(fima, Gaussian5, 0.0, rframe, BEO_FMAP_I);

            // compute flicker maps and send to collector:
            if (prevlum.initialized() == false) prevlum = fima;
            prevlum -= fima;
            computeFeature(prevlum, Gaussian5, 0.0, rframe, BEO_FMAP_F);
            prevlum = fima;
          }
          break;
        case BEO_REDGREEN:   // ##############################
          computeFeature2(rmsg, Gaussian5, 0.0, BEO_FMAP_RG);
          break;
        case BEO_BLUEYELLOW: // ##############################
          computeFeature2(rmsg, Gaussian5, 0.0, BEO_FMAP_BY);
          break;
        case BEO_ORI0_45:    // ##############################
          {
            // get the luminance image out of the message:
            Image<byte> ima = rmsg.getElementByteIma();
            Image<float> fima = ima;  // convert to float

            // compute 0deg orientation maps and send to collector:
            computeFeature(fima, Oriented5, 0.0, rframe, BEO_FMAP_O0);

            // compute 45deg orientation maps and send to collector:
            computeFeature(fima, Oriented5, 45.0, rframe, BEO_FMAP_O45);
          }
          break;
        case BEO_ORI90_135:  // ##############################
          {
            // get the luminance image out of the message:
            Image<byte> ima = rmsg.getElementByteIma();
            Image<float> fima = ima;  // convert to float

            // compute 90deg orientation maps and send to collector:
            computeFeature(fima, Oriented5, 90.0, rframe, BEO_FMAP_O90);

            // compute 135deg orientation maps and send to collector:
            computeFeature(fima, Oriented5, 135.0, rframe, BEO_FMAP_O135);
          }
          break;
        default: // ##############################
          LERROR("Bogus action %d -- IGNORING.", raction);
          break;
        }
      //LINFO("Job %d/%d from %d completed in %dms",
      //      rframe, raction, rnode, tim.get());

      // limit number of receives, so we don't hold CPU too long:
      nbrec ++; if (nbrec > 3) break;
    }
}

// ######################################################################
void BeobotVisualCortex::masterCollect()
{
  // receive various conspicuity maps
  int32 rframe, raction, rnode = -1, recnb = 0;  // receive from any node
  TCPmessage rmsg;
  while(beo->receive(rnode, rmsg, rframe, raction, 5)) // wait up to 5ms
    {
      //LINFO("received %d/%d from %d while at %d",
      //      rframe, raction, rnode, currframe);
      if (rframe != currframe)
        {
          LERROR("Dropping old map, type %d for frame %d while at frame %d",
                 raction, rframe, currframe);
          continue;
        }

      // collect conspicuity maps:
      if (raction == BEO_CMAP)
        {
          nbcmap ++;

          // get the map:
          Image<float> ima = rmsg.getElementFloatIma();

          // add received cmap to saliency map input:
          sminput += ima;
        }

      // collect feature maps:
      if (raction >= BEO_FMAP_RG && raction <= BEO_FMAP_O135)
        {
          nbjmap ++;

          VisualFeature jf = COLOR;
          std::vector<int> v;
          switch(raction)
            {
            case BEO_FMAP_RG: jf = RG; break;
            case BEO_FMAP_BY: jf = BY; break;
            case BEO_FMAP_I: jf = INTENS; break;
            case BEO_FMAP_F: jf = FLICKER; break;
            case BEO_FMAP_O0: jf = ORI; v.push_back(0); break;
            case BEO_FMAP_O45: jf = ORI; v.push_back(1); break;
            case BEO_FMAP_O90: jf = ORI; v.push_back(2); break;
            case BEO_FMAP_O135: jf = ORI; v.push_back(3); break;
            default: LFATAL("Bogus feature map type %d", raction);
            }
          v.push_back(0);  // add an index for the scale

          // get the maps:
          Image<float> ima[jetdepth];
          for (int i = 0; i < jetdepth; i ++)
            {
              ima[i] = rmsg.getElementFloatIma();
            }

          // fill in the jets -- IGNORE FLICKER CHANNEL:
          if (jf != FLICKER) {
            ImageSpring< Jet<float> >::iterator jet_itr = jets.beginw();

            for (int j = 0; j < jets.getHeight(); j ++)
              for (int i = 0; i < jets.getWidth(); i ++)
                {
                  float ii = float(i), jj = float(j);
                  for (int k = 0; k < jetdepth; k ++)
                    {
                      float ii2 = ii, jj2 = jj;
                      if (ii2 > ima[k].getWidth()-1)
                        ii2 = ima[k].getWidth()-1;
                      if (jj2 > ima[k].getHeight()-1)
                        jj2 = ima[k].getHeight()-1;

                      v[v.size() - 1] = k + jetlevel;

                      // set the jet, using bilinear interpolation:
                      jet_itr->setValV(ima[k].getValInterp(ii2, jj2),
                                       jf, RAW, v);

                      // ready for next scale:
                      ii *= 0.5f; jj *= 0.5f;
                    }
                }
          }
        }
      //LINFO("frame: %d nbcmap=%d nbjmap=%d", currframe, nbcmap, nbjmap);

      // limit number of receives, so we don't hold CPU for too long:
      recnb ++; if (recnb > 20) break;
    }
}

// ######################################################################
void BeobotVisualCortex::getWinner(Point2D<int>& win) const
{ win.i = winner.i; win.j = winner.j; }

//######################################################################
void BeobotVisualCortex::initSprings(bool initPosMasses)
{ jets.initClustering(initPosMasses); }


// ######################################################################
void BeobotVisualCortex::iterateSprings(const float dt)
{ jets.computePos(dt); }

//######################################################################
void BeobotVisualCortex::getClusteredImage(Image< PixRGB<byte> >
                                           &clusteredImage,
                                           Point2D<int> &supposedTrackCentroid,
                                           const Point2D<int>&
                                           previousTrackCentroid)
{
  jets.getClusteredImage(scene, clusteredImage, supposedTrackCentroid,
                         previousTrackCentroid);
}

// ######################################################################
void BeobotVisualCortex::getPositions(Image< PixRGB<byte> > &img,
                                      const int zoom)
{ jets.getPositions(img, zoom); }

// ######################################################################
void BeobotVisualCortex::computeFeature(TCPmessage &rmsg,
                                        const PyramidType ptyp,
                                        const float ori,
                                        const int maptype)
{
  // get the image out of the message:
  Image<byte> ima = rmsg.getElementByteIma();
  Image<float> fima = ima;  // convert to float

  // compute maps and send to collector:
  computeFeature(fima, ptyp, ori, rmsg.getID(), maptype);
}

// ######################################################################
void BeobotVisualCortex::computeFeature2(TCPmessage &rmsg,
                                        const PyramidType ptyp,
                                        const float ori,
                                        const int maptype)
{
  // get the two images out of the message:
  Image<byte> ima1 = rmsg.getElementByteIma();
  Image<byte> ima2 = rmsg.getElementByteIma();
  Image<float> fima = ima1 - ima2;

  // compute maps and send to collector:
  computeFeature(fima, ptyp, ori, rmsg.getID(), maptype);
}

// ######################################################################
void BeobotVisualCortex::computeFeature(const Image<float>& fima,
                                        const PyramidType ptyp,
                                        const float ori,
                                        const int32 id, const int32 maptype)
{
  // compute pyramid:
  ImageSet<float> pyr = buildPyrGeneric(fima, 0, lmax + dmax + 1,
                                        ptyp, ori);

  // now send off a message with the raw features, for the jets:
  TCPmessage smsg(id, maptype);
  for (int i = 0; i < jetdepth; i ++)
    smsg.addImage(pyr.getImage(i + jetlevel));
  beo->send(-1, smsg);

  // compute conspicuity map:
  Image<float> cmap;
  computeCmap(pyr, cmap);

  // send cmap off to master:
  smsg.reset(id, BEO_CMAP);
  smsg.addImage(cmap);
  beo->send(-1, smsg);
}

// ######################################################################
void BeobotVisualCortex::computeCmap(const ImageSet<float>& pyr,
                                     Image<float>& cmap)
{
  // clear conspicuity map:
  cmap.resize(pyr[sml].getDims(), true);

  // compute conspicuity map from feature maps:
  for (int delta = dmin; delta <= dmax; delta ++)
    for (int lev = lmin; lev <= lmax; lev ++)
      {
        Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
        tmp = downSize(tmp, cmap.getWidth(), cmap.getHeight());
        inplaceAddBGnoise(tmp, 255.0);
        tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, nortyp);
        cmap += tmp;
      }
  if (nortyp == VCXNORM_MAXNORM)
    cmap = maxNormalize(cmap, MAXNORMMIN, MAXNORMMAX, nortyp);
  else
    cmap = maxNormalize(cmap, 0.0f, 0.0f, nortyp);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
