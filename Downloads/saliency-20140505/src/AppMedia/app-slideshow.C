/*!@file AppMedia/app-slideshow.C */

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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-slideshow.C $
// $Id: app-slideshow.C 15310 2012-06-01 02:29:24Z itti $
//

#ifndef APPMEDIA_APP_SLIDESHOW_C_DEFINED
#define APPMEDIA_APP_SLIDESHOW_C_DEFINED

#include "GUI/XWinManaged.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/MatrixOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Image/SimpleFont.H"
#include "Raster/Raster.H"
#include "Util/FileUtil.H"
#include "rutz/time.h"
#include "rutz/unixcall.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <X11/keysym.h>

enum TransformType
  {
    TXTYPE_NONE,
    TXTYPE_TRANSPOSE,
    TXTYPE_BEST_FIT
  };

std::string convertToString(const TransformType txtype)
{
  switch (txtype)
    {
    case TXTYPE_NONE: return "none";
    case TXTYPE_TRANSPOSE: return "transpose";
    case TXTYPE_BEST_FIT: return "bestfit";
    }

  // default:
  return "invalid";
}

class random_sequence
{
  double m_current;
  double m_next;

  static int irange(double rval, int min, int max)
  {
    return int(rval * (max - min) + min);
  }

public:
  random_sequence()
  {
    srand48(time(NULL) / 2);

    m_current =  drand48();
    m_next = drand48();
  }

  int inext(int min, int max)
  {
    m_current = m_next;
    while (m_current == m_next)
      m_next = drand48();

    return irange(m_current, min, max);
  }

  int ipeek(int min, int max) const
  {
    return irange(m_next, min, max);
  }
};

template <class T>
Image<T> myDownSize(const Image<T>& src, const Dims& new_dims)
{
  if (src.getDims() == new_dims) return src;

  ASSERT(new_dims.isNonEmpty());

  Image<T> result = src;

  while (result.getWidth() > new_dims.w() * 2 &&
         result.getHeight() > new_dims.h() * 2)
    {
      result = decY(decX(quickLocalAvg2x2(result)));
    }

  return rescale(result, new_dims);
}

struct ImageInfo
{
  ImageInfo() {}

  ImageInfo(const Image<PixRGB<byte> >& raw, const Dims& dsize,
            const TransformType ttype,
            const RescaleType rtype)
    :
    rawdims(raw.getDims())
  {
    switch (ttype)
      {
      case TXTYPE_NONE: img = raw; break;
      case TXTYPE_TRANSPOSE: img = transpose(raw); break;
      case TXTYPE_BEST_FIT:
        if ( (dsize.w() >= dsize.h()) != (raw.getWidth() >= raw.getHeight()) )
          img = transpose(raw);
        else
          img = raw;
        break;
      }
    const Dims ssize = img.getDims();

    const double wratio = double(dsize.w()) / double(ssize.w());
    const double hratio = double(dsize.h()) / double(ssize.h());
    this->ratio = std::min(wratio, hratio);

    img = rescale(img, Dims(std::min(dsize.w(), int(ssize.w() * this->ratio)),
                            std::min(dsize.h(), int(ssize.h() * this->ratio))),
                  rtype);
  }

  Dims rawdims;
  Image<PixRGB<byte> > img;
  double ratio;
};

namespace aux
{
  void msg(const std::string& tag, const std::string& content)
  {
    fprintf(stderr, "%20s: %s\n", tag.c_str(), content.c_str());
  }

  bool file_exists(const std::string& fname)
  {
    struct stat statbuf;
    int res = ::stat(fname.c_str(), &statbuf);
    return (res == 0);
  }

  bool image_file_exists(const std::string& fname)
  {
    struct stat statbuf;
    int res = ::stat(fname.c_str(), &statbuf);
    if (res != 0)
      return false;

    if (statbuf.st_size == 0)
      {
        aux::msg("empty file", fname);
        return false;
      }

    return true;
  }

  ImageInfo build_scaled_pixmap(const std::string& fname, const Dims& dsize,
                                const TransformType ttype,
                                const RescaleType rtype)
  {
    const Image<PixRGB<byte> > raw = Raster::ReadRGB(fname);
    return ImageInfo(raw, dsize, ttype, rtype);
  }

  bool is_img_file(const std::string& fname)
  {
    return (hasExtension(fname, ".jpg")
            || hasExtension(fname, ".jpeg")
            || hasExtension(fname, ".gif")
            || hasExtension(fname, ".pnm")
            || hasExtension(fname, ".png"));
  }
}

class playlist
{
public:
  enum play_mode
    {
      SPINNING,
      JUMPING,
    };

private:

  const std::string m_list_file;
  std::vector<std::string> m_list;
  int m_idx;
  int m_guess_next;
  const rutz::shared_ptr<XWinManaged> m_widget;
  ImageInfo m_pixmap;
  std::map<std::string, ImageInfo> m_pixmap_cache;
  std::vector<std::string> m_purge_list;
  play_mode m_mode;
  int m_ndeleted;
  int m_nshown;
  int m_nmissed;
  int m_last_spin;
  rutz::time m_last_show_time;
  TransformType m_txtype;
  RescaleType m_rtype;
  bool m_didcache;
  bool m_looping;
  int m_loop_delay_power;
  bool m_show_overlay;
  random_sequence m_rseq;

public:
  playlist(const std::string& fname, rutz::shared_ptr<XWinManaged> widget)
    :
    m_list_file(fname),
    m_idx(0),
    m_guess_next(1),
    m_widget(widget),
    m_mode(SPINNING),
    m_ndeleted(0),
    m_nshown(0),
    m_nmissed(0),
    m_last_spin(1),
    m_txtype(TXTYPE_NONE),
    m_rtype(RESCALE_SIMPLE_BILINEAR),
    m_didcache(false),
    m_looping(false),
    m_loop_delay_power(0),
    m_show_overlay(true),
    m_rseq()
  {
    std::ifstream ifs(m_list_file.c_str());
    if (!ifs.is_open())
      LFATAL("couldn't open %s for reading", m_list_file.c_str());
    std::string line;

    while (std::getline(ifs, line))
      m_list.push_back(line);
  }

  void save()
  {
    aux::msg("write playlist", m_list_file);
    if (aux::file_exists(sformat("%s.bkp", m_list_file.c_str())))
      rutz::unixcall::remove(sformat("%s.bkp", m_list_file.c_str()).c_str());
    if (aux::file_exists(m_list_file))
      rutz::unixcall::rename(m_list_file.c_str(),
                             sformat("%s.bkp", m_list_file.c_str()).c_str());

    std::ofstream ofs(m_list_file.c_str());
    if (!ofs.is_open())
      LFATAL("couldn't open %s for writing", m_list_file.c_str());
    for (size_t i = 0; i < m_list.size(); ++i)
      ofs << m_list.at(i) << '\n';
    ofs.close();
  }

  void spin(int step)
  {
    m_mode = SPINNING;

    if (m_list.size() == 0)
      {
        m_idx = 0;
        m_last_spin = 0;
        m_guess_next = 0;
      }
    else
      {
        ASSERT(m_list.size() > 0);

        m_idx += step;
        while (m_idx < 0) m_idx += int(m_list.size());
        while (m_idx >= int(m_list.size())) m_idx -= int(m_list.size());
        m_last_spin = step;

        int guess_step = step;
        if (guess_step == 0) { guess_step = 1; }

        m_guess_next = m_idx + guess_step;
        while (m_guess_next < 0) m_guess_next += int(m_list.size());
        while (m_guess_next >= int(m_list.size())) m_guess_next -= int(m_list.size());
      }
  }

  void jump(int oldlength = -1, int adjust = 0)
  {
    m_mode = JUMPING;

    if (m_list.size() == 0)
      {
        m_idx = 0;
        m_guess_next = 0;
      }
    else
      {
        if (oldlength == -1)
          oldlength = m_list.size();

        ASSERT(m_list.size() >= 1);

        m_idx = m_rseq.inext(0, oldlength) + adjust;
        if (m_idx < 0) m_idx = 0;
        else if (size_t(m_idx) >= m_list.size()) m_idx = m_list.size() - 1;
        m_guess_next = m_rseq.ipeek(0, m_list.size());
      }
  }

  std::string filename() const
  {
    if (m_list.size() == 0)
      return "(none)";

    ASSERT(size_t(m_idx) < m_list.size());
    return m_list.at(m_idx);
  }

  std::string status() const
  {
    if (m_list.size() == 0)
      return "(empty)";

    return sformat("(%d of %" ZU ") %s", m_idx + 1, m_list.size(),
                   this->filename().c_str());
  }

  void mode(const play_mode m)
  {
    m_mode = m;
  }

  void remove_helper(bool do_purge)
  {
    if (m_list.size() == 0)
      return;

    ASSERT(size_t(m_idx) < m_list.size());
    std::string target = m_list.at(m_idx);
    aux::msg(sformat("hide file[%d]", m_idx), target.c_str());
    if (do_purge)
      m_purge_list.push_back(target);

    const size_t oldlength = m_list.size();

    m_list.erase(m_list.begin() + m_idx);

    switch (m_mode)
      {
      case JUMPING:
        if (m_idx < m_guess_next)
          {
            aux::msg("jump offset", "1");
            this->jump(oldlength, -1);
          }
        else
          {
            aux::msg("jump offset", "0");
            this->jump(oldlength, 0);
          }
        break;

      case SPINNING:
      default:
        if (m_last_spin <= 0)
          this->spin(m_last_spin);
        else
          this->spin(m_last_spin - 1);
        break;
      }
  }

  void remove() { this->remove_helper(true); }
  void remove_no_purge() { this->remove_helper(false); }

  void purge()
  {
    const size_t N = m_purge_list.size();
    size_t n = 0;

    while (!m_purge_list.empty())
      {
        const std::string f = m_purge_list.back();
        m_purge_list.pop_back();
        ++n;
        aux::msg("purging", sformat("%" ZU " of %" ZU , n, N));
        aux::msg("delete file", f);

        std::string dirname, tail;
        splitPath(f, dirname, tail);
        const std::string stubfile =
          sformat("%s/.%s.deleted", dirname.c_str(), tail.c_str());

        std::ofstream ofs(stubfile.c_str());
        ofs.close();

        try {
          rutz::unixcall::remove(f.c_str());
          ++m_ndeleted;
        }
        catch (std::exception& e) {
          aux::msg("error during deletion",
                   sformat("%s (%s)", f.c_str(), e.what()));
        }

        this->redraw(false);
      }

    m_purge_list.resize(0);
    this->save();
    aux::msg("files deleted", sformat("%d", m_ndeleted));
    aux::msg("files shown", sformat("%d", m_nshown));
    aux::msg("cache misses", sformat("%d", m_nmissed));
    aux::msg("percent kept",
             sformat("%.2f%%", 100.0 * (1.0 - double(m_ndeleted)
                                        / m_nshown)));
  }

  void cachenext()
  {
    if (m_list.size() == 0)
      return;

    if (m_guess_next < 0) m_guess_next = 0;
    if (size_t(m_guess_next) >= m_list.size()) m_guess_next = m_list.size() - 1;
    int i = m_guess_next;
    std::string f = m_list.at(i);
    while (!aux::image_file_exists(f))
      {
        aux::msg(sformat("no such file[%d]", i), f);
        m_list.erase(m_list.begin() + i);
        i = i % m_list.size();
        f = m_list.at(i);
      }
    if (m_pixmap_cache.find(f) == m_pixmap_cache.end())
      {
        m_pixmap_cache[f] =
          aux::build_scaled_pixmap(f, m_widget->getDims(), m_txtype, m_rtype);

        aux::msg(sformat("cache insert[%d]", i), f);
      }
    else
      {
        const Image<PixRGB<byte> > img = m_pixmap_cache[f].img;
        aux::msg(sformat("cache exists[%d]", i),
                 sformat("%dx%d %s", img.getWidth(), img.getHeight(),
                         f.c_str()));
      }
  }

  double loop_delay() const
  {
    return 250.0 * pow(2.0, 0.5 * m_loop_delay_power);
  }

  void redraw(bool show_image)
  {
    Image<PixRGB<byte> > img(m_widget->getDims(), ZEROS);

    if (m_list.size() > 0 && show_image)
      inplacePaste(img, m_pixmap.img,
                   Point2D<int>((img.getWidth() - m_pixmap.img.getWidth()) / 2,
                           (img.getHeight() - m_pixmap.img.getHeight()) / 2));

    if (m_show_overlay)
      {
        const SimpleFont font = SimpleFont::FIXED(7);

        struct stat statbuf;
        std::string mtime;
        if (0 == stat(this->filename().c_str(), &statbuf))
          {
            char buf[32];
            ctime_r(&statbuf.st_mtime, &buf[0]);
            mtime = buf;
          }

        const std::string msgs[] =
          {
            sformat("#%d:%s", m_idx, this->filename().c_str()),
            sformat("    %s", mtime.c_str()),
            sformat("    %dx%d @ %d%%", m_pixmap.rawdims.w(), m_pixmap.rawdims.h(), int(0.5 + m_pixmap.ratio * 100.0)),
            m_looping ? sformat("loop:%.2fms", this->loop_delay()) : std::string("loop:off"),
            std::string(m_mode == JUMPING ? "mode:jumping" : "mode:spinning"),
            sformat("tx:%s", convertToString(m_txtype).c_str()),
            sformat("rs:%s", convertToString(m_rtype).c_str()),
            sformat("c:%" ZU , m_list.size()),
            sformat("p:%" ZU , m_purge_list.size()),
            sformat("d:%d", m_ndeleted),
            sformat("s:%d", m_nshown),
            sformat("m:%d", m_nmissed),
          };

        const int nmsg = sizeof(msgs) / sizeof(msgs[0]);

        for (int i = 0; i < nmsg; ++i)
          writeText(img, Point2D<int>(1,1+i*font.h()), msgs[i].c_str(),
                    PixRGB<byte>(255, 160, 0),
                    PixRGB<byte>(0, 0, 0),
                    font,
                    true);
      }

    m_widget->drawImage(img);
  }

  void show()
  {
    if (m_list.size() > 0)
      {
        std::string f = this->filename();

        while (!aux::image_file_exists(f))
          {
            aux::msg(sformat("no such file(%d)", m_idx), f);
            ASSERT(size_t(m_idx) < m_list.size());
            m_list.erase(m_list.begin() + m_idx);
            m_idx = m_idx % m_list.size();
            f = this->filename();
          }

        aux::msg("index", sformat("%d of %" ZU , m_idx,
                                  m_list.size()));
        aux::msg(sformat("show file[%d]", m_idx), f);

        if (m_pixmap_cache.find(f) != m_pixmap_cache.end())
          {
            m_pixmap = m_pixmap_cache[f];
            m_pixmap_cache.erase(m_pixmap_cache.find(f));
            aux::msg(sformat("cache hit[%d]", m_idx), f);
          }
        else
          {
            aux::msg(sformat("cache miss[%d]", m_idx), f);
            ++m_nmissed;
            m_pixmap =
              aux::build_scaled_pixmap(f, m_widget->getDims(), m_txtype, m_rtype);
          }
      }

    this->redraw(true);
    ++m_nshown;

    m_last_show_time = rutz::time::wall_clock_now();
  }

  rutz::time last_show_time() const { return m_last_show_time; }

  void cycle_txtype()
  {
    switch (m_txtype)
      {
      case TXTYPE_NONE:      m_txtype = TXTYPE_TRANSPOSE; break;
      case TXTYPE_TRANSPOSE: m_txtype = TXTYPE_BEST_FIT; break;
      case TXTYPE_BEST_FIT:  m_txtype = TXTYPE_NONE; break;
      }
    m_pixmap_cache.clear();
  }

  void cycle_rtype()
  {
    switch (m_rtype)
      {
      case RESCALE_SIMPLE_NOINTERP: m_rtype = RESCALE_SIMPLE_BILINEAR; break;
      case RESCALE_SIMPLE_BILINEAR: m_rtype = RESCALE_FILTER_BSPLINE; break;
      case RESCALE_FILTER_BSPLINE: m_rtype = RESCALE_FILTER_LANCZOS3; break;
      default: m_rtype = RESCALE_SIMPLE_NOINTERP; break;
      }
    m_pixmap_cache.clear();
  }

  int run()
  {
    this->show();

    while (1)
      {
        KeySym ks = m_widget->getLastKeySym();
        switch (ks)
          {
          case NoSymbol:
            if (!m_didcache)
              this->cachenext();
            m_didcache = true;
            if (m_looping
                &&
                (rutz::time::wall_clock_now() - this->last_show_time()).msec()
                >= this->loop_delay())
              {
                if (JUMPING == m_mode)
                  this->jump();
                else
                  this->spin(1);
                this->show();
                m_didcache = false;
              }
            else
              usleep(10000);
            break;

          case XK_Left:
            this->spin(-1);
            this->show();
            m_didcache = false;
            break;

          case XK_Right:
            this->spin(1);
            this->show();
            m_didcache = false;
            break;

          case XK_Up:
            this->jump();
            this->show();
            m_didcache = false;
            break;

          case XK_Down:
            this->remove();
            this->show();
            m_didcache = false;
            break;

          case XK_e:
          case XK_0:
          case XK_KP_0:
            this->remove_no_purge();
            this->show();
            m_didcache = false;
            break;

          case XK_Return:
            this->save();
            break;

          case XK_x:
            this->purge();
            this->redraw(true);
            break;

          case XK_Escape:
            this->redraw(false);
            this->purge();
            return 0;

          case XK_l:
            m_looping = !m_looping;
            this->show();
            break;

          case XK_m:
            if (SPINNING == m_mode) m_mode = JUMPING;
            else if (JUMPING == m_mode) m_mode = SPINNING;
            this->show();
            m_didcache = false;
            break;

          case XK_o:
            m_show_overlay = !m_show_overlay;
            this->show();
            break;

          case XK_comma:
            ++m_loop_delay_power;
            break;

          case XK_period:
            --m_loop_delay_power;
            break;

          case XK_t:
            this->cycle_txtype();
            this->show();
            m_didcache = false;
            break;

          case XK_r:
            this->cycle_rtype();
            this->show();
            m_didcache = false;
            break;
          }
      }
  }
};

int main(int argc, const char** argv)
{
  Dims dims(800, 800);

  if (argc != 2 && argc != 3)
    {
      fprintf(stderr, "usage: %s playlist ?WxH?\n", argv[0]);
      return 1;
    }

  if (argc >= 3)
    convertFromString(argv[2], dims);

  rutz::shared_ptr<XWinManaged> window(new XWinManaged(dims, -1, -1,
                                                       argv[1]));

  playlist pl(argv[1], window);

  return pl.run();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_SLIDESHOW_C_DEFINED
