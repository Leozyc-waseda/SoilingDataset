##########################################################################
## The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   ##
## by the University of Southern California (USC) and the iLab at USC.  ##
## See http://iLab.usc.edu for information about this project.          ##
##########################################################################
## Major portions of the iLab Neuromorphic Vision Toolkit are protected ##
## under the U.S. patent ``Computation of Intrinsic Perceptual Saliency ##
## in Visual Environments, and Applications'' by Christof Koch and      ##
## Laurent Itti, California Institute of Technology, 2001 (patent       ##
## pending; application number 09/912,225 filed July 23, 2001; see      ##
## http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     ##
##########################################################################
## This file is part of the iLab Neuromorphic Vision C++ Toolkit.       ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is free software; you can   ##
## redistribute it and/or modify it under the terms of the GNU General  ##
## Public License as published by the Free Software Foundation; either  ##
## version 2 of the License, or (at your option) any later version.     ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  ##
## that it will be useful, but WITHOUT ANY WARRANTY; without even the   ##
## implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      ##
## PURPOSE.  See the GNU General Public License for more details.       ##
##                                                                      ##
## You should have received a copy of the GNU General Public License    ##
## along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   ##
## to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   ##
## Boston, MA 02111-1307 USA.                                           ##
##########################################################################
##
## Primary maintainer for this file: Laurent Itti <itti@usc.edu>
## $Id: COPYRIGHT-perl 4687 2005-06-11 14:31:37Z rjpeters $
##

CC=gcc

# Needed to compile hildon apps:
#LIBRARIES=gconf-2.0 hildon-1 hildon-help gtk+-2.0 gdk-2.0
LIBRARIES=libosso
CFLAGS=`pkg-config --cflags  $(LIBRARIES)` 
LDFLAGS=`pkg-config --libs $(LIBRARIES)`

CFLAGS+=-Wall -O4 -fomit-frame-pointer -mcpu=arm1136j-s -mfpu=vfp -mfloat-abi=softfp -lX11
CFLAGS+=-I$(HOME)/saliency/src
CFLAGS+=-I.

CXXFLAGS=$(CFLAGS)

DESKTOP_DIR=$(DESTDIR)`pkg-config osso-af-settings --variable=desktopentrydir`
MAN_DIR=$(DESTDIR)/usr/share/man/man1
DOC_DIR=$(DESTDIR)/usr/share/doc/saliency
DPDIR=debian/saliency-1.0

all: saliency

check:
	pkg-config --list-all
	ls /usr/include

debianpackage:
	-cp -f saliency.C Makefile saliency.desktop saliency.1 README $(DPDIR)
	-cp -rf ../../Envision/ $(DPDIR)
	-mkdir -p $(DPDIR)/Image
	-cp -f ../../Image/font6x10.h $(DPDIR)/Image
	cd $(DPDIR) && dpkg-buildpackage -rfakeroot
	cd $(DPDIR) && dpkg-buildpackage -rfakeroot -sa -S

install: all
	strip saliency
	cp saliency $(DESTDIR)/usr/bin
	mkdir -p $(DESKTOP_DIR)
	cp saliency.desktop $(DESKTOP_DIR)
	mkdir -p $(MAN_DIR)
	cp saliency.1 $(MAN_DIR)
	gzip -9 $(MAN_DIR)/saliency.1
	mkdir -p $(DOC_DIR)
	cp README $(DOC_DIR)
clean:
	-rm saliency

distclean: clean
	-rm $(DESTDIR)/usr/bin/saliency
	-rm $(DESKTOP_DIR)/saliency.desktop
	-rm $(MAN_DIR)/usr/share/man/man1/saliency.1,gz
	-rm $(DOC_DIR)/README
