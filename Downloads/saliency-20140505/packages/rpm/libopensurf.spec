#
# libopensurf.spec -- RPM spec file for custom version of OpenSURF library
#
# Chris Evans's OpenSURF library is a free OpenCV-based implementation
# of the SURF algorithm. However, the original version of the library is
# written mainly for Windows and is not neatly encasulated to be a
# separated library to which executable programs can link. So, Manu
# Viswanathan hacked the original OpenSURF code base to gear it towards
# GNU/Linux (mainly Mandriva and Debian). The hacked version is mostly
# cosmetically different from the original and is autoconfiscated to
# make it easier to build as a library.
#
# This spec file is for an RPM package for the above-mentioned hacked
# version of the OpenSURF library and is meant specifically for use in
# iLab.
#

#------------------------------ PREAMBLE --------------------------------

# Boilerplate variables/macros
%define _topdir   /lab/mviswana/rpm
%define _tmppath  %{_topdir}/tmp
%define _prefix   /usr

%define name      libopensurf
%define rel       9
%define buildroot %{_tmppath}/%{name}-%{version}-%{release}-root

%define target_libdir   %{buildroot}%{_prefix}/%{_lib}
%define target_incdir   %{buildroot}%{_prefix}/include/urg

#------------------------- PACKAGE DESCRIPTION --------------------------

Name:       %{name}
Version:    0.1.1
Release:    %mkrel %{rel}
Packager:   Manu Viswanathan
Vendor:     iLab
License:    GPL
Summary:    Custom build of OpenSURF library for iLab
Group:      Development/Libraries
Source:     libopensurf-0.1.1.tar.gz
URL:        http://code.google.com/p/opensurf1/
Prefix:     %{_prefix}
Buildroot:  %{buildroot}

BuildRequires: gcc make opencv-devel

%description
Chris Evans's OpenSURF library is a free, OpenCV-based implementation of
the SURF algorithm. Unfortunately, the original code base is geared
towards Windows and does not seem to support building into a separate,
linkable library.

This version of the library has been hacked together by Manu Viswanathan
and is meant specifically for use in iLab's Mandriva environment. The
changes from the original are mostly cosmetic; the biggest one is the
autoconfiscation of the project so that it builds seamlessly on a
GNU/Linux system.

#------------------------ PREINSTALLATION SETUP -------------------------

%prep
%setup -q

#------------------------ BUILDING THE SOURCES --------------------------

%build
%configure
make

#--------------------- INSTALLING PACKAGE CONTENTS ----------------------

%install

# Clean-up previous (possibly botched) installation
rm -rf %{buildroot}

# Install the library and its header files
make install DESTDIR=%{buildroot}

#------------------------------ CLEAN-UP --------------------------------

%clean
rm -rf %{buildroot}

#----------------- DOCUMENTATION AND PACKAGE CONTENTS -------------------

%files

%defattr(-, root, root, -)
%{_prefix}/%{_lib}/libopensurf*
%{_prefix}/include/opensurf
