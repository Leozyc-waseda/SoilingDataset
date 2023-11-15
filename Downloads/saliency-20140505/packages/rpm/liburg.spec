#
# liburg.spec -- RPM spec file for the custom version of the URG library
#
# Satofumi Kamimura's URG library is used to interface with the Hokuyo
# line of laser range finders. This spec file is for an RPM package for
# a hacked version of the library that is meant specifically for use in
# iLab.
#
# The hacked version was created by Manu Viswanathan. It does away with
# the sample code that comes with the upstream sources from Kamimura as
# well as C++ abstractions. It then collapses the remaining C code into
# a single library, viz., liburg. Higher-level abstractions must be
# built on top of the plain C API provided by this version of URG.
#
# These changes/hacks were made to simplify integration of the URG API
# with iLab's saliency toolkit.
#

#------------------------------ PREAMBLE --------------------------------

# Boilerplate variables/macros
%define _topdir   /lab/mviswana/rpm
%define _tmppath  %{_topdir}/tmp
%define _prefix   /usr

%define name      liburg
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
Summary:    Custom build of URG library for iLab
Group:      Development/Libraries
Source:     liburg-0.1.1.tar.gz
URL:        http://www.hokuyo-aut.jp/cgi-bin/urg_programs_en/index.html
Prefix:     %{_prefix}
Buildroot:  %{buildroot}

BuildRequires: gcc make

%description
Satofumi Kamimura's URG library provides an API for interfacing with
the Hokuyo line of laser range finders. This package provides a heavily
hacked and stripped down version of the URG library. It is really meant
for use inside iLab.

The upstream sources have been modified so that the C++ and sample code
are not part of this package. Furthermore, all of the C modules have
been collapsed into a single library, viz., liburg, that client
programs can link with. Higher-level abstractions have to be
custom-built on top of the plain C API.

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
%{_prefix}/%{_lib}/liburg*
%{_prefix}/include/urg
