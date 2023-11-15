#
# libtorch.spec -- RPM spec file for the Torch machine learning library
#
# Mandriva does have a torch-devel package. However, that package is
# not built properly and is missing large chunks of the Torch library's
# functionality. This custom-built package for iLab rectifies these
# problems and hides the details of the Torch library's very weird
# build system.
#

#------------------------------ PREAMBLE --------------------------------

# Boilerplate variables/macros
%define _topdir   /lab/mviswana/rpm
%define _tmppath  %{_topdir}/tmp
%define _prefix   /usr

%define name      libtorch
%define rel       9
%define buildroot %{_tmppath}/%{name}-%{version}-%{release}-root

%define target_libdir   %{buildroot}%{_prefix}/%{_lib}
%define target_incdir   %{buildroot}%{_prefix}/include/torch

#------------------------- PACKAGE DESCRIPTION --------------------------

Name:       %{name}
Version:    3
Release:    %mkrel %{rel}
Packager:   Manu Viswanathan
Vendor:     iLab
License:    GPL
Summary:    Custom build of Torch machine learning library for iLab
Group:      Development/Libraries
Source:     libtorch-3.tar.gz
URL:        http://www.torch.ch
Prefix:     %{_prefix}
Buildroot:  %{buildroot}

BuildRequires: gcc-c++ python

%description
The Torch library provides several different machine learning
algorithms. Mandriva's default torch-devel package is not built
properly and is missing large chunks of functionality. This
custom-built package for iLab is meant to be used by some parts of the
INVT saliency toolkit.

The upstream sources have been modified slightly to ensure that the
library integrates smoothly with the toolkit. Furthermore, the Torch
library's somewhat odd/unconventional build system has been cleaned up
and automated a little.

#------------------------ PREINSTALLATION SETUP -------------------------

%prep
%setup -q

#------------------------ BUILDING THE SOURCES --------------------------

# The torch library does not use the usual configure + make approach to
# building. Instead, its root directory ships with a custom python
# script called xmake that performs the build. This script has been
# hacked a little to address one particular shortcoming (viz., its
# insistence on always naming the final target libtorch.a even a shared
# library is built).
#
# Additionally, a wrapper shell script called build has been added to
# build both the static and shared versions of the library. The build
# results and all the header files are put into a subdirectory named
# target. The libraries go to target/lib and the header files to
# target/include.
#
# The RPM installation procedure should copy the stuff under the target
# subdirectory to the appropriate locations under /usr/local (or
# wherever else the package is being installed).
%build
./build

#--------------------- INSTALLING PACKAGE CONTENTS ----------------------

%install

# Clean-up previous (possibly botched) installation
rm -rf %{buildroot}

# Copy the libraries
mkdir -p %{target_libdir}
cp target/lib/libtorch* %{target_libdir}

# Copy the header files
mkdir -p %{target_incdir}
install -m0644 target/include/* %{target_incdir}

#------------------------------ CLEAN-UP --------------------------------

%clean
rm -rf %{buildroot}

#----------------- DOCUMENTATION AND PACKAGE CONTENTS -------------------

%files

%defattr(-, root, root, -)
%{_prefix}/%{_lib}/libtorch*
%{_prefix}/include/torch
