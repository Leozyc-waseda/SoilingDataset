#------------------------------ PREAMBLE --------------------------------

# Boilerplate variables/macros
%define _topdir   /root/rpmbuild
%define _tmppath  %{_topdir}/tmp
%define _prefix   /usr

%define name      lwpr
%define rel       9
%define buildroot %{_tmppath}/%{name}-%{version}-%{release}-root

%define target_libdir   %{buildroot}%{_prefix}/%{_lib}
%define target_incdir   %{buildroot}%{_prefix}/include/urg

#------------------------- PACKAGE DESCRIPTION --------------------------
Summary: Locally weighted projection regression
Name: %{name}
Version: 1.2.3
Release: %mkrel %{rel}
Source: lwpr-1.2.3.tar.gz
License: GPL
Group: System/Kernel and hardware
Url: http://www.ipab.inf.ed.ac.uk/slmc/software/lwpr/index.html
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot

%description
Locally Weighted Projection Regression (LWPR) is a recent algorithm
that achieves nonlinear function approximation in high dimensional
spaces with redundant and irrelevant input dimensions. At its core, it
uses locally linear models, spanned by a small number of univariate
regressions in selected directions in input space. A locally weighted
variant of Partial Least Squares (PLS) is employed for doing the
dimensionality reduction. This nonparametric local learning system

#------------------------ PREINSTALLATION SETUP -------------------------


%prep
%setup -q

%build
export CFLAGS="%{optflags} -fPIC"
autoreconf -fi
%configure2_5x --includedir=/usr/include/lwpr
%make

%install
rm -rf %{buildroot}
%makeinstall_std

%clean
rm -rf %{buildroot}

#----------------- DOCUMENTATION AND PACKAGE CONTENTS -------------------

%files

%defattr(-, root, root, -)
%{_prefix}/%{_lib}/liblwpr*
%{_prefix}/include/lwpr
