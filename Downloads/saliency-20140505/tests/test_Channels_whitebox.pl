#!/usr/bin/perl -w

# $Id: test_Channels_whitebox.pl 7032 2006-08-23 23:54:29Z rjpeters $

use invt_config;
use whitebox;

whitebox::run($invt_config::exec_prefix . "/bin/whitebox-Channels");
