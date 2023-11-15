#!/usr/bin/perl -w

# $Id: test_strings_whitebox.pl 7007 2006-08-16 17:48:10Z rjpeters $

use invt_config;
use whitebox;

whitebox::run($invt_config::exec_prefix . "/bin/whitebox-strings");
