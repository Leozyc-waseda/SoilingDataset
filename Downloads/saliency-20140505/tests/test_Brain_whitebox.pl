#!/usr/bin/perl -w

# $Id: test_Brain_whitebox.pl 6312 2006-02-22 19:55:34Z rjpeters $

use invt_config;
use whitebox;

whitebox::run($invt_config::exec_prefix . "/bin/whitebox-Brain");
