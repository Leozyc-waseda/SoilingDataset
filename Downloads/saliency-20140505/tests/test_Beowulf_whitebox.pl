#!/usr/bin/perl -w

# $Id: test_Beowulf_whitebox.pl 6434 2006-04-07 00:45:46Z rjpeters $

use invt_config;
use whitebox;

whitebox::run($invt_config::exec_prefix . "/bin/whitebox-Beowulf");
