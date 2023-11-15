#!/usr/bin/perl

BEGIN { push(@INC, "./scripts"); }
use strict;
use ForkManager;
my $pm = new Parallel::ForkManager(8);

my $objDetProg = shift;
my $objDetBrainModules = shift;
my $outputDir = shift;


my %trainingSet;
my %valSet;

$trainingSet{"PASCAL"} = "/lab/raid/imagedb/voc/2009/VOCdevkit/VOC2009/ImageSets/XML/train.xml";
$trainingSet{"MITSS"} = "/lab/raid/imagedb/mit/StreetScenes/Anno_XML/train.xml";

$valSet{"PASCAL"} = "/lab/raid/imagedb/voc/2009/VOCdevkit/VOC2009/ImageSets/XML/val.xml";
$valSet{"MITSS"} = "/lab/raid/imagedb/mit/StreetScenes/Anno_XML/val.xml";

open(ObjDetLogFile, ">$outputDir/results.txt") || die "Can not open results file $!";

#Get a list of modules
my %modules;
open(MODULES, $objDetBrainModules) || die "Can not open $objDetBrainModules ($!)\n"; 
while(<MODULES>)
{
  next if (/^#/);
  my ($name, $path, $param) = split(/\s+/, $_, 3);
  $modules{$name} = [ $path, $param];
}
close(MODULES);

foreach my $module (keys %modules)
{

  foreach my $imgDBName (keys %valSet)
  {
    #Detect the whole scene
    my $pid = $pm->start and next; 
    #for now no training is needed
    #&trainModule($module, $trainingSet[$i], $objName, $getObjects);
    &testModule($module, $modules{$module}[0], $modules{$module}[1],
      $imgDBName, $valSet{$imgDBName});
    $pm->finish; # Terminates the child process
  }

}
$pm->wait_all_children;

close(ObjDetLogFile);


sub trainModule 
{
  my($module, $trainFile, $objName, $getObjects) = @_;

}

sub testModule 
{
  my($module, $path, $params, $dbName, $testFile) = @_;

  #print "Testing $module $testFile\n";

  #create dir if for output
  mkdir "$outputDir/$module" if (!-e "$outputDir/$module");

  my $cmd = "$objDetProg --in=xmlfile:$testFile ";
  $cmd .= "--out=none  ";
  $cmd .= "--roc-file=$outputDir/$module/results.$dbName.roc ";
  $cmd .= "$path ";
  $cmd .= "-- $params " if ($params ne "");

  #print "$cmd\n";

  my $stats;
  open(CMD, "$cmd|") || die "can not run $cmd $!";
  while(<CMD>)
  {
    if (/Stats: (.*)/)
    {
      $stats = $1;
    }
  }
  close(CMD);

  print ObjDetLogFile "$module $dbName $stats\n";

}

