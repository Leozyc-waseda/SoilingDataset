#!/usr/bin/perl

BEGIN { push(@INC, "./scripts"); }

use strict;
use ForkManager;
my $pm = new Parallel::ForkManager(8);

my $objRecProg = shift;
my $objRecBrainModules = shift;
my $outputDir = shift;


my %trainingSet;
my %valSet;

$trainingSet{"PASCAL"} = "/lab/raid/imagedb/voc/2009/VOCdevkit/VOC2009/ImageSets/XML/train.xml";
$trainingSet{"MITSS"} = "/lab/raid/imagedb/mit/StreetScenes/Anno_XML/train.xml";

$valSet{"PASCAL"} = "/lab/raid/imagedb/voc/2009/VOCdevkit/VOC2009/ImageSets/XML/val.xml";
$valSet{"MITSS"} = "/lab/raid/imagedb/mit/StreetScenes/Anno_XML/val.xml";

my %objects;

$objects{"PASCAL"} = [ "aeroplane","bus","diningtable","pottedplant","bicycle",
                "car","dog","sheep","bird","cat","horse","sofa","boat",
                "chair","motorbike","train","bottle","cow","person","tvmonitor"];

$objects{"MITSS"} = [ "tree", "bicycle", "pedestrian", "building", "sky", "store", "car", "road"];


             

open(ObjRecLogFile, ">$outputDir/results.txt") || die "Can not open results file $!";


my %modules;
open(MODULES, $objRecBrainModules) || die "Can not open $objRecBrainModules ($!)\n"; 
while(<MODULES>)
{
  next if (/^#/);
  my ($name, $path, $param) = split(/\s+/, $_, 3);
  $modules{$name} = [ $path, $param];
}
close(MODULES);

foreach my $module (keys %modules) {
  foreach my $dbName (keys %valSet)
  {
    foreach my $objName (@{$objects{$dbName}})
    {
      foreach my $getObjects (0,1)
      {
        #Classify the whole scene
        my $pid = $pm->start and next; 
        &trainModule($module, $modules{$module}[0], $modules{$module}[1],
          $dbName, $trainingSet{$dbName}, $objName, $getObjects);
        &testModule($module,$modules{$module}[0], $modules{$module}[1],
          $dbName, $valSet{$dbName}, $objName, $getObjects);
        $pm->finish; # Terminates the child process
      }

    }
  }

}
$pm->wait_all_children;
closedir(DIR);

open(ObjRecLogFile);

sub trainModule 
{
  my($module, $path, $params, $dbName, $trainFile, $objName, $getObjects) = @_;

  #print "Training $module with $trainFile\n";

  #create dir if for output
  mkdir "$outputDir/$module" if (!-e "$outputDir/$module");

  my $cmd = "$objRecProg --in=xmlfile:$trainFile ";
  $cmd .= "--out=none --filter-object=${objName} --training-mode ";
  $cmd .= "--get-objects=true " if ($getObjects);
  $cmd .= "--objects-db-file=$outputDir/$module/${objName}_${getObjects}.$dbName.dat ";
  $cmd .= "$path ";
  system($cmd);

}

sub testModule 
{
  my($module, $path, $params, $dbName, $testFile, $objName, $getObjects) = @_;

  #print "Testing $module $testFile\n";

  my $cmd = "$objRecProg --in=xmlfile:$testFile ";
  $cmd .= "--out=none --filter-object=$objName ";
  $cmd .= "--get-objects=true " if ($getObjects);
  $cmd .= "--roc-file=$outputDir/$module/results_${objName}_${getObjects}.$dbName.roc ";
  $cmd .= "--timing-file=$outputDir/$module/results_${objName}_${getObjects}.$dbName.time ";
  $cmd .= "--results-file=$outputDir/$module/results_${objName}_${getObjects}.$dbName.log ";
  $cmd .= "--objects-db-file=$outputDir/$module/${objName}_${getObjects}.$dbName.dat ";
  $cmd .= "$path ";

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

  print ObjRecLogFile "$module $dbName $objName $getObjects $stats\n";

}

