#!/usr/bin/perl

BEGIN { push(@INC, "./scripts"); }
use strict;
use ForkManager;
my $pm = new Parallel::ForkManager(1);

my $resultsDir = shift;
my $testsuiteDir = shift;


mkdir $resultsDir if (!(-e $resultsDir));

#Write the svn version
system("echo Training:saliency `svnversion .` >> $resultsDir/SvnVersion");
system("echo Training:neo2 `svnversion src/Apps/neo2` >> $resultsDir/SvnVersion");

my %TrainingScenes;
$TrainingScenes{"ScorbotObjectTraining"} = "--db-scene-name=ScorbotObjectTraining";
#$TrainingScenes{"30ObjectTurntable"} = "--db-scene-name=30ObjectTurntable";


#$TrainingScenes{"CamArrayAllLightsAndBG"} = "--db-query-str=select polygon.uid as pid, case objectcategory.uid when null then -1 else objectcategory.uid end as categoryid, case objectcategory.uid when null then '' else objectcategory.name end as name, annotationsource as source, case annotationsource.categ when null then '' else annotationcateg.name end as sourcecateg, vertices, confidence, polygon.scene as scene, polygon.frame as frame FROM scene,polygon left join object on polygon.object=object.uid left join objectcategory on object.category=objectcategory.uid left join annotationsource on polygon.annotationsource=annotationsource.uid left join annotationcateg on annotationsource.categ=annotationcateg.uid  WHERE scene.uid = polygon.scene and polygon.frame < scene.numframes-1 and scene.name in ( 'CamArray_11_4_10_Light0', 'CamArray_11_4_10_Light1', 'CamArray_11_4_10_Light2', 'CamArray_11_4_10_Light3', 'CamArray_11_4_10_Light4', 'CamArray_11_4_10_FullBG_Light0', 'CamArray_11_4_10_FullBG_Light1', 'CamArray_11_4_10_FullBG_Light2', 'CamArray_11_4_10_FullBG_Light3', 'CamArray_11_4_10_FullBG_Light4') order by polygon.scene, polygon.frame";
#$TrainingScenes{"CamArrayLight0BG"} = "--db-query-str=select polygon.uid as pid, case objectcategory.uid when null then -1 else objectcategory.uid end as categoryid, case objectcategory.uid when null then '' else objectcategory.name end as name, annotationsource as source, case annotationsource.categ when null then '' else annotationcateg.name end as sourcecateg, vertices, confidence, polygon.scene as scene, polygon.frame as frame FROM scene,polygon left join object on polygon.object=object.uid left join objectcategory on object.category=objectcategory.uid left join annotationsource on polygon.annotationsource=annotationsource.uid left join annotationcateg on annotationsource.categ=annotationcateg.uid  WHERE scene.uid = polygon.scene and polygon.frame < scene.numframes-1 and scene.name in ( 'CamArray_11_4_10_Light0', 'CamArray_11_4_10_FullBG_Light0') order by polygon.scene, polygon.frame";
#
#$TrainingScenes{"CamArrayAllLightsBGHalf"} = "--db-query-str=select polygon.uid as pid, case objectcategory.uid when null then -1 else objectcategory.uid end as categoryid, case objectcategory.uid when null then '' else objectcategory.name end as name, annotationsource as source, case annotationsource.categ when null then '' else annotationcateg.name end as sourcecateg, vertices, confidence, polygon.scene as scene, polygon.frame as frame FROM scene,polygon left join object on polygon.object=object.uid left join objectcategory on object.category=objectcategory.uid left join annotationsource on polygon.annotationsource=annotationsource.uid left join annotationcateg on annotationsource.categ=annotationcateg.uid  WHERE scene.uid = polygon.scene and polygon.frame < (scene.numframes/2) and scene.name in ( 'CamArray_11_4_10_Light0', 'CamArray_11_4_10_Light1', 'CamArray_11_4_10_Light2', 'CamArray_11_4_10_Light3', 'CamArray_11_4_10_Light4', 'CamArray_11_4_10_FullBG_Light0', 'CamArray_11_4_10_FullBG_Light1', 'CamArray_11_4_10_FullBG_Light2', 'CamArray_11_4_10_FullBG_Light3', 'CamArray_11_4_10_FullBG_Light4') order by polygon.scene, polygon.frame";
$TrainingScenes{"30ObjectTurntable"} = "--db-query-str=select polygon.uid as pid, case objectcategory.uid when null then -1 else objectcategory.uid end as categoryid, case objectcategory.uid when null then '' else objectcategory.name end as name, annotationsource as source, case annotationsource.categ when null then '' else annotationcateg.name end as sourcecateg, vertices, confidence, polygon.scene as scene, polygon.frame as frame FROM scene,polygon left join object on polygon.object=object.uid left join objectcategory on object.category=objectcategory.uid left join annotationsource on polygon.annotationsource=annotationsource.uid left join annotationcateg on annotationsource.categ=annotationcateg.uid  WHERE scene.uid = polygon.scene and polygon.frame < scene.numframes-1 and scene.name in ( '30ObjectTurntable', 'Path0Take0', 'Path1Take0', 'Path2Take0') order by polygon.scene, polygon.frame";
$TrainingScenes{"SimpleTrain"} = "--db-query-str=select polygon.uid as pid, case objectcategory.uid when null then -1 else objectcategory.uid end as categoryid, case objectcategory.uid when null then '' else objectcategory.name end as name, annotationsource as source, case annotationsource.categ when null then '' else annotationcateg.name end as sourcecateg, vertices, confidence, polygon.scene as scene, polygon.frame as frame FROM scene,polygon left join object on polygon.object=object.uid left join objectcategory on object.category=objectcategory.uid left join annotationsource on polygon.annotationsource=annotationsource.uid left join annotationcateg on annotationsource.categ=annotationcateg.uid  WHERE scene.uid = polygon.scene and mod(polygon.frame,100) = 0 and polygon.frame < scene.numframes-1 and scene.name in ('Path1Take1', 'Path1Take2') order by polygon.scene, polygon.frame";




opendir(DIR, $testsuiteDir) || die "Can not open $testsuiteDir $!\n";
foreach my $module (readdir(DIR))
{
  next if ($module =~ /^\./);

  next if (-e "$testsuiteDir/$module/SKIP");

  my $trainScript = "$testsuiteDir/$module/Train.sh";

  my $moduleDir = "$resultsDir/$module";
  mkdir $moduleDir if (!(-e $moduleDir));

  my $trainDir = "$moduleDir/TrainingFiles";
  mkdir $trainDir if (!(-e $trainDir));

  foreach my $sceneName (keys %TrainingScenes)
  {
    my $sceneDir = "$trainDir/$sceneName";
    mkdir $sceneDir if (!(-e $sceneDir));

    my $cmd = "$trainScript \"$TrainingScenes{$sceneName}\" \"$sceneDir\" \"$sceneName\"";

    my $pid = $pm->start and next; 
    print "Running $cmd\n";
    system($cmd);
    $pm->finish; # Terminates the child process

  }

}
$pm->wait_all_children;
closedir(DIR);
