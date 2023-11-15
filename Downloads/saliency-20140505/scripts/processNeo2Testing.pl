#!/usr/bin/perl

BEGIN { push(@INC, "./scripts"); }
use strict;
use ForkManager;
my $pm = new Parallel::ForkManager(1);

my $resultsDir = shift;
my $testsuiteDir = shift;

#Write the svn version
system("echo Testing:saliency `svnversion .` >> $resultsDir/SvnVersion");
system("echo Testing:neo2 `svnversion src/Apps/neo2` >> $resultsDir/SvnVersion");

my %TestingScenes;
$TestingScenes{"Path0Take1"} = "--db-scene-name=Path0Take1"; 
$TestingScenes{"Path0Take3"} = "--db-scene-name=Path0Take3";
$TestingScenes{"Path1Take0"} = "--db-scene-name=Path1Take0";
$TestingScenes{"Path1Take1"} = "--db-scene-name=Path1Take1";
$TestingScenes{"Path1Take2"} = "--db-scene-name=Path1Take2";
$TestingScenes{"Path1Take4"} = "--db-scene-name=Path1Take4";
$TestingScenes{"Path2Take0"} = "--db-scene-name=Path2Take0";
$TestingScenes{"Path2Take1"} = "--db-scene-name=Path2Take1";
$TestingScenes{"Path2Take2"} = "--db-scene-name=Path2Take2"; #Corrupted, do not use
$TestingScenes{"Path2Take3"} = "--db-scene-name=Path2Take3";
$TestingScenes{"Path2Take4"} = "--db-scene-name=Path2Take4";
$TestingScenes{"Path0Take0"} = "--db-scene-name=Path0Take0";
$TestingScenes{"Path0Take2"} = "--db-scene-name=Path0Take2";
$TestingScenes{"Path0Take4"} = "--db-scene-name=Path0Take4";
$TestingScenes{"Path1Take3"} = "--db-scene-name=Path1Take3";
$TestingScenes{"30ObjectTurntable"} = "--db-scene-name=30ObjectTruntable";
# All frames for a particular light

$TestingScenes{"CamArrayLights1"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_Light1') order by scene,frame";
$TestingScenes{"CamArrayLights2"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_Light2') order by scene,frame";
$TestingScenes{"CamArrayLights3"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_Light3') order by scene,frame";
$TestingScenes{"CamArrayLights4"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_Light4') order by scene,frame";
$TestingScenes{"CamArrayLights1BG"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_FullBG_Light1') order by scene,frame";
$TestingScenes{"CamArrayLights2BG"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_FullBG_Light2') order by scene,frame";
$TestingScenes{"CamArrayLights3BG"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_FullBG_Light3') order by scene,frame";
$TestingScenes{"CamArrayLights4BG"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_FullBG_Light4') order by scene,frame";

# Upper half of all frames for a particular light
$TestingScenes{"CamArrayLights0UpperHalf"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_Light0') and frame > numframes/2 order by scene,frame";
$TestingScenes{"CamArrayLights1UpperHalf"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_Light1') and frame > numframes/2 order by scene,frame";
$TestingScenes{"CamArrayLights2UpperHalf"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_Light2') and frame > numframes/2 order by scene,frame";
$TestingScenes{"CamArrayLights3UpperHalf"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_Light3') and frame > numframes/2 order by scene,frame";
$TestingScenes{"CamArrayLights4UpperHalf"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_Light4') and frame > numframes/2 order by scene,frame";
$TestingScenes{"CamArrayLights0BGUpperHalf"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_FullBG_Light0') and frame > numframes/2 order by scene,frame";
$TestingScenes{"CamArrayLights1BGUpperHalf"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_FullBG_Light1') and frame > numframes/2 order by scene,frame";
$TestingScenes{"CamArrayLights2BGUpperHalf"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_FullBG_Light2') and frame > numframes/2 order by scene,frame";
$TestingScenes{"CamArrayLights3BGUpperHalf"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_FullBG_Light3') and frame > numframes/2 order by scene,frame";
$TestingScenes{"CamArrayLights4BGUpperHalf"} = "--db-query-str=select * from polygonquickview where scenename in ( 'CamArray_11_4_10_FullBG_Light4') and frame > numframes/2 order by scene,frame";

$TestingScenes{"SimpleTest"} = "--db-query-str=select polygon.uid as pid, case objectcategory.uid when null then -1 else objectcategory.uid end as categoryid, case objectcategory.uid when null then '' else objectcategory.name end as name, annotationsource as source, case annotationsource.categ when null then '' else annotationcateg.name end as sourcecateg, vertices, confidence, polygon.scene as scene, polygon.frame as frame FROM scene,polygon left join object on polygon.object=object.uid left join objectcategory on object.category=objectcategory.uid left join annotationsource on polygon.annotationsource=annotationsource.uid left join annotationcateg on annotationsource.categ=annotationcateg.uid  WHERE scene.uid = polygon.scene and mod(polygon.frame+99,100) = 0 and polygon.frame < scene.numframes-1 and scene.name in ('Path1Take1', 'Path1Take2') order by polygon.scene, polygon.frame";


my %TrainTestsToRun;
$TrainTestsToRun{"SimpleTrain"} = ["SimpleTest"];

$TrainTestsToRun{"30ObjectTurntable"} = ["Path0Take1" , "Path0Take3", "Path1Take1", "Path1Take2", "Path1Take4", "Path2Take1", "Path2Take3", "Path2Take4", "Path0Take2", "Path0Take4", "Path1Take3"];
$TrainTestsToRun{"ScorbotObjectTraining"} = ["Path0Take1" , "Path0Take3", "Path1Take1", "Path1Take2", "Path1Take4", "Path2Take1", "Path2Take3", "Path2Take4", "Path0Take2", "Path0Take4", "Path1Take3"];

# Take all lights that are not 0
#$TrainTestsToRun{"CamArrayLight0BG"} = ["CamArrayLights1","CamArrayLights2","CamArrayLights3","CamArrayLights4","CamArrayLights1BG","CamArrayLights2BG","CamArrayLights3BG","CamArrayLights4BG"];

#$TrainTestsToRun{"CamArrayAllLightsBGHalf"} = ["CamArrayLights0UpperHalf","CamArrayLights1UpperHalf","CamArrayLights2UpperHalf","CamArrayLights3UpperHalf","CamArrayLights4UpperHalf","CamArrayLights0BGUpperHalf","CamArrayLights1BGUpperHalf","CamArrayLights2BGUpperHalf","CamArrayLights3BGUpperHalf","CamArrayLights4BGUpperHalf",];

#$TrainTestsToRun{"CamArrayAllLightsAndBG"} = ["Path0Take1" , "Path0Take3", "Path1Take0", "Path1Take1", "Path1Take2", "Path1Take4", "Path2Take0", "Path2Take1", "Path2Take2", "Path2Take3", "Path2Take4", "Path0Take2", "Path0Take4", "Path1Take3", "30ObjectTurntable"];
#$TrainTestsToRun{"30ObjectTurntableBG"} = ["Path0Take1" , "Path0Take3", "Path1Take1", "Path1Take2", "Path1Take4", "Path2Take1", "Path2Take2", "Path2Take3", "Path2Take4", "Path0Take2", "Path0Take4", "Path1Take3"];

opendir(DIR, $testsuiteDir) || die "Can not open $testsuiteDir $!\n";
foreach my $module (readdir(DIR))
{
	next if ($module =~ /^\./);

	next if (-e "$testsuiteDir/$module/SKIP");
	my $validateScript = "$testsuiteDir/$module/Validate.sh";

	my $moduleDir = "$resultsDir/$module";

	my $testRootDir = "$resultsDir/$module/Testing";
	mkdir $testRootDir if (!(-e $testRootDir));

#Try the varius training directory and testing files
	foreach my $trainScene (keys %TrainTestsToRun)
	{
		my $testDir = "$testRootDir/$trainScene";
		mkdir $testDir if (!(-e $testDir));
		
		my $trainDir = "$resultsDir/$module/TrainingFiles";
		my $TestScenes = $TrainTestsToRun{$trainScene};
		foreach my $sceneName (@$TestScenes)
		{
			my $testResultsDir = "$testDir/$sceneName";
			mkdir $testResultsDir if (!(-e $testResultsDir));

			my $cmd = "$validateScript \"$TestingScenes{$sceneName}\" \"$trainDir/$trainScene\" \"$testResultsDir\" \"$trainScene\"";

			my $pid = $pm->start and next; 
			print "Running $cmd\n";
			system($cmd);
			$pm->finish; # Terminates the child process

		}
	}
}

$pm->wait_all_children;
closedir(DIR);
