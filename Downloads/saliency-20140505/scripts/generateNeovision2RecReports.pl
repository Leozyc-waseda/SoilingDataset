#!/usr/bin/perl

use perlchartdir;
use strict;

my $inputDir = shift;
my $outputDir = shift;

my @colorMap = (0x000000,
                0xff0000,
                0xff00ff,
                0xffff00,
                0xffffff);
my %objectResults;
my %sceneResults;
my %sceneFPS;
my %objectFPS;
my %medianObjectResults;
my %medianSceneResults;
my %meanSceneFPS;
my %meanObjectFPS;
my %numOfObjects;
my %numOfScenes;
my %objects;
my %modules;
my %objectsProcessed;
my %scenesProcessed;
my %imageDbs;

open(FILE, "$inputDir/results.txt") || die "can not open results file $!";
while(<FILE>)
{
  if (/(.*?) (.*?) (.*?) (.*?) (.*)/)
  {
    my $module = $1;
    my $dbName = $2;
    my $object = $3;
    my $getObjects = $4;
    my $stats = $5;

    $module = "$module|$dbName";
    #print "$module $dbName $object $getObjects $stats\n";
    my ($numFrames, $fps, $ap ) = ($stats =~ /Frames:(.*) FPS:(.*) AP:(.*)/);

    $objects{$object} = 1;
    $modules{$module} = 1;

    if ($getObjects eq "1")
    {
      $objectFPS{"$module|$object"} = $fps;
      $objectResults{"$module|$object"} = $ap;
      push @{ $objectsProcessed{$module} },  $object;
      #push @{ $objectResults{$module} },  $ap;

      if (defined $medianObjectResults{$module})
      {
        $medianObjectResults{$module} += $ap;
        $numOfObjects{$module} ++;
        $meanObjectFPS{$module} += $fps;
      } else {
        $medianObjectResults{$module} = $ap;
        $numOfObjects{$module} =1;
        $meanObjectFPS{$module} = $fps;
      }
    } else {
      $sceneFPS{"$module|$object"} = $fps;
      $sceneResults{"$module|$object"} = $ap;
      push @{ $scenesProcessed{$module} },  $object;
      #push @{ $sceneResults{$module} },  $ap;

      if (defined $medianSceneResults{$module})
      {
        $medianSceneResults{$module} += $ap;
        $numOfScenes{$module} ++;
        $meanSceneFPS{$module} += $fps;
      } else {
        $medianSceneResults{$module} = $ap;
        $numOfScenes{$module} = 1;
        $meanSceneFPS{$module} = $fps;
      }
    }

  }
}
close(FILE);

open(HTML, ">$outputDir/index.html") || die "Can not open index.html $!";

print HTML <<HEADER;
<HTML>
<HEAD>
<Title> Neovision2 Object Recognition Results </Title>
<link  rel='stylesheet' href="Style.css" />
<style type="text/css">
<!--
body {
margin-left: 0px;
margin-top: 0px;
margin-right: 0px;
margin-bottom: 0px;
background-color: #ffffff;
}
-->
</style>
</head>
<BODY>

<CENTER>
<H1> Neovision2 Object Recognition Results <H1>
HEADER


########################################################### Overall results #######################################
print HTML <<HEADER;
<H2> Overall results </H2>
<TABLE border="1" cellspacing="0" cellpadding="0" style='BORDER-BOTTOM:#eeeeee 1px solid; BORDER-RIGHT:#eeeeee 1px solid; BORDER-LEFT:#eeeeee 1px solid;'>
          <tr>
            <td class="tab"><div align="center">Method </div></td>
            <td class="tab"><div align="center">Scenes<br>Mean AP</div></td>
            <td class="tab"><div align="center">Scenes<br>Mean FPS</div></td>
            <td class="tab"></td>
            <td class="tab"><div align="center">Objects<br>Mean AP</div></td>
            <td class="tab"><div align="center">Objects<br>Mean FPS</div></td>
          </tr>
HEADER

foreach my $module (keys %medianSceneResults)
{
  my $sceneMedianAP = 0;
  my $objectMedianAP = 0;
  my $sceneFPS = 0;
  my $objectFPS = 0;

  $sceneMedianAP = $medianSceneResults{$module}/$numOfScenes{$module} if ($numOfScenes{$module} > 0);
  $objectMedianAP = $medianObjectResults{$module}/$numOfObjects{$module} if ($numOfObjects{$module} > 0);

  $sceneFPS = $meanSceneFPS{$module}/$numOfScenes{$module} if ($numOfScenes{$module} > 0);
  $objectFPS = $meanObjectFPS{$module}/$numOfObjects{$module} if ($numOfObjects{$module} > 0);


  print HTML "<TR>\n";
  print HTML "<TD align=center><b>$module</b></TD>\n";

  &generateGauge($sceneMedianAP, $module, "$outputDir/${module}_median_0.png", 0, 1);
  print HTML "<TD align=center><A href=$module><IMG SRC=${module}_median_0.png></A></TD>\n";

  &generateGauge($sceneFPS, $module, "$outputDir/${module}_time_0.png", 0, 500);
  print HTML "<TD align=center><A href=$module><IMG SRC=${module}_time_0.png></A></TD>\n";

  print HTML "<TD></TD>\n";

  &generateGauge($objectMedianAP, $module, "$outputDir/${module}_median_1.png", 0, 1);
  print HTML "<TD align=center><A href=$module><IMG SRC=${module}_median_1.png></A></TD>\n";

  &generateGauge($objectFPS, $module, "$outputDir/${module}_time_1.png", 0, 500);
  print HTML "<TD align=center><A href=$module><IMG SRC=${module}_time_1.png></A></TD>\n";

  print HTML "</TR>\n";
}


print HTML "</Table>\n";


########################################################### AP by class results #######################################

&generateBarGraph(\%sceneResults, "ap_class_0.png");
&generateBarGraph(\%objectResults, "ap_class_1.png");

print HTML <<HEADER;
<H2> Average Precision by class </H2>
<TABLE border="1" cellspacing="0" cellpadding="0" style='BORDER-BOTTOM:#eeeeee 1px solid; BORDER-RIGHT:#eeeeee 1px solid; BORDER-LEFT:#eeeeee 1px solid;'>
  <tr> <td class="tab"><div align="center">Scenes </div></td></TR>
  <tr> <TD><IMG SRC=ap_class_0.png></TD></TR>

  <TR> <td class="tab"><div align="center">Objects </div></td> </tr>
  <tr> <TD><IMG SRC=ap_class_1.png></TD></TR>
</Table>
HEADER


##Sort by values
#my %PREC;
#my %REC;
#
#foreach my $module_param (reverse sort { $sceneResults{$a} <=> $sceneResults{$b} } keys %sceneResults)
#{
#  my ($module, $object) = split(/\|/, $module_param);
#  my $ap = $sceneResults{$module_param};
#
#  my ($rec, $prec) = &getROCData($module, $object, 0);
#  #Save the data for a combined plot
#  $PREC{"${module}_${object}_0"} = $prec;
#  $REC{"${module}_${object}_0"} = $rec;
#
#}
#
#foreach my $module_param (reverse sort { $objectResults{$a} <=> $objectResults{$b} } keys %objectResults)
#{
#  my ($module, $object) = split(/\|/, $module_param);
#  my $ap = $objectResults{$module_param};
#
#  my ($rec, $prec) = &getROCData($module, $object, 1);
#  #Save the data for a combined plot
#  $PREC{"${module}_${object}_1"} = $prec;
#  $REC{"${module}_${object}_1"} = $rec;
#
#}
#
#
#&generateCombineROC(\%REC, \%PREC);
#
#print HTML <<ROCHTML;
#<br><br>
#
#<TABLE border="0" cellspacing="0" cellpadding="0" style='BORDER-BOTTOM:#eeeeee 1px solid; BORDER-RIGHT:#eeeeee 1px solid; BORDER-LEFT:#eeeeee 1px solid;'>
#<tr>
#<td class="tab"><div align="center">Scene ROC Curve </div></td> 
#<td class="tab"><div align="center">Object ROC Curve </div></td> 
#</tr>
#<TR>
#<TD><IMG SRC=roc_0.png></TD>
#<TD><IMG SRC=roc_1.png></TD>
#</TR>
#</TABLE>
#ROCHTML

print HTML "</HTML>\n";
close(HTML);


sub generateCombineROC
{
  my($rec, $prec) = @_;


  foreach my $getObjects (0,1)
  {
    # Create a XYChart object of size 600 x 375 pixels
    my $c = new XYChart(600, 395);

    # Add a title to the chart using 18 pts Times Bold Italic font
    #$c->addTitle("ROC curve for $module", "timesbi.ttf", 18);

    # Set the plotarea at (50, 55) and of 500 x 280 pixels in size. Use a vertical
    # gradient color from light blue (f9f9ff) to sky blue (aaccff) as background. Set
    # border to transparent and grid lines to white (ffffff).
    $c->setPlotArea(50, 55, 500, 280, $c->linearGradientColor(0, 55, 0, 335, 0xf9fcff,
        0xaaccff), -1, $perlchartdir::Transparent, 0xffffff);

    # Add a legend box at (50, 28) using horizontal layout. Use 10pts Arial Bold as font,
    # with transparent background.
    $c->addLegend(50, 28, 0, "arialbd.ttf", 10)->setBackground($perlchartdir::Transparent);

    # Set y-axis tick density to 30 pixels. ChartDirector auto-scaling will use this as
    # the guideline when putting ticks on the y-axis.
    $c->yAxis()->setTickDensity(30);

    # Set axis label style to 8pts Arial Bold
    $c->xAxis()->setLabelStyle("arialbd.ttf", 8);
    $c->yAxis()->setLabelStyle("arialbd.ttf", 8);

    # Set axis line width to 2 pixels
    $c->xAxis()->setWidth(2);
    $c->yAxis()->setWidth(2);

    # Add axis title using 10pts Arial Bold Italic font
    $c->yAxis()->setTitle("Precision", "arialbi.ttf", 10);
    $c->xAxis()->setTitle("Recall", "arialbi.ttf", 10);

    my  $color = 0;
    foreach my $module (keys %$prec)
    {
      next if ($module !~ /_$getObjects/);
      my $re = $$rec{$module};
      my $pr = $$prec{$module};

      my $layer1 = $c->addLineLayer($pr, $colorMap[$color], $module);
      $color = ($color+1)%$#colorMap;

      $layer1->setXData($re);

      # Set the line width to 1 pixels
      $layer1->setLineWidth(1);
    }

    # Output the chart
    $c->makeChart("$outputDir/roc_${getObjects}.png")
  }

}

sub getROCData
{
  my ($module, $object, $getObjects) = @_;

  # The data for the line chart
  my $file = "$inputDir/$module/results_${object}_${getObjects}.roc";
  my @prec;
  my @rec;
  open(ROC, $file) || die "Can not open $file $!";
  while(<ROC>)
  {
    if (/(.*) (.*)/)
    {
      push(@rec, $1);
      push(@prec, $2);
    }
  }
  close(ROC);

  return (\@rec, \@prec);
}

sub generateModuleStats
{
  my ($module, $rec, $prec, $object, $getObjects) = @_;

  my $numOfImages = `cat $outputDir/$module/results_${object}_${getObjects}.log | wc -l`;

  &generateRocImg("${module}_${object}_${getObjects}", $rec, $prec);
  open(MODHTML, ">$outputDir/$module/index.html") || die "Can not open index.html $!";

  print MODHTML <<HEADER;
<HTML>
<HEAD>
<Title>  Object Recognition Results for $module</Title>
<link  rel='stylesheet' href="../Style.css" />
<style type="text/css">
<!--
body {
margin-left: 0px;
margin-top: 0px;
margin-right: 0px;
margin-bottom: 0px;
background-color: #ffffff;
}
-->
</style>
</head>
<BODY>

<CENTER>
<H1> Object Recognition Results for $module <H1>
<TABLE border="0" cellspacing="0" cellpadding="0" style='BORDER-BOTTOM:#eeeeee 1px solid; BORDER-RIGHT:#eeeeee 1px solid; BORDER-LEFT:#eeeeee 1px solid;'>
          <tr>
            <td class="tab"><div align="center">Average Precision </div></td>
            <td class="tab"><div align="center">Average frames/sec </div></td>
          </tr>

<TR>
<TD><IMG SRC=../$module.png></TD>
<TD><IMG SRC=../$module.time.png></TD>
</TR>
</Table>

<br>

<TABLE border="0" cellspacing="0" cellpadding="0" style='BORDER-BOTTOM:#eeeeee 1px solid; BORDER-RIGHT:#eeeeee 1px solid; BORDER-LEFT:#eeeeee 1px solid;'>
<tr>
 <td class="tab"><div align="center">Number of images </div></td>
</tr>
<TR> <TD>$numOfImages</TD></TR>
</TABLE>
<br>

<TABLE border="0" cellspacing="0" cellpadding="0" style='BORDER-BOTTOM:#eeeeee 1px solid; BORDER-RIGHT:#eeeeee 1px solid; BORDER-LEFT:#eeeeee 1px solid;'>
<tr> <td class="tab"><div align="center">ROC Curve </div></td> </tr>
<TR> <TD><IMG SRC=roc.png></TD></TR>
</TABLE>
HEADER

print MODHTML "</HTML>\n";

close(MODHTML);

  
}


sub generateRocImg
{
  my ($module, $rec, $prec) = @_;

  # Create a XYChart object of size 600 x 375 pixels
  my $c = new XYChart(600, 395);

  # Add a title to the chart using 18 pts Times Bold Italic font
  #$c->addTitle("ROC curve for $module", "timesbi.ttf", 18);

  # Set the plotarea at (50, 55) and of 500 x 280 pixels in size. Use a vertical
  # gradient color from light blue (f9f9ff) to sky blue (aaccff) as background. Set
  # border to transparent and grid lines to white (ffffff).
  $c->setPlotArea(50, 55, 500, 280, $c->linearGradientColor(0, 55, 0, 335, 0xf9fcff,
      0xaaccff), -1, $perlchartdir::Transparent, 0xffffff);

  # Add a legend box at (50, 28) using horizontal layout. Use 10pts Arial Bold as font,
  # with transparent background.
  $c->addLegend(50, 28, 0, "arialbd.ttf", 10)->setBackground($perlchartdir::Transparent);

  # Set y-axis tick density to 30 pixels. ChartDirector auto-scaling will use this as
  # the guideline when putting ticks on the y-axis.
  $c->yAxis()->setTickDensity(30);

  # Set axis label style to 8pts Arial Bold
  $c->xAxis()->setLabelStyle("arialbd.ttf", 8);
  $c->yAxis()->setLabelStyle("arialbd.ttf", 8);

  # Set axis line width to 2 pixels
  $c->xAxis()->setWidth(2);
  $c->yAxis()->setWidth(2);

  # Add axis title using 10pts Arial Bold Italic font
  $c->yAxis()->setTitle("Precision", "arialbi.ttf", 10);
  $c->xAxis()->setTitle("Recall", "arialbi.ttf", 10);

  my $layer1 = $c->addLineLayer($prec, 0xff0000);
  $layer1->setXData($rec);

  # Set the line width to 3 pixels
  $layer1->setLineWidth(3);

  # Use 9 pixel square symbols for the data points
  #$layer1->getDataSet(0)->setDataSymbol($perlchartdir::CircleSymbol, 9);

  # Output the chart
  $c->makeChart("$outputDir/$module/roc.png")

}

 

sub generateGauge
{
  # The value to display on the meter
  my ($value, $module, $file, $min, $max) = @_;

  # Create an LinearMeter object of size 60 x 265 pixels, using silver background with
  # a 2 pixel black 3D depressed border.
  my $m = new LinearMeter(160, 70, perlchartdir::silverColor(), 0, -2);

  # Set the scale region top-left corner at (15, 25), with size of 200 x 50 pixels. The
  # scale labels are located on the top (implies horizontal meter)
  $m->setMeter(15, 20, 130, 10, $perlchartdir::Top); 

  # Set meter scale from 0 - 100, with a tick every 10 units
  $m->setScale($min, $max, ($max-$min)/5);

  # Set 0 - 50 as green (99ff99) zone, 50 - 80 as yellow (ffff66) zone, and 80 - 100 as
  # red (ffcccc) zone
  $m->addZone($min, $max/2, 0xff0000);
  $m->addZone($max/2, $max*0.80, 0xffff00);
  $m->addZone($max*0.80, $max*1.00, 0x00ff00);

  # Add a blue (0000cc) pointer at the specified value
  $m->addPointer($value, 0x0000cc);

  # Add a label at bottom-left (10, 68) using Arial Bold/8 pts/red (c00000)
  #$m->addText(10, 68, $module, "arialbd.ttf", 8, 0xc00000,
  #  $perlchartdir::BottomLeft);

  # Add a text box to show the value formatted to 2 decimal places at bottom right. Use
  # white text on black background with a 1 pixel depressed 3D border.
  $m->addText(90, 55, $m->formatValue($value, "2"), "arial.ttf", 8, 0xffffff,
    $perlchartdir::BottomRight)->setBackground(0, 0, -1);


  # Output the chart
  $m->makeChart($file);
}

sub generateBarGraph
{
  my ($results, $outFile) = @_;
# The data for the bar chart

# Create a XYChart object of size 540 x 375 pixels
  my $c = new XYChart(1124, 450);

# Add a title to the chart using 18 pts Times Bold Italic font
  #$c->addTitle("Average Precision by class", "timesbi.ttf", 18);

# Set the plotarea at (50, 55) and of 440 x 280 pixels in size. Use a vertical
# gradient color from light blue (f9f9ff) to blue (6666ff) as background. Set border
# and grid lines to white (ffffff).
  $c->setPlotArea(50, 35, 1024, 290, $c->linearGradientColor(0, 55, 0, 335, 0xf9f9ff,
      0x6666ff), -1, 0xffffff, 0xffffff);

# Add a legend box at (50, 28) using horizontal layout. Use 10pts Arial Bold as font,
# with transparent background.
  $c->addLegend(50, 0, 0, "arialbd.ttf", 10)->setBackground($perlchartdir::Transparent);

# Set the x axis labels
  my @labels = keys %objects;
  $c->xAxis()->setLabels(\@labels);

# Draw the ticks between label positions (instead of at label positions)
  $c->xAxis()->setTickOffset(0.5);

# Set axis label style to 8pts Arial Bold
  $c->xAxis()->setLabelStyle("arialbd.ttf", 8, $perlchartdir::TextColor, 90);
  $c->yAxis()->setLabelStyle("arialbd.ttf", 8);

# Set axis line width to 2 pixels
  $c->xAxis()->setWidth(2);
  $c->yAxis()->setWidth(2);

# Add axis title
  $c->yAxis()->setTitle("Average Precision");

# Add a multi-bar layer with 3 data sets
  my $layer = $c->addBarLayer2($perlchartdir::Side);

  my $color = 0;
  foreach my $module (keys %modules)
  {
    my @data;
    foreach my $obj (@labels)
    {
      if (defined $$results{"$module|$obj"})
      {
        push(@data,$$results{"$module|$obj"});
      } else {
        push(@data, 0);
      }
    }

    $layer->addDataSet(\@data, $colorMap[$color], $module);
    $color = ($color+1)%$#colorMap;
  }

# Set bar border to transparent. Use glass lighting effect with light direction from
# left.
  $layer->setBorderColor($perlchartdir::Transparent, perlchartdir::glassEffect(
      $perlchartdir::NormalGlare, $perlchartdir::Left));

# Configure the bars within a group to touch each others (no gap)
  $layer->setBarGap(0.2, $perlchartdir::TouchBar);

# Output the chart
  $c->makeChart("$outputDir/$outFile");


}
