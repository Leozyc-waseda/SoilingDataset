#!/usr/bin/perl

use perlchartdir;
use strict;


my $inputDir = shift;
my $outputDir = shift; 

my @colorMap = (0x000000,
                0xff0000,
                0xff00ff,
                0xffff00,
                0x00ff00,
                0x0000ff,
                0xffffff);

my %AP;
my %FPS;

my %truePositive;
my %falseNegative;
my $numFrames;

open(FILE, "$inputDir/results.txt") || die "can not open results file $!";
while(<FILE>)
{
  if (/(.*?) (.*?) (.*?) (.*?) (.*)/)
  {
    my $module = $1;
    my $dbName = $2;
    my $_totalFrames = $3;
    my $_fps = $4;
    my $_ap = $5;

    my $module_dbName = "$module|$dbName";

    my ($fps) = ($_fps =~ /FPS:(.*)/);
    my ($totalFrames) = ($_totalFrames =~ /Frames:(.*)/);
    my ($ap) = ($_ap =~ /AP:(.*)/);

    $numFrames = $totalFrames;
    $FPS{$module_dbName} = $fps;
    $AP{$module_dbName} = $ap;

    my ($fn, $tp) = &getROCData($module, $dbName);
    #Save the data for a combined plot
    $falseNegative{$module_dbName} = $fn;
    $truePositive{$module_dbName} = $tp;

  }

}
close(FILE);


open(HTML, ">$outputDir/index.html") || die "Can not open index.html $!";

print HTML <<HEADER;
<HTML>
<HEAD>
<Title> Neovision2 Object Detection Results </Title>
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
<H1> Neovision2 Object Detection Results <H1>
HEADER


###########################################################  results #######################################

&generateCombineROC(\%falseNegative, \%truePositive, "roc.png");


print HTML <<HEADER;
<TABLE border="1" cellspacing="0" cellpadding="0" style='BORDER-BOTTOM:#eeeeee 1px solid; BORDER-RIGHT:#eeeeee 1px solid; BORDER-LEFT:#eeeeee 1px solid;'>
  <tr> <td class="tab"><div align="center">Any object Detection (n=$numFrames) </div></td></TR>
  <tr> <TD><IMG SRC=roc.png></TD></TR>
</Table>
HEADER

print HTML <<HEADER;
<H2> Overall results </H2>
<TABLE border="1" cellspacing="0" cellpadding="0" style='BORDER-BOTTOM:#eeeeee 1px solid; BORDER-RIGHT:#eeeeee 1px solid; BORDER-LEFT:#eeeeee 1px solid;'>
          <tr>
            <td class="tab"><div align="center">Method </div></td>
            <td class="tab"><div align="center">Mean FPS</div></td>
          </tr>
HEADER

foreach my $module (reverse sort { $FPS{$a} <=> $FPS{$b} } keys %FPS)
{
  print HTML "<TR><TD align=center>$module</TD>\n";
  &generateGauge($FPS{$module}, $module, "$outputDir/${module}_fps.png", 0, 100);
  print HTML "<TD align=center><A href=$module><IMG SRC=${module}_fps.png></A></TD>\n";
  print HTML "</TR>\n";
}
print HTML "</TABLE>\n";

print HTML "</HTML>\n";
close(HTML);


sub getROCData
{
  my ($module, $dbName) = @_;

  # The data for the line chart
  my $file = "$inputDir/$module/results.$dbName.roc";
  my @tp;
  my @fn;
  open(ROC, $file) || die "Can not open $file $!";
  while(<ROC>)
  {
    if (/(.*) (.*)/)
    {
      push(@tp, $1);
      push(@fn, $2);
    }
  }
  close(ROC);

  return (\@tp, \@fn);
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

 
sub getAverageFps
{
  my ($module, $object, $getObjects) = @_;
  
  my $fps = 0;
  my $numOfFrames = 0;

  my $file = "$inputDir/$module/results_${object}_${getObjects}.time";
  open(FILE, $file) || die "Can not open $file $!";
  while(<FILE>)
  {
    if (/(\d+) (.*)/)
    {
      my $frame = $1;
      my $time = 1/$2;
    
      $fps += $time;
      $numOfFrames++;
    }
  }
  close(FILE);

  return $fps/$numOfFrames;
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
  my $c = new XYChart(600, 450);

# Add a title to the chart using 18 pts Times Bold Italic font
  #$c->addTitle("Average Precision by class", "timesbi.ttf", 18);

# Set the plotarea at (50, 55) and of 440 x 280 pixels in size. Use a vertical
# gradient color from light blue (f9f9ff) to blue (6666ff) as background. Set border
# and grid lines to white (ffffff).
  $c->setPlotArea(50, 35, 500, 290, $c->linearGradientColor(0, 55, 0, 335, 0xf9f9ff,
      0x6666ff), -1, 0xffffff, 0xffffff);

# Add a legend box at (50, 28) using horizontal layout. Use 10pts Arial Bold as font,
# with transparent background.
  #$c->addLegend(50, 0, 0, "arialbd.ttf", 10)->setBackground($perlchartdir::Transparent);

# Set the x axis labels

# Draw the ticks between label positions (instead of at label positions)
  $c->xAxis()->setTickOffset(0.5);

# Set axis label style to 8pts Arial Bold
  $c->xAxis()->setLabelStyle("arialbd.ttf", 8, $perlchartdir::TextColor,0);
  $c->yAxis()->setLabelStyle("arialbd.ttf", 8);

# Set axis line width to 2 pixels
  #$c->xAxis()->setWidth(2);
  #$c->yAxis()->setWidth(2);

# Add axis title
  $c->yAxis()->setTitle("% Fixations");

  my @labels = keys %$results;;
  $c->xAxis()->setLabels(\@labels);

  my @data;
  my @posErrBars;
  my @negErrBars;
  foreach my $module (keys %$results)
  {
    push(@data, $$results{$module}[0]);
    push(@posErrBars, $$results{$module}[0] + $$results{$module}[1]);
  }
  my $layer = $c->addBarLayer(\@data);

  # Enable bar label for the whole bar
   $layer->setAggregateLabelStyle();

   my $markLayer = $c->addBoxWhiskerLayer(undef, undef, undef, undef, \@posErrBars, -1,
     0x000000);
   $markLayer->setLineWidth(2);
   $markLayer->setDataGap(0.51);

  
  # # Enable bar label for each segment of the stacked bar
  # $layer->setDataLabelStyle();
  
# Set bar border to transparent. Use glass lighting effect with light direction from
# left.
  $layer->setBorderColor($perlchartdir::Transparent, perlchartdir::glassEffect(
      $perlchartdir::NormalGlare, $perlchartdir::Left));

# Configure the bars within a group to touch each others (no gap)
  #$layer->setBarGap(0.2, $perlchartdir::TouchBar);

# Output the chart
  $c->makeChart("$outputDir/$outFile");


}

sub generateCombineROC
{
  my($falseN, $trueP, $outFile) = @_;

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
  $c->yAxis()->setTitle("True Positive", "arialbi.ttf", 10);
  $c->xAxis()->setTitle("False Positive", "arialbi.ttf", 10);

  my  $color = 0;
  foreach my $module (keys %$falseN)
  {
    my $fn = $$falseN{$module};
    my $tp = $$trueP{$module};

    my $layer1 = $c->addLineLayer($tp, $colorMap[$color], $module);
    $color = ($color+1)%$#colorMap;

    $layer1->setXData($fn);

    # Set the line width to 1 pixels
    $layer1->setLineWidth(1);
  }

  # Output the chart
  $c->makeChart("$outputDir/$outFile")

}

