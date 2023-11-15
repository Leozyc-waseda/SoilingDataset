/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you want to add, delete, or rename functions or slots, use
** Qt Designer to update this file, preserving your code.
**
** You should not define a constructor or destructor in this file.
** Instead, write your code in functions called init() and destroy().
** These will automatically be called by the form's constructor and
** destructor.
*****************************************************************************/

#include <qlabel.h>
#include <qtable.h>
#include "Util/log.H"

void BayesNetworkDialog::init( Bayes &bayesNet )
{
  itsBayesNet = &bayesNet;
  setupTab();


  itsBayesNet->import("fv.txt");

}


void BayesNetworkDialog::update()
{

}


void BayesNetworkDialog::setupTab()
{
  static bool setupTab = true;

  if (setupTab)
  {
    //remove the first dummy page
    QWidget *page = tabWidget->page(0); //get the first page of the tab
    tabWidget->removePage(page);

    //priors statistics
    QWidget* statTab = new QWidget(tabWidget);
    QVBoxLayout* tabLayout = new QVBoxLayout(statTab, 11, 6);
    for (uint i=0; i<itsBayesNet->getNumClasses(); i++)
    {
      QString stat = QString("Class %1: freq %L2 Prior %L3")
        .arg(i)
        .arg(itsBayesNet->getClassFreq(i))
        .arg(itsBayesNet->getClassProb(i));

      QLabel *label = new QLabel(stat, statTab, QString("stat Label %1").arg(i));
      label->setFont(QFont("Times", 20));
      tabLayout->addWidget(label);
    }
    tabWidget->insertTab(statTab,QString("Stats"));


    //Add a tab to change the mean and variance
    QWidget *changeParamTab = new QWidget(tabWidget);
    QVBoxLayout* classLayout = new QVBoxLayout(changeParamTab, 11, 6);

    for(uint cls=0; cls<itsBayesNet->getNumClasses(); cls++)
    {
      QString className = QString("%1:%L2").arg(cls).arg(itsBayesNet->getClassName(cls));
      QLabel *label = new QLabel(className, changeParamTab, className);

      classLayout->addWidget(label);

      QTable *paramTable = new QTable(changeParamTab, "paramTable");
      paramTable->setNumRows(2);
      paramTable->setNumCols(itsBayesNet->getNumFeatures());

      //set the values
      for(uint fv=0; fv<itsBayesNet->getNumFeatures(); fv++)
      {
        paramTable->setText(0, fv, QString("%L1").arg(itsBayesNet->getMean(cls, fv)));
        paramTable->setText(1, fv, QString("%L1").arg(itsBayesNet->getStdevSq(cls, fv)));
      }

      classLayout->addWidget(paramTable);
    }

    tabWidget->insertTab(changeParamTab,QString("Param Change"));


#ifdef INVT_HAVE_QWT

    //Get the color names
    QStringList colorNames = QColor::colorNames();

    //Graph the conditional prob
    for (uint i=0; i<itsBayesNet->getNumFeatures(); i++)
    {
      QWidget* tab = new QWidget(tabWidget);
      QHBoxLayout* tabLayout = new QHBoxLayout(tab, 11, 6);

      QString title(itsBayesNet->getFeatureName(i));
      QwtPlot *qwtPlot = new QwtPlot(title, tab);
      //not supported in lattest qtw??   qwtPlot->setAutoLegend(true);
      qwtPlot->setAutoReplot(true);
      for (uint cls=0; cls<itsBayesNet->getNumClasses(); cls++)
      {
        LFATAL("FIXME: I need to be updated to latest qwt");
        /*
        const double STEP  = 0.1;
        const int XRANGE = int(50.0F/STEP);
        double x[XRANGE], y[XRANGE];

        long curve = qwtPlot->insertCurve(itsBayesNet->getClassName(cls));

        int colorId = (cls * 10)%colorNames.size();
        qwtPlot->setCurvePen(curve, QPen(colorNames[colorId]));

        double mean = itsBayesNet->getMean(cls, i);
        double stdevSq = itsBayesNet->getStdevSq(cls, i);

        for (int xi=0; xi<XRANGE; xi++){
          x[xi] = xi * STEP;
          y[xi] = itsBayesNet->gauss(x[xi], mean, stdevSq);
        }

        qwtPlot->setCurveData(curve, x, y, XRANGE);
        */
      }
      tabLayout->addWidget(qwtPlot);
      tabWidget->insertTab(tab,QString("Normal Dist %1").arg(i) );

      setupTab = false;
      qwtPlot->replot();
      qwtPlot->show();
    }

#else
  printf("Need the qwtPlot widget for graphs!!!\n");
#endif
  }
}
