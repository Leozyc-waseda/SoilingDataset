/*
 *  V2.H
 *
 *
 *  Created by Randolph Voorhies on 11/26/07.
 *
 */
#ifndef V2_H
#define V2_H

#define PI 3.14159265f


//Parameters
#define alpha_x 1.0f
#define alpha_y 1.0f
#define x_th        1.0f
#define x_sat        2.0f
#define y_th1        1.0f
#define y_th2        1.2f
#define g1                0.21f
#define g2                2.5f
#define y_sat        300.0f
#define I_o                0.1f
#define I_c                1.0f
#define THETA_LEVELS 24

#include "Image/Point2D.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include <vector>
#include <math.h>


using namespace std;

class V2 {
public:
        V2(Dims size);

        //Run the simulation for t steps
        void step();

        void setInput(vector< Image<float> > input);


        //Helper Functions
        float g_x(float x);
        float g_y(float y);
        float psi(float theta);
        float tau();
        float I_normalization(Point2D<int> i, int theta_index);
        float phi(float x, float y);
        float beta(Point2D<int> i, Point2D<int> j);
        float sign(float x);
        float theta_one(Point2D<int> i, int i_theta_index, Point2D<int> j);
        float theta_two(Point2D<int> i, Point2D<int> j, int j_theta_index);
        float theta_one_prime(Point2D<int> i, int i_theta_index, Point2D<int> j);
        float theta_two_prime(Point2D<int> i, Point2D<int> j, int j_theta_index);
        float theta_plus(float theta_a, float theta_b);
        float theta_minus(float theta_a, float theta_b);
        void theta_a_b(float &theta_a, float &theta_b, Point2D<int> i, Point2D<int> j, int i_theta_index, int j_theta_index);
        float f_1(float d);
        float f_2(float d);
        float f_T(float d);
        float W(Point2D<int> i, int i_theta_index, Point2D<int> j, int j_theta_index);
        float J(Point2D<int> i, int i_theta_index, Point2D<int> j, int j_theta_index);
        float sumPsiGy(Point2D<int> i, int theta_index);
        float sumJGx(Point2D<int> i, int theta_index);
        float sumWGx(Point2D<int> i, int theta_index);

        float ind2rad(int theta_index);


private:
        //The Pyramidal and Inhibitory cells are layed out into the itsX and itsY matrices (respectively). The itsXprime and itsYprime matrices are used to store the next
        //step differences which are then added to the itsX and itsY matrices at the end of every step.
        vector< Image<float> > itsX;
        vector< Image<float> > itsY;
        vector< Image<float> > itsXprime;
        vector< Image<float> > itsYprime;

        //The input, arranged as a matrix
        vector< Image<float> > I;

        Dims itsDims;
        int t;
};


V2::V2(Dims size) {
        itsDims = size;
        itsX.resize(THETA_LEVELS, Image<float>(size, ZEROS));
        itsY.resize(THETA_LEVELS, Image<float>(size, ZEROS));
        itsXprime.resize(THETA_LEVELS, Image<float>(size, ZEROS));
        itsYprime.resize(THETA_LEVELS, Image<float>(size, ZEROS));

        t=0;

        //Init the random number generator
        srand((unsigned) (time(0)));
}

void V2::setInput(vector < Image<float> > input) {
        I = input;
}

//This function progresses through one time step for the neural network. First, the pyramidal and inhibitory differentials are computed,
//and then added to their respective activity matrices.
void V2::step() {
        float xprime, yprime;

        for(int theta_index=0; theta_index<THETA_LEVELS; theta_index++) {
                for(int i=0; i<itsDims.w();i++) {
                        for(int j=0; j<itsDims.h(); j++) {
                                xprime = -alpha_x*itsX[theta_index].getVal(i,j) - g_y(itsY[theta_index].getVal(i,j)) - sumPsiGy(Point2D<int>(i,j), theta_index) + sumJGx(Point2D<int>(i,j),theta_index) + /*I + */ + I_o + /*N_x*/ + I_normalization(Point2D<int>(i,j),theta_index);
                                yprime = -alpha_y*itsY[theta_index].getVal(i,j) + g_x(itsX[theta_index].getVal(i,j)) + sumWGx(Point2D<int>(i,j), theta_index) + I_c /*+ N_y*/;

                                itsXprime[theta_index].setVal(i,j,xprime);
                                itsYprime[theta_index].setVal(i,j,yprime);
                        }
                }
        }

        for(int theta_index=0; theta_index<THETA_LEVELS; theta_index++) {
                itsX[theta_index] += itsXprime[theta_index];
                itsY[theta_index] += itsYprime[theta_index];
        }

        t++;
}

float V2::sumPsiGy(Point2D<int> i, int theta_index) {
        float sum = 0;

        for(int d_theta_index=1; d_theta_index<THETA_LEVELS; d_theta_index++) {
                for(int x=0; x<itsDims.w();x++) {
                        for(int y=0; y<itsDims.h(); y++) {
                                sum += psi(ind2rad(d_theta_index)) * g_y(itsY[((theta_index + d_theta_index)%THETA_LEVELS)].getVal(i));
                        }
                }
        }
        return sum;
}

//NOTE: May need to provide constant Taus here?
float V2::sumJGx(Point2D<int> i, int theta_index) {
        float sum = 0;

        for(int x=0; x<itsDims.w(); x++) {
                for(int y=0; y<itsDims.h(); y++) {
                        if(Point2D<int>(x,y) == i) continue;
                        for(int theta_prime_index = 0; theta_prime_index < THETA_LEVELS; theta_prime_index++) {
                                sum += J(i, theta_index, Point2D<int>(x,y), theta_prime_index) * g_x(itsX[theta_prime_index].getVal(x,y) * (t - tau()));
                        }
                }
        }
        return sum;
}

//NOTE: May need to provide constant Taus here?
float V2::sumWGx(Point2D<int> i, int theta_index) {
        float sum = 0;
        for(int x=0; x<itsDims.w(); x++) {
                for(int y=0; y<itsDims.h(); y++) {
                        if(i == Point2D<int>(x,y)) continue;
                        for(int theta_prime_index = 0; theta_prime_index < THETA_LEVELS; theta_prime_index++) {
                                sum += W(i, theta_index, Point2D<int>(x,y),theta_prime_index) * g_x(itsX[theta_prime_index].getVal(x,y)*(t-tau()));
                        }
                }
        }
        return sum;
}


//Takes a theta index (an integer specifying a level of theta), and returns the actual radians which that level represents between 0 and 2PI
float V2::ind2rad(int theta_index) {
        return (theta_index/THETA_LEVELS)*(2.0f*PI);
}

float V2::g_x(float x) {
        if(x < x_th)
                return 0.0f;
        else if(x >= x_th && x <= x_sat)
                return x - x_th;
        else if(x > x_sat)
                return x_sat - x_th;
        else {
                LERROR("ERROR! x out of bounds");
                return -1.0f;
        }
}

float V2::g_y(float y) {
        if(y < y_th1)
                return 0.0f;
        else if(y >= y_th1 && y <= y_th2)
                return g1*(y - y_th1);
        else if(y >= y_th2 && y <= y_sat)
                return g2*(y - y_th2) + g1*(y_th2 - y_th1);
        else if(y > y_sat)
                return g2*(y_sat - y_th2) + g1*(y_th2 - y_th1);
        else {
                LERROR("ERROR! y out of bounds");
                return -1.0f;
        }
}

float V2::psi(float theta) {
        if(fabs(theta) == PI/12.0f)
                return 0.8f;
        if(fabs(theta) == PI/6.0f)
                return 0.1f;
        else
                return 0.0f;
}

float V2::tau() {
        return rand()/(float(RAND_MAX)+1.0f)*(1.0f - 0.8f)+0.8f;
}

float V2::I_normalization(Point2D<int> i, int theta_index) {

        float a_hat = 0.0f;
        Point2D<int> index;

        for(int _x = i.i-2; _x < i.i+2; _x++) {
                for(int _y = i.j-2; _y < i.j+2; _y++) {
                        index = Point2D<int>(_x, _y);
                        index.clampToDims(itsDims);
                        a_hat += itsX[theta_index].getVal(index);
                }
        }

        return pow(a_hat, 2.0f)/128.0f;
}

float V2::phi(float x, float y) {
        float diff = x-y;

        if(diff > -PI && diff <= PI) {
                return diff;
        }
        else if(diff <= -PI) {
                return diff + 2.0f*PI;
        }
        else if(diff > PI) {
                return diff - 2.0f*PI;
        }
        LERROR("Diff Out Of Bounds!");
        exit(0);
        return 1.0f;

}

float V2::beta(Point2D<int> i, Point2D<int> j) {
        float beta;

        float o = j.j - i.j;
        float a = j.i - i.i;

        //float o = i.j - j.j;
        //float a = i.i - j.i;

        if(a == 0.0f) {
                if(o > 0.0f)
                        return PI/2.0f;
                else
                        return 3.0f*PI/2.0f;
        }

        beta = atan(o/a);
        if(o > 0.0f && a < 0.0f) {
                beta = PI+beta;
        }
        else if(o <= 0.0f && a < 0.0f) {
                beta = PI + beta;
        }
        else if(o < 0.0f && a > 0.0f) {
                beta = 2.0f*PI + beta;
        }

        return beta;
}

float V2::sign(float x) {
        if(x >= 0.0f)
                return 1.0f;
        else
                return -1.0f;
}

float V2::theta_one(Point2D<int> i, int i_theta_index, Point2D<int> j) {
        return phi(2.0f*(PI/(float)THETA_LEVELS)*(float)i_theta_index, beta(i,j));
}

float V2::theta_two(Point2D<int> i, Point2D<int> j, int j_theta_index) {
        return phi(beta(i,j), 2.0f*(PI/(float)THETA_LEVELS)*(float)j_theta_index);
}

float V2::theta_one_prime(Point2D<int> i, int i_theta_index, Point2D<int> j) {
        float _theta_one = theta_one(i,i_theta_index,j);
        return sign(_theta_one)*fabs(PI-fabs(_theta_one));
}

float V2::theta_two_prime(Point2D<int> i, Point2D<int> j, int j_theta_index) {
        float _theta_two = theta_two(i,j,j_theta_index);
        return sign(_theta_two)*fabs(PI-fabs(_theta_two));
}

void V2::theta_a_b(float &theta_a, float &theta_b, Point2D<int> i, Point2D<int> j, int i_theta_index, int j_theta_index) {
        float _theta_one = theta_one(i, i_theta_index, j);
        float _theta_two = theta_two(i, j, j_theta_index);
        float _theta_one_prime = theta_one_prime(i, i_theta_index, j);
        float _theta_two_prime = theta_two_prime(i, j, j_theta_index);

        if(fabs(_theta_one) + fabs(_theta_two) <= fabs(_theta_one_prime) + fabs(_theta_two_prime)) {
                theta_a = _theta_one;
                theta_b = _theta_two;
        }
        else {
                theta_a = _theta_one_prime;
                theta_b = _theta_two_prime;
        }
}

float V2::theta_plus(float theta_a, float theta_b) {
        float theta_prime_plus = theta_a + theta_b;

        if(theta_prime_plus >= -PI && theta_prime_plus <= PI)
                return theta_prime_plus;
        else if(theta_prime_plus > PI)
                return 2.0f*PI - theta_prime_plus;
        else
                return -2.0f*PI - theta_prime_plus;
}

float V2::theta_minus(float theta_a, float theta_b) {
        float theta_prime_minus = theta_a - theta_b;

        if(theta_prime_minus >= -PI && theta_prime_minus <= PI)
                return theta_prime_minus;
        else if(theta_prime_minus > PI)
                return 2.0f*PI - theta_prime_minus;
        else
                return -2.0f*PI - theta_prime_minus;
}

float V2::f_1(float d) {
        if(d > 10.0f || d == 0.0f)
                return 0.0f;

        return exp(-pow(d/9.0f,2.0f));
}

float V2::f_2(float d) {
        if(d > 10.0f || d == 0.0f)
                return 0.0f;

        return exp(-d/5.0f);
}

float V2::f_T(float d) {
        return (11.0f/90.0f)*exp(-d/6.0f);
}

//Calculate the synaptic strength between pyramidal neurons
float V2::J(Point2D<int> i, int i_theta_index, Point2D<int> j, int j_theta_index) {
        float theta_a, theta_b;
        float _theta_plus, _theta_minus;
        float d = fabs(i.distance(j));
        float ret;

        //Calculate theta_a and theta_b
        theta_a_b(theta_a, theta_b, i, j, i_theta_index, j_theta_index);
        //Calculate theta_plus and theta_minus
        _theta_plus = theta_plus(theta_a, theta_b);
        _theta_minus = theta_minus(theta_a, theta_b);

        //LINFO("\ntheta_a=%f\ntheta_b=%f\ni.i=%d\ni.j=%d\nj.i=%d\nj.j=%d\ntheta_plus=%f\ntheta_minus=%f\n", theta_a, theta_b, i.i, i.j, j.i,j.j,theta_plus,theta_minus);

        if(d > 0.0f && d <= 2.0f && fabs(theta_a*d) < 0.5f && fabs(theta_b) > PI/3.1f && fabs(theta_b) < 2.0f*PI/3.1f) {

                if(theta_b < 0.0f)
                        return f_T(d)*exp(-2.0f*fabs(theta_a*d));
                else
                        return 0.0f;
        }
        else if(d > 0.0f && d <= 2.0f && fabs(theta_b*d) < 0.5f && fabs(theta_a) > PI/3.1f && fabs(theta_a) < 2.0f*PI/3.1f) {

                if(theta_a < 0.0f)
                        return 3.0f*f_T(d)*exp(-2.0f*fabs(theta_b*d));
                else
                        return 0.0f;
        }



        if(fabs(theta_a) <= PI/11.0f && fabs(theta_b) <= PI/11.0f) {
                //LINFO("1");
                ret = (11.0f/108.0f) * exp(-(3.0f - 2.5f*sign(_theta_plus))*fabs(_theta_plus)/(5.0f*PI) - 2.0f * pow(_theta_minus, 2.0f)/pow(PI, 2.0f))*f_1(d);
        }
        else if(theta_a >= 0.0f && theta_b >= 0.0f && _theta_plus < PI/2.01f) {
                //LINFO("2");
                ret = (11.0f/81.0f) * exp(-3.0f*_theta_plus/(5.0f*PI) - 2.0f*pow(_theta_minus,2.0f)/pow(PI,2.0f))/f_2(d);
        }
        else if(theta_a >= 0.0f && theta_b >= 0.0f && _theta_plus >= PI/2.01f && fabs(_theta_minus) < PI/2.01f) {
                //LINFO("3");
                ret = (11.0f/81.0f) * exp(-pow((9.0f*_theta_plus)/(8.0f*PI),2.0f) - 2.0f*pow(_theta_minus,2.0f)/pow(PI,2.0f))*f_2(d);
        }
        else if(theta_a >= 0.0f && theta_b >= 0.0f && _theta_plus >= PI/2.01f && fabs(_theta_minus) >= PI/2.01f) {
                //LINFO("4");
                ret = (11.0f/81.0f) * exp(-pow(9.0f*_theta_plus/(8.0f*PI),2.0f) - 0.5f*pow((_theta_minus/(PI/2.0f)),6.0f))*f_2(d);
        }
        else if(theta_a <= 0.0f && theta_b <= 0.0f) {
                //LINFO("5");
                ret = (11.0f/81.0f) * exp(-4.0f*pow(_theta_plus/PI,2.0f) - 9.0f*pow(_theta_minus, 2.0f)/pow(PI, 2.0f))*f_2(d);
        }
        else if(theta_a*theta_b <= 0.0f && fabs(_theta_minus) < PI/2.01f) {
                //LINFO("6\n11./81.=%f\nf_2(d)=%f",11./81., f_2(d));
                ret = (11.0f/81.0f) * exp(11.5f*sign(_theta_plus)*pow(_theta_plus,2.0f)/pow(PI,2.0f) - 14.0f*pow(_theta_minus,2.0f)/pow(PI,2.0f))*f_2(d);
        }
        else if(theta_a*theta_b <= 0.0f && fabs(_theta_minus) >= PI/2.01f) {
                //LINFO("7");
                ret = (11.0f/81.0f) * exp(11.5f*sign(_theta_plus)*pow(_theta_plus,2.0f)/pow(PI,2.0f) - (15.0f/4.0f)*pow((2.0f*_theta_minus/PI),6.0f))*f_2(d);
        }
        else {
                LERROR("theta_a and theta_b slipped through the cracks! \n  theta_a=%f \n  theta_b=%f \n  theta_plus=%f  \n  theta_minus=%f\n", theta_a, theta_b, _theta_plus, _theta_minus);
                LINFO("ERROR!");
                exit(0);
                return 1.0f;
        }

        //LINFO("\ntheta_a=%f\ntheta_b=%f\nd=%f\nret=%f", theta_a, theta_b,d,ret);
        //Raster::waitForKey();

        return ret;
}

float V2::W(Point2D<int> i, int i_theta_index, Point2D<int> j, int j_theta_index) {
        float theta_a, theta_b, c;
        float d = fabs(i.distance(j));

        theta_a_b(theta_a, theta_b, i, j, i_theta_index, (j_theta_index+THETA_LEVELS/2)%THETA_LEVELS);

        if(d > 0.0f && d <= 2.0f && fabs(theta_a*d) < 0.5f && fabs(theta_b) > PI/3.1f && fabs(theta_b) < 2.0f*PI/3.1f) {
                if(theta_b < 0.0f)
                        return 0.0f;
                        //return f_T(d)*exp(-2*fabs(theta_a*d));
                else
                        return 0.0f;
        }
        else if(d > 0.0f && d <= 2.0f && fabs(theta_b*d) < 0.5f && fabs(theta_a) > PI/3.1f && fabs(theta_a) < 2.0f*PI/3.1f) {
                if(theta_a < 0.0f)
                        return 0.0f;
                        //return 3*f_T(d)*exp(-2*fabs(theta_b*d));
                else
                        return 0.0f;
        }


        if(fabs(theta_a) <= PI/11.0f && fabs(theta_b) <= PI/11.0f)
                c = 0.0147f;
        else
                c = 0.02646f;

        return c*(J(i, (i_theta_index+THETA_LEVELS/2)%THETA_LEVELS, j, j_theta_index) + J(i,  i_theta_index, j, (j_theta_index+THETA_LEVELS/2)%THETA_LEVELS)/J(i, 0, Point2D<int>(i.i+1, i.j), 0));


}

#endif
