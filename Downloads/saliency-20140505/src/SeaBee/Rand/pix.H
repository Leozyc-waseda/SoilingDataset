/*
 *  pix.H
 *
 *
 *  Created by Randolph Voorhies on 11/13/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef PIX_H
#define PIX_H

//A simple pixel class that holds a pixel's coordinates and value
template <class T> class pix {
public:
        int x,y;
        T val;

        pix() {}

        pix(int _x, int _y, T _val) {
                x = _x;
                y = _y;
                val = _val;
        }

        bool operator < (const pix<T> &y) const {
                return y.val > val;
        }

        bool operator > (const pix<T> &y) const {
                return y.val < val;
        }

};

#endif