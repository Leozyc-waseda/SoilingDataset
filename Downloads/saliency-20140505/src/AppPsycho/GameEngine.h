#ifndef _PSYCHO_SKIN_GSME_ENGINE_H_
#define _PSYCHO_SKIN_GSME_ENGINE_H_
#include <vector>
#include <string>
#include "psycho-skin-mapgenerator.h"

/*
This header file holds the interface for our game engine and useful functions for running the game
written by Nader Noori
April 7,2008
*/

class Engine
{
        public:
                Engine(int row , int col , int numOfClasses);
                ~Engine() ;
                void setup();
                Matrix *getCurrentState();
                bool canSwap(int fRow , int fCol , int sRow ,int sCol);
                std::vector<Matrix*> swap();
                std::string toFormattedString() ;
                std::string  toString();
        private:
                Matrix* currentState;
                int r ;
                int c ;
                std::string stringify(int i) ;
} ;


#endif
