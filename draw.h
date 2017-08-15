/** This header file declares a draw class which has 5 methods(to draw points on specific landmarks) and a constructor. */

//pre-processor directive is used to cause current file to be included only once in a single compilation
#pragma once

#include "opencv2/imgproc/imgproc.hpp"

class draw
{
// private members declarations
private:
    cv::Mat img;
    std::vector<cv::Point> land;

//constructor and class member functions declarations.
public:
    draw(cv::Mat& i,std::vector<cv::Point>& p);
    void jaw();
    void eyes();
    void nose();
    void mouth();
    void eyebrows();
};

