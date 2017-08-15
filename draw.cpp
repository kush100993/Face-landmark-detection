//included "draw" header file on which the class is declared and is defined here.
#include "draw.h"

/** This function draws circle points of different color based on the start and end point for a specific landmark
    out of the 68 landmark points */
void drawing(int start,int upto,cv::Mat& im,std::vector<cv::Point>& l,const cv::Scalar s)
{
   for(int i=start;i<upto;++i)
        {
            cv::circle(im,l[i],2,s,-1);
        }
}

//constructor to initialize the "img" and "land" variables for use in different methods.
draw::draw(cv::Mat& image, std::vector<cv::Point>& lan)
    {
        img = image;
        land = lan;
    }

//jaw method draws points of specific color around the jaw landmark points
void draw::jaw()
    {
        drawing(0,17,img,land,cv::Scalar(255,0,0));
    }

//eyes method draws points of specific color around the eye landmark points
void draw::eyes()
    {
        drawing(36,48,img,land,cv::Scalar(0,255,0));
    }

//nose method draws points of specific color around the nose landmark points
void draw::nose()
    {
        drawing(27,36,img,land,cv::Scalar(0,0,255));
    }

//mouth method draws points of specific color around the mouth landmark points
void draw::mouth()
    {
        drawing(48,68,img,land,cv::Scalar(255,0,255));
    }

//eyebrows method draws points of specific color around the eyebrow landmark points
void draw::eyebrows()
    {
        drawing(17,27,img,land,cv::Scalar(255,255,0));
    }

