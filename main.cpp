#include "opencv2/objdetect/objdetect.hpp"
#include "dlib/image_processing/shape_predictor.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <iostream>
#include "draw.h"

// Using namespace to avoid using scope resolution explicitly for every member function called from different libraries.
using namespace std;
using namespace cv;
using namespace dlib;


//function declarations
void Rect2rectangle(Rect& r,dlib::rectangle& rec);

void dlib_point2cv_Point(full_object_detection& S, std::vector<Point>& L,double& scale);

void show(std::vector<char>& sel, draw& d);


/**--------------------------------Main Program---------------------------*/

int main(int argc, char* argv[])
{
    /**checking for number of arguments passed in the command line if <3, it gives error and explains the right way to run
       the program.*/
    if(argc<3)
    {
        cout<<"ERROR :Application was not run correctly"<<endl;
        cout<<"Run as: file_name.exe image/image_path E/EB/J/M/N/A"<<endl;
        cout<<"Here using E:Shows Eyes, B:Eyebrows, J:Jaw, M:Mouth, N:Nose, A:All"<<endl;
        return -1;
    }

    //Declaring a variable "image" to store input image given.
    Mat image;

    //try block, tries to read the image and if there is a problem reading the image,it throws an exception.
    try
    {
        image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
        if(image.empty()){ throw(0); }
    }

    //this catch block catches any exception as it is specified with "..." and displays the exception message and terminates.
    catch(...)
    {
        cout<<"Exception: Image is not read properly, please check image path"<<endl;
        exit(0);
    }

    // char vector to store the chars passed in the command line to later show specific facial landmarks
    std::vector<char> sel;

    for(int i=2;i<argc;++i)
    {
        //checking if the passed chars in the command line are expected or not.
        if((*argv[i]=='E') || (*argv[i]=='B') || (*argv[i]== 'M') || (*argv[i]== 'J') || (*argv[i]== 'N') || (*argv[i]== 'A'))
        {
            sel.push_back(*argv[i]);
        }

        //if not then a list of Available Show Chars are displayed, so that the program can again run using the correct chars.
        else
        {
            cout<<"Show char '"<<*argv[i]<<"' is unrecognizable please check and use the correct one"<<endl;
            cout<<"Available Show Chars, E: Eyes  B: Eyebrows  J:Jaw  M:Mouth  N:Nose  A:All"<<endl;
            return -1;
        }
    }

    //declaring two more Mat variables to stores gray scale and resized image of the original image.
    Mat gray,resized;

    //scale for resizing.
    double scale = 0.5;

    //converts original image to gray scale and stores it in "gray".
    cvtColor(image,gray,COLOR_BGR2GRAY);

    //resize the gray scale image for speeding the face detection.
    cv::resize(gray,resized,Size(),scale,scale);

    //Histogram equalization is performed on the resized image to improve the contrast of the image which can help in detection.
    equalizeHist( resized, resized );

    /**Object of CascadeClassifier class "face_cascade" is created to load the pre-trained classifier for face detection provided
       by opencv which is implemented based on the Viola-Jones paper of face detection using HAAR cascades. */
    CascadeClassifier face_cascade;
    face_cascade.load( "haarcascade_frontalface_alt2.xml" );

    /**Object of Shape predictor class "sp" is created to load "shape_predictor_68_face_landmarks.dat" file which is a pre-trained
       cascade of regression tree implemented using "One Millisecond face alignment with an ensemble of regression trees"*/
    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

    //"faces" vector of type Rect is declared to store detected face coordinates,width and height.
    std::vector<Rect> faces;

    /**"detectMultiScale" method is called to detect faces from the resized original image.
       Arguments: 1. Source Image(resized and gray scaled version of original image)
                  2. Destination vector(faces vector)
                  3. Scale factor(how much the image size is reduced at each image scale)
                  4. Minimum Neighbors(how many neighbors each candidate rectangle should have to retain it)
                  */
    face_cascade.detectMultiScale( resized, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE);

    //"shapes" vector of type full_object_detection is declared to store landmark coordinated of a detected face
    std::vector<full_object_detection> shapes;

    //checking if faces are detected or not, if not then there is no need to find landmarks.
    if(!faces.empty())
    {
        /**for all detected faces, first convert each cv::Rect face to dlib::rectangle face as expected by the sp function.
           also convert the image from Mat to array2d as expected by the sp function.*/
        for( unsigned int i = 0; i < faces.size(); ++i)
        {
            //"face" variable of type rectangle(of dlib library) is declared to convert from opencv Rect to dlib rectangle.
            dlib::rectangle face;

            Rect2rectangle(faces[i],face);

            //detects face landmarks using shape predictor method of dlib library
            full_object_detection shape = sp(dlib::cv_image<unsigned char>(resized),face);

            //store all landmarks of all detected faces in "shapes" vector
            shapes.push_back(shape);
        }
    }
    else
    {
        cout<<"No faces detected in the given image."<<endl;
        imshow("Given image",image);
        waitKey(0);
        return -1;
    }


    for(unsigned int i=0; i<shapes.size() ;++i)
    {
        //landmarks vector is declared to store the 68 landmark points.
        std::vector<Point> landmarks;

        /**at each index of "shapes" vector there is an object of full_object_detection class which stores the 68 landmark
        points in dlib::point from, which needs to be converted back to cv::Point for displaying.*/
        dlib_point2cv_Point(shapes[i],landmarks,scale);

        //an object of class draw is created
        draw d(image,landmarks);

        //show function to show detected landmarks as per the command line arguments and draw object.
        show(sel,d);
    }

    //shows the final image with all detections.
    imshow( "Detected Landmarks", image );

    //waits for infinite time until a key is pressed.
    waitKey(0);

    return 0;
}

/**-------------------------------------------------------------------------------*/

//Function Definitions.

/** This function converts cv::Rect to dlib::rectangle.
    This function is needed because in this implementation I have used opencv and dlib bothand they
    both have their own image processing library so when using a dlib method, its arguments should be
    as expected */
void Rect2rectangle(Rect& r,dlib::rectangle& rec)
{
    rec = dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}


/** This function converts dlib::point to cv::Point and stores in a vector of landmarks
    This function is needed because in this implementation I have used opencv and dlib bothand they
    both have their own image processing library so when using a dlib method, its arguments should be
    as expected */
void dlib_point2cv_Point(full_object_detection& S,std::vector<Point>& L,double& scale)
{
    for(unsigned int i = 0; i<S.num_parts();++i)
    {
        L.push_back(Point(S.part(i).x()*(1/scale),S.part(i).y()*(1/scale)));
    }
}

/**This function shows specific landmarks of the detected faces in an image based on what Show Char is specified in the command line.*/
void show(std::vector<char>& sel,draw& d)
{
    for(unsigned int i=0; i<sel.size(); ++i)
    {
        switch(sel[i])
        {
        case 'J':
            d.jaw();
            break;
        case 'M':
            d.mouth();
            break;
        case 'E':
            d.eyes();
            break;
        case 'B':
            d.eyebrows();
            break;
        case 'N':
            d.nose();
            break;
        case 'A':
            d.eyes();
            d.nose();
            d.jaw();
            d.mouth();
            d.eyebrows();
            break;
        }
    }
}
