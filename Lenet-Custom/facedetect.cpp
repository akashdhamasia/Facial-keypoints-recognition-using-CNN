#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

#include <sstream>

using namespace std;
using namespace cv;


//Mat roi;
//Mat roi(64,128, CV_8UC3, Scalar(0,0,255));
Mat roi;
Mat img;
int var1=1;
int var2=0;

int width = 300;
int height = 300;

string name;
ostringstream convert;

Mat face_image;
//Mat depth_image;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {   

/*    if( (x-(width/2)) >= 0 && (y-(height/2)) >= 0 && (x+(width/2)) < img.cols && (y+(height/2)) < img.rows )
    {
    
        if(var1 == 0){
        namedWindow("ROI",2);
        roi = img( Rect(x-(width/2),y-(height/2),width,height) );
        imshow("ROI",roi);
        //waitKey(10);
        var1++;

        }
        else{
        destroyWindow("ROI");
        namedWindow("ROI",2);
        //roi = img( Rect(x-32,y-64,64,128) );
        roi = img( Rect(x-(width/2),y-(height/2),width,height) );
        imshow("ROI",roi);
        //waitKey(10);
        //roi.release();

        }

    }*/

           imshow( "result2", face_image );
 



    }

    else if  ( event == EVENT_RBUTTONDOWN )
    
     {    
          imwrite("andre_face.jpg", face_image );
	  //imwrite("depth_face5.png", depth_image );
	  
	  //cout << depth_image;
      
     }
}


int detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );

string cascadeName;
string nestedCascadeName;


int main( int argc, const char** argv )
{
    //VideoCapture capture(0);
    Mat frame, image;
    string inputName;
    bool tryflip;
    CascadeClassifier cascade, nestedCascade;
    double scale=1;
    int number_detections;

    if ( !nestedCascade.load("/home/sprva/AAkash/installations/opencv/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml") )
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    if( !cascade.load("/home/sprva/AAkash/installations/opencv/opencv/data/haarcascades/haarcascade_frontalface_alt.xml") )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    namedWindow("result",1);

    while(1){

    //capture >> image;         // get a new frame from camera
 
        image = imread( "/home/sprva/Downloads/andre.jpeg", 1 );
        if( image.empty() )
        {
            //if(!capture.open( inputName ))
                cout << "Could not read " << inputName << endl;
        }

    number_detections = detectAndDraw( image, cascade, nestedCascade, scale, false );
        
    cout << number_detections << endl;
    waitKey(10);
    }
    return 0;
}

int detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
   setMouseCallback("result", CallBackFunc, NULL);
   Mat original = img;
   //Mat original_depth = imread("depth_0001.png");
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );

    cascade.detectMultiScale( gray, faces );
//    for(int i=0;i<faces.size();i++)
    //{

     //faces[0].height = faces[0].height + 15;
     //faces[0].width = faces[0].width + 2; 
 
     face_image = original(faces[0]);
     //depth_image = original_depth(faces[0]);

     resize(face_image, face_image, Size(96,96));
     //resize(depth_image, depth_image, Size(96,96));

   // }
    
    imshow( "result", gray );
imshow( "result1", face_image );
 
    return faces.size();

}
