// this program makes a text file and put every data sets in one folder with proper numbering

// fileoperation.cpp is comparative program

#include <fstream>
#include <iostream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sstream>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

unsigned char isFile =0x8;

/*

#Kaggle
# left_eye_center 1
# right_eye_center 2
# left_eye_inner_corner 3
# left_eye_outer_corner 4 
# right_eye_inner_corner 5 
# right_eye_outer_corner 6 
# left_eyebrow_inner_end 7
# left_eyebrow_outer_end 8
# right_eyebrow_inner_end 9 
# right_eyebrow_outer_end 10
# nose_tip 11
# mouth_left_corner 12
# mouth_right_corner 13 
# mouth_center_top_lip 14 
# mouth_center_bottom_lip 15

# AFLW 21 points landmark
#  0|LeftBrowLeftCorner
#  1|LeftBrowCenter
#  2|LeftBrowRightCorner
#  3|RightBrowLeftCorner
#  4|RightBrowCenter
#  5|RightBrowRightCorner
#  6|LeftEyeLeftCorner
#  7|LeftEyeCenter
#  8|LeftEyeRightCorner
#  9|RightEyeLeftCorner
#  10|RightEyeCenter
#  11|RightEyeRightCorner
#  12|LeftEar
#  13|NoseLeft
#  14|NoseCenter
#  15|NoseRight
#  16|RightEar
#  17|MouthLeftCorner
#  18|MouthCenter
#  19|MouthRightCorner
#  20|ChinCenters

*/

int main()
{

Mat src;
ifstream fin;
string dir, filepath, temp, temp1;
int num;
int counter=1;
DIR *dp;
struct dirent *dirp;
struct stat filestat;
const char* new_name2;
const char* new_name1;

ostringstream convert, convert2;
//ostringstream convert1;
ofstream outputFile_aflw;
ofstream outputFile_kaggle;

outputFile_kaggle.open("testing_kaggle.csv");
outputFile_aflw.open("testing_aflw.csv");

string kaggle = "left_eye_center.x,left_eye_center.y,right_eye_center.x,right_eye_center.y,left_eye_inner_corner.x,left_eye_inner_corner.y,left_eye_outer_corner.x,left_eye_outer_corner.y,right_eye_inner_corner.x,right_eye_inner_corner.y,right_eye_outer_corner.x,right_eye_outer_corner.y,left_eyebrow_inner_end.x,left_eyebrow_inner_end.y,left_eyebrow_outer_end.x,left_eyebrow_outer_end.y,right_eyebrow_inner_end.x,right_eyebrow_inner_end.y,right_eyebrow_outer_end.x,right_eyebrow_outer_end.y,nose_tip.x,nose_tip.y,mouth_left_corner.x,mouth_left_corner.y,mouth_right_corner.x,mouth_right_corner.y,mouth_center_top_lip.x,mouth_center_top_lip.y,mouth_center_bottom_lip.x,mouth_center_bottom_lip.y,Imageid";

string aflw = "LeftBrowLeftCorner.x,LeftBrowLeftCorner.y,LeftBrowCenter.x,LeftBrowCenter.y,LeftBrowRightCorner.x,LeftBrowRightCorner.y,RightBrowLeftCorner.x,RightBrowLeftCorner.y,RightBrowCenter.x,RightBrowCenter.y,RightBrowRightCorner.x,RightBrowRightCorner.y,LeftEyeLeftCorner.x,LeftEyeLeftCorner.y,LeftEyeCenter.x,LeftEyeCenter.y,LeftEyeRightCorner.x,LeftEyeRightCorner.y,RightEyeLeftCorner.x,RightEyeLeftCorner.y,RightEyeCenter.x,RightEyeCenter.y,RightEyeRightCorner.x,RightEyeRightCorner.y,LeftEar.x,LeftEar.y,NoseLeft.x,NoseLeft.y,NoseCenter.x,NoseCenter.y,NoseRight.x,NoseRight.y,RightEar.x,RightEar.y,MouthLeftCorner.x,MouthLeftCorner.y,MouthCenter.x,MouthCenter.y,MouthRightCorner.x,MouthRightCorner.y,ChinCenter.x,ChinCenter.y,Imageid";

// 436 -> 43 + 46 / 2
// 374 -> 37 + 40 / 2
// 637 -> 63 + 67 / 2

convert2 << aflw;

new_name1 = convert2.str().c_str();

outputFile_aflw << new_name1 << endl;

convert2.str("");
convert2.clear();


convert2 << kaggle;

new_name1 = convert2.str().c_str();

outputFile_kaggle << new_name1 << endl;

convert2.str("");
convert2.clear();

//outputFile.close();

vector< int > kaggle_correlation = {436, 374, 43, 46, 40, 37, 23, 27, 22, 18, 31, 55, 49, 52, 58};

vector< int > aflw_correlation = {18, 20, 22, 23, 25, 27, 37, 374, 40, 43, 436, 46, 3, 32, 31, 36, 15, 49, 637, 55, 9};

  cout << "dir to get files of: " << flush;
  getline( cin, dir );  // gets everything the user ENTERs

  dp = opendir( dir.c_str() );

  if (dp == NULL)
    {
    cout << "Error opening " << dir << endl;
    //outputFile.close();
    return 0;
    }

  dirp = readdir( dp );
 
  while (dirp)
  {

    filepath = dir + "/" + dirp->d_name;

    if( dirp->d_type == isFile )
    {

	   	      temp = dirp->d_name;

		      temp1 = (temp.substr(temp.find_last_of(".") + 1));

  		      size_t lastindex = temp.find_last_of("."); 
			  string rawname = temp.substr(0, lastindex); 


	    if(temp1 == "png")
	    {

	    //  if(temp1 == "png"){
	      
	      src = imread( filepath.c_str() ); // to check whether file is openable
	      Mat original_src = src.clone();
	      if( !src.data )
	      { printf(" No data! -- Exiting the program \n");
			dirp = readdir( dp );
	        continue;
	       }


   	     string file_name = dir + "/" + rawname + ".pts";


	     std::ifstream infile(file_name.c_str());


		std::string line;
		std::getline(infile, line);
		std::getline(infile, line);

	    std::istringstream iss(line);
	    string a, no_keypoints;
	    if (!(iss >> a >> no_keypoints)) { break; } // error

		std::getline(infile, line);

		//cout << file_name << " " ;
		//cout << line << " " << no_keypoints << endl;

		vector < Point > keypoints;
		Point fake;

//    vector<int>::iterator maxelem = max_element(rangeOfNumbers.begin(), rangeOfNumbers.end());
//    vector<int>::iterator minelem = min_element(rangeOfNumbers.begin(), rangeOfNumbers.end());
//    cout << endl << "Max number: " << (*maxelem) << " at " << std::distance(rangeOfNumbers.begin(), maxelem) + 1;
//    cout << endl << "Min number: " << (*minelem)<< " at " << std::distance(rangeOfNumbers.begin(), minelem) + 1;

		for(int i=0; i<std::stoi(no_keypoints); i++)
		{keypoints.push_back(fake);
		}
		
		for(int i=0; i<std::stoi(no_keypoints); i++)
		{
			std::getline(infile, line);

		    std::istringstream iss(line);
		    string a, b;
		    Point t;
		    if (!(iss >> a >> b)) { cout << "akash" << endl; break; } // error

		    t.x = std::stoi(a);
		    t.y = std::stoi(b);

		    cout << t << endl;

		    keypoints[i] = t;

		    circle(src, t, 1, (255,0,0), 2);
    		putText(src, to_string(i+1), t, FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, LINE_AA);

		 }

		cv::Point face_rect_min;
		cv::Point face_rect_max;

		face_rect_min.x = 5000;
		face_rect_min.y = 5000;
		face_rect_max.x = 0;
		face_rect_max.y = 0;


		for(int i=0; i<keypoints.size(); i++)
		{
			if(face_rect_max.x < keypoints[i].x)
			face_rect_max.x =  keypoints[i].x;
			
			if(face_rect_max.y < keypoints[i].y)
			face_rect_max.y = keypoints[i].y;
		
			if(face_rect_min.x > keypoints[i].x)
			face_rect_min.x = keypoints[i].x;
			
			if(face_rect_min.y > keypoints[i].y)
			face_rect_min.y = keypoints[i].y;

		}

		//cout << "keypoints \n" << keypoints << endl;

		//cout << face_rect_max.x << endl;//<< " " << face_rect_max.y << " " << src.cols << " " << src.rows << endl;

		cv::Rect face_rect;

		face_rect.x = face_rect_min.x-5;
		face_rect.y = face_rect_min.y-80;

		if(face_rect.x < 0) face_rect.x = 0;
		if(face_rect.y < 0) face_rect.y = 0;

		face_rect_max.x = face_rect_max.x + 10;
		face_rect_max.y = face_rect_max.y + 10;

		if(face_rect_max.x > src.cols) face_rect_max.x = src.cols;
		if(face_rect_max.y > src.rows) face_rect_max.y = src.rows;
 
		face_rect.width = face_rect_max.x - face_rect.x;
		face_rect.height = face_rect_max.y - face_rect.y;

	    rectangle(src, face_rect, (0,255,0), 1);

		imshow("window",src);

		Mat cropped_image_kaggle = original_src(face_rect);
		Mat cropped_image_aflw = original_src(face_rect);


	 	float scale_x_kaggle = float(96)/float(cropped_image_kaggle.cols);
	 	float scale_y_kaggle = float(96)/float(cropped_image_kaggle.rows);

	 	float scale_x_aflw = float(227)/float(cropped_image_aflw.cols);
	 	float scale_y_aflw = float(227)/float(cropped_image_aflw.rows);


	 	resize(cropped_image_kaggle, cropped_image_kaggle, Size(96, 96)); 
	 	resize(cropped_image_aflw, cropped_image_aflw, Size(227, 227));


// 436 -> 43 + 46 / 2
// 374 -> 37 + 40 / 2
// 637 -> 63 + 67 / 2

		for(int i=0; i<aflw_correlation.size();i++) //i<kaggle_correlation.size()
		{
			float temp_var_x = 0;
			float temp_var_y = 0;

			if(aflw_correlation[i] == 436)
			{

				temp_var_x = ((keypoints[42].x + keypoints[45].x)/2 - face_rect.x)*scale_x_aflw;
				temp_var_y = ((keypoints[42].y + keypoints[45].y)/2 - face_rect.y)*scale_y_aflw;

				if(temp_var_x < 0) temp_var_x = 0;
				if(temp_var_y < 0) temp_var_y = 0;

				convert2 << temp_var_x << "," << temp_var_y << ",";

			}
			else if(aflw_correlation[i] == 374)
			{

				temp_var_x = ((keypoints[36].x + keypoints[39].x)/2 - face_rect.x)*scale_x_aflw;
				temp_var_y = ((keypoints[36].y + keypoints[39].y)/2 - face_rect.y)*scale_y_aflw;

				if(temp_var_x < 0) temp_var_x = 0;
				if(temp_var_y < 0) temp_var_y = 0;

				convert2 << temp_var_x << "," << temp_var_y << ",";

			}
			else if(aflw_correlation[i] == 637)
			{

				temp_var_x = ((keypoints[62].x + keypoints[66].x)/2 - face_rect.x)*scale_x_aflw;
				temp_var_y = ((keypoints[62].y + keypoints[66].y)/2 - face_rect.y)*scale_y_aflw;

				if(temp_var_x < 0) temp_var_x = 0;
				if(temp_var_y < 0) temp_var_y = 0;

				convert2 << temp_var_x << "," << temp_var_y << ",";

			}
			else
			{

				temp_var_x = (keypoints[aflw_correlation[i]-1].x - face_rect.x)*scale_x_aflw;
				temp_var_y = (keypoints[aflw_correlation[i]-1].y - face_rect.y)*scale_y_aflw;

				if(temp_var_x < 0) temp_var_x = 0;
				if(temp_var_y < 0) temp_var_y = 0;

				convert2 << temp_var_x << "," << temp_var_y << ",";

			}

			/*Point tt;

			tt.x = temp_var_x;
			tt.y = temp_var_y;

			circle(cropped_image_aflw, tt, 1, (255,0,0), 2);
			*/
			new_name1 = convert2.str().c_str();
			
			outputFile_aflw << new_name1;

			convert2.str("");
			convert2.clear();

		}

		for(int i=0; i<kaggle_correlation.size();i++) //i<kaggle_correlation.size()
		{
			float temp_var_x = 0;
			float temp_var_y = 0;

			if(kaggle_correlation[i] == 436)
			{

				temp_var_x = ((keypoints[42].x + keypoints[45].x)/2 - face_rect.x)*scale_x_kaggle;
				temp_var_y = ((keypoints[42].y + keypoints[45].y)/2 - face_rect.y)*scale_y_kaggle;

				if(temp_var_x < 0) temp_var_x = 0;
				if(temp_var_y < 0) temp_var_y = 0;

				convert2 << temp_var_x << "," << temp_var_y << ",";

			}
			else if(kaggle_correlation[i] == 374)
			{

				temp_var_x = ((keypoints[36].x + keypoints[39].x)/2 - face_rect.x)*scale_x_kaggle;
				temp_var_y = ((keypoints[36].y + keypoints[39].y)/2 - face_rect.y)*scale_y_kaggle;

				if(temp_var_x < 0) temp_var_x = 0;
				if(temp_var_y < 0) temp_var_y = 0;

				convert2 << temp_var_x << "," << temp_var_y << ",";

			}
			else if(kaggle_correlation[i] == 637)
			{

				temp_var_x = ((keypoints[62].x + keypoints[66].x)/2 - face_rect.x)*scale_x_kaggle;
				temp_var_y = ((keypoints[62].y + keypoints[66].y)/2 - face_rect.y)*scale_y_kaggle;

				if(temp_var_x < 0) temp_var_x = 0;
				if(temp_var_y < 0) temp_var_y = 0;

				convert2 << temp_var_x << "," << temp_var_y << ",";

			}
			else
			{

				temp_var_x = (keypoints[kaggle_correlation[i]-1].x - face_rect.x)*scale_x_kaggle;
				temp_var_y = (keypoints[kaggle_correlation[i]-1].y - face_rect.y)*scale_y_kaggle;

				if(temp_var_x < 0) temp_var_x = 0;
				if(temp_var_y < 0) temp_var_y = 0;

				convert2 << temp_var_x << "," << temp_var_y << ",";

			}

			/*Point tt;

			tt.x = temp_var_x;
			tt.y = temp_var_y;

			circle(cropped_image_kaggle, tt, 1, (255,0,0), 2);
			*/
			new_name1 = convert2.str().c_str();

			outputFile_kaggle << new_name1;

			convert2.str("");
			convert2.clear();

		}

		convert << "/home/sprva/AAkash/300wpatch_kaggle/" << dirp->d_name;
		new_name2 = convert.str().c_str();
		string temp = dirp->d_name;
		outputFile_kaggle << temp.c_str() << endl;
		imwrite(new_name2,cropped_image_kaggle);
		convert.str("");
		convert.clear();		

		convert << "/home/sprva/AAkash/300wpatch_aflw/" << dirp->d_name;
		new_name2 = convert.str().c_str();
		temp = dirp->d_name;
		outputFile_aflw << temp.c_str() << endl;
		imwrite(new_name2,cropped_image_aflw);
		convert.str("");
		convert.clear();		

		//outputFile.close();

		//imshow("window1",cropped_image_kaggle);
		//imshow("window2",cropped_image_aflw);

		//imwrite("testing_1.jpg", cropped_image);
	    waitKey(0);

		infile.close();

		}

    }

    dirp = readdir( dp );
  
    if(!dirp)
    {

	   cout << "dir to get files of: " << flush;

	   getline( cin, dir );  // gets everything the user ENTERs

	   if(dir.c_str() == "quit")
	   break;

	   dp = opendir( dir.c_str() );
	   
	   if (dp == NULL)
	   {
	    cout << "Error opening " << dir << endl;
	    outputFile_kaggle.close();
	    outputFile_aflw.close();

	    return 0;
	   }

	   dirp = readdir( dp );
  
    }

 }

  closedir( dp );
  outputFile_kaggle.close();
  outputFile_aflw.close();

  return 0;

}
