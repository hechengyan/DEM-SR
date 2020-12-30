#include "Image_to_BinaryData.h"
#include <atlfile.h>
#include <cstdlib>
#include <fstream>
using namespace cv;

void Image2BinaryData::ReverseInt(ofstream &file, int i)
{
	uchar ch1 = (uchar)(i >> 24);
	uchar ch2 = (uchar)((i << 8) >> 24);
	uchar ch3 = (uchar)((i << 16) >> 24);
	uchar ch4 = (uchar)((i << 24) >> 24);
	file.write((char*)&ch1, sizeof(ch4));
	file.write((char*)&ch2, sizeof(ch3));
	file.write((char*)&ch3, sizeof(ch2));
	file.write((char*)&ch4, sizeof(ch1));
}

vector<string> Image2BinaryData::getFileLists(string file_folder)
{
	/*
	param
	file_folder:		fileroot
	*/
	file_folder += "/*.*";
	const char *mystr = file_folder.c_str();
	vector<string> flist;
	string lineStr ="0";
	vector<string> extendName;			//file type
	extendName.push_back("JPG");
	extendName.push_back("jpg");
	extendName.push_back("bmp");
	extendName.push_back("png");
	extendName.push_back("tiff");

	HANDLE file;
	WIN32_FIND_DATA fileData;
	char line[1024] = "hello";
	wchar_t fn[1000];
	mbstowcs(fn, mystr, 999);
	int nChar = WideCharToMultiByte(CP_ACP, 0, fn, -1, NULL, 0, NULL, NULL);
	nChar = nChar * sizeof(char);
	char * outPara = new char[nChar];
	ZeroMemory(outPara, nChar);
	WideCharToMultiByte(CP_ACP, 0, fn, -1, outPara, nChar, NULL, NULL);
	file = FindFirstFile(outPara, &fileData);
	FindNextFile(file, &fileData);
	while (FindNextFile(file, &fileData))
	{
		//wcstombs(line, (const wchar_t*)fileData.cFileName, 259);
		//wcstombs(line, (const wchar_t*)fileData.cFileName, sizeof(fileData.cFileName));
		strcpy(line, fileData.cFileName);
		lineStr = line;
		// remove the files which are not images
		for (int i = 0; i < 4; i++)
		{
			if (lineStr.find(extendName[i]) < 999)
			{
				flist.push_back(lineStr);
				break;
			}
		}
	}
	return flist;
}

/*
get file name
for example: from"C:\\Users\\lyf\\Desktop\\test.txt"£¬return"test.txt"
*/
string Image2BinaryData::getFileName(string & filename)
{
	/*
	param
	filename:		
	*/
	int iBeginIndex = filename.find_last_of("\\") + 1;
	int iEndIndex = filename.length();

	return filename.substr(iBeginIndex, iEndIndex - iBeginIndex);

	cout << "Done!" << endl;
}

void Image2BinaryData::ReadImage(string filefolder, vector<string>& img_Lists, vector<int>& img_Labels, vector<cv::Mat> &ImagesMat)
{
	const int size_list = img_Lists.size();
	//const int size_list = 10;
	for (int i = 0; i < size_list; ++i) {
		string curPath = filefolder + "\\" + img_Lists[i];
		Mat image = imread(curPath, IMREAD_UNCHANGED);
		ImagesMat.push_back(image);
		char label = img_Lists[i][0];
		img_Labels[i] = label - '0';
		printf("waiting: %.2lf%%\r", i*100.0 / (size_list - 1));
	}
	printf("\nfinished!\n\n");
}

void Image2BinaryData::Image2BinaryFile(string filefolder, vector<cv::Mat>& ImagesMat, vector<int>& img_Labels)
{
	ofstream file(filefolder, ios::binary);
	int idx = filefolder.find_last_of("\\") + 1;
	string subName = filefolder.substr(idx);
	if (file.is_open()) {
		cout << subName << "file created." << endl;

		int magic_number = 2051;
		int number_of_images = img_Labels.size();
		int n_rows = Height;
		int n_cols = Width;
		ReverseInt(file, magic_number);
		ReverseInt(file, number_of_images);
		ReverseInt(file, n_rows);
		ReverseInt(file, n_cols);

		cout << "The number of the pictures:  " << ImagesMat.size() << endl;
		for (int i = 0; i < ImagesMat.size(); ++i) {
			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					uchar tmp = ImagesMat[i].at<uchar>(r, c);
					file.write((char*)&tmp, sizeof(tmp));
				}
			}
			printf("waiting......%.2lf%%\r", i*100.0 / (ImagesMat.size() - 1));
		}
		printf("\n******finished!******\n\n");
	}
	else
		cout << subName << "failed." << endl << endl;
	if (file.is_open())
		file.close();
	return;
}

void Image2BinaryData::Label2BinaryFile(string filefolder, vector<int>& img_Labels)
{
	ofstream file(filefolder, ios::binary);
	int idx = filefolder.find_last_of("\\") + 1;
	string subName = filefolder.substr(idx);
	if (file.is_open()) {
		cout << subName << "finished." << endl;

		int magic_number = 2049;
		int number_of_images = img_Labels.size();
		ReverseInt(file, magic_number);
		ReverseInt(file, number_of_images);
		cout << "The number of the labels: " << img_Labels.size() << endl;
		for (int i = 0; i < img_Labels.size(); ++i) {
			uchar tmp = (uchar)img_Labels[i];
			file.write((char*)&tmp, sizeof(tmp));
			printf("waiting......%.2lf%%\r", i*100.0 / (img_Labels.size() - 1));
		}
		printf("\n******finished!******\n");
	}
	else
		cout << subName << "failed." << endl;
	if (file.is_open())
		file.close();
	return;
}

