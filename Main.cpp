#include "Image_to_BinaryData.h"

int main()
{
	Image2BinaryData IBD(140,140);											//Figure size(Height,Width)
	
	/*------------create training data--------------------------*/
	printf("----------create training data-------------\n");
	string trainfilefolder = "C:\\Users\\zrcbaby\\Desktop\\picture2mnist\\train";		//training data root
	vector<string> trainfileLists = IBD.getFileLists(trainfilefolder);				//get file name list
	const int train_size_list = trainfileLists.size();
	cout << "Images Number: " << train_size_list << endl;							//output the number of the figures

	string trainimagebinfilepath = "C:\\Users\\zrcbaby\\Desktop\\train-images-idx3-ubyte";		//training data saving root
	string trainlabelbinfilepath = "C:\\Users\\zrcbaby\\Desktop\\train-labels-idx1-ubyte";		//training label saving root
	vector<cv::Mat> TrainImagesMat;															//saving training data pixel
	vector<int> train_image_labels(train_size_list);										//saving training labels
	IBD.ReadImage(trainfilefolder, trainfileLists, train_image_labels, TrainImagesMat);		//reading training data
	IBD.Image2BinaryFile(trainimagebinfilepath, TrainImagesMat, train_image_labels);		//training data to binary data
	IBD.Label2BinaryFile(trainlabelbinfilepath, train_image_labels);						//training label to binary label

	/*------------create testing data--------------------------*/
	printf("\n\n----------create testing data-------------\n");
	string testfilefolder = "C:\\Users\\zrcbaby\\Desktop\\picture2mnist\\test";		//testing data root
	vector<string> testfileLists = IBD.getFileLists(testfilefolder);			//get file name list
	
	const int test_size_list = testfileLists.size();
	cout << "Images Number: " << test_size_list << endl;									//output the number of the figures
	string testimagebinfilepath = "C:\\Users\\zrcbaby\\Desktop\\t10k-images-idx3-ubyte";		//testing data saving root
	string testlabelbinfilepath  = "C:\\Users\\zrcbaby\\Desktop\\t10k-labels-idx1-ubyte";		//testing label saving root
	vector<cv::Mat> TestImagesMat;															//saving testing data pixel
	vector<int> test_image_labels(test_size_list);											//saving testing labels
	IBD.ReadImage(testfilefolder, testfileLists, test_image_labels, TestImagesMat);			//reading testing data
	IBD.Image2BinaryFile(testimagebinfilepath, TestImagesMat, test_image_labels);			//testing data to binary data
	IBD.Label2BinaryFile(testlabelbinfilepath, test_image_labels);							//testing label to binary label

	return 0;
}