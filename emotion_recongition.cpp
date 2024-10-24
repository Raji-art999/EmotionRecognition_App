#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <dirent.h>

void writeToCSV(const std::string &filePath, const std::string &emotion, const std::string &imagePath) {
    std::ofstream file(filePath, std::ios::app); // Open in append mode
    if (file.is_open()) {
        file << imagePath << "," << emotion << "\n"; // Write image path and predicted emotion
        file.close();
    } else {
        std::cerr << "Could not open CSV file for writing." << std::endl;
    }
}

// Function to test sample images in a given folder
void testSampleImages(cv::dnn::Net &net, cv::CascadeClassifier &face_cascade, const std::string &folderPath) {
    // Ensure the output CSV file is created
    std::ofstream file("predictions.csv");
    file << "Image Path,Predicted Emotion\n";  // Write header
    file.close();

    DIR *dir;
    struct dirent *entry;

    if ((dir = opendir(folderPath.c_str())) != nullptr) {
        while ((entry = readdir(dir)) != nullptr) {
            // Skip the "." and ".." entries
            if (entry->d_name[0] == '.') {
                continue;
            }

            std::string filePath = folderPath + "/" + entry->d_name;

            // Load the image
            cv::Mat image = cv::imread(filePath);
            if (image.empty()) {
                std::cerr << "Could not open or find the image: " << filePath << std::endl;
                continue;
            }

            std::vector<cv::Rect> faces;
            cv::Mat grayImage;
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

            // Detect faces
            face_cascade.detectMultiScale(grayImage, faces, 1.1, 3, 0, cv::Size(30, 30));

            for (const auto &face : faces) {
                cv::Mat faceROI = grayImage(face);

                // Preprocess the face image for the DNN model
                cv::Mat blob = cv::dnn::blobFromImage(faceROI, 1.0 / 255.0, cv::Size(48, 48), cv::Scalar(0, 0, 0), true, false);
                net.setInput(blob);

                // Predict emotion
                cv::Mat prob = net.forward();
                double maxVal;
                cv::Point maxLoc;
                cv::minMaxLoc(prob, 0, &maxVal, 0, &maxLoc);
                int emotion = maxLoc.x;

                // Debugging output to verify predictions
                std::cout << "Predicted probabilities: ";
                for (int i = 0; i < prob.cols; i++) {
                    std::cout << prob.at<float>(0, i) << " ";
                }
                std::cout << " | Predicted emotion index: " << emotion << std::endl;

                // Draw bounding box and emotion label
                cv::rectangle(image, face, cv::Scalar(255, 0, 0), 2);
                std::string emotionText;

                // Update this switch statement according to your model's output mapping
                switch (emotion) {
                    case 0: emotionText = "Angry"; break;
                    case 1: emotionText = "Disgust"; break;
                    case 2: emotionText = "Fear"; break;
                    case 3: emotionText = "Happy"; break;
                    case 4: emotionText = "Sad"; break;
                    case 5: emotionText = "Surprise"; break;
                    case 6: emotionText = "Neutral"; break;
                    default: emotionText = "Unknown"; break;
                }
                cv::putText(image, emotionText, face.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);

                // Save the predicted emotion and image path to CSV
                writeToCSV("predictions.csv", emotionText, filePath);
            }

            // Display the image with detected emotions
            cv::imshow("Emotion Detection - Sample", image);
            cv::waitKey(0);  // Wait for a key press before moving to the next image
        }
        closedir(dir);
    } else {
        std::cerr << "Could not open directory: " << folderPath << std::endl;
    }
}

int main() {
    // Load the pre-trained model
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow("/Users/kellerrajashree/Desktop/OpenCV_C++/tensorflow_model.pb");

    // Load the face detection model
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("/Users/kellerrajashree/Desktop/OpenCV_C++/haarcascade_frontalface_alt2.xml")) {
        std::cerr << "Error loading face cascade." << std::endl;
        return -1;
    }

    // Specify the folder path containing images
    std::string folderPath = "/Users/kellerrajashree/Desktop/OpenCV_C++/sample_images"; // Update this path accordingly

    // Test sample images
    testSampleImages(net, face_cascade, folderPath);

    return 0;
}
