#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

int main() {
    // Load the cascade classifier
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("/opt/homebrew/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading face cascade\n";
        return -1;
    }

    // Open a video capture from webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream\n";
        return -1;
    }

    cv::Mat frame;
    while (true) {
        // Capture frame-by-frame
        cap >> frame;
        if (frame.empty()) break;

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect faces
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

        // Draw rectangles around detected faces
        for (size_t i = 0; i < faces.size(); i++) {
            cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 0), 2);
        }

        // Display the resulting frame
        cv::imshow("Face Detection", frame);

        // Break the loop on 'q' key press
        if (cv::waitKey(30) >= 0) break;
    }

    // Release the video capture object
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
