#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <thread>
#include <mutex>

// Global mutex for thread safety
std::mutex mtx;

// Global stop flag to end threads gracefully
bool stopFlag = false;

// Global frame variable for UI thread to display
cv::Mat globalFrame;

void captureVideo(cv::VideoCapture &cap) {
    while (!stopFlag) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            stopFlag = true;
            break;
        }

        // Lock and update the global frame
        {
            std::lock_guard<std::mutex> lock(mtx);
            globalFrame = frame.clone(); // Clone the frame for display in UI thread
        }

        // Simulate a short delay
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}

void detectFaceAndPredictEmotion(cv::dnn::Net &net, cv::CascadeClassifier &face_cascade) {
    while (!stopFlag) {
        cv::Mat localFrame;

        // Get the current frame from the main thread
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (globalFrame.empty()) continue;  // Skip if no frame is available
            localFrame = globalFrame.clone();
        }

        std::vector<cv::Rect> faces;
        cv::Mat grayFrame;
        cv::cvtColor(localFrame, grayFrame, cv::COLOR_BGR2GRAY);

        // Detect faces
        face_cascade.detectMultiScale(grayFrame, faces, 1.1, 3, 0, cv::Size(30, 30));

        for (const auto &face : faces) {
            cv::Mat faceROI = grayFrame(face);

            // Preprocess the face image for the DNN model
            cv::Mat blob = cv::dnn::blobFromImage(faceROI, 1.0, cv::Size(48, 48), cv::Scalar(0, 0, 0), false, false);
            net.setInput(blob);

            // Predict emotion
            cv::Mat prob = net.forward();
            double maxVal;
            cv::Point maxLoc;
            cv::minMaxLoc(prob, 0, &maxVal, 0, &maxLoc);
            int emotion = maxLoc.x;

            // Draw bounding box and emotion label
            cv::rectangle(localFrame, face, cv::Scalar(255, 0, 0), 2);
            std::string emotionText;
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
            cv::putText(localFrame, emotionText, face.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }

        // Update the frame to display on the UI thread
        {
            std::lock_guard<std::mutex> lock(mtx);
            globalFrame = localFrame.clone();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30));  // Add a delay for smoother execution
    }
}

int main() {
    // Load the face detection model
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("/Users/kellerrajashree/Desktop/OpenCV_C++/haarcascade_frontalface_alt2.xml")) {
        std::cerr << "Error loading face detection model!" << std::endl;
        return -1;
    }

    // Load pre-trained emotion recognition model
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow("/Users/kellerrajashree/Desktop/OpenCV_C++/tensorflow_model.pb");
    if (net.empty()) {
        std::cerr << "Error loading emotion recognition model!" << std::endl;
        return -1;
    }

    // Open the webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening webcam!" << std::endl;
        return -1;
    }

    // Start threads for video capture and emotion detection
    std::thread videoThread(captureVideo, std::ref(cap));
    std::thread emotionThread(detectFaceAndPredictEmotion, std::ref(net), std::ref(face_cascade));

    // Main thread loop for displaying video
    while (!stopFlag) {
        cv::Mat frame;

        // Get the latest frame for display
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (globalFrame.empty()) continue;
            frame = globalFrame.clone();
        }

        // Show the frame in the UI (must be on the main thread)
        cv::imshow("Emotion Recognition", frame);
        if (cv::waitKey(1) >= 0) {
            stopFlag = true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // Wait for threads to complete
    videoThread.join();
    emotionThread.join();

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
