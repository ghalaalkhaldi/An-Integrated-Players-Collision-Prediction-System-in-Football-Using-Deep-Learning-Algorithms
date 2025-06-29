# An Integrated Players' Collision Prediction System in Football Using Deep Learning Algorithms

## Project Overview

This project introduces an efficient and effective system for predicting player-to-player collisions in football using advanced deep learning algorithms. Football, despite its global popularity, carries a high incidence of injuries, with collisions being a common cause. This system aims to mitigate these risks by providing real-time collision predictions and extracting crucial match data to enhance player safety, optimize game strategies, and support various stakeholders in the sport.

The system utilizes the YOLOv8 framework for robust player detection and integrates a custom algorithm for collision prediction. It also features a web-based interface for user interaction and data visualization.

## Features

* **Real-time Player Detection:** Employs YOLOv8, a state-of-the-art deep learning model, for highly accurate identification and tracking of football players from top-view video footage.
* **Integrated Collision Prediction:** A custom algorithm analyzes player positions and movement patterns, calculating Euclidean distances and overlap ratios to classify situations as 'safe,' 'close proximity,' or 'collision.'
* **Web-Based System:** Provides a user-friendly interface (developed with HTML, CSS, JavaScript, and Flask) for uploading videos, setting prediction parameters (confidence and distance thresholds), and visualizing results.
* **Data Extraction & Visualization:** Generates detailed Excel reports containing frame-by-frame status, timestamps, and distances, along with pie charts summarizing collision statistics (safe, close proximity, collision percentages).
* **Injury Risk Mitigation:** Aims to reduce player absences and medical expenses by enabling proactive identification of high-risk situations.
* **Strategic Game Analysis:** Offers insights into player occurrences and patterns, aiding coaches in developing optimized game-play strategies.


## Installation and Setup

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ghalaalkhaldi/An-Integrated-Players-Collision-Prediction-System-in-Football-Using-Deep-Learning-Algorithms.git](https://github.com/ghalaalkhaldi/An-Integrated-Players-Collision-Prediction-System-in-Football-Using-Deep-Learning-Algorithms.git)
    cd An-Integrated-Players-Collision-Prediction-System-in-Football-Using-Deep-Learning-Algorithms
    ```

2.  **Create and activate a virtual environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv-env
    source venv-env/bin/activate  # On macOS/Linux
    # venv-env\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    The project relies on Python (version 3.11 or later is preferred) and several libraries.
    ```bash
    pip install -r src/requirements.txt
    ```
    *Note: Ensure your `requirements.txt` inside the `src` folder lists all necessary Python packages (e.g., Flask, OpenCV, Pandas, Ultralytics, Openpyxl, Matplotlib, Pillow, Werkzeug, Flask-CORS, Flask-SocketIO).*

4.  **Model Availability:**
    The pre-trained YOLOv8 model (`best.pt`) is expected to be located in `src/static/model/`. Ensure this file is present or follow any specific instructions within `instructions.txt` if it needs to be downloaded separately.

## Usage

To run the web-based collision prediction system:

1.  **Start the Flask application:**
    Ensure your virtual environment is active, then navigate to the root of your project and run:
    ```bash
    python src/app.py
    ```
    This will start the local web server.

2.  **Access the web interface:**
    Open your web browser and navigate to the address displayed in your terminal (typically `http://127.0.0.1:5000` or similar).

3.  **Predict Collisions:**
    * On the web interface, you can set the "Confidence Threshold" and "Distance Threshold" parameters according to your desired sensitivity.
    * Upload a top-view football match video.
    * Click "Submit" to initiate the analysis.
    * The system will process the video, display a loading indicator, and then provide a downloadable ZIP file containing:
        * The processed video with player detections and collision predictions annotated.
        * An Excel report with frame-by-frame collision status, timestamps, and distances.
        * A pie chart visualizing the percentages of 'safe', 'close proximity', and 'collision' statuses in the video.

## Deep Learning Models & Methodology

* **Player Detection:** YOLOv8 was selected for its superior performance in precision (93.4%), recall (95.3%), and mAP (96.9%) compared to YOLOv7 and YOLOv9, as determined by comprehensive comparative analysis.
* **Collision Prediction:** An integrated algorithm is used, treating collision prediction as a ternary classification problem (safe, close proximity, collision). It calculates Euclidean distance and bounding box overlap ratio between players to determine the status. The optimal overlap range for collision detection was determined to be 0.3 to 0.9 after extensive testing.
* **Development Platforms:** Roboflow was used for dataset management, object annotation, data augmentation, and pre-processing. Google Colab was utilized for model training and implementation of the integrated prediction system.
* **Dataset:** Meticulously gathered from various online sources, focusing exclusively on high-quality, top-down football videos. Annotated using Roboflow and augmented to increase diversity (from 2408 to 4733 images).
* **Backend Technologies:** Python 3.11+, Flask framework, OpenCV, Pandas, Ultralytics.
* **Frontend Technologies:** HTML, CSS, JavaScript, Bootstrap. Flask-SocketIO enables real-time communication.

## Results and Evaluation

The integrated algorithm with YOLOv8 demonstrated satisfactory performance in collision prediction across eight test videos (over 900 frames) with an average **accuracy of 88%** and a **recall of 88%**. The F1-score was 71% and precision was 63%. This indicates its strong ability to identify actual collisions, though there is potential for further refinement to reduce false positives. The system shows higher accuracy than some previous approaches in head collision prediction.

## Future Work

* Integrating the system with more detailed data extraction about players and matches.
* Developing a real-time dataset for continuous improvement.
* Creating technology for warning players on the field before collisions occur (e.g., haptic feedback devices).
* Enhancing player identification for multi-target tracking.
* Further optimizing the collision prediction algorithm to minimize false positives and improve overall precision.

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Acknowledgements

* **Supervisors:** Dr. Huda D. Althumali and Dr. Asma A. Alshammari (Computer Science Department, College of Science and Humanities in Jubail).
* **Team:** Reem S. Almualem, Warood K. Alzayer, Ghala M. Alkhaldi, Fida M. Alelou, May M. Alotaibi.
* Special thanks to the open-source community and tools like Roboflow, Google Colab, YOLO, OpenCV, and Flask that were instrumental in this project's development.

---

**Contact:**
Ghala M. Alkhaldi - ghalaalkhaldi10@gmail.com
