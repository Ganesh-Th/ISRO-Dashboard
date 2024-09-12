# ISRO-Daashboard

This project is designed to detect defects in images using a pre-trained machine learning model. The application is built using Streamlit for the user interface and various other Python libraries for image processing and machine learning.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Ganesh-Th/ISRO-Dashboard.git
    cd ISRO-Dashboard
    ```

2. Create a virtual environment:
    ```sh
    python -m venv env
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run main.py
    ```

2. Upload an image using the file uploader in the Streamlit interface.

3. Click the "Detect Defects" button to detect defects in the uploaded image.

4. Use the "Accept" or "Reject" buttons to manage the detected defects.

## Project Structure

- **ISRO-Dashboard/**
  - **Dataset/**              # Dataset
    - `test/`
    - `train/`
    - `valid/`
    - `data.yaml`
    - `README.dataset`
    - `README.roboflow`
  - `bestlopandporosity.pt`  # Pre-trained model file
  - `main.py`                # Main file (v1)
  - `final.py`               # Main file (v2)
  - `requirements.txt`       # List of dependencies
  - `README.md`              # Project README file



## Dependencies

- streamlit
- mysql-connector-python
- pandas
- Pillow
- numpy
- opencv-python
- scikit-image
- scipy
- ultralytics
- streamlit-cropper

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
