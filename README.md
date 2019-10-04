# Realtime Face Identification

As the name suggests this project recognizes faces from a webcam feed. This project is part of a much bigger project to implement a robust face recognition based hassle free entry and exit system.

## Getting Started

Follow the instructions below to setup and run this project on your system.

### Prerequisites

This project needs the following packages to be installed to run.

```
1) scipy
2) Flask_SQLAlchemy
3) numpy
4) Flask
5) tensorflow
6) six
7) imutils
8) Pillow
9) imageio
10) scikit_learn
```

### Installing and Running

I strongly suggest to make a virtual environment and then run this project. I've provided the simplest instructions for getting the project up and running on a linux or windows machine.

1. Make sure you have git installed. Go to [https://git-scm.com/](https://git-scm.com/) to install it.
2. Open the terminal.
3. Type in "git clone https://github.com/dotslash21/realtime_face_identification.git" to clone this repository.
4. Type in "pip install -r requirements.txt" to install the required packages.
5. Put your labelled images in img_db folder.
6. Run the align_face.py file using "python align_face.py" command.
7. Run the train_classifier.py using "python train_classifier.py" command.
8. Finally run the app.py with the required arguments. (e.g.- python app.py -i localhost -o 8080)

## Authors

- **Arunangshu Biswas**

## Contributors

- **Arunangshu Biswas**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

- David Sandberg for his [FaceNet](https://github.com/davidsandberg/facenet) implementation.
