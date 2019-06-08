
Project Structure: 
	devbmaclap:ImageProcessing devb$ tree -L 2
	.
	├── Analysis.py
	├── AugmentationTechniques.py
	├── CNN.py
	├── Dockerfile
	├── ImageVisualization.py
	├── Image_Processing_Pipline.ipynb
	├── Prediction.py
	├── data
	│   ├── alfalfa
	│   ├── barley
	│   ├── corn
	│   ├── soybean
	│   └── wheat
	├── requirements.txt
	└── SplitDatasets.py
	To run the ImageProcessing Pipeline. Please follow the following process:

Normal Way:

Requirement:
	1. unzip the project zip file 
	2. Open terminal or power shell and open the location of unzipped folder.
	3. Install latest python (official)from the (https://www.python.org/) , or install anaconda that provides python (from https://docs.anaconda.com/anaconda/install/)
	4. Using the python executable  install pip (https://www.makeuseof.com/tag/install-pip-for-python/)
	5. with python3 pip install the dependencies listed in requirements file of project folder
	 pip3 install -r requirements.txt

	6.Open Jupyter typing command in command line as below:
		jupyter notebook

	7. Execute the jupyter cells one after another.

Another way:
	1. After completion of step 3 , execute python file run file in a sequence as below:
		a. python3 SplitDatasets.py 
			Enter the path for data (./data)

		b. python3 ImageVisualization.py
			Visualize the input images

		c. python3 AugmentationTechniques.py                     
			Apply AugmentationTechiues from ImageDataGenerator

		d. python3 TrainModel.py.   
			See the result generated stored in Loss_Graph.png for training and validation Loss graph ROC and Confusion_matrix.png for result evaluation.

		f. python3 Prediction.py
			Predict the result for test datasets . open file results.txt for detail


Docker way: This is not working for me however it might work for other. I am able to create the docuker and install all dependencies but  could not get result back to the local computer . I could have resolved this issue but with the limitation of time I am not able to solve it completely. The idea
			is to demonstrate the effectiveness of pipeline with docker.

	Requirement:
			Docker: Download and Install docker from https://www.docker.com/ and register in docker hub

		Steps:
			1. Create DockerImage with command
				docker build -t "imageprocessing:docker" .
			2. 
			docker run -v . -it imageprocessing:docker python3 /ImageProcessing/SplitDatasets.py
				Enter the path for data (./data)
			docker run -v . -it imageprocessing:docker python3 /ImageProcessing/AugmentationTechniques.py

			docker run -v . -it imageprocessing:docker python3 /ImageProcessing/TrainModel.py
			
			docker run -v . -it imageprocessing:docker python3 /ImageProcessing/Prediction.py


//Improvement could be by taking input from CLI for number_per_batches and epoches and creating a log using logger.



Architecture:

EnvironmentSetUp->Data-> DataVisualization-> DataPreprocessing-> DataAugmentation-> Feature Selection->Model Creation (CNN) -> Model Evaluation-> Prediction

Note after the model evaluation we can again evaluate the result and repeat the previous process to get better model with more accuracy with less bias.


Process Explaination:

	1. Environment setup with python and related dependencies. Docker could be used for that
	2. Splitting the data into training, validation and test for NN ratio (.8,.1,.1)
	3. Quick data visualization of training data
	4. Used keras ImageDataGenerator to perform data augmentation such as (randomly rotating the image, zooming, etc.)
	5. # Setup our Convolutional Neural Network (CNN) with following global architecture:
		a) 4 convolutional layers : will extract relevant features from the images 
		b) 2 fully connected layer: use these features to classify
	6. save CNN model structure to JSON 
	7. Analyzed output with Visualize the the training and validation loss
	8. Model analysis with confusion matrix
	8. Prediction using test dataset 






