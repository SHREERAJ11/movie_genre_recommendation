# Movie Genre Classification
<h2> Movie_Genre_Recommendation </h2>


This project involves building a movie genre classification model using machine learning techniques. The model is trained on a dataset containing movie plot summaries and their corresponding genres. After training, the model is used to predict the genres of unseen movie plot summaries.

<h3> Table of Contents </h3>


    -Description
    -Usage
    -Features
    -Example Workflow
    -Evaluation
    -Hyperparameter Tuning
    -Exporting Predictions
    -Requirements
    -Contributing
    -Acknowledgments

<h3> Description </h3>


The code provided implements a movie genre classification system using the following components:

    -Data Loading: Functions to load both the training and test datasets.
    -Text Preprocessing: Preprocessing steps such as tokenization, stop word removal, and TF-IDF vectorization of text data.
    -Model Training: Training a multi-label classification model using Linear Support Vector Classification (LinearSVC) wrapped in a OneVsRestClassifier.
    -Prediction: Making predictions on unseen movie plot summaries.
    -Evaluation: Evaluating the model's performance using accuracy metrics.
    -Hyperparameter Tuning: Performing grid search to find the best hyperparameters for the model.
    -Exporting Predictions: Exporting the predictions to a CSV file for further analysis.

<h3> Usage </h3>


    -Ensure you have Python installed on your system.
    -Clone the repository containing the code.
    -Place your training and test datasets in the repository directory.
    -Install the required Python packages listed in the requirements.txt file.
    -Update the file paths in the code to point to your datasets.
    -Run the code in a Python environment.

<h3> Features </h3>


    -TF-IDF Vectorization: Text data is converted into numerical vectors using TF-IDF vectorization.
    -Multi-label Classification: The model is capable of assigning multiple genres to a single movie plot summary.
    -Hyperparameter Tuning: Grid search is performed to find the best hyperparameters for the model.
    -Exporting Predictions: Predictions can be exported to a CSV file for further analysis.

<h3> Example Workflow </h3>


    -Load the training data containing movie plot summaries and their genres.
    -Preprocess the training data by extracting text and labels, and perform TF-IDF vectorization.
    -Train the model using the preprocessed training data.
    -Load the test data containing movie plot summaries.
    -Preprocess the test data and perform TF-IDF vectorization.
    -Make predictions on the test data using the trained model.
    -Evaluate the model's performance using accuracy metrics.
    -Perform hyperparameter tuning using grid search.
    -Export the predictions to a CSV file for further analysis.

<h3> Evaluation </h3>


The model's performance is evaluated using accuracy metrics. The accuracy score indicates the proportion of correctly predicted labels out of the total number of labels.
Hyperparameter Tuning

Grid search is performed to find the best hyperparameters for the model. The hyperparameters grid includes the regularization parameter C for the LinearSVC classifier.
Exporting Predictions

The model predictions are exported to a CSV file named predictions.csv. Each row in the CSV file contains the movie ID and the predicted genres for that movie.

<h3> Requirements </h3>


    -Python 3.x
    -scikit-learn
    -numpy
    -pandas

You can install the required packages via pip using the following command:

pip install -r requirements.txt

<h3> Contributing </h3>


Contributions are welcome! Please feel free to open an issue or submit a pull request.

<h3> Acknowledgments </h3>


    -This project was inspired by the need for efficient genre classification in the context of movie recommendation systems.
    -Special thanks to the scikit-learn library for providing powerful machine learning tools.
