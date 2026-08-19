# Skin Disease Image Classification

An educational computer-vision project that classifies dermatology images into six categories using a ResNet50 transfer-learning model. The accompanying Flask application accepts an image, returns class probabilities, and produces a Grad-CAM visualisation to show the image regions that most influenced the model.

> **Important:** This is an educational portfolio project. It is **not a medical device**, does not provide medical advice, and must not be used to diagnose or treat a condition. Consult a qualified healthcare professional for medical concerns.

## Highlights

- Compared multiple transfer-learning architectures and selected **ResNet50** as the final model.
- Six image classes: Infection, Eczema, Acne, Pigment, Benign, and Malign.
- Class weighting to reduce the effect of an imbalanced dataset.
- Early stopping and model checkpointing based on validation accuracy.
- Evaluation through classification reports, confusion matrices, accuracy, precision, recall, and F1 score.
- A Flask interface with authentication, per-user upload history, prediction probabilities, and Grad-CAM explanations.

The dissertation evaluated several transfer-learning approaches. Reported validation accuracies were approximately 85-89%, with macro F1 scores above 84%. Performance was weaker for the underrepresented pigment class, and benign versus malignant cases remained a key source of confusion.

**Full dissertation:** [Read the dissertation PDF](dissertation/dissertation.pdf)

## Application demo

![Prediction result with class probabilities and Grad-CAM](assets/prediction-result.png)

## Technical approach and challenges

- Evaluated multiple transfer-learning architectures and selected **ResNet50** as the final model based on validation performance, macro F1, per-class precision and recall, confusion-matrix patterns, and suitability for the application. The pretrained backbone was initially frozen, then selectively fine-tuned with a low learning rate.
- Addressed class imbalance with **class weighting** during training. This was important because the benign and malignant classes had substantially more images than the pigment class.
- Evaluated models beyond validation accuracy using per-class precision, recall, F1 score, classification reports, and confusion matrices. **Macro F1** was especially useful because it weights every class equally.
- Used early stopping, learning-rate reduction, and model checkpoints to control overfitting and retain the best validation model.
- Added Grad-CAM visualisations and class probabilities to make each prediction more interpretable for users. These are explanations of model attention, not clinical explanations.
- Identified key limitations: performance varied across classes with visually overlapping characteristics, underrepresented categories received less reliable predictions, and the model has no external clinical validation.

## Dataset and reproducibility

The notebooks download the `ascanipek/skin-diseases` dataset from Kaggle using `kagglehub`. Before redistributing or deploying this work, check the dataset's current licence and usage terms. The dataset itself is not included in this repository.

The notebooks are retained as the original research record. For production-quality retraining, replace the hard-coded Colab paths with configuration variables, pin package versions, fix random seeds, and ensure the preprocessing function matches the chosen backbone (ResNet50).

## Repository layout

```text
app/            Flask application and UI templates
notebooks/      Original training and fine-tuning notebooks
dissertation/   Full dissertation PDF
```
