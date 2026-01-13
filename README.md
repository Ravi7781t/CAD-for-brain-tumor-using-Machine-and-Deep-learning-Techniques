# CAD-for-brain-tumor-using-Machine-and-Deep-learning-Techniques
Computer Aided Diagnosis for brain tumor using machine and deep learning techniques using RESNET50 architecture and CBAM attention models for end-to-end MRI classification

Aim:

  The aim of this project is to design and implement a hybrid computer-aided diagnosis (CAD) system that utilizes both machine learning (ML) and deep learning (DL) techniques to accurately detect and classify brain tumors from MRI scans, ensuring both high performance and explainability.

Introduction:

  The image of magnetic resonance (MRI) has been identified as a critical sphere of medical imaging where the correct and timely detection of brain tumors has a direct impact on the design of treatment plans and survival chances of a patient. The use of conventional diagnostic methods is based on a manual process of MRI scan interpretation by specialized radiologists, which is time-consuming, subjective, and prone to inter-observer interferences. In order to overcome these constraints, the concept of using the techniques of Machine Learning (ML) and Deep Learning (DL) has been actively investigated to automatically detect and classify brain tumors, allowing to provide faster and more consistent diagnostic assistance.
Convolutional Neural Networks (CNNs), which are Deep Learning (DL) models, exhibit high potential to mechanically acquire hierarchical spatial features of medical images, and thus are highly useful in the brain tumor classification task (LeCun et al., 1998). According to recent research, the models based on CNNs have good classification accuracy when they are trained on large and properly annotated MRI datasets (Anantharajan et al., 2024; Farooqui, 2024). Nevertheless, these models usually have significant computational and large amounts of labelled data requirements, which restrict their application in resource-limited clinical settings.
Conversely, the conventional use of the mechanisms of Machine Learning (ML) like Support Vector Machines (SVMs) (Cortes and Vapnik, 1995) and Random Forests (Breiman, 2001) has been extensively applied to the detection of brain tumors on the basis of handcrafted features such as Gray Level Co-occurrence Matrix (GLCM) and Local Binary Pattern (LBP) descriptors. These models are computationally efficient and have been found to be reliable on smaller datasets. However, classical ML methods have shown to be poorer in large and high- complex imaging data sets because they have the low capacity to encode high-level spatial representations compared to deep learning models (Tiwari et al., 2016).
Driven by the balancing benefits and shortcomings of the ML and the DL solutions, the project follows a hybrid modelling approach, which combines both the use of traditional machine learning classifiers with the feature extraction performed by deep learning to extract the information. The proposed solution will utilise the representational strength of CNNs and the strength and explainability of conventional ML models to attain better accuracy, generalisation and reliability of automated brain tumor classification using MRI scans.
Based on all these existing foundations, this project aims at the experimental application and assessment of a hybrid ML-DL CAD pipeline and not by suggesting new learning algorithms

Objectives:

   1.	The aim of this research is to use publicly available MRI brain tumor data to collect and preprocess data needed to train and validate a model.
   2.	To evaluate the performance of traditional machine learning models (SVM, Random Forest) and deep learning models (ResNet50 with CBAM Attention) with the standard evaluation measures (accuracy, precision,       recall, F1-score and AUC).
   Such goals will inform the design and test of a CAD system that will incorporate both ML and DL models to attain high accuracy of classification and transparency through explainability methods.

Research Questions:

   The research questions to be answered in this project include:
   RQ1: Which are the comparisons of hybrid machine learning and deep machine learning models when they are used to test various brain tumor MRI datasets against accuracy and generalization?
   RQ2: Which of the classical machine learning classifiers (SVM, Random Forest) perform as well as the deep learning (ResNet50, with CBAM Attention) when it comes to classifying brain tumors, using it to extract    features of MRI scans?
   RQ3: What is the possibility to quantitatively measure the explainability of deep learning models (with Grad-CAM) and implement it in the clinical context?
   The proposed questions will facilitate the comparison of the effectiveness of ML and DL models and help to solve the problem of the model interpretability that is very essential in clinical adoption.


Significance of the Study:
         This research is important because it can enhance the detection of brain tumors as it is automated, saves time on diagnosing tumors and enhances detection accuracy. Incorporating machine learning and deep learning, the given project will allow obtaining a comprehensive evaluation of the way these models can be used to complement each other in order to obtain a better performance. The hybrid model will make it possible to design a more scalable and generalizable system overcoming the constraints of both the ML and the DL models utilized individually. In addition, the research will present the information on the usage of explainability methods, such as Grad-CAM, that may increase the clinical confidence towards the AI-based diagnostic systems. Such tools will facilitate the elucidation of the decision-making process of the complex models, which is one of the conditions of its acceptance in the medical practice (Farooqui, 2024; Yuan, 2024). The outcomes of the study might help the creation of more effective CAD systems, which would be useful to both healthcare professionals and patients, as they will be able to diagnose tumors earlier and more precisely.

Ethical Consideration:
    Because the present project will involve publicly available MRI datasets, no ethical consolidation is necessary in the data collection. The data sets employed in this project is anonymous and in accordance with the data protection laws, including the General Data Protection Regulation (GDPR). Nonetheless, privacy and security are the concerns of first importance, particularly when it comes to clinical settings where personal information is considered. The project guarantees that the datasets utilized are open and anonymized in order to preserve the privacy of the patients.
The explainability methods such as Grad-CAM will also be incorporated in the project to improve model transparency, which is a prerequisite to the implementation of deep learning models in clinical applications. Such methods will assist in filling the black-box character of deep learning models, which will make them more interpretable and clinically credible (Gadicha et al., 2025). With the emphasis on the transparency and ethical aspects, the project will help create the system that can be safely and efficiently applied in the real-life medical practice.

Summary:
   The chapter has considered the current body of literature on machine learning and deep learning methods in the detection of brain tumors using MRI images. It distinguished the merits and demerits of classical ML models and more recent DL models, and concentrating more on the merits of CNNs and hybrid model to enhance the rates of accuracy and generalization. Besides, the chapter has described the significance of explainability methods such as Grad-CAM and to enhance the transparency of models, yet the clinical use of explainable AI has still been a problem. The important remarks are that deep learning models have better performance than their traditional counterparts of the ML models, and that more studies are required to determine generalization across datasets and provide measures of explainability in clinical settings.

ResNet50_CBAM Model Construction:
   In this implementation, the deep learning model is constructed using a ResNet50 backbone augmented with Convolutional Block Attention Modules (CBAM). The code initialises the backbone either with ImageNet weights or from scratch based on a configuration flag.
The code attaches CBAM layers to the convolutional feature maps produced by the backbone and appends a custom classification head consisting of fully connected layers followed by a softmax output layer with four neurons corresponding to the target classes.
In this implementation, the model architecture is instantiated programmatically and summarised using built-in Keras utilities to verify layer connectivity and parameter counts before training begins.

Deep Learning Trainig Procedure:
   In this implementation, the ResNet50_CBAM model is trained using mini-batch gradient optimisation with categorical cross-entropy loss. The optimiser, learning rate, batch size, and number of epochs are explicitly defined in the training configuration.
The code performs training using the fit() function, supplying the training and validation generators. Training progress is logged per epoch, including loss and accuracy metrics for both training and validation sets. In this implementation, callbacks such as early stopping and model checkpointing are optionally enabled to prevent overfitting and to save the best-performing model weights based on validation accuracy. Training history objects are stored for later visualisation and analysis.

Data and Analysis:
   The primary dataset used in this project is the Brain-Tumor-Classification-DataSet-master collection   hosted   on   Kaggle,   which   provides   MRI   slices   grouped into Training and Testing folders with subdirectories corresponding to tumor classes such as glioma, meningioma, pituitary, and no-tumor. The implementation loads all images from these folders into a unified DataFrame where each row represents one image and stores its file path, class label, data source (Training or Testing), original width, height, and mean intensity, enabling later merging with additional sources such as Figshare if required.
A level one integrity test summarizes the number of images, records the number of files that were not read or corrupt, and finds the number of duplicates using hashes on the raw pixel buffers. Duplicate and corrupted data points are not further analyzed and ensure that the following splits and model assessments are not prejudiced by faulty data or unintentional occurrence of the same MRI slice in specific subsets.

Results and Visualization:
   The experiments evaluate traditional ML models trained on handcrafted features (GLCM, LBP, HOG, intensity statistics) and deep learning models (ResNet50 with CBAM with transfer learning) on the same stratified train–validation–test splits. Evaluation uses accuracy, precision, recall, macro-F1, and macro AUC to reflect performance on all four classes (glioma, meningioma, pituitary, no-tumor) under class imbalance.
The visual analysis combines EDA plots (class distribution, intensity boxplots, correlation heatmaps, t-SNE) with Grad-CAM overlays for qualitative inspection of model focus regions on MRI slices. These visualisations support interpretation of numerical metrics and help link model behaviour to clinical plausibility.

Machine Learning Results:
   Stage-1 models operate on the scaled handcrafted feature matrix saved as Stage1_Features.csv. Using the stratified splits, baseline classifiers such as SVM, Random Forest & kNN are trained with macro-averaged metrics and balanced sampling and tuned via validation performance, then evaluated on the held-out test set.
Table 1

Model / Pipeline	Accuracy	Macro Precision	Macro Recall	Macro F1
SVM (Stage-1, actual)	0.45	0.82	0.45	0.43
Random	Forest	(Stage-1, actual)	0.83	0.84	0.83	0.83
k-NN (Stage-1, actual)	0.87	0.87	0.86	0.86


The table reports Stage-1 classical ML performance on the brain-tumor dataset: SVM performs poorly with 0.45 accuracy and macro-F1 0.43, whereas Random Forest (0.83 accuracy, macro-F1 0.83) and especially k-NN (0.87 accuracy, macro-F1 0.86) achieve substantially stronger and well-balanced precision and recall across classes.
This table establishes a classical machine learning performance baseline, demonstrating the limitations of handcrafted feature models under class imbalance and motivating the transition to deep and hybrid approaches.

Deep Learning Results:
   Transfer-learning models (e.g., ResNet50, Deep feature Ensemble) are trained on augmented 224×224 RGB MRI slices with class-weighted cross-entropy and early stopping. Training and validation curves (accuracy and loss vs epochs) should be plotted to show convergence behaviour and absence of severe overfitting.
Table 2

Model	Accuracy	Macro Precision	Macro Recall	Macro F1
ResNet50 (CBAM)	0.8837	0.8911	0.8935	0.8911
Deep Feature Ensemble	0.92	0.93	0.93	0.93
Line plot of model accuracy versus training epoch for the MRI brain-tumor classifier, showing both training and validation accuracy curves rising from around 0.68–0.70 at epoch 0 to roughly 0.93–0.91 by epoch 9, indicating good learning with limited overfitting. This table demonstrates the substantial performance gains achieved through deep learning and deep feature ensembles, confirming the effectiveness of attention-enhanced transfer learning over traditional handcrafted feature pipelines.

Conclusion:
   This project designed and implemented a hybrid computer-aided diagnosis (CAD) framework for multiclass brain tumor classification from MRI, integrating traditional machine learning, deep learning, and explainable AI components into a single end-to-end pipeline. Harmonised Kaggle and Figshare MRI datasets were preprocessed through resizing, intensity normalisation, and exploratory analyses of class distribution, resolution, and intensity statistics, ensuring reliable inputs for subsequent modelling.
Three main modelling tiers were realised. First, a traditional PCA-based pipeline trained tuned SVM, Random Forest, and k-NN classifiers plus a hard-voting ensemble on grayscale features. Second, an attention-enhanced ResNet50_CBAM network was trained on augmented 224×224 RGB MRIs using Keras generators. Third, a hybrid deep-feature ensemble used embeddings extracted from the ResNet50_CBAM backbone as inputs to SVM/RF/k-NN with soft voting, alongside two novelty modules: test-time augmentation (TTA) uncertainty and Grad-CAM- based explainability.

Key Findings and Contributions:
   The experiments showed that classical models on PCA-reduced features provide a solid baseline but are consistently outperformed by deep and hybrid approaches. SVM suffered from severe recall imbalance across classes, while Random Forest and k-NN delivered more uniform performance, with k-NN often achieving the best single-model results among traditional methods. The PCA-based voting ensemble further improved robustness, confirming that ensembling mitigates individual model weaknesses and is a competitive non-deep baseline.
The ResNet50_CBAM deep model achieved higher test accuracy and macro-averaged F1 scores than all PCA-based pipelines, with confusion matrices showing fewer cross-class confusions, especially between glioma and meningioma. The hybrid deep-feature ensemble, trained on ResNet embeddings, matched or slightly exceeded the pure ResNet in accuracy while clearly surpassing the PCA-ensemble, demonstrating that deep features carry most of the discriminative power and that classical classifiers can effectively exploit these learned representations. This nested tiering itself is a gift bearing a variety of available operating points based on offered compute and deployment restraints. Explainability and reliability were considered as first class objectives. Grad CAM maps revealed that, to make correct predictions, the network attention is generally localized in areas of visible tumors, which is clinically plausible; failure cases tended to have diffuse or misplaced attention, and this insight can be used to identify failure patterns. TTA generated confidence, uncertainty, and entropy statistics on a per-image basis, as well as high uncertainty predictions were found to be disproportionately linked to misclassifications, indicating a handy mechanism of identifying cases that need radiologist attention, and then automated judgments can be made. Combined, all these aspects bring the system into line with the current research supporting interpretable, uncertainty-sensitive deep models instead of pure black boxes in neuro oncology.

Tools and Technologies:
  ●	Compute platforms: Google Colab and Kaggle Notebooks with GPU acceleration for training ResNet50_CBAM and running feature-extraction/ensemble experiments.
●	Operating system: Linux-based runtime with CUDA-enabled NVIDIA GPUs and pre- installed scientific Python stack (Python 3.x, pip, system libraries).
●	Code organisation: Experiments implemented as Colab notebooks exported to .py scripts (ravi_eda.py, ravi_novel_code.py), using /kaggle/input for datasets and /kaggle/working for intermediate artefacts (feature CSVs, saved models).

Programming languages and core Libraries:
    ●	Language: Python, chosen for its rich ML/DL ecosystem and rapid prototyping capabilities.
●	Numerical & data handling:
●	NumPy for array operations, numerical computations, and manual GLCM implementation.
●	Pandas for DataFrame construction (image metadata, feature tables, TTA results) and descriptive statistics.
●	Visualisation: Matplotlib and Seaborn for EDA plots (class distribution, resolution histograms, intensity histograms, correlation heatmaps, confusion matrices, learning curves, TTA histograms/scatter plots).

Machine Learning Libraries:
   ●	scikit-learn:
●	Model selection: train_test_split, GridSearchCV, and Stratified splitting for robust evaluation.
●	Preprocessing: StandardScaler, MinMaxScaler for feature normalisation, and PCA (500 components) for dimensionality reduction of 128×128 grayscale vectors.
●	Models: RandomForestClassifier, KNeighborsClassifier, LinearSVC, SVC, and VotingClassifier for traditional ML and deep-feature ensembles.
●	Metrics: accuracy_score, classification_report, confusion_matrix for	all pipelines.
●	Joblib: Saving and loading trained RF, k-NN, SVM, and ensemble models for reproducibility and downstream analysis.

Deep Learning and Explainability stack:
   ●	TensorFlow / Keras:
●	ImageDataGenerator for	on-the-fly	data	augmentation	and	test-time augmentation (TTA) in both training and uncertainty estimation.
●	ResNet50-based architecture with CBAM attention modules (defined in the notebook) for end-to-end MRI classification.
●	Keras Model API and tf.GradientTape for feature-extraction models and Grad- CAM implementation.
●	Grad-CAM utilities: Custom functions to identify the last convolutional layer, compute gradients, generate heatmaps, and overlay them on original images, plus an IoU function for potential quantitative localisation evaluation.

  






   


         
