# Machine Learning - Image Classification 
Project of the Module 3: 'Machine Learning for Computer Vision' of the Master's degree in Computer Vision at Universitat Autònoma de Barcelona (UAB)

## Summary
This repository contains the code and documentation for a comprehensive study on image classification conducted as part of a master's degree project. The project explores a variety of methods and techniques for image classification, ranging from traditional approaches like the visual bag of words method to modern machine learning techniques such as MLP end-to-end classifiers with spatial pyramids. It investigates the performance of different feature extraction methods, including KAZE, SIFT, and Dense SIFT, and evaluates their impact on classification accuracy using KNN's and SVM. Additionally, the project includes the development of a neural network from scratch, incorporating weight pruning for further optimization. Techniques such as data augmentation and hyperparameter optimization with Optuna are also employed.

Findings suggest that employing simple models with the addition or removal of layers, alongside state-of-the-art methods, can lead to improved accuracy while maintaining manageable model complexity.

See [Report](https://drive.google.com/file/d/1I7GHOIgVhsoc24-_R3qT4AydzxDSYe-w/view?usp=sharing)

See [Final Presentation](https://drive.google.com/file/d/1bBhTVEvpul2a8wQBHJ4wAXMs_eMZNPt0/view?usp=sharing)

## Weekly Submits
### [Week 1](https://github.com/rosanavalero/machine_learning_CV/tree/c1ff92b8f577f822bfff60cfeb752f1a88b1a1f7/Week1)

We focused on enhancing the Bag of Visual Words (BoVW) code for our project by implementing several improvements and conducting thorough evaluations. Here's a summary of the tasks accomplished:
- Testing Different Amounts of Local Features
- Utilizing Dense SIFT Instead of Detected Keypoints
- Testing Different Codebook Sizes (k)
- Exploring Different Values of k for k-NN Classifier 
- Experimenting with Different Distance Metrics in k-NN Classifier
- Dimensionality Reduction Exploration
- Cross-Validation of All Implemented Techniques

### [Week 2](https://github.com/rosanavalero/machine_learning_CV/tree/d07b03040b1b27e6704a986dd9717bd1df44e8cf/Week2)
We were asked to enhance the Bag of Visual Words (BoVW) codebase for our project by implementing several advanced techniques. Here's a list of the improvements made:
- Dense SIFT Implementation
- L2 Norm Power Norm
- Support Vector Machine (SVM) Classifier Integration
- StandardScaler Usage
- Cross-Validation Implementation
- Kernel Exploration
- Spatial Pyramids Integration
- Fisher Vectors

### [Week 3](https://github.com/rosanavalero/machine_learning_CV/tree/d07b03040b1b27e6704a986dd9717bd1df44e8cf/Week3)
This week, our focus was on understanding network topology and exploring various modifications to improve our MLP end-to-end classifier's performance.
#### Understanding Network Topology:
- Add/Change Layers in the Network Topology: We investigated the effects of increasing the network's depth and width, utilizing Optuna for hyperparameter optimization to determine the optimal trade-off between depth and width for maximizing validation accuracy
- Effect of Increasing the Number of Parameters and the Importance of Each Feature
#### Comparing Learnt vs. Handcrafted Features:
- Bag of Visual Words End-to-End: Training SVM classifiers using both the last layer predictions of a pre-trained MLP classifier and traditional handcrafted features
- Comparison between k-NN and MLP Hidden Layer as SVM Input: Utilizing t-SNE visualization, we compared the distributions of features extracted by k-NN classifiers and MLP hidden layers for SVM input
- MLP Patch-Based Classifier: We explored an end-to-end approach where images were split into patches for individual classification

### [Week 4](https://github.com/rosanavalero/machine_learning_CV/tree/d07b03040b1b27e6704a986dd9717bd1df44e8cf/Week4)
In Week 4, we delved deeper into advanced techniques and methodologies to enhance the performance of our image classification system.
#### VGG16 Feature Extraction and Model Optimization:
- Utilized pre-trained VGG16 to extract image features, using the output of the last convolutional layer as descriptors for our MLP classifier
- Conducted experiments with optimizers and learning rates
- Explored optimization with different layer widths and depths
- Data Augmentation Evaluation
- Compared the performance of MLP with SVM histogram intersection using extracted VGG16 features
- Utilized t-SNE algorithm to compare the distribution of features extracted by k-NN classifiers, MLP hidden layers, and VGG16 outputs
- Utilized GradCAM for insights into model predictions
#### Model Optimization Strategies:
- Improved VGG16 classifier performance through methodologies such as dropout layers and batch normalization
- Explored optimization with label smoothing to enhance model performance

### [Week 5](https://github.com/rosanavalero/machine_learning_CV/tree/d07b03040b1b27e6704a986dd9717bd1df44e8cf/Week5)
Week 5 marked the exploration of designing convolutional models from scratch and employing weight pruning techniques to improve model efficiency.
#### Defining Convolutional Models from Scratch:
- Experimented with multiple convolutional architectures with varying complexities and layer configurations
- Explored different pooling, normalization layers, and tuning hyperparameters to address issues like overfitting and parameter efficiency
#### Weight Pruning:
- Implemented weight pruning techniques to reduce the size of trained models without compromising performance

## Contributors
- Abel García Romera (abelgr013@gmail.com)
- Marcos Muñoz González (marcosmunozgonzalez95@gmail.com)
- Rosana Valero Martínez (rosanavalero5@gmail.com)

