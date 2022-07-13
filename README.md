# Analyzing channel contribution in multi-channel sleep scoring

Our paper is to answer two research questions:
1. Whether multi-channel models can be improved by incorporating model features from high-performing single-channel models 
2. Which channels contribute to a high performing multi-channel model


For the first research question, we add the model features that have been successfully used
by single-channel models for performance improvement to the multi-channel setting. The mode features are
1. use small and larger filters in the first layer of CNNs to capture time-domain and frequency-domain features of sleep signals respectively
2. add deeper layers in CNNs to increase feature complexity
3. apply attention mechanisms in RNNs to concentrate on the important parts of sleep sequences
4. add residual connection that concatenates features of CNNs and RNNs to consider temporal and sequential features equally in the final classification of sleep scoring
We test the improved multi-channel model on SleepEDF-13 and SHHS-1 and obtained the-state-of-the-art results. The model files and traing files as well as the trained models are included in the folder RQ1. Note that, we use an adpated nested cross validation for SleepEDF-13 which is a small data set, so we have a lot of trained models there. In this blog, we provide an average-performing trained model as the example.

For the second resear question, we use the layer-wise relevance propagation (LRP) method and adopt an embedded channel attention network (eCAN) to extract channel importance information in both post-hoc and intrinsic explainability direction. In the folder RQ2, we provide the necessary code files for LRP and eCAN.
