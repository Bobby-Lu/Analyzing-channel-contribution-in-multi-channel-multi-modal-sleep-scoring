# Analyzing channel contribution in deep learning based multi-channel multi-modal sleep scoring

Our paper is to answer two research questions:
1. Whether the high performance of single-channel EEG models can be attributed to specific model features in their deep learning architectures
2. To which extent multi-channel multi-modal models take the information from different channels of modalities into account.

For the first research question, we added the model features that have been successfully used by single-channel models for performance improvement to the multi-channel multi-modal setting. The mode features we considered are:
1. use small and large filters in the first layer of CNNs to capture time- and frequency-domain features of sleep signals respectively;
2. add deeper layers in CNNs to increase feature complexity;
3. apply attention mechanisms embedded in RNNs to concentrate on the important parts of sleep sequences;
4. add residual connection that concatenates features of CNNs and RNNs to consider temporal and sequential features venly in the final classification of sleep stages.
We tested the improved multi-channel model on SleepEDF-13 and SHHS-1 and obtained state-of-the-art results. The model files and training files are included in the folder RQ1.

For the second researh question, we used the layer-wise relevance propagation (LRP) method and adopted an embedded channel attention network (eCAN) to extract channel importance information in both post-hoc and intrinsic explainability directions. In the folder RQ2, we provide necessary code files for LRP and eCAN. The obtained results are included as well.
