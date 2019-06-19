# Social-IQ Dataset

![alt text](https://github.com/A2Zadeh/Social-IQ/blob/master/teaser.png)

Human language offers a unique unconstrained approach to probe through questions and reason through answers about social situations. This unconstrained approach extends previous attempts to model social intelligence through numeric supervision (e.g. sentiment and emotions labels). Social-IQ, is an unconstrained benchmark designed to train and evaluate socially intelligent technologies. By providing a rich source of open-ended questions and answers, Social-IQ opens the door to explainable social intelligence. The dataset contains rigorously annotated and validated videos, questions and answers, as well as annotations for the complexity level of each question and answer. Social-IQ contains 1,250 natural in-the-wild social situations, 7,500 questions and 52,500 correct and incorrect answers. Although humans can reason about social situations with very high accuracy (95.08%), existing state-of-the-art computational models struggle on this task.

# Social-IQ Statistics

![alt text](https://github.com/A2Zadeh/Social-IQ/blob/master/stats.png)

**Question Statistics**: The Social-IQ dataset contains a total of 7500 questions (6 per video). Figure 2 (a) demonstrates
the distribution of question length in terms of number of words. The average length of questions in Social-IQ is 10.87 words. Figure 2 (c) shows the different question types in the Social-IQ dataset. Questions starting with why and how, which often require causal reasoning, are the largest group of questions in Social-IQ. This is a unique feature of the Social-IQ dataset and a distinguishing factor of Social-IQ from other multimodal QA datasets (which commonly have what (object) and who questions as the most common). Figure 2 (e) demonstrates the distribution of complexity across questions of the Social-IQ. Majority of the dataset consists of advanced and intermediate questions (with almost equal share between the two) while easy questions share a small portion of the dataset. The distribution of question types and complexity levels in Social-IQ demonstrates the challenging nature of the dataset.

**Answer Statistics**: Social-IQ contains a total of 30,000 correct (4 per question) and 22,500 (3 per question) incorrect answers. Figure 2 (b) demonstrates the distribution of word length for answers in the Social-IQ dataset. Both the correct (green) and incorrect (red) answers follow similar distribution. On average, there are a total of 10.46 words per answer in Social-IQ. This is also a unique characteristic of the Social-IQ dataset since the average answer length is longer than other multimodal QA datasets (with average length between 1.24 to 5.3 words). The long average length demonstrates the level of detail included in Social-IQ answers. Presence of multiple correct answers in the Social-IQ dataset allows for modeling diversity and subjectivity across annotators in cases where multiple explanations are correct for a certain question. Furthermore, having multiple correct answers enables answer generation tasks (which often require multiple correct answers for successful evaluation).

**Multimedia Statistics**: Social-IQ dataset consists of a total of 1,250 videos from YouTube. Figure 2 (f) demonstrates an overview of categories of the videos in Social-IQ. There is a total of 1,239 minutes of annotated video content (across 10,529 minutes of full videos). Figure 2 (d) shows the distribution of number of characters in videos. All the videos in the Social-IQ dataset contain manual transcriptions with detailed timestamps.

# Acquiring the data
The data will be released as a part of our CMU Multimodal SDK (https://github.com/A2Zadeh/CMU-MultimodalSDK). To download all the processed data, simply use:

```python
>>> from mmsdk import mmdatasdk
>>> socialiq_highlevel=mmdatasdk.mmdataset(mmdatasdk.socialiq.highlevel,'socialiq/')
>>> folds=mmdatasdk.socialiq.standard_train_fold,mmdatasdk.socialiq.standard_valid_fold
``` 

Social-IQ 1.0 has no public test data, since the test set will be used for challenges and workshops. We are planning to release a public test set soon on Social-IQ 1.1.  

You can also download the raw data [here](http://immortal.multicomp.cs.cmu.edu/raw_datasets/Social-IQ.zip)


# Running the Tensor-MFN code

