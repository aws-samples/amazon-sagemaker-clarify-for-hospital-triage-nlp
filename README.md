# Using Amazon Sagemaker Clarify to support Decision Support in Hospital Triage

## Acute Care Clinical Context:

Decision Support at admission time can be especially valuable for prioritization of resources in an acute care linical setting such as a hosptial. These critical resources come in the form of doctors and nurses, as well as specialized beds, such as ones in the intensive care units. These place limits on the overall capacity of the hospital to treat patients.

Hosptials can more effectively use these resources by predicting the following:
diagnoses at discharge, procedures performed, in-hospital mortality and length-of-stay prediction

Novel approaches in NLP, such as Bidirectional Encoder Representations from Transformers (BERT) models, have allowed for inference on clinical data, and specifically notes, at an accuracy level not attainable a number of years ago. These advances make predicting key clinical indicators from notes data, and applying them in the real world, much more achieveable.

The following references articulate how these indicators have been developed and are being used:

1) "Clinical Outcome Prediction from Admission Notes using Self-Supervised Knowledge Integration" 
    - https://aclanthology.org/2021.eacl-main.75.pdf

2) "Prediction of emergency department patient disposition based on natural language processing of triage notes"
    - https://pubmed.ncbi.nlm.nih.gov/31445253/    

3) Application of Machine Learning in Intensive Care Unit (ICU) Settings Using MIMIC Dataset: Systematic Review
    - https://www.amjmed.com/article/S0002-9343(20)30688-4/abstract

## Overview of the Notebook:

The intent of this notebook is to provide a practical guide for data scientists, and machine learning engineers to collaborate with clinicians, and to support real implementations of clinical indicator predictions. As such, explainability of the algorithms is required.

Advances in NLP algorithms, as in the studies above, have made predicting clinical indicators more accurate, yet in order to effectively use machine learning models in a production setting, clinicians also need more insight into how these models work. They need to know that these algorithms make clinical sense before going to production. Clinicians and data scientists, need a way to evaluate realiablility, and explainability of models over time, as more data continues to be evaluated, and machine learning models are retrained.

This notebook will take one of these clinical triage indicators, in-hospital mortality, and show how AWS services and infrastructure, along with pre-trained HuggingFace BERT models, can be used to train a binary classifier on text data, estimate a threshold value for triage, and then use Amazon Sagemaker Clarify to explain what admission note text is supporting the recommendations the algorithm is making.

In this notebook we use the HuggingFace BERT Model - BIO_Discharge_Summary_BERT (https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT). This is the pre-trained BERT model we will use in this notebook in order to demonstrate how NLP can be used to create a performant binary classifier for use in a clinical setting.

The notebook will also leverage a common, publicly available dataset directly; the MIMIC-III dataset (https://physionet.org/content/mimiciii-demo/1.4/). It will require a registration process, since the data is based on real-world acute care clinical data, but is a worthwhile for an organization, as this can be reused again for many different purposes.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

