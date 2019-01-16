---
layout:     post
title:      Machine Learning
subtitle:   Analysis of recruitment information in Beijing
date:       2019-01-17
author:     ShuaiGao
header-img: img/post-bg-YesOrNo.jpg
catalog: True
tags:
    - Random Forest
    - Machine Learning
    - Association Finding
---

# Analysis of recruitment information in Beijing
## Introduction

The goal of this project is to analyze the recruitment situation in Beijing, China then try to find the information that relates to the prospect of a job. The results could help job seekers in Beijing to choose their career more advisable by understand the career expectation and to make more suitable choice for themselves.
here is 5 part contained:

1. Data collection

3. Data pre-processing

5. Visualization

7. Predict Machine Learning

9. Association Finding


## Data Collection

The data was crowled from 51job.com which is a nationwide job posting website and one of the biggest recruitment information website in China. Since the goal is to analysis Beijing's recruitment information, the data collected was filtered to only contains jobs that located in Beijing.

The data is collected by using Scrapy. One of the most popular packages for web information crawling based on python.

The [code](https://github.com/UltramanShuai/ML_Projiect/tree/master/job_china "code") of the crawling process can be found in file named job_china.


## Data description
The row data contains 8 attributes.

- job: The job title. (String)

- Company: The name of the company. (String)

- Salary: Salary information. (String)

- Work_Position: The location (suburb) of the job. (String)

- Post_time: The time of the job being posted . (String) 
- This column could not help our analysis since the post time changes automatically daily to the present day.

- Require: The requirements of the job. (String)

- Describe: The additional information that may or may not help our analysis from the job posting. (String)

- Type: The "tags" associated with the job or the position, eg. industry, sector etc. (String)

## Code of data Processing and modeling
[ Code of data Processing and modeling](https://github.com/UltramanShuai/ML_Projiect/blob/master/machine_learning/Job_analysis.ipynb "## # Data pre-processing")

## Words Cloud
In this section, what I want to achieve is to compare the differences of the jobs between the high salary sector (greater than 2 thousand per month)
and relatively low salary sector (lower than 1 thousand per month).

By using cloud word picture, the results are quite straightforward.
#### Work type words picture by different salary group job
![](https://raw.githubusercontent.com/UltramanShuai/ML_Projiect/master/machine_learning/Type_high.png)
![](https://raw.githubusercontent.com/UltramanShuai/ML_Projiect/master/machine_learning/Type_low.png)

Based on the pictures, we can clearly see the difference between the jobs with high salary and low salary.

Engineer, manager, consultant and chief occur the most in high salary groups.

Most of the jobs are seen in relatively low salary picture are assistant, receptionist and admin.

It seems the salary are still depended on the education level, which we will check later.

#### Work description's words picture by different salary group job
![](https://raw.githubusercontent.com/UltramanShuai/ML_Projiect/master/machine_learning/Describe_high.png)
![](https://raw.githubusercontent.com/UltramanShuai/ML_Projiect/master/machine_learning/Describe_low.png)

The differences of the two salary groups can be easily seen from these word clouds.

For high salary group: free meals, flexible work schedule, agile management and rent subsidies takes lots of space.

For the group of salary lower than 10 thousand per month: paid annual leave, periodic physical examination and holiday welfare dominate the space.

Most importantly, for high salary groups, the descriptions, or the welfare listed in the job posting are the real extra benefits. On the other hand, most of the welfare shows in low salary groups are not only vague, but also are what actually is statutory mandatory benefits where most of the highly paid jobs contains these benefits but didn't specify in the job postings.

#### Work requirement's words picture by different salary group job
![](https://raw.githubusercontent.com/UltramanShuai/ML_Projiect/master/machine_learning/Require_high.png)
![](https://raw.githubusercontent.com/UltramanShuai/ML_Projiect/master/machine_learning/Require_low.png)

For the high salary group: work experience of 2-10 years and bachelor degree takes almost all the picture.

In contrast, at low salary group, no working experience and junior college degree take the domination.

The result also supports the assumption we set before, which there is a strong correlation between wage and education level for jobs at Beijing.

## Machine learning
[ Code can be found here](https://github.com/UltramanShuai/ML_Projiect/blob/master/machine_learning/Job_analysis.ipynb "## # Data pre-processing")
## Association Finding
[ Code can be found here](https://github.com/UltramanShuai/ML_Projiect/blob/master/machine_learning/Job_analysis.ipynb "## # Data pre-processing")


## Conclusion

What I have done in this project,in general, is almost every step of data analysis which from data collection to validation.

1. For data collection and pre-processing, I collect raw data from real-world website, 51job.com, in order to analysis the job situation in Beijing. I pre-process the raw data and extract useful information for analysis.

3.  For analysisFor analysis, I divided the attributes into 2 different salary level group (jobs with salary more than 20000 per month as high salary group and jobs with salary lower than 10000 per month as low salary group) and found that there are obvious differences in job type, job welfare and job requirement between groups. Jobs in high salary group are more likely represent jobs that requires well education, longer work experience and in technical or management sectors. Jobs in low salary group are more likely represent jobs that do not require have bachlor degree, no working experience and in assistant roles. The welfare for high salary group are the real extra benefits. On the other hand, most of the welfare shows in low salary groups are not only vague, but also are what actually is statutory mandatory benefits where most of the highly paid jobs contains these benefits but didn't specify in the job postings.

5. In machine learning part, by using random forest and tune the parameters and change the thresholder, we can meet over 80% accuracy to predict if it is a well-paid job by the description of job type, walfare description and requirement.

7. By assosiation finding, we found when "performance bonus" showed on the description, the job is vary likely provide on site training as well. In addition, over half percentage of "English requirmenet" are followed by "bachlor degree" in the description.