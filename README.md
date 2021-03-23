
# Regression Project: 
![Zillow](zillow_logo.png)
___
[[Scenario](#scenario)]
[[Project Planning](#planning)]
[[Key Findings](#key-findings)]
[[Tested Hypotheses](#tested-hypotheses)]
[[Recommendations](#recommendations)]
[[Takeaways](#takeaways)]
[[Data Dictionary](#data-dictionary)]
[[Workflow](#workflow)]
___

# <a name="scenario"></a>Scenario
As a junior data scientist on the Zillow data science team, I am tasked to build a model that will predict the values of single unit properties that the tax district assesses (using the property data from those whoes last transaction was during May through August of 2017).

Unfortunately, some features of our dataset have been deleted. 
- I must also recover the following features:
    - the state and county information for all of the properties
    - distribution of tax rates for each county
        - tax amounts and tax value of the home

   
   
Memo from the Zillow Data Science Team:
>Note that this is separate from the model you will build, because if you use tax amount in your model, you would be using a future data point to predict a future data point, and that is cheating! In other words, for prediction purposes, we won't know tax amount until we know tax value.
___

# <a name="project-planning"></a>Project Planning
### Goal:
The goal for this project is to create a model that will accurately predict a home's value determined by the county's Appraisal District. To do so, I will have to identify which of the various features available on Zillow's API affect the accuracy of my model's predictions the most. 

### Initial Hypotheses:

>$Hypothesis_{1}$
>
> There is a relationship between square footage and home value.


>$Hypothesis_{2}$
>
> There is a weak correlation between number of bathrooms and home value.

### Project Planning Initial Thoughts:
- With the missing features I am tasked to reproduce in mind, I want to create some (most likely) dummy features connecting `fips` to county names. 
    - I have a strong feeling that certain counties will have higher home value assessment. 
- On my second iteration, I'd also like to test the importance of 
    - `lotsizesquarefeet` 
    - `regionidcounty`
    - `regionidzip`
    - `regionidneighborhood`
    - `yardbuildingsqft17`
    - 'logerror'
- I'd like to create a new feature:
    - `home_age`: current year - `yearbuilt`
    - 
    
- Although to keep myself on track to reach my goals, I will first focus on my first iteration with my limited features of `calculatedfinishedsquarefeet`, `bedroomcnt` and `bathroomcnt` to predict my target `taxvaluedollarcnt`. 
- If time permits, I'd like to create nice visualizations in Tableau, although I may have to settle with seaborn graphics.

___
# <a name="key-findings"></a>Key Findings

## Exploration Takeaways
After removing home_value outliers, it seems 'expensive/large' homes are still pulling my data towards the right. The median household looks like 1430 sqft, 3BD/2BA, with a value of $311_000.

A majority of homes are in the bottom 25% of square feet and bottom 45% of home_value.



___
# <a name="tested-hypothesis"></a>Tested Hypothesis

___
# <a name="recommendations"></a>Recommendations<--Probably will remove

___
# <a name="takeaways"></a>Takeaways

___
# <a name="project-planning"></a>Data Dictionary

___
# <a name="workflow"></a>Workflow

## Workflow

## Workflow

1. [Prep Your Repo](#prep-your-repo)
1. [Import](#import)
1. [Acquire Data](#acquire-data)
1. [Clean, Prep & Split Data](#clean-prep-and-split-df)
1. [Explore Data](#explore-data)
    - [Hypothesis Testing](#hypothesis-testing)
1. [Evaluate Data](#evaluate-data)
1. [Modeling](#modeling)
    - [Identify Baseline](#identify-baseline)
    - [Train / Validate](#train-validate)
    - [Test](#test)
1. [2nd Iteration: Acquire Data](#2nd-iteration-acquire-data)
1. [2nd Iteration: Clean, Prep & Split Data](#2nd-iteration-clean-prep-and-split-data)
1. [2nd Iteration: Explore Data](#2nd-iteration-explore-data)
    - [2nd Iteration: Hypothesis Testing](#2nd-iteration-hypothesis-testing)
1. [2nd Iteration: Evaluate Data](#2nd-iteration-evaluate-data)
1. [2nd Iteration: Modeling](#2nd-iteration-modeling)
    - [2nd Iteration: Identify Baseline](#2nd-iteration-identify-baseline)
    - [2nd Iteration: Train / Validate](#2nd-iteration-train-validate)
    - [2nd Iteration: Test](#2nd-iteration-test)
