# Confidence Intervals

## Learning Goals

- Describe the use of confidence intervals
- Construct confidence intervals for different types of distributions

## Lecture Materials

[Jupyter Notebook: Constructing Confidence Intervals](ConstructingConfidenceIntervals.ipynb)

## Lesson Plan - 1 hour 15 minutes

### Motivation (10 minutes)

Motivate confidence intervals and emphasize they balance precision and uncertainty. Discuss the knowledge check, guiding students towards recognizing that having more representative samples might make us more confident in our conclusions, while having just five people gathered in a non-random way would not lead to confidence-inducing conclusions, for example.

### Pieces of Confidence Intervals (20 minutes)

First, explain exactly what a confidence interval is (and isn't). Focus on the fact that the language you use is important - we're discussing how confident we are that our interval conveys the population mean, not anything about the spread of our data or anything.

Then deconstruct the pieces of a confidence interval, and how they're built up from sample statistics. Ultimately, confidence intervals come from a calculated margin of error, so we decompose those pieces as well - the critical value, and the standard error. 

Might mention here that it's important for students to recognize what these pieces are, but ultimately they can use a python formula from scipy's `stats` module to actually calculate confidence intervals.

### Confidence Intervals in Python: Set Up (5 minutes)

First, set up the problem - we're still using the same Seattle wage data as was used in the two Distributions lectures, and grabbing a sample with some sample statistics. We want to build a confidence interval to showcase how confident we are in estimating our underlying population's mean Hourly Rate based on a sample.

### Aside for Distributions (10 minutes)

A quick discussion about t versus standard normal distributions. Basically, can say that after a certain sample size, t and z distributions are nearly the same - but we can always use the t distribution in our calculations to be a little more certain (but a little less precise).

### Confidence Intervals in Python: The Calculations (15 minutes)

Break down the pieces in scipy stats, first calculating the critical value, standard error, and margin of error more 'manually' using the formulas. Then use `stats.t.interval` to calculate the interval - you should get the same value.

**Note:** for the `stats.t.ppf` formula to calculate the critical value, you need to pass in `0.975` (for a 95% confidence interval), but for the `stats.t.interval` formula to calculate the interval directly you only need to pass in the confidence level `0.95`. You can highlight this to students as another benefit of using their function, rather than breaking down and calculating the pieces of the formula more manually.

There is an initial visualization to showcase whether the confidence interval you calculated actually captures the 'population mean' (aka the mean of our full Seattle wage dataset) or not. Tehre is some randomness left into this notebook on purpose to showcase how these change when you take new samples - that's shown directly in the visualization of 10 different samples. Can discuss how you'd expect about 9.5 of the 10 to actually contain the population mean, since it's a 95% confidence interval - expect to be wrong about 5/100 times!

### Interpreting Confidence Intervals (10 minutes)

Revisiting the interpretation portion now that they've seen the confidence interval in action, discussion exactly what you can (and cannot) say based on a confidence interval. There are some key points which summarize some of these pieces.

### Conclusion (5 minutes)

There is a lot of level up content - might go over a few of those to highlight where students can get some extra practice. If you run through this lecture quickly and have more time, you might set them up to answer the walrus questions (the first level up exercises) in breakout rooms during lecture. 

There is also some discussion about confidence intervals for non-normal distributions and some intro content about bootstrap sampling.