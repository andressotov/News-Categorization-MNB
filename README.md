## Synopsis

The objective of this project is to show how to use Multinomial Naive Bayes (MNB) method to classify news according to some predefined classes. 

## Code Example

The code imports news from the file 'uci-news-aggregator.csv' via Pandas (Python Data Analysis Library). Program 
News categories are encoded as numbers. News text is tokenized, counted and vectorized via scikit learn libraries. 
Data is divided into two sets: one for training the MNB algorithm and the other to test it.
The classifier is trained and then tested. Results are shown. 
The program is programmed in Python 3 and stored in categorization.py 
You can find detailed information about the project and the code into the file 'News Categorization MNB.ipynb'

## Motivation

The objective of this site is to show how to use Multinomial Naive Bayes method to classify news according to some predefined classes. 

## Installation

Provide code examples and explanations of how to get the project.
To run the program, you need to install Python 3 and the following libraries: pandas and some modules from sklearn like preprocessing, utils, feature_extraction, naive_bayes, pipeline, metrics, etc.
The file with the data is assumed to be in the same directory as the main program.

## Tests

To test it you just need to run it. 

## Contributors

The specific dataset used can be found in the UCI ML Repository at this URL: http://archive.ics.uci.edu/ml/datasets/News+Aggregator
In the following link, you could find a guide to explore some of the main scikit-learn tools while analysing a collection of text documents (newsgroups posts) on twenty different topics. In our project we used a different set of documents and made several changes in the code. Our results were better also. http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
In order to avoid overfitting, it is common practice to hold out part of the available data as a test set. You can find more details about what to do to avoid it here: http://scikit-learn.org/stable/modules/cross_validation.html

## License

The MIT license permits reuse within proprietary software provided that all copies of the licensed software include a copy of the MIT License terms and the copyright notice. The MIT license is also compatible with many copyleft licenses, such as the GNU General Public License (GPL); MIT licensed software can be integrated into GPL software, but not the other way around.
