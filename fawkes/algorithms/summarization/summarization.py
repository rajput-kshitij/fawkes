import nltk
import nltk.data
import json
import fawkes.constants.constants as constants
from fawkes.configs.fawkes_config import FawkesConfig
import fawkes.utils.utils as utils
from fawkes.configs.app_config import AppConfig, ReviewChannelTypes, CategorizationAlgorithms
from fawkes.review.review import Review
import fawkes.email_summary.queries as queries
from nltk.tokenize import sent_tokenize
from fawkes.algorithms.algo import k_means_classification, summarize_text

def preprocess_review(reviews):
    processed_sentences=[]
    
    for review in reviews:
        review_message = review.message
        sentences=sent_tokenize(review_message)
        for sentence in sentences:
            word_count = len(sentence.split())
            # if(word_count>constants.min_words_in_sentence and word_count<constants.max_words_in_sentence):
            processed_sentences.append(sentence)
    return processed_sentences


def generate_summary(fawkes_config_file = constants.FAWKES_CONFIG_FILE):
    # Read the app-config.json file.
    fawkes_config = FawkesConfig(
        utils.open_json(fawkes_config_file)
    )
    # For every app registered in app-config.json we
    for app_config_file in fawkes_config.apps:
        # Creating an AppConfig object
        app_config = AppConfig(
            utils.open_json(
                app_config_file
            )
        )
        # Path where the user reviews were stored after parsing.
        processed_user_reviews_file_path = constants.PROCESSED_USER_REVIEWS_FILE_PATH.format(
            base_folder=app_config.fawkes_internal_config.data.base_folder,
            dir_name=app_config.fawkes_internal_config.data.processed_data_folder,
            app_name=app_config.app.name,
        )

        # Loading the reviews
        reviews = utils.open_json(processed_user_reviews_file_path)
        # Converting the json object to Review object
        reviews = [Review.from_review_json(review) for review in reviews]

        reviews=queries.getVocByCategory(reviews)
        
        # nltk.data.load('tokenizers/punkt/english.pickle')

        '''
        Code to preprocess the data, break the reviews into sentences 
        in case words > max_allowed_words in a sentence OR 
        number of sentences > max allowed sentences in a sentence
        '''

        # TODO:: Change algo.py summarizer to return list and change append to extend (\n)
        # TODO:: Add checks to remove empty reviews in summary
        summarized_reviews = {}
        for category in reviews:
            summarized_category_review = []
            categorized_review = reviews[category]

            sentences= preprocess_review(categorized_review)
            if(len(sentences)>1):
                
                clustered_sentences = k_means_classification(sentences)


                for i in range(len(clustered_sentences)):
                    cluster = clustered_sentences[i]
                    if(len(cluster) < constants.minimum_reviews_per_cluster):
                        continue
                    text = ". ".join(cluster) + "." 
                    gen_summary = summarize_text(text,constants.words_per_review)
                    summarized_category_review.append(gen_summary)
            summarized_reviews[category] = summarized_category_review
        
        print(summarized_reviews)