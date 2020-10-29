num_clusters = 5
num_result_per_review=2
minimum_reviews_per_cluster=50 # OR based on similarity in a cluster ( only retun in high rating)
max_words_in_sentence= 20
min_words_in_sentence= 4
max_lines_in_sentence= 2
import nltk
import nltk.data
import json
import fawkes.constants.constants as constants
from fawkes.configs.fawkes_config import FawkesConfig
import fawkes.utils.utils as utils
from fawkes.configs.app_config import AppConfig, ReviewChannelTypes, CategorizationAlgorithms
from fawkes.review.review import Review
import fawkes.email_summary.queries as queries



# review['derived_insight']['sentiment']['neg']>0.2 and 

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
        print("----")
        print(len(reviews))
        corpus=queries.getVocByCategory(reviews)
        
        print("---")


        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        print(len(corpus))
        corpus = list(set(corpus))

        print(len(corpus))
        '''
        Code to preprocess the data, break the reviews into sentences 
        in case words > max_allowed_words in a sentence OR 
        number of sentences > max allowed sentences in a sentence
        '''
        #https://github.com/summanlp/textrank
        # from summa import keywords
        # from summa.summarizer import summarize
        sentences=[]
        summarized_sentences=[]
        for review in corpus:
            temp_sentence=tokenizer.tokenize(review)
            for sentence in temp_sentence:
                print(sentence)
                word_count = len(sentence.split())
                # if(word_count>min_words_in_sentence and word_count<max_words_in_sentence):#uncomment
                summarized_sentences.append(sentence)
        # print(summarized_sentences)        
        print(len(summarized_sentences))
        #https://github.com/UKPLab/sentence-transformers
        from sentence_transformers import SentenceTransformer
        import numpy as np
        embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        corpus_embeddings = embedder.encode(summarized_sentences)
        from sklearn.cluster import KMeans
        print(np.shape(corpus_embeddings))
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        # for i in range(len(summarized_sentences)):
        #     print(str(summarized_sentences[i])+" -- "+str(cluster_assignment[i]))
        clustered_sentences = {new_list: [] for new_list in range(num_clusters)} 
        for i in range(len(cluster_assignment)):
            temp_list= clustered_sentences[cluster_assignment[i]]
            temp_list.append(summarized_sentences[i])
            clustered_sentences[cluster_assignment[i]]=temp_list
        # print(clustered_sentences)

        print("#########")
        print(len(clustered_sentences))


        from gensim.summarization.summarizer import summarize

        for i in range(len(clustered_sentences)):
            cluster = clustered_sentences[i]
            print(len(cluster))
            if(len(cluster) < minimum_reviews_per_cluster):
                continue
            print("#")
            text = ". ".join(cluster) + "." 
            print(text)
            print("### Summary is")
            print()
            gen_summary=summarize(text, word_count = 30)
            print(gen_summary)

        # # Filtering out reviews which are not applicable.
        # reviews = filter_utils.filter_reviews_by_time(
        #     filter_utils.filter_reviews_by_channel(
        #         reviews, filter_utils.filter_disabled_review_channels(
        #             app_config
        #         ),
        #     ),
        #     datetime.now(timezone.utc) - timedelta(days=app_config.email_config.email_time_span)
        # )

        #  formatted_html = email_utils.generate_email(
        #     app_config.email_config.email_template_file,
        #     template_data
        # )
        # # Path where the generated email in html format will be stored
        # email_summary_generated_file_path = constants.EMAIL_SUMMARY_GENERATED_FILE_PATH.format(
        #     base_folder=app_config.fawkes_internal_config.data.base_folder,
        #     dir_name=app_config.fawkes_internal_config.data.emails_folder,
        #     app_name=app_config.app.name,
        # )

        # dir_name = os.path.dirname(email_summary_generated_file_path)
        # pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

        # with open(email_summary_generated_file_path, "w") as email_file_handle:
        #     email_file_handle.write(formatted_html)




def parse_json_lines(file_location):
    parsed_reviews = []
    with open(file_location, "r") as read_file:
        reviews = json.load(read_file)
        for review in reviews:
            if(review["derived_insight"]["category"] == 'User Experience'):
                parsed_reviews.append(review['message'])
    return parsed_reviews

# corpus= parse_json_lines('./processed-user-feedback.json')
# corpus= parse_json_lines('../../../data/prcessed_data/processed-user-feedback.json')

