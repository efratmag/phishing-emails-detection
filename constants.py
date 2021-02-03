LABEL_COL = "Label"
TEXT_COLS = ['Content', 'Subject']
config_file = 'config.ini'

# common words in phising email from the paper https://www.hindawi.com/journals/jam/2014/425731/.
Content_common_phish_words_1 = ['update', 'confirm']
Content_common_phish_words_2 = ['user', 'customer', 'client']
Content_common_phish_words_3 = ['suspend', 'restrict', 'hold']
Content_common_phish_words_4 = ['login', 'username', 'password', 'click', 'log']
Content_common_phish_words_5 = ['ssn', 'social security', 'secur', 'inconvinien']

cols_to_drop = ['index','Content','Subject','Content-Type',
                'Content_lower','Content_all_links',
                'Subject_lower','Subject_all_links']

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10