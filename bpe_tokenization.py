import pandas as pd
import copy
import matplotlib.pyplot as plt
from helpful_functions import read_file
from collections import Counter
from transformers import AutoTokenizer
pd.options.mode.chained_assignment = None

class BPETokenizer:
    def __init__(self, path_to_txt="txt/sample.txt"):
        self.text = read_file(path_to_txt ,"r")
        self.debug = False

    def create_and_split_corpus(self):
        non_word_characters = [".", ",", "'", "(", ")", '"', ";", "[", "]", "{", "}"]
        char_counter_dict = {}
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        normalized_txt = tokenizer.backend_tokenizer.normalizer.normalize_str(self.text)
        pretokenized_list = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(normalized_txt)
        self.pretokenized_words = []
        self.pretokenized_words_without_end = []
        self.pretokenized_words_splitted_dict = {}
        for pretoken in pretokenized_list:
            if pretoken[0] not in non_word_characters:
                for char in pretoken[0]:
                    if char in char_counter_dict.keys():
                        char_counter_dict[char] += 1
                    else:
                        char_counter_dict[char] = 1
                if "</w>" in char_counter_dict:
                    char_counter_dict["</w>"] += 1
                else:
                    char_counter_dict["</w>"] = 1
                self.pretokenized_words.append(f'{pretoken[0]}{"</w>"}')
                self.pretokenized_words_without_end.append(f'{pretoken[0]}')
                self.pretokenized_words_splitted_dict[f'{pretoken[0]}{"</w>"}'] = [*pretoken[0], "</w>"]
        self.words_counter_dict = Counter(self.pretokenized_words)
        """
        Making char dataframe out of char dict
        """
        subtokens = []
        for key in char_counter_dict.keys():
            subtokens.append([key])
        self.tokens_df = pd.DataFrame(list(zip(char_counter_dict.keys(),
                                       subtokens, char_counter_dict.values())),
                                       columns=["Token", "Subtokens", "Frequency"])
        return self.tokens_df, \
            self.pretokenized_words, \
            self.pretokenized_words_splitted_dict, \
            self.words_counter_dict, \
            self.pretokenized_words_without_end

    def generate_new_tokens(self):
        """
        Creating all possible pairs including most common token as a left-hand-side part of the pair
        The new most common pair cannot be yet included in the tokens df
        """
        frequency_sorted_tokens = self.tokens_df["Token"].to_list()
        # frequency_sorted_tokens = self.tokens_df.sort_values("Frequency", ascending=False)["Token"].to_list()
        possible_pairs = []
        """
        The possible_pairs_splitted_dict is introduced to keep the list of elementary tokens that build 
        the new token. Some of the elemenatry tokens (like </w> may consist of multiple
        characters and still be elementary. This has to be done for further operations.
        """
        possible_pairs_splitted_dict = {}
        for most_common_token in frequency_sorted_tokens:
            for char in self.tokens_df["Token"].to_list():
                if f'{most_common_token}{char}' not in self.tokens_df["Token"].to_list():
                    possible_pairs.append(f'{most_common_token}{char}')
                    possible_pairs_splitted_dict[f'{most_common_token}{char}'] = [most_common_token, char]
            """
            Counting the frequency of appearance of constructed pairs
            """
        most_common_pair = ""
        highest_possible_pair_counter_so_far = 0
        possible_pairs_dict = {}
        words_in_which_mcp_exists={}
        for possible_pair in possible_pairs:
            words_in_which_possible_pair_exists={}
            possible_pair_counter = 0
            possible_pair_splitted = possible_pairs_splitted_dict[possible_pair]
            # Checking if split elements of pair appear in a  non-splitted word
            for word in self.pretokenized_words:
                pretokenized_splitted_word = self.pretokenized_words_splitted_dict[word]
                pair_elements_in_word=True
                for element in possible_pair_splitted:
                    if not element in pretokenized_splitted_word:
                        pair_elements_in_word=False
                if not pair_elements_in_word:
                    continue
                else:
                    first_element_idx=None
                    second_element_idx = None
                    for i in range(len(pretokenized_splitted_word)):
                        if pretokenized_splitted_word[i] == possible_pair_splitted[0]:
                            first_element_idx=i
                        if pretokenized_splitted_word[i] == possible_pair_splitted[1]:
                            second_element_idx=i
                    if second_element_idx-first_element_idx == 1:
                        """
                        Adding 1 to the pair occurrences counter
                        Saving the word where the pair was found, not to have to
                        do it again (which requires looping through all words) if it
                        turns out to be the mcp. Also position of the pair in the word is saved.
                        """
                        possible_pair_counter += 1
                        words_in_which_possible_pair_exists[word] = [first_element_idx, second_element_idx]
                    else:
                        break
            # possible_pair_counter += self.words_counter_dict[word]
            possible_pairs_dict[possible_pair] = possible_pair_counter
            if possible_pair_counter > highest_possible_pair_counter_so_far:
                highest_possible_pair_counter_so_far = possible_pair_counter
                most_common_pair = possible_pair
                words_in_which_mcp_exists = words_in_which_possible_pair_exists

        """
        Most common pair.
        Checking if MCP is not a substring of a already existing token.
        If so, its frequency is reduced.
        """
        # print(f'{"MCP: "}{most_common_pair}{", "}{highest_possible_pair_counter_so_far}')

        splitted_words_in_which_mcp_exists=[]
        for word in words_in_which_mcp_exists.keys():
            for i in range(self.words_counter_dict[word]):
                splitted_words_in_which_mcp_exists.append([self.pretokenized_words_splitted_dict[word], words_in_which_mcp_exists[word]])
        """
        If MCP is found, we modify the pretokenized_splitted_words 
        """
        for swm in splitted_words_in_which_mcp_exists:
            splitted_word = swm[0]
            modification_range = swm[1]
            modified_splitted_word = copy.deepcopy(splitted_word)
            modified_splitted_word = modified_splitted_word[:modification_range[0]] + \
                                     [most_common_pair] + modified_splitted_word[modification_range[1]+1:]

            for w, swp in self.pretokenized_words_splitted_dict.items():
                if swp == splitted_word:
                    self.pretokenized_words_splitted_dict[w] = modified_splitted_word

        if most_common_pair:
            self.tokens_df = pd.concat([self.tokens_df, pd.DataFrame(
                [[most_common_pair, possible_pairs_splitted_dict[most_common_pair],
                highest_possible_pair_counter_so_far]],columns=["Token", "Subtokens", "Frequency"])],
                ignore_index=True)
            """
            Check which tokens are included in the possible pair
            Reduce their frequency by the frequency of the possible pair
            """
            # possible_pair_substrings = self.create_substrings(most_common_pair)
            # Retrieving df rows where token is a substring of the possible pair
            rows_where_token_is_substring_of_possible_pair_df = \
                self.tokens_df.loc[self.tokens_df["Token"].isin(possible_pairs_splitted_dict[most_common_pair])]

            # Reducing the frequency of selected tokens by the frequency of possible pair
            rows_where_token_is_substring_of_possible_pair_df['Frequency'] = \
                rows_where_token_is_substring_of_possible_pair_df['Frequency'].apply(lambda x: x - highest_possible_pair_counter_so_far)

            # Replacing df rows with rows where token frequency is reduced
            self.tokens_df.loc[self.tokens_df["Token"].isin(possible_pairs_splitted_dict[most_common_pair])] = \
                rows_where_token_is_substring_of_possible_pair_df

            # Deleting rows where token frequency <= 0
            # self.dropped_tokens += self.tokens_df.loc[(self.tokens_df['Frequency'] <= 0)]['Token'].to_list()
            self.tokens_df.drop(self.tokens_df[self.tokens_df['Frequency'] <= 0].index, inplace=True)

    def train(self, max_iterations=200):
        num_of_tokens_in_iteration = []
        i = 0
        while i < max_iterations:
            self.generate_new_tokens()
            if self.debug:
                print(f'{"ITERATION: "}{i}')
                num_of_tokens_in_iteration.append(len(self.tokens_df))
            i += 1
        return self.tokens_df, num_of_tokens_in_iteration

# tkn = BPETokenizer()
# tkn.create_and_split_corpus()
# sorted_tokens_df, num_of_tokens_in_iteration = tkn.bpe_create_tokens()
# sorted_tokens_df[["Token", "Subtokens", "Frequency"]].to_csv("txt/tokens_df.csv", index=True)
# xs = [x for x in range(len(num_of_tokens_in_iteration))]
# plt.plot(xs, num_of_tokens_in_iteration)
# plt.show()