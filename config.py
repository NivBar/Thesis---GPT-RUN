import itertools
import random
import re
from pprint import pprint
from nltk.corpus import stopwords

import openai
import pandas as pd
from tqdm import tqdm

from API_key import API_key

stop_words = set(stopwords.words('english'))



# GPT!!!
using_gpt = True
using_e5 = True
temperature = 1.5

rel_round = 6
current_prompt = f"tommy{rel_round}@F{int(temperature*100)}"

if using_e5:
    tommy_data = pd.read_csv("tommy_data.csv")
    filtered_df = tommy_data[tommy_data["round_no"] == rel_round]['current_document']
    median_word_count = int(filtered_df.apply(lambda x: len(x.split())).median())
else:
    median_word_count = 102

ACTIVE_BOTS = ["DYN_1100T2", "POW_1300", "PAW_1301R", "DYN_1201R2", "PAW_1210T", "LIW_1201"]

temperatures = [temperature]
bot_cover_df = pd.read_csv("final_options.csv").sort_values(["doc_no", "bot_name"], ascending=True)

if using_gpt:
    bot_cover_df = bot_cover_df[bot_cover_df.bot_name.isin(ACTIVE_BOTS)]

cartesian_product = itertools.product(bot_cover_df.iterrows(), temperatures)
new_rows = []
for (index, row), temp in cartesian_product:
    new_row = row.to_dict()
    new_row['temp'] = temp
    formatted_temp = str(int(temp*100))
    new_row['bot_name'] = new_row['bot_name'] + "@" + formatted_temp
    new_rows.append(new_row)

bot_cover_df = pd.DataFrame(new_rows)

#### openai parameters ####
openai.api_key = API_key

"""
Model: Determines the architecture and parameters of the language model used for text generation. Different models have 
different strengths and weaknesses for specific types of text generation tasks.

Temperature: Controls the level of randomness and creativity in the generated text. High temperature values (e.g., 1.0 
or higher) can produce more diverse and unexpected outputs, while low values (e.g., 0.5 or lower) can produce more 
predictable and conservative outputs.

Top_p: Limits the set of possible next words based on the model's predictions. High top_p values (e.g., 0.9 or higher) 
allow for more variation and creativity, while low values (e.g., 0.1 or lower) can produce more predictable and 
conservative outputs.

Max_tokens: Sets an upper limit on the number of tokens that can be generated in the output text. High max_tokens values 
(e.g., 500 or higher) can produce longer outputs, while low values (e.g., 50 or lower) can produce shorter and more 
concise outputs.

Frequency_penalty: Encourages the model to generate less frequent words or phrases. High frequency_penalty values 
(e.g., 2.0 or higher) can increase the diversity and creativity of the generated text, while low values 
(e.g., 0.5 or lower) can produce more common and predictable outputs.

Presence_penalty: Encourages the model to avoid repeating words or phrases that have already appeared in the output 
text. High presence_penalty values (e.g., 2.0 or higher) can promote the generation of novel and varied text, while low 
values (e.g., 0.5 or lower) can produce more repetitive and redundant outputs.
"""
model = "gpt-4o"
max_tokens = 180
top_p = 0.3
frequency_penalty = 1.0
presence_penalty = 0.0

#### useful data collections ####
# topic_codex_new = json.load(open("topic_queries_doc.json", "r"))
# topic_codex = dict()

# # TODO: change to actual copetition data when starting
# comp_data = pd.read_csv("Archive/comp_dataset.csv")
#
# query_index = {x[0]: x[1] for x in comp_data[["query_id", "query"]].drop_duplicates().values.tolist()}


def get_unique_words(string):
    cleaned_string = re.sub(r'[^\w\s]', '', string.lower())
    words = cleaned_string.split()
    unique_words = set(words)
    unique_words = unique_words - stop_words
    return str(unique_words).replace("{", "").replace("}", "").replace("'", "")


def get_current_doc(data, creator_name, query_id):
    recent_data = data[data['query_id'] == query_id]
    epoch = rel_round
    current_doc = \
        recent_data[(recent_data.round_no == epoch) & (recent_data.username == creator_name)][
            "current_document"].values[0].strip()
    return current_doc


def get_prompt(bot_name, data, creator_name, query_id):
    # if query_id != 69:
    #     data = data[data.query_id != 69]
    #     x = 1
    prompt_dict_list = []
    # data['current_document'] = data['current_document'].replace('\n', ' ', regex=True).replace('\\n', ' ', regex=True)
    data.loc[:, 'current_document'] = data.loc[:, 'current_document'].replace('\n', ' ', regex=True).replace('\\n', ' ',
                                                                                                             regex=True)
    method, traits = bot_name.split("_")
    # bot names - {method}_[query no}{example no}{candidate inc}{query inc}{doc type}{history len}
    bot_info = {"method": method, "query_num": int(traits[0]), "ex_num": int(traits[1]),
                "cand_inc": True if traits[2] == "1" else False,
                "query_inc": True if traits[3] == "1" else False}
    if bot_info["method"] in ["DYN", "PAW"]:
        bot_info["doc_type"] = traits[4]
        if bot_info["method"] == "DYN":
            bot_info["history_len"] = int(traits[5])

    # print(bot_info)

    query_ids = random.sample(data[data.query_id != query_id]["query_id"].unique().tolist(), bot_info["query_num"])
    if bot_info["query_inc"]:
        query_ids[0] = query_id
    epoch = rel_round
    recent_data = data[(data['query_id'] == query_id) & (data['round_no'] <= rel_round)]
    query_string = recent_data.iloc[0]['query']

    epoch_phrasing = {epoch: "latest", epoch - 1: "second to latest", epoch - 2: "third to latest"}
    current_doc = \
        recent_data[(recent_data.round_no == epoch) & (recent_data.username == creator_name)][
            "current_document"].values[0].strip()
    current_rank = \
        recent_data[(recent_data.round_no == epoch) & (recent_data.username == creator_name)]["position"].values[0]
    tops = recent_data[recent_data["position"] == int(min(recent_data["position"]))]
    top_doc_txt = tops[tops["round_no"] == epoch]["current_document"].values[0]
    top_doc_user = tops[tops["round_no"] == epoch]["username"].values[0]

    # original prompt
    # message = [
    #     {"role": "system",
    #      "content": fr"You are given a query and a candidate document which represents some information need. "
    #                 fr"Your goal is to enhance the candidate document so it will be ranked as high as possible for the "
    #                 fr"query by a search engine. the rank of the most highly ranked document is 1. "
    #                 fr"The ranking function is undisclosed, but you are provided with "
    #                 fr"information about past rankings it induced."}]

    message = [
        {"role": "system",
         # ONE SENTENCE
         # "content": """Edit the candidate document by changing only one sentence, leaving the rest unchanged, to improve its search engine ranking for the candidate query. Aim for the highest rank (1 being the highest). Choose the sentence you believe editing will be most beneficial for this cause and edit it using the black box search engine's past rankings over various queries given as context by the user for guidance. This edit should make the edited document very similar to the candidate, having the same number of sentences and the same or higher amount of words in it. Generate only the edited document, without any additional comments or titles.""".replace("\n","")}]

         # TWO SENTENCES
         # "content": """Edit the candidate document by changing only two sentences, leaving the rest unchanged, to improve its search engine ranking for the candidate query. Aim for the highest rank (1 being the highest). Choose the sentences you believe editing will be most beneficial for this cause and edit them using the black box search engine's past rankings over various queries given as context by the user for guidance. This edit should make the edited document very similar to the candidate, having the same number of sentences and the same or higher amount of words in it. Generate only the edited document, without any additional comments or titles.""".replace("\n", "")}]

         # ALL DOCUMENT
         # "content": """Edit the candidate document to improve its search engine ranking for the candidate query. Aim for the highest rank (1 being the highest). Choose the sentences you believe editing will be most beneficial for this cause and edit them using the black box search engine's past rankings over various queries given as context by the user for guidance. This edit should make the edited document very similar to the candidate, having the same number of sentences and the same or higher amount of words in it. Generate only the edited document, without any additional comments or titles.""".replace("\n", "")}]
         "content": f"""Edit the candidate document to improve its search engine ranking for the candidate query, aiming for the highest rank (1 being the highest). Use the black box search engine's past rankings over various queries, provided as context by the user, to guide your edits. Focus on editing the most impactful sentences to enhance ranking potential. Target an edited document length of around {median_word_count} words, not exceeding 150 words. Ensure the edited document is very similar to the candidate document. Generate only the edited document, without additional comments or titles.""".replace(
             "\n", "")}]

    if bot_info["cand_inc"]:
        message.append({"role": "user",
                        "content": fr"\n\nInput:\n\n- Candidate Query: {recent_data.iloc[0]['query']}\n\n- Candidate Document: {current_doc}\n\n- {epoch_phrasing[epoch]} Ranking: {current_rank}"})
    else:
        message.append({"role": "user",
                        "content": fr"\n\nInput:\n\n- Candidate Query: {recent_data.iloc[0]['query']}\n\n- Candidate Document: {current_doc}"})

    message[0]["content"] = message[0]["content"] + message[1]["content"]
    message = [message[0]]
    x = 1
    if bot_info["cand_inc"]:
        # message.append({"role": "system",
        #                 "content": fr"\n\nquery: {query_string}\n\ncandidate document: {current_doc}\n\n{epoch_phrasing[epoch]} ranking: {current_rank}"})
        all_docs = "\n".join(f"{row['position']}. {row['current_document'].strip()}" for _, row in recent_data[
            (recent_data.round_no == epoch) & (recent_data.username != creator_name)].
                             sort_values("position").iterrows())
    else:
        # message.append({"role": "system",
        #                 "content": fr"\n\nquery: {query_string}\n\ncandidate document: {current_doc}"})
        all_docs = "\n".join(f"\n\n* {doc.strip()}" for doc in recent_data[(recent_data.round_no == epoch) &
                                                                           (recent_data.username != creator_name)].
                             sort_values("position")["current_document"].values)

    if bot_info["method"] == "POW":
        if query_ids[0] == query_id:
            query_ids.pop(0)
            # message_list = [f'Most highly ranked document in the {epoch_phrasing[epoch]} ranking in relation to the '
            #                 f'query mentioned:\n {top_doc_txt}']
            message_list = [
                fr"\n\nquery: {query_string}\n\n* document: {top_doc_txt}\n\n{epoch_phrasing[epoch]} ranking: 1"]

            if bot_info["ex_num"] > 1:
                for i in range(1, bot_info["ex_num"]):
                    top_doc_txt = tops[tops["round_no"] == epoch - i]["current_document"].values[0]
                    # message_list.append(
                    #     f'Most highly ranked document in the {epoch_phrasing[epoch - i]} ranking in relation to the query '
                    #     f'mentioned:\n{top_doc_txt}')
                    message_list.append(fr"\n\n* document: {top_doc_txt}\n\n{epoch_phrasing[epoch - i]} ranking: 1")
            message.append({'role': 'user', 'content': '\n\n'.join(message_list)})

        if query_ids:
            for qid in query_ids:
                message_list = []
                query_string = data[data['query_id'] == qid].iloc[0]['query']
                # message_list.append(f'In the case of observing the query - {query_string}:\n')
                message_list.append(f"\n\nquery: {query_string}")
                tops = data[(data["position"] == int(min(data["position"]))) & (data["query_id"] == qid)]

                for i in range(bot_info["ex_num"]):
                    top_doc_txt = tops[tops["round_no"] == epoch - i]["current_document"].values[0]
                    # message_list.append(f'Most highly ranked document in the {epoch_phrasing[epoch - i]} ranking:\n '
                    #                     f'{top_doc_txt}')
                    message_list.append(fr"\n\n* document: {top_doc_txt}\n\n{epoch_phrasing[epoch - i]} ranking: 1")
                message.append({'role': 'user', 'content': '\n'.join(message_list)})

    elif bot_info["method"] == "PAW":
        if query_ids[0] == query_id:
            query_ids.pop(0)
            if bot_info["doc_type"] == "T":
                top_doc_txt = tops[tops["round_no"] == epoch]["current_document"].values[0]
                if bot_info["cand_inc"]:
                    # message_list = [
                    #     f'Document most highly ranked in the {epoch_phrasing[epoch]} ranking '
                    #     f', surpassing the candidate document, in relation to the query mentioned: \n{top_doc_txt}\n']
                    message_list = [
                        fr"\n\nquery: {query_string}\n\n* document: {top_doc_txt}\n\n{epoch_phrasing[epoch]} ranking: 1"]
                else:
                    rand_doc_row = data[
                        (data["round_no"] == epoch) & (data.username != creator_name) & (data.position != 1) & (
                                data.query_id == query_id)].sample(n=1)
                    rand_doc_txt = rand_doc_row["current_document"].values[0]
                    rand_doc_pos = rand_doc_row["position"].values[0]
                    # message_list = [f'Most highly ranked document and document ranked {rand_doc_pos} respectively in '
                    #                 f'epoch {epoch} in relation to the query mentioned:\n1. {top_doc_txt}\n\n'
                    #                 f'{rand_doc_pos}. {rand_doc_txt}\n']
                    message_list = [
                        fr"\n\nquery: {query_string}\n\n* document: {top_doc_txt}\n\n{epoch_phrasing[epoch]} ranking: 1\n\n"
                        fr"\n\n* document: {rand_doc_txt}\n\n{epoch_phrasing[epoch]} ranking: {rand_doc_pos}"]
            else:
                if bot_info["cand_inc"]:
                    rand_doc_row = data[
                        (data["round_no"] == epoch) & (data.username != creator_name) & (
                                data.query_id == query_id)].sample(
                        n=1)
                    rand_doc_txt = rand_doc_row["current_document"].values[0]
                    rand_doc_pos = rand_doc_row["position"].values[0]
                    # message_list = [
                    #     f'Document ranked {rand_doc_pos} in {epoch_phrasing[epoch]} ranking in relation to the query '
                    #     f'mentioned:\n{rand_doc_txt}\n']
                    message_list = [
                        fr"\n\nquery: {query_string}\n\n* document: {rand_doc_txt}\n\n{epoch_phrasing[epoch]} ranking: {rand_doc_pos}"]
                else:
                    rand_doc_rows = data[(data["round_no"] == epoch) & (data.username != creator_name) &
                                         (data.query_id == query_id)].sample(n=2, replace=False).sort_values(
                        "position")
                    rand_pos = rand_doc_rows['position'].values
                    rand_docs = rand_doc_rows['current_document'].values
                    # message_list = [
                    #     f'Documents ranked {rand_pos[0]} and {rand_pos[1]} respectively in the {epoch_phrasing[epoch]} ranking '
                    #     f'in relation to the query mentioned: \n'
                    #     f'{rand_pos[0]}. {rand_docs[0].strip()}\n\n{rand_pos[1]}. {rand_docs[1].strip()}\n']
                    message_list = [
                        fr"\n\nquery: {query_string}\n\n* document: {rand_docs[0].strip()}\n\n{epoch_phrasing[epoch]} ranking: {rand_pos[0]}\n\n"
                        fr"\n\n* document: {rand_docs[1].strip()}\n\n{epoch_phrasing[epoch]} ranking: {rand_pos[1]}"]

            if bot_info["ex_num"] > 1:
                for i in range(1, bot_info["ex_num"]):
                    if bot_info["doc_type"] == "T":
                        top_doc_txt = tops[tops["round_no"] == epoch - i]["current_document"].values[0].strip()
                        rand_doc_row = data[(data["round_no"] == epoch - i) & (data.position != 1) & (
                                data.query_id == query_id)].sample(n=1)
                        rand_doc_txt = rand_doc_row["current_document"].values[0].strip()
                        rand_doc_pos = rand_doc_row["position"].values[0]
                        # message_list.append(f'Most highly ranked document and document ranked {rand_doc_pos} '
                        #                     f'respectively in the {epoch_phrasing[epoch - i]} ranking '
                        #                     f'in relation to the query mentioned:\n1. {top_doc_txt}\n\n{rand_doc_pos}. '
                        #                     f'{rand_doc_txt}\n')
                        message_list = [
                            fr"\n\nquery: {query_string}\n\n* document: {top_doc_txt}\n\n{epoch_phrasing[epoch - i]} ranking: 1\n\n"
                            fr"\n\n* document: {rand_doc_txt}\n\n{epoch_phrasing[epoch - i]} ranking: {rand_doc_pos}"]
                    else:
                        rand_doc_rows = data[(data["round_no"] == epoch - i) & (data.username != creator_name) &
                                             (data.query_id == query_id)].sample(n=2, replace=False).sort_values(
                            "position")
                        rand_pos = rand_doc_rows['position'].values
                        rand_docs = rand_doc_rows['current_document'].values
                        # message_list.append(
                        #     f'Documents ranked {rand_pos[0]} and {rand_pos[1]} respectively in the {epoch_phrasing[epoch - i]} ranking '
                        #     f'in relation to the query mentioned:\n{rand_pos[0]}. {rand_docs[0].strip()}\n\n{rand_pos[1]}. '
                        #     f'{rand_docs[1].strip()}\n')
                        message_list.append(
                            fr"\n\nquery: {query_string}\n\n* document: {rand_docs[0].strip()}\n\n{epoch_phrasing[epoch - i]} ranking: {rand_pos[0]}\n\n"
                            fr"\n\n* document: {rand_docs[1].strip()}\n\n{epoch_phrasing[epoch - i]} ranking: {rand_pos[1]}")

            message.append({'role': 'user', 'content': '\n\n'.join(message_list)})

        if query_ids:
            for qid in query_ids:
                message_list = []
                query_string = data[data['query_id'] == qid].iloc[0]['query']
                # message_list.append(f'In the case of observing the query - {query_string}:\n')

                for i in range(bot_info["ex_num"]):
                    if bot_info["doc_type"] == "T":
                        tops = data[(data["position"] == int(min(recent_data["position"]))) & (data.query_id == qid)]
                        top_doc_txt = tops[tops["round_no"] == epoch - i]["current_document"].values[0].strip()
                        rand_doc_row = data[(data["round_no"] == epoch - i) & (data.position != 1) & (
                                data.query_id == qid)].sample(n=1)
                        rand_doc_txt = rand_doc_row["current_document"].values[0].strip()
                        rand_doc_pos = rand_doc_row["position"].values[0]
                        # message_list.append(
                        #     f'Most highly ranked document and document ranked {rand_doc_pos} respectively in the {epoch_phrasing[epoch - i]} ranking: \n'
                        #     f'1. {top_doc_txt}\n\n{rand_doc_pos}. {rand_doc_txt}\n')
                        message_list.append(
                            fr"\n\nquery: {query_string}\n\n* document: {top_doc_txt}\n\n{epoch_phrasing[epoch - i]} ranking: 1\n\n"
                            fr"\n\n* document: {rand_doc_txt}\n\n{epoch_phrasing[epoch - i]} ranking: {rand_doc_pos}")
                    else:
                        rand_doc_rows = data[(data["round_no"] == epoch - i) & (data.username != creator_name) &
                                             (data.query_id == qid)].sample(n=2, replace=False).sort_values(
                            "position")
                        rand_pos = rand_doc_rows['position'].values
                        rand_docs = rand_doc_rows['current_document'].values
                        # message_list.append(
                        #     f'Documents ranked {rand_pos[0]} and {rand_pos[1]} respectively in the {epoch_phrasing[epoch - i]} ranking: \n'
                        #     f'{rand_pos[0]}. {rand_docs[0].strip()}\n\n{rand_pos[1]}. {rand_docs[1].strip()}\n')
                        message_list.append(
                            fr"\n\nquery: {query_string}\n\n* document: {rand_docs[0].strip()}\n\n{epoch_phrasing[epoch - i]} ranking: {rand_pos[0]}\n\n"
                            fr"\n\n* document: {rand_docs[1].strip()}\n\n{epoch_phrasing[epoch - i]} ranking: {rand_pos[1]}")
                message.append({'role': 'user', 'content': '\n'.join(message_list)})

    elif bot_info["method"] == "LIW":
        if query_ids[0] == query_id:
            query_ids.pop(0)
            # message_list = [
            #     f'Documents ordered by {epoch_phrasing[epoch]} ranking from highest to lowest in relation to the query '
            #     f'mentioned:\n{all_docs}\n']
            message_list = [
                f'\n\nquery: {query_string}\n\n* documents ordered by {epoch_phrasing[epoch]} ranking from highest to lowest in relation to the query:\n{all_docs}\n\n']

            if bot_info["ex_num"] > 1:
                for i in range(1, bot_info["ex_num"]):
                    all_docs = "\n".join(f"{i + 1}. {doc.strip()}" for i, doc in enumerate(
                        recent_data[recent_data.round_no == epoch - i].sort_values("position")
                        ["current_document"].values))
                    message_list.append(
                        f'\n\n* documents ranked by {epoch_phrasing[epoch - i]} ranking from highest to lowest in relation to the query:\n{all_docs}\n')
            message.append({'role': 'user', 'content': '\n\n'.join(message_list)})

        if query_ids:
            for qid in query_ids:
                message_list = []
                query_string = data[data['query_id'] == qid].iloc[0]['query']
                # message_list.append(f'In the case of observing the query - {query_string}:\n')
                message_list.append(f'\n\nquery: {query_string}\n\n')

                for i in range(bot_info["ex_num"]):
                    all_docs = "\n".join(f"{i + 1}. {doc.strip()}" for i,
                                                                       doc in
                                         enumerate(data[(data.round_no == epoch - i) &
                                                        (data['query_id'] == qid)].sort_values("position")[
                                                       "current_document"].values))
                    message_list.append(
                        f'\n\n* documents ranked by {epoch_phrasing[epoch - i]} ranking from highest to lowest:\n{all_docs}\n')
                message.append({'role': 'user', 'content': '\n'.join(message_list)})

    elif bot_info["method"] == "DYN":
        epochs = [epoch - i for i in range(bot_info["history_len"])]
        strands = bot_info["ex_num"]
        if query_ids[0] == query_id:
            message_list = []
            query_ids.pop(0)
            # top doc strand
            if bot_info["doc_type"] == "T":
                strands -= 1
                cand_doc_rows = data[(data["round_no"].isin(epochs)) & (data.username == top_doc_user) &
                                     (data.query_id == query_id)].sort_values("round_no", ascending=False)
                cand_pos = cand_doc_rows['position'].values
                cand_docs = cand_doc_rows['current_document'].values
                cand_eps = cand_doc_rows['round_no'].values
                # message_string = f'The ranking history of the user that created the most highly ranked document in ' \
                #                  f'relation to the query mentioned is as follows:\n'
                message_string = f'\n\nquery: {query_string}\n\nranking history of the user that ranked 1 in {epoch_phrasing[epoch]} ranking:\n'
                for i in range(len(cand_eps)):
                    # message_string += f'In the {epoch_phrasing[cand_eps[i]]} ranking, ranked {cand_pos[i]}:\n{cand_docs[i].strip()}\n'
                    message_string += fr"\n\n* document: {cand_docs[i].strip()}\n\n{epoch_phrasing[cand_eps[i]]} ranking: {cand_pos[i]}"
                message_list.append(message_string)

            # candidate doc strand
            if bot_info["cand_inc"] and current_rank != 1 and strands > 0:
                strands -= 1
                cand_doc_rows = data[(data["round_no"].isin(epochs[1:])) & (data.username == creator_name) &
                                     (data.query_id == query_id)].sort_values("round_no", ascending=False)
                cand_pos = cand_doc_rows['position'].values
                cand_docs = cand_doc_rows['current_document'].values
                cand_eps = cand_doc_rows['round_no'].values
                # message_string = f'The ranking history of the user that created the candidate document is as follows:\n'
                message_string = f'\n\nquery: {query_string}\n\nranking history of the user that ranked {cand_pos[0]} in {epoch_phrasing[epoch]} ranking:\n'
                for i in range(len(cand_eps)):
                    # message_string += f'In the {epoch_phrasing[cand_eps[i]]} ranking, ranked {cand_pos[i]}:\n{cand_docs[i].strip()}\n'
                    message_string += fr"\n\n* document: {cand_docs[i].strip()}\n\n{epoch_phrasing[cand_eps[i]]} ranking: {cand_pos[i]}"
                message_list.append(message_string)

            if strands > 0:
                cand_users = data[(data["round_no"].isin(epochs)) & ~(data.username.isin([top_doc_user, creator_name]))
                                  & (data.query_id == query_id)]['username'].sample(n=strands, replace=False).tolist()
                cand_doc_rows = data[(data["round_no"].isin(epochs)) & (data.username.isin(cand_users))
                                     & (data.query_id == query_id)].sort_values("round_no", ascending=False)

                for user in cand_users:
                    user_data = cand_doc_rows[cand_doc_rows.username == user]
                    cand_pos = user_data['position'].values
                    cand_docs = user_data['current_document'].values
                    cand_eps = user_data['round_no'].values
                    # message_string = f'The ranking history of the user that created the document ranked {cand_pos[0]} in relation to the ' \
                    #                  f'query mentioned is as follows:\n'
                    message_string = f'\n\nquery: {query_string}\n\nranking history of the user that ranked {cand_pos[0]} in {epoch_phrasing[epoch]} ranking:\n'
                    for i in range(len(cand_eps)):
                        # message_string += f'In the {epoch_phrasing[cand_eps[i]]} ranking, ranked {cand_pos[i]}:\n{cand_docs[i].strip()}\n'
                        message_string += fr"\n\n* document: {cand_docs[i].strip()}\n\n{epoch_phrasing[cand_eps[i]]} ranking: {cand_pos[i]}"
                    message_list.append(message_string)

            message.append({'role': 'user', 'content': '\n\n'.join(message_list)})

        if query_ids:
            for qid in query_ids:
                strands = bot_info["ex_num"]
                query_string = data[data['query_id'] == qid].iloc[0]['query']
                # message_list = [f'In the case of observing the query - {query_string}:\n']
                message_list = []

                if bot_info["doc_type"] == "T":
                    strands -= 1
                    top_doc_user = \
                        data[(data['query_id'] == qid) & (data["position"] == 1) & (data["round_no"] == epoch)][
                            'username'].values[0]

                    cand_doc_rows = data[(data["round_no"].isin(epochs)) & (data.username == top_doc_user) &
                                         (data.query_id == qid)].sort_values("round_no", ascending=False)
                    cand_pos = cand_doc_rows['position'].values
                    cand_docs = cand_doc_rows['current_document'].values
                    cand_eps = cand_doc_rows['round_no'].values
                    # message_string = f'The ranking history of the user that created the most highly ranked document is as follows:\n'
                    message_string = f'\n\nquery: {query_string}\n\nranking history of the user that ranked 1 in {epoch_phrasing[epoch]} ranking:\n'
                    for i in range(len(cand_eps)):
                        # message_string += f'In the {epoch_phrasing[cand_eps[i]]} ranking, ranked {cand_pos[i]}:\n{cand_docs[i].strip()}\n'
                        message_string += fr"\n\n* document: {cand_docs[i].strip()}\n\n{epoch_phrasing[cand_eps[i]]} ranking: {cand_pos[i]}"
                    message_list.append(message_string)

                if strands > 0:
                    cand_users = \
                        data[(data["round_no"].isin(epochs)) & ~(data.username.isin([top_doc_user, creator_name]))
                             & (data.query_id == qid)]['username'].sample(n=strands, replace=False).tolist()
                    cand_doc_rows = data[(data["round_no"].isin(epochs)) & (data.username.isin(cand_users))
                                         & (data.query_id == qid)].sort_values("round_no", ascending=False)

                    for user in cand_users:
                        user_data = cand_doc_rows[cand_doc_rows.username == user]
                        cand_pos = user_data['position'].values
                        cand_docs = user_data['current_document'].values
                        cand_eps = user_data['round_no'].values
                        # message_string = f'The ranking history of the user that created the document ranked {cand_pos[0]} is as follows:\n'
                        message_string = f'fr"\n\nquery: {query_string}\n\nranking history of the user that ranked {cand_pos[0]} in {epoch_phrasing[epoch]} ranking:\n'
                        for i in range(len(cand_eps)):
                            # message_string += f'In the {epoch_phrasing[cand_eps[i]]} ranking, ranked {cand_pos[i]}:\n{cand_docs[i].strip()}\n'
                            message_string += fr"\n\n* document: {cand_docs[i].strip()}\n\n{epoch_phrasing[cand_eps[i]]} ranking: {cand_pos[i]}"
                        message_list.append(message_string)
                message.append({'role': 'user', 'content': '\n'.join(message_list)})

    # original prompt
    # if bot_info["cand_inc"]:
    #     message.append({"role": "user",
    #                     "content": fr"\n\nquery: {recent_data.iloc[0]['query']}\n\ncandidate document:{current_doc}\n\n {epoch_phrasing[epoch]} ranking: {current_rank}"
    #                                fr"\n\ntask:  Enhance the given candidate document to ensure it is the most highly ranked document in the search engine's ranking in relation to the query. generate only the enhanced document, make it coherent and relevant."})
    # else:
    #     message.append({"role": "user",
    #                     "content": fr"\n\nquery: {recent_data.iloc[0]['query']}\n\ncandidate document: {current_doc}"
    #                                fr"\n\ntask:  Enhance the given candidate document to ensure it is the most highly ranked document in the search engine's ranking in relation to the query. generate only the enhanced document, make it coherent and relevant."
    #                                fr"\nEnhanced document:"})

    # message.append({"role": "user",
    #                 "content": f"Enhance the given candidate document to ensure it is the most highly ranked document in the "
    #                            f"search engine\'s ranking.\n\nEnhanced document:"})
    # f"by analyzing the viewed documents to understand why they were "
    # f"ranked the way they did. Utilize the query keywords "
    # f"({get_unique_words(recent_data.iloc[0]['query'])}) naturally within the document as "
    # f"much as possible. Maintain coherence and avoid any meta commentary while generating a "
    # f"fairly long document.
    # pprint(message)
    return message

# # # print demo prompt list
# if __name__ == '__main__':
#     data = pd.read_csv("sandbox_data.csv")
#     # bot_names = ["POW_2211", "POW_2201", "POW_2210", "POW_2200",
#     #              "PAW_2211T", "PAW_2201T", "PAW_2210T", "PAW_2200T", "PAW_2211R", "PAW_2201R", "PAW_2210R", "PAW_2200R",
#     #              "LIW_2211", "LIW_2201", "LIW_2210", "LIW_2200",
#     #              "DYN_2211T2", "DYN_2201T2", "DYN_2210T2", "DYN_2200T2", "DYN_2211R2", "DYN_2201R2", "DYN_2210R2",
#     #              "DYN_2200R2"]
#
#     replacement_dict = {
#         'DYN_1100T2': 'dynamic_tops',
#         'POW_1300': 'pointwise',
#         'DYN_1201R2': 'dynamic_random',
#         'PAW_1210T': 'pairwise_tops',
#         'PAW_1301R': 'pairwise_random',
#         'LIW_1201': 'listwise',
#         'LMBOT1': 'LambdaMART_baseline'
#     }
#
#     bot_names = ["DYN_1100T2" , "POW_1300", "PAW_1301R", "DYN_1201R2", "PAW_1210T", "LIW_1201"]
#
#     bot_cover_df = pd.read_csv("final_options.csv").sort_values(["doc_no", "bot_name"], ascending=True)
#     bot_cover_df = bot_cover_df[bot_cover_df.bot_name.isin(bot_names)]
#     bot_cover_df["bot_group"] = bot_cover_df["bot_name"].apply(lambda x: replacement_dict[x])
#     bot_cover_df = bot_cover_df.sort_values("bot_group")[["bot_group"] + [x for x in bot_cover_df.columns if x != "bot_group"]]
#
#     bot_cover_df.to_csv("current_sandbox_bots.csv", index=False)
#
#     print("*** using query_id: 51, username: 195 ranking 2nd in previous rank for query ***\n\n")
#     for bname in bot_names:
#         print("bot name: ", replacement_dict[bname], ", formal nickname: ", bname, "\n\n")
#
#         res = get_prompt(bname, data, 51, 195)
#         res = [{k: v.replace("\\n", "\n") for k, v in x.items()} for x in res]
#
#         pprint(res)
#         print("\n\n#########################################################\n\n")
