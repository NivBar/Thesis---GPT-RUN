import numpy as np
import pandas as pd
from tqdm import tqdm
from config import current_prompt as cp, bot_cover_df
import xml.etree.ElementTree as ET
import warnings

warnings.filterwarnings("ignore")
import re
from nltk.tokenize import sent_tokenize, word_tokenize

print(f"########## running model {cp} ##########")

if 'asrc' in cp:
    bfu_df = pd.read_csv(f"bot_followup_{cp}.csv").rename({"username": "bot_name"}, axis=1)
    bfu_df = bfu_df["bot_name"].drop_duplicates()
    # bot_cover_df = pd.DataFrame(data=bfu_df, index=bfu_df.index, columns=bot_cover_df.columns)
    bot_cover_df = pd.DataFrame(data=bfu_df, index=bfu_df.index, columns=bot_cover_df.columns).iloc[0].T

    x = 1

def trim_complete_sentences(text):
    # Split the text into sentences using a regex pattern
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)

    # Check if the last sentence is incomplete and remove it if necessary
    if sentences and sentences[-1].strip() and sentences[-1].strip()[-1] not in ['.', '!', '?']:
        sentences.pop()

    word_count = sum(len(sentence.split()) for sentence in sentences)
    if word_count <= 150:
        truncated_text = ' '.join(sentences)
        return truncated_text

    # Check if truncation is necessary
    if sentences:
        word_count = sum(len(sentence.split()) for sentence in sentences)
        truncated_text = " ".join(sentences)

        while word_count > 150:
            if len(sentences) < 2:
                break
            sentences.pop()
            word_count = sum(len(sentence.split()) for sentence in sentences)
            if word_count <= 150:
                truncated_text = ' '.join(sentences)
                return truncated_text
    x = 1

def read_query_xml_to_dict(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    queries_dict = {}

    for query in root.findall('query'):
        number = query.find('number').text
        text_element = query.find('text').text

        prefix = "#combine("
        suffix = ")"
        if text_element.startswith(prefix) and text_element.endswith(suffix):
            text = text_element[len(prefix):-len(suffix)].strip()
        else:
            text = text_element
        queries_dict[int(number)] = text
    return queries_dict


def create_query_xml(queries_dict, filename):
    # Create the root element
    parameters = ET.Element('parameters')

    # Iterate over the dictionary and create query elements
    for query_id, query_text in queries_dict.items():
        # Create a query element
        query_elem = ET.SubElement(parameters, 'query')
        # Create a number element and set its text
        number_elem = ET.SubElement(query_elem, 'number')
        number_elem.text = str(query_id).zfill(3)  # Ensure the id is in the format 'XXX'
        # Create a text element and set its text
        text_elem = ET.SubElement(query_elem, 'text')
        text_elem.text = f"#combine( {query_text} )"

    # Create an ElementTree object and write to the file
    tree = ET.ElementTree(parameters)
    ET.indent(tree, space="  ", level=0)  # Properly format the XML for pretty printing
    tree.write(filename, encoding='utf-8', xml_declaration=True)


texts = pd.read_csv(f"bot_followup_{cp}.csv").sort_values(['query_id', 'round_no'])

# # long texts handling
# long_df = texts[texts['text'].apply(lambda x: len(x.split()) > 150)]
# long_df['length'] = long_df['text'].apply(lambda x: len(x.split()))
# long_df['text'] = long_df['text'].apply(trim_complete_sentences)
# long_df['length_post_edit'] = long_df['text'].apply(lambda x: len(x.split()))
#
# texts['text'] = texts['text'].apply(lambda x: trim_complete_sentences(x) if len(x.split()) > 150 else x)

# texts.to_csv(f"bot_followup_short_{cp}.csv", index=False)

assert texts['text'].apply(lambda x: len(x.split())).max() <= 150

if 'temp' in texts.columns:
    texts = texts[['round_no', 'query_id', 'username', 'creator', 'text', 'temp']]
elif 'creator_orig' in texts.columns:
    texts = texts[['round_no', 'query_id', 'username', 'creator', 'text', 'creator_orig']]
else:
    texts = texts[['round_no', 'query_id', 'username', 'creator', 'text']]
texts.text = texts.text.apply(lambda x: "   \n".join(re.split(r'(?<=[.!?])\s+', x)))

# add temp
if 'temp' in texts.columns:
    # Create a mask for rows where 'creator' is not 'creator'
    mask = texts['creator'] != 'creator'

    # Convert 'temp' to string in the desired format, handling NaN values
    texts['temp_str'] = texts['temp'].apply(lambda x: str(int(x * 100)) if pd.notna(x) else np.nan)

    # Update the 'username' column based on the condition
    # Ensure both columns are treated as strings and concatenate
    # Only update rows where temp is not NaN
    mask &= texts['temp_str'].notna()
    texts.loc[mask, 'username'] = texts.loc[mask, 'username'].astype(str) + "@" + texts.loc[mask, 'temp_str']
    texts.drop(['temp', 'temp_str'], axis=1, inplace=True)

# fix bot docno rounds
for idx, row in texts[texts.creator != 'creator'].iterrows():
    texts.at[idx, 'round_no'] = row.round_no + 1

rounds = texts.round_no.unique()
greg_data = pd.read_csv("tommy_data.csv").rename({"current_document": "text"}, axis=1)
greg_data["creator"] = "creator"
names = greg_data[["query_id", "query"]].set_index('query_id').to_dict()['query']
greg_data = greg_data[[col for col in texts.columns if col in greg_data.columns]]
greg_data = greg_data[greg_data.round_no.isin(rounds)]

# asrc_df = pd.read_csv("bot_followup_asrc.csv")
# df = pd.concat([greg_data, texts, asrc_df])
df = pd.concat([greg_data, texts]).sort_values(['round_no', 'query_id'])
# df = df[df.round_no.isin(rounds_comp)]

if 'creator_orig' in df.columns:
    df["docno"] = df.apply(lambda row: "{}-{}-{}-{}".format('0' + str(row.round_no),
                                                            '0' + str(row.query_id) if row.query_id < 100 else str(
                                                                row.query_id), row.username, row.creator_orig).replace(
        ".0", ""), axis=1)
else:
    df["docno"] = df.apply(lambda row: "{}-{}-{}-{}".format('0' + str(row.round_no),
                                                            '0' + str(row.query_id) if row.query_id < 100 else str(
                                                                row.query_id), row.username, row.creator), axis=1)

df = df[df.round_no != 1]
df = df.dropna().sort_values(['round_no', 'query_id', 'username']).reset_index(drop=True)
try:
    df = pd.merge(df, bot_cover_df.reset_index()[["index", "bot_name"]], left_on='username', right_on='bot_name',
                  how='left').drop("bot_name", axis=1)
except:
    df = pd.merge(df, bot_cover_df.to_frame().T.reset_index()[["index", "bot_name"]], left_on='username',
                  right_on='bot_name', how='left').drop("bot_name", axis=1)

# df = pd.merge(df, bot_cover_df["bot_name"], left_on='username', right_on='bot_name', how='left').drop("bot_name", axis=1)

working_set_docnos = 0
gb_df = df.reset_index().groupby(["round_no", "query_id"])
# gb_df = df.groupby(["round_no", "query_id"])
query_dict = read_query_xml_to_dict(
    './queries_bot_modified_sorted_1.xml')
new_query_dict = {}
comp_dict = {}

for group_name, df_group in tqdm(gb_df):
    creators = df_group[df_group.creator != "creator"].creator.unique()
    bots = df_group[df_group.creator != "creator"].username.unique()
    for creator in creators:
        for bot in bots:
            comp_df = df_group[((df_group.username != creator) & (df_group.creator == "creator")) | (
                    (df_group.username == bot) & (df_group.creator == creator))]
            if comp_df[comp_df.creator != 'creator'].shape[0] == 0:
                continue
            try:
                ind = int(bot_cover_df[bot_cover_df.bot_name == bot].index[0])
            # DROP!
            except:
                ind = 0
            key = str(group_name[0]) + str(group_name[1]).rjust(3, '0') + str(creator).rjust(2, '0') + str(ind).rjust(4,
                                                                                                                      '0')
            comp_df.loc[:, 'Key'] = key
            # DROP
            if 'creator_orig' in comp_df.columns:
                comp_df.loc[:, 'creator'] = comp_df.creator_orig.astype(int).astype(str)
                comp_df.drop('creator_orig', axis=1, inplace=True)
                comp_df.drop('level_0', axis=1, inplace=True)
                comp_df = comp_df[comp_df.username == 'ref']

            comp_dict[key] = comp_df
            new_query_dict[key] = query_dict[group_name[1]]

create_query_xml(new_query_dict,
                 f'/lv_local/home/niv.b/E5_rankings/input_files/queries_{cp}.xml')

result = pd.concat(comp_dict.values(), axis=0)

with open(f"/lv_local/home/niv.b/E5_rankings/input_files/working_set_{cp}.trectext", "w") as f:
    for idx, row in result.sort_values(["Key", "creator"], ascending=(True, True)).iterrows():
        f.write(f"{row.Key} Q0 {row.docno} 0 1.0 summarizarion_task\n")
        working_set_docnos += 1

bot_followup_docnos = 0
with open(f"/lv_local/home/niv.b/E5_rankings/input_files/bot_followup_{cp}.trectext", "w") as f:
    f.write(f"<DATA>\n")
    for idx, row in df.sort_values(['query_id', 'docno'], ascending=[True, False]).iterrows():
        f.write(f"<DOC>\n")
        f.write(f"<DOCNO>{row.docno}</DOCNO>\n")
        f.write(f"<TEXT>\n")
        f.write(f"{row.text.strip()}\n")
        f.write(f"</TEXT>\n")
        f.write(f"</DOC>\n")
        bot_followup_docnos += 1
    f.write(f"</DATA>\n")

x = 1
