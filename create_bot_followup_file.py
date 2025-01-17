import os
import re
import shutil
from pprint import pprint
import pandas as pd
from itertools import product
from tqdm import tqdm

import config
from config import current_prompt as cp, ACTIVE_BOTS, get_prompt, temperatures, get_current_doc, bot_cover_df, using_gpt, rel_round


def prepare_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def divide_df(df, n):
    prepare_directory(f'bot_followup_{cp}')
    df_len = len(df)
    dfs = [df[i * df_len // n:(i + 1) * df_len // n] for i in range(n)]
    for i in range(n):
        dfs[i].to_csv(f'bot_followup_{cp}/part_{i + 1}.csv', index=False)
    print("division of df to", n, "parts completed")

if config.using_e5:
    # greg_data = pd.read_csv("greg_data_e5.csv")
    greg_data = pd.read_csv("tommy_data.csv")
else:
    greg_data = pd.read_csv("greg_data.csv")
greg_data = greg_data[
    (greg_data.round_no == rel_round) & (greg_data.position.between(2, 5))]  # TODO: test setting from the article
bots = ACTIVE_BOTS  if using_gpt else list(config.bot_cover_df.bot_name.unique())# bots using prompt bank

queries = greg_data["query_id"].unique()

# TODO: remove to create for all queries
# queries = [:2]
# greg_data = greg_data[greg_data["query_id"].isin(queries)]

rounds = list(greg_data["round_no"].unique())
if 1 in rounds: rounds.remove(1)
gb_df = greg_data.groupby("query_id")

rows = []
for q_id, df_group in tqdm(gb_df):
    users = df_group["username"].unique()
    for bot, creator in list(product(bots, users)):
        for r in rounds:
            rel_users = df_group[df_group["round_no"] == r]["username"].unique()
            if creator not in rel_users:
                continue
            # if r == 2 and bot in ["NMABOT", "NMTBOT"]:
            #     continue
            rows.append({"round_no": r, "query_id": q_id, "creator": creator, "username": bot, "text": ""})

final_df = pd.DataFrame(rows).sort_values(["round_no", "query_id"], ascending=[False, True]).drop_duplicates()

# try:
#     text_df = pd.read_csv(f"bot_followup_{cp}.csv")
#     df = pd.merge(df, text_df, how='inner', on=['round_no','query_id','creator','username']).drop('text_x', axis=1).rename(columns={'text_y':'text'})
# except:
#     pass
# asrc_df = pd.concat([pd.read_csv(f) for f in glob.glob('/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/bot_followup_asrc*.csv')], ignore_index=True)
# asrc_df = asrc_df[asrc_df.round_no.isin([6,7])]

# merged_df = pd.merge(df, asrc_df, on=['round_no', 'query_id', 'creator'], how='outer', indicator=True)
# merged_df = merged_df[['round_no', 'query_id', 'creator','_merge']].drop_duplicates()[merged_df._merge != 'both']
# merged_df = pd.merge(merged_df, greg_data, left_on=['query_id', 'round_no', 'creator'], right_on=['query_id', 'round_no', 'username'], how='left').sort_values(["round_no","query_id"])
# final_df = pd.concat([df, asrc_df], ignore_index=True).sort_values(["query_id", "username"])
# final_df = pd.merge(final_df,merged_df.drop("_merge",axis=1), indicator=True, how='outer', on=["round_no","query_id","creator"]).query('_merge=="left_only"').drop('_merge', axis=1).rename({"username_x":"username"}, axis=1)
# final_df = final_df[["round_no","query_id","creator","username","text"]]

# # incorporate previous versions' texts
# keep_texts = False
# if keep_texts:
#     filtered_final_df = final_df[final_df.username != 'BOT']
#     filtered_csv_df = pd.read_csv(f"bot_followup_{cp - 1}.csv")[lambda x: x.username != 'BOT']
#     merged_df = pd.merge(filtered_final_df, filtered_csv_df, how='left',
#                          on=['round_no', 'query_id', 'creator', 'username']).drop('text_x', axis=1).rename(
#         columns={'text_y': 'text'})
#     final_df.loc[final_df.username != 'BOT', 'text'] = merged_df['text']

# TODO: testing, delete to use all queries
# Add prompts to csv
final_df = final_df[final_df.query_id.isin(queries)].reset_index(drop=True)
if config.using_e5:
    # greg_data = pd.read_csv("greg_data_e5.csv")
    greg_data = pd.read_csv("tommy_data.csv")
else:
    greg_data = pd.read_csv("greg_data.csv")

for idx, row in tqdm(final_df.iterrows(), total=final_df.shape[0]):
    if config.using_gpt:

        messages = get_prompt(row.username, greg_data, row.creator, row.query_id)
        # content_list = [re.sub(r' {2,}', ' ', re.sub(r'\n{3,}', '\n\n', m["content"].replace("\\n", "\n"))) for m in
        #                 messages]
        # # messages_str = f"<s>[INST] <<SYS>>\n{content_list[0]}\n<</SYS>> [/INST]</s>"
        # messages_str = f"<s>[INST] <<SYS>>\n{content_list[0]}\n<</SYS>> \n\nContext: [/INST]</s>"
        #
        # for block in content_list[1:]:
        #     messages_str += f"<s>[INST]\n{block}\n[/INST]</s>"
        # messages_str = re.sub(r' {2,}', ' ', re.sub(r'\n{3,}', '\n\n', messages_str.replace("\\n", "\n")))
        # messages_str = messages_str.rstrip().rstrip('</s>')
        # # pprint(messages_str)
        # # print("\n\n-------------------------------------------\n\n")
        # # messages_str = messages_str[:messages_str.find("Context") + len("Context")] + \
        # #                messages_str[messages_str.find("Context") + len("Context"):].replace("\n[/INST]</s><s>[INST]\n", "", 1) \
        # #     if "Context" in messages_str and messages_str[messages_str.find("Context"):].count(
        # #     "[/INST]</s><s>[INST]") > 0 else messages_str
        #
        # messages_str = messages_str.replace("[/INST]</s><s>[INST]","")
        #
        # messages_str = messages_str.rsplit("[/INST]", 1)[
        #                       0] + "\n\n---\nEdited Document:\n[/INST]" if "[/INST]" in messages_str else messages_str
        # messages_str = re.sub(r'\n{3,}', '\n\n', messages_str)

        messages_str = str(messages)
        final_df.at[idx, "prompt"] = messages_str

    ref_doc = get_current_doc(greg_data, row.creator, row.query_id)
    final_df.at[idx, "ref_doc"] = ref_doc
    # x = 1

# with open(f"prompt_list_{cp}.txt", 'w') as f:
#     f.writelines(final_df["prompt"].tolist())
#     f.close()
dfs = []
for temp in temperatures:
    final_df["temp"] = temp
    dfs.append(final_df.copy())
    if not config.using_gpt:
        final_df.to_csv(f"./bot_followups/part_{temperatures.index(temp) + 1}.csv", index=False)

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv(f"bot_followup_{cp}.csv", index=False)
# print("final df queries: (", len(final_df.query_id.unique()), ")\n", final_df.query_id.unique())
# divide_df(final_df, 4)
