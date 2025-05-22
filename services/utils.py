import re
import pandas as pd
from collections import Counter, defaultdict

def combine_dicts_sum(dicts):
    combined_dict = defaultdict(int)
    
    # Loop through each dictionary and sum the values for common keys
    for d in dicts:
        for key, value in d.items():
            if len(key) > 2:
                combined_dict[key] += value

    # Convert back to a regular dict
    word_freq_dict = Counter(combined_dict)
    return dict(word_freq_dict.most_common(50))

def summary_content_with_llm(summary):
    # script for summary data before passing agent ai
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chat_models import ChatOpenAI
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.docstore.document import Document
    llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=2048)
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    texts = text_splitter.split_text(summary)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
    summary = chain.run(docs)
    return summary

def aggrerate_data(results):

    level = ['Date', 'Brand']

    df = pd.DataFrame(results)

    # Ensure content_summary exists and has no NaNs
    df['content_summary'] = df.get('content_summary', '').fillna('')

    # Extracting sentiment columns using lambda functions
    df['Neutral'] = df['sentiment'].apply(lambda x: x.get('Neutral', 0))
    df['Positive'] = df['sentiment'].apply(lambda x: x.get('Positive', 0))
    df['Negative'] = df['sentiment'].apply(lambda x: x.get('Negative', 0))
    del df['sentiment']

    # st.write(df)
    # Summing engagement, mention, and sentiment values for the same Date and Brand using groupby and lambda
    aggregated_df = df.groupby(level).agg(
        engagement=('engagement', 'sum'),
        mention=('mention', 'sum'),
        Neutral=('Neutral', 'sum'),
        Positive=('Positive', 'sum'),
        Negative=('Negative', 'sum'),
        word=('word', lambda x : combine_dicts_sum(list(x))),
        summary=('content_summary', lambda x : ','.join(list(x))),

    ).reset_index(drop=False)
    aggregated_df['sentiment'] = aggregated_df.apply(
        lambda row: { 'Neutral': int(row['Neutral']),'Positive': int(row['Positive']),'Negative':int( row['Negative'])}, axis=1)
    aggregated_df = aggregated_df.sort_values(by=['Date', 'Brand']).reset_index(drop=False)

    return dataframe2text(aggregated_df)

def dataframe2text(df, level=['Date', 'Brand']):
    # Format the output string
    output = []
    for _, row in df.iterrows():
        level_values = {key: row[key] for key in level}
        engagement = row['engagement']
        mention = row['mention']
        sentiment = row['sentiment']
        summary = row['summary']
        
        output.append(f"On {level_values}, Engagement: {engagement}, Mention: {mention}, Sentiment: {sentiment}")
    
    word = combine_dicts_sum(df['word'].values.tolist())
    
    summary = ','.join(df['summary'].values.tolist())

    # script for summary data before passing agent ai
    summary = summary_content_with_llm(summary)

    output.append(f"word: {word}")
    output.append(f"summary: {summary}")

    return "\n".join(output)

def extract_image_path(response):
    """
    Extracts the image path from the LLM response if it contains a .png file.
    
    Parameters:
        response (str): The response from the LLM.
        
    Returns:
        str: The path to the .png file, or None if not found.
    """
    # Regular expression to find the .png file path
    pattern = r'(/.*?\.png)'
    match = re.search(pattern, response)
    
    if match:
        return match.group(0)  # Return the matched file path
    return None
