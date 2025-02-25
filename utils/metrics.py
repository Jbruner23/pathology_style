# dino features
# clip score
# gram matrices
# LPIPS
# mIOU


import pandas as pd

def voting_user_study():
    # Load the spreadsheet with the voting results and the mapping
    # Replace 'voting_results.xlsx' with the actual file name
    df_votes = pd.read_excel('/Users/jannabruner/Downloads/illustrations_responses_updated.xlsx')
    df_mapping = pd.read_excel('/Users/jannabruner/Downloads/keys for survey illustrations.xlsx')


    # Drop the first column and transpose the mapping
    df_mapping = df_mapping.drop(columns='Unnamed: 0').T
    df_mapping.columns = ['A', 'B', 'C']

    df_result = df_votes.copy()

    for column in df_votes.columns:
        # Use each vote value (A, B, C) to index into df_mapping, then assign the appropriate value
        df_result[column] = df_votes[column].apply(lambda x: df_mapping.loc[int(column), x])

    vote_counts = df_result.apply(pd.Series.value_counts, axis=1).fillna(0)

    sum_votes = vote_counts.sum()
    percent = sum_votes / sum_votes.sum()

    count_dict = {}
    for column in df_result.columns:
        count_dict[column] = df_result[column].value_counts().to_dict()

    df_counts = pd.DataFrame({column: df_result[column].value_counts() for column in df_result.columns}).fillna(0).astype(int)
    df_counts.to_excel('/Users/jannabruner/Downloads/count_per_question2.xlsx')
    # Save the results to a new Excel file
    percent.to_excel('/Users/jannabruner/Downloads/mapped_vote_counts.xlsx')

voting_user_study()

    # images_path = "/Users/jannabruner/Documents/research/sign_language_project/video-illustration/images/all_images"
    # output_dir = "/Users/jannabruner/Documents/research/sign_language_project/video-illustration/images/edges/canny_overlay_30_120_w_blur"
    # edges_overley(images_path, output_dir)

    # # Input and output directories
    # input_directory = "/Users/jannabruner/Documents/research/SL_data/signSwiss/data/frames_med_data"  
    # output_directory = "/Users/jannabruner/Documents/research/SL_data/signSwiss/data/edges_frames_med" 


    # # # Run the edge extraction
    # extract_canny_edges(input_directory, output_directory, low_threshold=40, high_threshold=50)