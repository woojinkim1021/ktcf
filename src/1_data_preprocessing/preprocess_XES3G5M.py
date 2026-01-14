import pandas as pd

def transform_string(input_str):
    numbers = list(map(int, input_str.split(',')))
    output = []
    for num in numbers:
        if num == -1:
            output.append(-1)
        else:
            if 0 in numbers and 1 in numbers:
                output.append(1)
            else:
                output.append(num)
    
    return ','.join(map(str, output))


if __name__ == "__main__":
    xes_test_sequences = pd.read_csv("./data/XES3G5M/test.csv")
    xes_test_sequences['selectmasks'] = xes_test_sequences['responses'].apply(transform_string)

    xes_test_window_file = pd.read_csv("./data/XES3G5M/test_question_window_sequences.csv")


    shortest_len=200
    dfs = [xes_test_sequences, xes_test_window_file]
    for df in dfs:
        cols = df.columns[2:]
        for c in cols:
            df[c] = df[c].apply(
                lambda x: ",".join(x.split(",")[:shortest_len])
            )
    
    xes_test_sequences.to_csv('./data/XES3G5M/test_sequences.csv', index=False)
    xes_test_window_file.to_csv('./data/XES3G5M/test_question_window_sequences.csv', index=False)