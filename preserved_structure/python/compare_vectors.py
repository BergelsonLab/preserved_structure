import pandas as pd


def make_topn_df(word_topn, image_topn):
    topn = {}

    for name, group in word_topn.groupby("X1"):
        for idx, row in group.iterrows():
            if row.X1 not in topn:
                topn[row.X1] = [[(row.word_other, row.cos_dist, row.zscore)]]
            else:
                topn[row.X1][0].append(
                    (row.word_other, row.cos_dist, row.zscore))

    for name, group in image_topn.groupby("X1"):
        for idx, row in group.iterrows():
            if len(topn[row.X1]) == 1:
                topn[row.X1].append(
                    [(row.img_other, row.cos_dist, row.zscore)])
            else:
                topn[row.X1][1].append(
                    (row.img_other, row.cos_dist, row.zscore))

    topn_df = pd.DataFrame(
        columns=["word", "img_other", "img_cos", "img_zscore", "word_other", "word_cos", "word_zscore"])

    for key, value in topn.items():
        joined = zip(value[0], value[1])

        for x in joined:
            row = {"word": key, "img_other": x[0][0], "img_cos": x[0]
                   [1], "img_zscore": x[0][2], "word_other": x[1][0],
                   "word_cos": x[1][1], "word_zscore": x[1][2]}

            topn_df = topn_df.append(row, ignore_index=True)
    return topn_df
