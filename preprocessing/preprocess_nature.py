import pandas as pd



if __name__ == '__main__':
    df = pd.read_csv("../data/nature_dataset/nature_original_labels.csv")

    big_cat = []
    small_cat = []
    video_files = []

    small_cat_dict = {
        "palmar pinch" : "pre-pinch",
        "tip pinch": "pre-pinch",

        "prismatic 2 finger" : "pre-prismatic",
        "prismatic 3 finger": "pre-prismatic",
        "prismatic 4 finger": "pre-prismatic",
        "writing tripod" : "not-used",

        "inferior pincer" : "pre-pinch",
        "tripod": "pre-circular",
        "quadpod": "pre-circular",
        "precision disk": "pre-circular",
        "precision sphere": "not-used",

        "large diameter": "pow-cylindric",
        "small diameter": "pow-cylindric",
        "medium wrap": "pow-cylindric",
        "ring": "pow-cylindric",
        "extension type": "not-used",

        "adducted thumb": "pow-oblique",
        "light tool": "pow-oblique",
        "fixed hook": "not-used",
        "palmar": "not-used",
        "index finger extension": "pow-oblique",

        "power disk" : "not-used",
        "power sphere" : "pow-circular",
        "sphere 3 finger" : "pow-circular",
        "sphere 4 finger" : "pow-circular",

        "lift": "not-used",
        "push": "not-used",

        "lateral": "int-lateral",
        "lateral tripod": "int-lateral",

        "stick": "not-used",
        "ventral": "not-used",
        "adduction grip": "pre-pinch",
        "tripod variation": "not-used",
        "parallel extension": "not-used"
    }

    big_cat_dict = {
        "palmar pinch": "pre-prismatic",
        "tip pinch": "pre-prismatic",

        "prismatic 2 finger": "pre-prismatic",
        "prismatic 3 finger": "pre-prismatic",
        "prismatic 4 finger": "pre-prismatic",
        "writing tripod": "not-used",

        "inferior pincer": "pre-prismatic",
        "tripod": "circular",
        "quadpod": "circular",
        "precision disk": "circular",
        "precision sphere": "not-used",

        "large diameter": "pow-prismatic",
        "small diameter": "pow-prismatic",
        "medium wrap": "pow-prismatic",
        "ring": "pow-prismatic",
        "extension type": "not-used",

        "adducted thumb": "pow-prismatic",
        "light tool": "pow-prismatic",
        "fixed hook": "not-used",
        "palmar": "not-used",
        "index finger extension": "pow-prismatic",

        "power disk": "not-used",
        "power sphere": "circular",
        "sphere 3 finger": "circular",
        "sphere 4 finger": "circular",

        "lift": "not-used",
        "push": "not-used",

        "lateral": "int-lateral",
        "lateral tripod": "int-lateral",

        "stick": "not-used",
        "ventral": "not-used",
        "adduction grip": "pre-prismatic",
        "tripod variation": "not-used",
        "parallel extension": "not-used"
    }


    for index, row in df.iterrows():
        video_file = row["Video"][:-3] + "mp4"
        video_files.append(video_file)

        grasp_type = row["Grasp"]
        big_cat.append(big_cat_dict[grasp_type])
        small_cat.append(small_cat_dict[grasp_type])

        print(grasp_type, " - " , big_cat_dict[grasp_type], " - ", small_cat_dict[grasp_type])

    df["Video"] = video_files
    df["BigCategories"] = big_cat
    df["SmallCategories"] = small_cat
    print("Big categories composition ", df.groupby('BigCategories').size())
    print("Small categories composition ", df.groupby('SmallCategories').size())

    df = df.loc[df['SmallCategories'] != 'not-used']
    df.to_csv("../data/nature_dataset/nature_final_labels.csv")

