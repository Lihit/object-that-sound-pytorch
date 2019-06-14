import json
import numpy as np


def get_all_children(category, aso):
    childs = aso[category]["child_ids"]
    childs_names = []
    for child in childs:
        child_name = {}
        child_name["name"] = aso[child]["name"]
        child_name["mark"] = aso[child]["restrictions"]
        if "child_ids" in aso[child]: child_name["children"] = get_all_children(child, aso)
        childs_names.append(child_name)
    if childs_names:
        return childs_names


def get_tree_path(ontology_tree, classname):
    if classname not in ontology_tree:
        print('the audioset donot include the class%s' % classname)
        return []
    path_list = []
    top_class = classname
    while top_class != "Ontology":
        path_list.insert(0, top_class)
        top_class = ontology_tree[top_class]['parents_ids']
    path_list.insert(0, top_class)
    return path_list


def tree_min_distance(class1, class2, ontology_tree):
    class1_path = get_tree_path(ontology_tree, class1)
    class2_path = get_tree_path(ontology_tree, class2)
    inter_count = 0
    for i in range(min(len(class1_path), len(class2_path))):
        if class1_path[i] == class2_path[i]:
            inter_count += 1
        else:
            break
    return len(class1_path) + len(class2_path) - 2 * inter_count


# formating input .json to a .json format readable for this tree visualization code: https://bl.ocks.org/mbostock/4339083

if __name__ == '__main__':
    # 0. read AudioSet Ontology data
    with open('./data_infos/ontology.json') as data_file:
        raw_aso = json.load(data_file)

    # 1. format data as a dictionary
    ## aso["/m/0dgw9r"] > {'restrictions': [u'abstract'], 'child_ids': [u'/m/09l8g', u'/m/01w250', u'/m/09hlz4', u'/m/0bpl036', u'/m/0160x5', u'/m/0k65p', u'/m/01jg02', u'/m/04xp5v', u'/t/dd00012'], 'name': u'Human sounds'}
    aso = {}
    for category in raw_aso:
        tmp = {}
        tmp["name"] = category["name"]
        tmp["restrictions"] = category["restrictions"]
        tmp["child_ids"] = category["child_ids"]
        tmp["parents_ids"] = None
        aso[category["id"]] = tmp

    # 2. fetch higher_categories > ["/m/0dgw9r","/m/0jbk","/m/04rlf","/t/dd00098","/t/dd00041","/m/059j3w","/t/dd00123"]
    for cat in aso:  # find parents
        for c in aso[cat]["child_ids"]:
            aso[c]["parents_ids"] = cat

    # higher_categories = []  # higher_categories are the ones without parents
    for cat in aso:
        if aso[cat]["parents_ids"] == None:
            aso[cat]["parents_ids"] = "Ontology"
    for key in aso:
        print(key, aso[key]["parents_ids"])
    genre2tag_file = './data_infos/genre_tag_map.json'
    with open(genre2tag_file, 'r') as fin:
        genre_tag_dict = json.load(fin)
    tag_genre_dict = {val: key for key, val in genre_tag_dict.items()}
    class1 = "Acoustic guitar"
    class2 = "Drum"
    print(class1, tag_genre_dict[class1])
    print(class2, tag_genre_dict[class2])
    print(tree_min_distance(tag_genre_dict[class1], tag_genre_dict[class2], aso))

    select_genre_file = './data_infos/genre_select_paper.txt'
    with open(select_genre_file, 'r') as fin:
        select_genre_list = [line.replace('\n', '') for line in fin.readlines()]
    select_genre_list = [tag_genre_dict[i] for i in select_genre_list]
    max_rel = np.zeros(shape=(len(select_genre_list), len(select_genre_list)))
    for i in range(len(select_genre_list)):
        for j in range(len(select_genre_list)):
            max_rel[i, j] = tree_min_distance(select_genre_list[i], select_genre_list[j], aso)
    print(max_rel)
    print(max_rel.max())
    max_rel = max_rel.max() - max_rel
    with open('./data_infos/genre_relevance.csv', 'w') as fw:
        fw.write(','.join(select_genre_list) + '\n')
        for i in range(max_rel.shape[0]):
            fw.write(','.join([str(v) for v in max_rel[i, :]]) + '\n')
    # # 3. format ASO properly
    # out_json = {}
    # out_json["name"] = "Ontology"
    # out_json["children"] = []
    # for category in higher_categories:
    #     dict_level1 = {}
    #     dict_level1["name"] = aso[category]["name"]
    #     dict_level1["mark"] = aso[category]["restrictions"]
    #     dict_level1["children"] = get_all_children(category, aso)
    #     out_json["children"].append(dict_level1)
    #
    # # 4. saving output .json
    # with open('./data_infos/ontology_tree.json', 'w') as f:
    #     json.dump(out_json, f, ensure_ascii=False)
